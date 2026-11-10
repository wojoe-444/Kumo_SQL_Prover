"""Runtime errors as an outcome of the SMT prover.

BigQuery raises an error for a division by zero, an INT64 overflow, a failed ``CAST`` and the like, and it
promises no evaluation order between ``WHERE``, ``ON`` and the select list: an operation that can fail may run
on a row that a filter would have dropped. Only the branches of ``CASE``, ``IF``, ``COALESCE`` and ``NULLIF``
(which BigQuery evaluates lazily) and the ``SAFE_`` functions keep an operation from running.

So a query's **error exposure** is the set of operations that can fail, each with the condition under which it
fails and the guard that must hold for it to run, over every row of its ``FROM`` (filters do not guard). The
compiler records each as a :class:`Site` and assumes none of them fires when it proves two queries return the same
rows: the proof is equivalence on the databases where neither query errors. This module then compares the two
exposures and says what the rewrite does to errors (:class:`ErrorReport`):

* ``none``: neither query has an operation that can fail, so the proof is exact;
* ``same``: every operation that can fail in one query is covered by one in the other (the same operation, on the
  same row, under a weaker guard): the queries fail on the same databases, up to BigQuery's choice of order;
* ``refines``: every operation that can fail in the rewrite is covered by one in the original, and not the other
  way round: the rewrite never fails where the original succeeds, and the original fails somewhere the rewrite does not;
* ``introduces``: some operation of the rewrite can fail on a database where the original succeeds, shown by a
  concrete database (only when no uninterpreted function stands in for a value of the failing condition);
* ``unknown``: none of the above could be established (a failing condition over an uninterpreted function, a
  solver limit, an operation the original has in a different shape).

One site covers another when there is a map from the covering query's rows to the covered site's rows (rows of the
same table, as the prover maps occurrences for a bag proof) under which the covered site's condition implies
the covering site's: if the rewrite fails on a row tuple, the original fails on the tuple the map picks from the
same database. A map needs every table of the original's ``FROM`` to appear in the rewrite site's ``FROM``,
since the original only fails where every one of its tables has a row.

A ``SUM`` fails on a group of rows, not on one row tuple: it is a site with a :class:`~kumo_sql_prover.smt_group_sums.GroupSum`,
matched with another by the groups' rows and values (``smt_group_sums``), never by a map of single rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools

from . import smt_group_sums
from .solver_lock import bounded_solver

try:  # pragma: no cover
    import z3
except ImportError:  # pragma: no cover
    z3 = None

NONE, SAME, REFINES, INTRODUCES, UNKNOWN = "none", "same", "refines", "introduces", "unknown"
_MAX_MAPS = 2000
_NEAREST = 4  # sites of the same kind and text tried first as the ones that cover a site (``_Comparison.covered``)


@dataclass
class Site:
    """One operation that can raise a runtime error.

    ``fire`` is TRUE on a row exactly when the operation runs on it (its guards hold) and fails; ``scope`` maps the uid of
    every table occurrence of its ``FROM`` (and enclosing ``FROM`` clauses) to the occurrence; ``certain`` is
    ``fire`` when it mentions no uninterpreted function, so a model of it is a database on which the operation really fails.
    """

    kind: str
    text: str
    fire: object
    scope: dict = field(default_factory=dict)
    certain: bool = False
    under: object = None  # implies the failure for a concrete database (``fire`` itself when ``certain``)
    group: object = None  # a SUM over a group of rows (``smt_group_sums.GroupSum``): ``fire`` then only says the group is not empty

    def describe(self) -> str:
        return f"{self.kind}: {self.text}"


@dataclass(frozen=True)
class ErrorReport:
    verdict: str
    detail: str
    sites: tuple[str, ...] = ()
    original_sites: tuple[str, ...] = ()


def uninterpreted_free(formula) -> bool:
    """Whether ``formula`` applies no uninterpreted function (constants are fine)."""

    seen: set = set()
    stack = [formula]
    while stack:
        node = stack.pop()
        if node.get_id() in seen:
            continue
        seen.add(node.get_id())
        if z3.is_app(node):
            if node.num_args() and node.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                return False
            stack.extend(node.children())
    return True


def _outermost(formulas: list) -> list | None:
    """The outermost applications of uninterpreted functions in ``formulas`` (what is inside one is not looked at), or
    ``None`` when the formulas hold something this walk does not read (a quantifier)."""

    seen: set = set()
    found: list = []
    stack = list(formulas)
    while stack:
        node = stack.pop()
        if node.get_id() in seen:
            continue
        seen.add(node.get_id())
        if not z3.is_app(node):
            return None
        if node.num_args() and node.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            found.append(node)
        else:
            stack.extend(node.children())
    return found


def abstracted(formulas: list, known: list | None = None) -> tuple[list, bool] | None:
    """``formulas`` with each outermost application of an uninterpreted function read as a free constant, and whether
    that reading is exact. ``None`` when there is nothing to read so.

    An operation of a long expression fails under a formula that holds every operand before it: the value of
    ``c0 + .. + c38`` is ``Add(Add(..))`` over operands put in a canonical order, and an unknown failure is
    ``fails:Add(those operands)``. Whether the operation can fail at all rarely depends on what is inside.

    * A model of the formulas is a model of the abstract ones (give each constant its term's value). So when the
      abstract formulas cannot hold, neither can the formulas: always.
    * The converse needs the replaced terms to be applications of different functions. Then a model of the abstract
      formulas gives one of the formulas: let each function return its constant's value whatever its arguments are.
      Every application of an uninterpreted function is one of the replaced terms or inside one, so nothing else
      depends on the functions. That is the ``True`` in the result: the two readings can hold in the same cases.
      Two applications of one function (``F(x)`` and ``F(y)``) could be forced equal by their arguments, which the
      constants forget, so the reading is then only the first half.

    ``known`` are the outermost applications of the first ``len(formulas) - 1`` formulas when the caller has them
    already (``_Comparison._context``): only the last formula is then walked."""

    terms = _outermost(formulas) if known is None else _outermost(formulas[-1:])
    if terms is None:
        return None
    seen = {term.get_id() for term in terms}
    terms = terms + [term for term in known or [] if term.get_id() not in seen]
    if not terms:
        return None
    free = [(term, z3.FreshConst(term.sort(), "free")) for term in terms]
    # A formula with no application at all is not changed by the substitution: the known ones are left as they are.
    untouched = len(formulas) - 1 if known == [] else 0
    abstract = formulas[:untouched] + [z3.substitute(formula, *free) for formula in formulas[untouched:]]
    exact = len({term.decl().get_id() for term in terms}) == len(terms) and _outermost(abstract[untouched:]) == []
    return abstract, exact


def _occ_pairs(src, dst) -> list:
    pairs = []
    for name, v in list(src.cols.items()):
        target = dst.col(name)
        pairs.append((v.null, target.null))
        pairs.append((v.val, target.val))
    return pairs


def _maps(src_scope: dict, dst_scope: dict):
    """Every way to send each occurrence of ``src_scope`` to an occurrence of the same table in ``dst_scope``."""

    choices = []
    for occ in src_scope.values():
        options = [d for d in dst_scope.values() if d.table == occ.table]
        if not options:
            return
        choices.append([(occ, d) for d in options])
    for count, combo in enumerate(itertools.product(*choices)):
        if count >= _MAX_MAPS:
            raise _TooMany()
        yield list(combo)


class _TooMany(Exception):
    pass


class _Comparison:
    def __init__(self, prover, facts):
        self.prover = prover
        self.facts = list(facts)
        self.limited = False
        self._contexts: dict[tuple, tuple] = {}

    def _context(self, scope: dict) -> tuple[list, list | None]:
        """What holds of the rows of ``scope`` whatever is asked about them (declared types, constraints, the facts of
        the proof), with the outermost applications of uninterpreted functions in it (``_outermost``)."""

        key = tuple(scope)
        if key not in self._contexts:
            occs = list(scope.values())
            assertions = [*self.prover._typing(occs), *self.prover._constraint_facts(occs), *self.facts]
            self._contexts[key] = (assertions, _outermost(assertions))
        return self._contexts[key]

    def _solver(self, scope: dict, formula, want: str):
        solver = bounded_solver(self.prover.timeout_ms)
        solver.add(*self._context(scope)[0])
        solver.add(formula)
        return solver

    def _abstract(self, scope: dict, formula):
        """A solver for the question ``formula`` about the rows of ``scope`` without what is inside its operands, and
        whether that reading is exact (``abstracted``); ``None`` when the question has nothing to leave out."""

        assertions, known = self._context(scope)
        free = None if known is None else abstracted([*assertions, formula], known)
        if free is None:
            return None
        solver = bounded_solver(self.prover.timeout_ms)
        solver.add(*free[0])
        return solver, free[1]

    def satisfiable(self, site: Site) -> bool | None:
        # "Can this operation run and fail?" without what is inside its operands, when that is the same question. In
        # a sum they are every term before it, each put in a canonical order, and the answer took up to a second for
        # each ``+``.
        abstract = self._abstract(site.scope, site.fire)
        solver = abstract[0] if abstract is not None and abstract[1] else self._solver(site.scope, site.fire, "sat")
        result = solver.check()
        if result == z3.unknown:
            self.limited = True
            return None
        return result == z3.sat

    def coverers(self, site: Site, others: list[Site]):
        """The conditions under which ``others`` fail on the rows of ``site``'s tuple, one per map."""

        terms = []
        for other in others:
            if other.group is not None:  # a sum over a group fails on the group, not on one row tuple
                continue
            for mapping in _maps(other.scope, site.scope):
                pairs = [p for src, dst in mapping for p in _occ_pairs(src, dst)]
                terms.append(z3.substitute(other.fire, *pairs) if pairs else other.fire)
        return terms

    def covered(self, site: Site, others: list[Site], near: int = 0) -> bool:
        """Whether ``site`` fails only where one of ``others`` does.

        Two queries that share an expression record the same operations in the same order, so the sites of the same
        kind and text nearest to position ``near`` are tried first: when they cover ``site``, all of ``others`` do, and
        the rest need not be written over ``site``'s rows (one formula for each of them and each row map, which for a
        long sum is every ``+`` of the other query). Otherwise all of ``others`` are tried, as before."""

        if site.group is not None:
            return smt_group_sums.covered(self, site, others)
        same = [i for i, other in enumerate(others) if other.group is None and other.kind == site.kind and other.text == site.text]
        nearest = [others[i] for i in sorted(same, key=lambda i: abs(i - near))[:_NEAREST]]
        if nearest and len(nearest) < sum(other.group is None for other in others):
            limited = self.limited
            if self._covered_by(site, nearest):
                return True
            self.limited = limited  # a limit met on the way does not count: the full question is asked next
        return self._covered_by(site, others)

    def _covered_by(self, site: Site, others: list[Site]) -> bool:
        try:
            terms = self.coverers(site, others)
        except _TooMany:
            self.limited = True
            return False
        target = z3.Or(*terms) if terms else z3.BoolVal(False)
        question = z3.And(site.fire, z3.Not(target))
        # Each side simplified again: the solver orders the operands of ``and``, ``or`` and ``=`` by when they were
        # made, so the same operand of the other query, written over these rows, is otherwise another term.
        abstract = self._abstract(site.scope, z3.And(z3.simplify(site.fire), z3.Not(z3.Or(*[z3.simplify(term) for term in terms])))) if terms else None
        if abstract is not None:
            # The same question without what is inside the operands: when the two sites hold the same operand terms
            # it is settled here, and "no" is final when the reading is exact.
            answer = abstract[0].check()
            if answer == z3.unsat:
                return True
            if answer == z3.sat and abstract[1]:
                return False
        result = self._solver(site.scope, question, "sat").check()
        if result == z3.unsat:
            return True
        if result == z3.unknown:
            self.limited = True
        return False

    def witness(self, site: Site, others: list[Site]) -> bool:
        """A database on which ``site`` fails and no operation of ``others`` does."""

        if site.group is not None:
            return smt_group_sums.witness(self, site, others)
        if site.under is None or not all(o.certain for o in others) or any(o.group is not None for o in others):
            return False
        try:
            terms = self.coverers(site, others)
        except _TooMany:
            return False
        target = z3.Or(*terms) if terms else z3.BoolVal(False)
        return self._solver(site.scope, z3.And(site.under, z3.Not(target)), "sat").check() == z3.sat


def compare(prover, original: list[Site], rewrite: list[Site], facts=()) -> ErrorReport:
    """What the rewrite does to the errors of the original (see the module docstring)."""

    check = _Comparison(prover, facts)
    live_original = [s for s in original if check.satisfiable(s) is not False]
    live_rewrite = [s for s in rewrite if check.satisfiable(s) is not False]
    described = tuple(s.describe() for s in live_rewrite)
    original_described = tuple(s.describe() for s in live_original)
    if not live_original and not live_rewrite:
        return ErrorReport(NONE, "neither query has an operation that can raise a runtime error")
    new = [s for i, s in enumerate(live_rewrite) if not check.covered(s, live_original, i)]
    lost = [s for i, s in enumerate(live_original) if not check.covered(s, live_rewrite, i)]
    if not new and not lost:
        return ErrorReport(SAME, "the two queries can raise the same runtime errors", described, original_described)
    if not new:
        return ErrorReport(
            REFINES,
            "the rewrite raises no runtime error that the original cannot; the original can fail where the rewrite returns rows: "
            + "; ".join(s.describe() for s in lost[:3]),
            described,
            original_described,
        )
    shown = [s for s in new if check.witness(s, live_original)]
    if shown:
        return ErrorReport(
            INTRODUCES,
            "the rewrite can raise a runtime error on a database where the original returns rows: " + "; ".join(s.describe() for s in shown[:3]),
            described,
            original_described,
        )
    return ErrorReport(
        UNKNOWN,
        "the rewrite has an operation that can fail and the original has no matching one" + (" (a solver limit was reached)" if check.limited else ""),
        described,
        original_described,
    )
