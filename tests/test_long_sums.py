"""A sum costs what its length warrants (workstream issue 2, rows 10 and 22).

``c0 + c1 + .. + cn`` is n nodes inside each other, and each ``+`` is an operation that can overflow. Two things made
such a query slow whatever ``timeout_ms`` said:

* Each node asked for the type of everything below it, twice (``_class_of`` for its overflow,
  ``smt_numeric.exact_class`` for its scale), so compiling four hundred terms took twenty seconds before the solver
  and its time limit started. The compiler now keeps the type of an arithmetic node while the operands of the
  outermost one are compiled (``_Compiler._operands``).
* Each ``+`` cost two solver checks about a formula as deep as the sum before it ("can it run and fail?", "does the
  other query fail there too?"): forty columns compared with themselves took a minute or two. Both questions are now
  asked without what is inside the operands where that cannot change the answer (``smt_errors.abstracted``, and
  ``_may_fail`` says "the operands are known" of the operands as written), and the second is first put to the
  nearest operations of the same kind and text (``_Comparison.covered``)."""

import pytest

z3 = pytest.importorskip("z3")

from kumo_sql_prover import smt_equivalence, smt_errors, smt_numeric  # noqa: E402
from kumo_sql_prover.smt_equivalence import SmtStatus, _Compiler, prove_equivalent_smt  # noqa: E402

TERMS = 60


def _columns(count: int = TERMS) -> list[str]:
    return [f"c{i}" for i in range(count)]


def _sum(columns: list[str], operator: str = "+") -> str:
    return "SELECT " + f" {operator} ".join(columns) + " AS s FROM t"


def _compile(sql: str, columns: list[str], kind: str | None, monkeypatch) -> dict[str, int]:
    """Compile ``sql`` and count how often a type was asked for."""

    asked = {"class": 0, "exact": 0}
    class_of, exact_class = _Compiler._class_of, smt_numeric.exact_class

    def counted_class_of(self, *args):
        asked["class"] += 1
        return class_of(self, *args)

    def counted_exact_class(*args):
        asked["exact"] += 1
        return exact_class(*args)

    monkeypatch.setattr(_Compiler, "_class_of", counted_class_of)
    monkeypatch.setattr(smt_numeric, "exact_class", counted_exact_class)
    types = {"t": dict.fromkeys(columns, kind)} if kind else None
    _Compiler({"t": columns}, False, "bigquery", types).compile(sql)
    return asked


@pytest.mark.parametrize("kind", [None, "INT64", "NUMERIC", "FLOAT64"])
@pytest.mark.parametrize("operator", ["+", "-", "*"])
def test_the_types_asked_for_grow_with_the_length_of_a_sum_not_its_square(kind, operator, monkeypatch):
    columns = _columns()
    asked = _compile(_sum(columns, operator), columns, kind, monkeypatch)
    # Before: about TERMS squared of each (3,600 here). A node now asks for each operand a fixed number of times.
    assert asked["class"] <= 12 * TERMS, asked
    assert asked["exact"] <= 12 * TERMS, asked


SCHEMA = {"t": ["x", "y", "z", "n", "m", "f", "s"], "u": ["x", "k"]}
TYPES = {"t": {"x": "INT64", "y": "INT64", "z": "INT64", "n": "NUMERIC", "m": "BIGNUMERIC", "f": "FLOAT64", "s": "STRING"}, "u": {"x": "NUMERIC", "k": "INT64"}}

# Shapes where the same node could be read in more than one scope, or where the type of a name depends on the scope.
SCOPED = [
    "SELECT x + y + z + 1 AS a, x + y + z + 1.5 AS b FROM t",
    "SELECT n + m + x + 2 AS a, n * 2 + n * 3 - m AS b FROM t",
    "SELECT n + 0.1 + n + 0.25 + x AS a FROM t",
    "SELECT x + y AS x, x + y + z AS w FROM t",
    "SELECT x + y + z AS w FROM t ORDER BY w + x + y + z",
    "SELECT x + y AS w, SUM(n + m + n) AS total FROM t GROUP BY x + y HAVING SUM(n + m + n) + SUM(x + y + z) > 3",
    "SELECT x + y + (SELECT MAX(u.x + u.k + u.x) FROM u WHERE u.k = t.x + t.y + t.z) AS a FROM t",
    "SELECT t.x + u.x + t.y + u.k + t.n AS a FROM t JOIN u ON t.x + t.y + t.z = u.k + u.k + u.k",
    "SELECT CASE WHEN x + y + z > 0 THEN n + m + n ELSE x + y + z END + n + m AS a FROM t",
    "SELECT -(x + y + z) + -(n + m + n) + ROUND(n + n + n, 2) + CAST(x + y + z AS NUMERIC) AS a FROM t",
    "SELECT x + y + z AS a FROM t UNION ALL SELECT x + k + x FROM u",
    "WITH w AS (SELECT x + y + z AS x, n + n + n AS n FROM t) SELECT x + n + x + n FROM w",
    "SELECT (x + y + z) / (n + m + n) + f + f + f AS a, x - y - z - n AS b, x * y * z * n AS c FROM t",
    "SELECT * REPLACE (x + y + z AS x) FROM t WHERE x + y + z > n + m + n",
]


@pytest.fixture
def verified(monkeypatch):
    """Work out every kept type again where it is used; count the uses and keep any answer that differs."""

    record = {"uses": 0, "stale": []}
    verify = _Compiler._verify_kept

    def checking(self, kind, e, env, agg, aliases, kept):
        record["uses"] += 1
        try:
            verify(self, kind, e, env, agg, aliases, kept)
        except AssertionError as error:
            record["stale"].append(str(error))

    monkeypatch.setattr(smt_equivalence, "_VERIFY_KEPT_CLASSES", True)
    monkeypatch.setattr(_Compiler, "_verify_kept", checking)
    return record


@pytest.mark.parametrize("types", [TYPES, None], ids=["typed", "untyped"])
def test_a_kept_type_is_the_type_found_by_working_it_out_again(types, verified):
    for sql in SCOPED:
        for exact in (False, True):
            compiler = _Compiler(SCHEMA, exact, "bigquery", types)
            try:
                compiler.compile(sql)
            except smt_equivalence.Unsupported:
                pass  # a shape the compiler declines is still compiled up to that point
            assert not compiler._operand_depth and not compiler._kept_classes, sql  # nothing outlives the expression
    assert not verified["stale"], verified["stale"][:3]
    assert verified["uses"] > 50  # the shapes above do reach the kept types


def test_nothing_is_kept_after_a_failure_inside_an_operand():
    compiler = _Compiler(SCHEMA, False, "bigquery", TYPES)
    with pytest.raises(smt_equivalence.Unsupported):
        compiler.compile("SELECT x + y + (z + no_such_column) + n FROM t")
    assert not compiler._operand_depth and not compiler._kept_classes


@pytest.mark.parametrize(
    "left, right, status",
    [
        ("SELECT n + m + n + 1 AS a FROM t", "SELECT n + m + n + 1 AS a FROM t", SmtStatus.PROVEN_EQUIVALENT),
        ("SELECT n * 2 + n * 3 AS a FROM t", "SELECT n * 3 + n * 2 AS a FROM t", SmtStatus.PROVEN_EQUIVALENT),
        ("SELECT n + n + 1 AS a FROM t", "SELECT n + n + 2 AS a FROM t", SmtStatus.NOT_EQUIVALENT),
        ("SELECT x + y + z AS a FROM t WHERE x + y + z > 0", "SELECT x + y + z AS a FROM t WHERE NOT (x + y + z <= 0)", SmtStatus.PROVEN_EQUIVALENT),
    ],
)
def test_verdicts_on_sums_are_what_they_were(left, right, status, verified):
    assert prove_equivalent_smt(left, right, schema=SCHEMA, types=TYPES, timeout_ms=4000).status == status
    assert not verified["stale"], verified["stale"][:3]


# --- what the error comparison asks about each operation that can fail ----------------------------


def _counted(monkeypatch) -> dict[str, int]:
    """Count the questions the error comparison puts to the solver in full, the ones it reads without the operands'
    insides, and the formulas it writes over another site's rows."""

    count = {"full": 0, "exact": 0, "written": 0}
    coverers, solver, abstract = smt_errors._Comparison.coverers, smt_errors._Comparison._solver, smt_errors._Comparison._abstract

    def counted_coverers(self, site, others):
        terms = coverers(self, site, others)
        count["written"] += len(terms)
        return terms

    def counted_solver(self, *args):
        count["full"] += 1
        return solver(self, *args)

    def counted_abstract(self, scope, formula):
        result = abstract(self, scope, formula)
        if result is not None:
            count["exact"] += result[1]
            assert not any(z3.is_app(node) and node.num_args() and node.decl().kind() == z3.Z3_OP_UNINTERPRETED for node in _nodes(result[0].assertions()))
        return result

    monkeypatch.setattr(smt_errors._Comparison, "coverers", counted_coverers)
    monkeypatch.setattr(smt_errors._Comparison, "_solver", counted_solver)
    monkeypatch.setattr(smt_errors._Comparison, "_abstract", counted_abstract)
    return count


def _nodes(formulas):
    seen, stack = set(), list(formulas)
    while stack:
        node = stack.pop()
        if node.get_id() not in seen:
            seen.add(node.get_id())
            stack.extend(node.children())
            yield node


@pytest.mark.parametrize("kind", [None, "INT64", "FLOAT64"])
@pytest.mark.parametrize("operator", ["+", "*", "-"])
def test_a_sum_compared_with_itself_is_not_asked_about_its_operands(operator, kind, monkeypatch):
    count = _counted(monkeypatch)
    columns = _columns(8)
    sql = _sum(columns, operator)
    types = {"t": dict.fromkeys(columns, kind)} if kind else None
    result = prove_equivalent_smt(sql, sql, schema={"t": columns}, types=types, timeout_ms=4000)
    assert result.status is SmtStatus.PROVEN_EQUIVALENT
    assert result.errors.verdict == smt_errors.SAME
    operations = 2 * (len(columns) - 1)
    assert len(result.errors.sites) + len(result.errors.original_sites) == operations
    # "Can it run and fail?" for each operation, exactly, without what is inside its operands. "Does the other query
    # fail there too?" is settled the same way unless the solver wrote the two operands differently (which depends on
    # what the process made before), and then put in full once: at most one full question for each operation, where
    # there were two.
    assert count["exact"] >= operations
    assert count["full"] <= operations
    # Against the operations nearest to it, not all seven of the other query.
    assert count["written"] <= 2 * smt_errors._NEAREST * operations


def test_an_operation_both_queries_have_is_matched_whatever_else_changed(monkeypatch):
    count = _counted(monkeypatch)
    left = "SELECT x + y AS s, x * z AS p FROM t WHERE z > 1"
    right = "SELECT x + y AS s, x * z AS p FROM t WHERE NOT z <= 1"
    result = prove_equivalent_smt(left, right, schema={"t": ["x", "y", "z"]}, timeout_ms=4000)
    assert result.status is SmtStatus.PROVEN_EQUIVALENT and result.errors.verdict == smt_errors.SAME
    assert count["full"] == 0 and count["written"] == 4


def test_an_operation_the_nearest_ones_do_not_cover_is_put_to_all_of_them(monkeypatch):
    count = _counted(monkeypatch)
    # Every ``/`` has the same kind; the text is cut at 120 characters, so the six on the right read the same.
    long = " + ".join(["x"] * 32)
    left = f"SELECT ({long}) / y AS q FROM t"
    right = "SELECT " + ", ".join(f"({long}) / {divisor} AS q{i}" for i, divisor in enumerate(["z", "z", "z", "z", "z", "y"])) + " FROM t"
    compiler = _Compiler({"t": ["x", "y", "z"]}, False, "bigquery", {"t": dict.fromkeys("xyz", "INT64")})
    compiler.compile(left)
    mine = [site for site in compiler.sites if site.kind == "division by zero"]
    start = len(compiler.sites)
    compiler.compile(right)
    theirs = [site for site in compiler.sites[start:] if site.kind == "division by zero"]
    assert len(mine) == 1 and len(theirs) == 6 and {site.text for site in theirs} == {mine[0].text}
    check = smt_errors._Comparison(smt_equivalence._Prover(4000, None, compiler.types), compiler.typed_facts())
    count["written"] = 0
    assert check.covered(mine[0], theirs, near=0)  # only the sixth divides by y: found by asking all six
    assert count["written"] == smt_errors._NEAREST + 6
    assert not check.covered(mine[0], theirs[:5], near=0) and not check.limited


def _same_answer(formulas: list, abstract: list) -> object:
    plain, free = z3.Solver(), z3.Solver()
    plain.add(*formulas)
    free.add(*abstract)
    assert plain.check() == free.check()
    return plain.check()


def test_an_operand_read_as_a_free_constant_gives_the_same_answer_when_its_function_is_applied_once():
    x, y = z3.Ints("x y")
    add = z3.Function("Add/2", z3.IntSort(), z3.IntSort(), z3.IntSort())
    fails = z3.Function("fails:Add/1", z3.IntSort(), z3.BoolSort())
    deep = add(add(add(x, y), y), x)  # the function inside its own operand: only the outermost application counts
    for formulas, expected in (
        ([y < 3, z3.And(x > 0, fails(deep))], z3.sat),
        ([y < 3, z3.And(x > 0, x < 0, fails(deep))], z3.unsat),
        ([z3.And(deep + x > 2**63, x < 5), deep > 9], z3.sat),
        ([z3.And(deep + x > 2**63, x < 5), deep > 9, x > 7], z3.unsat),
        ([z3.And(deep > 5, fails(deep)), z3.Not(fails(deep))], z3.unsat),  # one term in two places: one constant
    ):
        abstract, exact = smt_errors.abstracted(formulas)
        assert exact and not smt_errors._outermost(abstract)
        assert _same_answer(formulas, abstract) == expected


def test_two_applications_of_one_function_are_only_read_one_way():
    x, y = z3.Ints("x y")
    fails = z3.Function("fails:Add/1", z3.IntSort(), z3.BoolSort())
    # Cannot hold as written (x = y, so the two applications agree). Two free constants forget that.
    formulas = [z3.Not(fails(y)), z3.And(x == y, fails(x))]
    abstract, exact = smt_errors.abstracted(formulas)
    assert not exact
    plain, free = z3.Solver(), z3.Solver()
    plain.add(*formulas)
    free.add(*abstract)
    assert plain.check() == z3.unsat and free.check() == z3.sat
    # The half that always holds: what cannot hold without the insides cannot hold with them.
    impossible = [z3.And(fails(x), x > 3), z3.And(fails(y), x < 2)]
    abstract, exact = smt_errors.abstracted(impossible)
    assert not exact
    assert _same_answer(impossible, abstract) == z3.unsat


def test_a_question_with_no_unknown_function_is_left_alone():
    x = z3.Int("x")
    for formulas in ([x > 9], [x > 9, z3.Bool("fails:now")]):
        assert smt_errors.abstracted(formulas) is None
    assert smt_errors.abstracted([z3.ForAll([x], x > 9)]) is None  # not read


def test_what_is_known_of_the_rows_is_walked_once():
    x, y = z3.Ints("x y")
    add = z3.Function("Add/2", z3.IntSort(), z3.IntSort(), z3.IntSort())
    rank = z3.Function("rank", z3.IntSort(), z3.IntSort())
    context, question = [x > 0, y > 0], add(x, y) > 5
    abstract, exact = smt_errors.abstracted([*context, question], [])
    assert exact and abstract[0] is context[0] and abstract[1] is context[1] and not smt_errors._outermost(abstract)
    # An application in what is known is replaced there too, and counts for exactness.
    context = [rank(x) > 0]
    known = smt_errors._outermost(context)
    abstract, exact = smt_errors.abstracted([*context, rank(x) + add(x, y) > 5], known)
    assert exact and not smt_errors._outermost(abstract)
    abstract, exact = smt_errors.abstracted([*context, rank(y) > 5], known)
    assert not exact and not smt_errors._outermost(abstract)


def test_saying_the_operands_are_known_does_not_need_their_order():
    """``_may_fail`` on values in canonical order, with the operands as written: the same condition, shorter."""

    compiler = _Compiler({"t": ["x", "y"]}, False, "bigquery")
    V = smt_equivalence._value_sort()
    a = smt_equivalence._Val(z3.Bool("a#null"), z3.Const("a", V))
    b = smt_equivalence._Val(z3.Bool("b#null"), z3.Const("b", V))
    swap = smt_equivalence._pair_lt(b, a)
    pair = [
        smt_equivalence._Val(z3.If(swap, b.null, a.null), z3.If(swap, b.val, a.val)),
        smt_equivalence._Val(z3.If(swap, a.null, b.null), z3.If(swap, a.val, b.val)),
    ]
    long, short = compiler._may_fail("Add:None", pair), compiler._may_fail("Add:None", pair, operands=[a, b])
    solver = z3.Solver()
    solver.add(long != short)
    assert solver.check() == z3.unsat
    assert not long.arg(0).eq(short.arg(0)) and short.arg(0).eq(z3.And(z3.Not(a.null), z3.Not(b.null)))
