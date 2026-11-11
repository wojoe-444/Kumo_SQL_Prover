"""Two window computations over inputs the prover shows equal are read over one input.

A select with window functions is kept whole by the prover and matched by its text
(``smt_equivalence.WINDOW_SOURCE_ASSUMPTION``: window functions over the same input with the same text give the
same values). So two spellings of the rows a window reads never met, however small the difference:

    SELECT p.title, ROW_NUMBER() OVER (PARTITION BY p.kind ORDER BY p.day DESC) FROM p JOIN u ON p.owner = u.id
    SELECT t.title, ROW_NUMBER() OVER (PARTITION BY t.kind ORDER BY t.day DESC) FROM (SELECT * FROM p, u WHERE p.owner = u.id) AS t

``unify(left, right, prove=...)`` works on the two normalized queries after the prover has left them unproven.

**Step 1, an identity on one query.** A select ``W`` with window functions

    SELECT e1 AS n1, .., f(a1) OVER (PARTITION BY a2 ORDER BY a3) AS v FROM <sources> WHERE .. GROUP BY .. HAVING ..

is

    SELECT i.c1 AS n1, .., f(i.c2) OVER (PARTITION BY i.c3 ORDER BY i.c4) AS v
    FROM (SELECT e1 AS c1, a1 AS c2, a2 AS c3, a3 AS c4 FROM <sources> WHERE .. GROUP BY .. HAVING ..) AS i

where the inner select ``J`` (the window's *input*) lists every select item without a window and every expression a
window reads (function arguments, ``PARTITION BY`` and ``ORDER BY`` keys). Window functions are evaluated over the
rows left after ``FROM``, ``WHERE``, ``GROUP BY`` and ``HAVING`` (one row per group in a grouped select), and ``J``
returns exactly those rows, one output row each, holding the values the outer select reads. The outer select then
computes the same windows over the same bag of rows with the same partition keys, sort keys and arguments.
Duplicates stay duplicates (``J`` has no ``DISTINCT``), NULL keys are the same NULLs, and with no rows both are empty
(a global aggregate keeps its one row: every aggregate ``W`` evaluates before its windows is in an item of ``J`` or
in the ``HAVING`` that ``J`` keeps).

**Step 2, a proof.** For a windowed select of each query, ``prove(J_left, J_right)`` asks the prover itself whether
the two inputs return the same bag of rows, column by column in the listed order. The columns are listed in the
order in which the select that reads ``W`` uses them, so the two queries list them alike when they read alike.

**Step 3, a replacement.** When the proof succeeds, both selects are written in the step-1 form and the right one
reads ``J_left`` in place of ``J_right``. That changes nothing: a derived table in ``FROM`` is used only through
its rows and column positions, and those are proven equal. The prover is then run again on the two texts, in
which the windows now read literally the same input.

Conditions checked (the select is left alone otherwise):

* neither query has a ``WITH``; the windowed select is the query itself or a derived table reached from it through
  derived tables only (never inside a scalar, ``EXISTS`` or ``IN`` subquery or a ``LATERAL``), so no column in it can
  name a row of an enclosing query and the input means the same wherever it is written;
* the select has only a select list, ``FROM``, joins, ``WHERE``, ``GROUP BY`` and ``HAVING``: no ``DISTINCT``,
  ``QUALIFY``, named ``WINDOW``, ``ORDER BY``, ``LIMIT`` or ``OFFSET``;
* every select item has a name of its own and is either free of windows or reads columns only inside its windows;
* a window is a plain function call (optionally ``IGNORE`` / ``RESPECT NULLS``, ``DISTINCT`` inside) with value
  arguments: no ``FILTER``, no ``ORDER BY`` inside the call, no column in the frame bounds;
* nothing in the select can differ between two evaluations (``RANDOM()``, an unknown function), and no input
  expression holds a subquery;
* every qualified column of the select names a source declared inside it.

Nothing is assumed beyond what the prover already assumes for windows: a sub-proof's assumptions are passed on.
At most ``MAX_PROOFS`` sub-proofs are tried for a pair, on inputs of at most ``MAX_CHARS`` characters.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from .ast_utils import faithful_sql

MAX_PROOFS = 4
MAX_CHARS = 6000
INPUT_ALIAS = "kqwi"
_SELECT_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having", "kind"})
_INPUT_ARGS = ("from_", "from", "joins", "where", "group", "having")
_UNSTABLE = (exp.Rand, exp.Randn, exp.Uuid, exp.CurrentTimestamp, exp.CurrentDate, exp.CurrentTime, exp.CurrentDatetime, exp.Anonymous, exp.UDTF, exp.Unnest, exp.Explode, exp.Lateral)
_NOT_A_VALUE = (exp.Order, exp.Ordered, exp.Query, exp.Subquery, exp.Window, exp.Filter, exp.Lambda, exp.Star, exp.WithinGroup)


def _own_windows(select: exp.Select) -> list[exp.Window]:
    return [w for w in select.find_all(exp.Window) if w.find_ancestor(exp.Select) is select]


def _reads_rows(node: exp.Expression) -> bool:
    return any(isinstance(n, (exp.Column, exp.AggFunc)) for n in node.walk())


def _reachable_through_derived_tables(select: exp.Select) -> bool:
    """Whether ``select`` is the query or a derived table under derived tables only: no enclosing row is visible to it."""

    node = select
    while node.parent is not None:
        derived = node.parent
        if not isinstance(derived, exp.Subquery) or any(value for key, value in derived.args.items() if key not in ("this", "alias")):
            return False  # a set operation branch, a parenthesized query with its own tail
        holder = derived.parent
        if not isinstance(holder, (exp.From, exp.Join)) or not isinstance(holder.parent, exp.Select):
            return False  # a scalar, EXISTS or IN subquery, a LATERAL
        node = holder.parent
    return True


class _Split:
    """The step-1 form of one windowed select: the input ``J`` and the outer select over it."""

    def __init__(self, select: exp.Select, input_select: exp.Select, outer: exp.Select):
        self.select = select
        self.input = input_select
        self.outer = outer

    def skeleton(self) -> tuple:
        return tuple(sorted((item.this if isinstance(item, exp.Alias) else item).sql() for item in self.outer.expressions)), len(self.input.expressions)

    def tables(self) -> tuple:
        return tuple(sorted(t.name.lower() for t in self.input.find_all(exp.Table)))


def _reader_rank(select: exp.Select) -> dict[str, int]:
    """Output name -> the position at which the select that reads ``select`` first uses it."""

    holder = select.parent
    reader = holder.find_ancestor(exp.Select) if isinstance(holder, exp.Subquery) else None
    if reader is None or not holder.alias:
        return {}
    alias = holder.alias.lower()
    rank: dict[str, int] = {}
    parts = list(reader.expressions) + [reader.args.get(key) for key in ("where", "group", "having", "qualify", "order")] + list(reader.args.get("joins") or [])
    for part in parts:  # in the order the reader is written: its select list first
        for column in part.find_all(exp.Column, bfs=False) if part is not None else ():
            if column.find_ancestor(exp.Select) is reader and column.table.lower() == alias:
                rank.setdefault(column.name.lower(), len(rank))
    return rank


def split(select: exp.Select) -> _Split | None:
    """The step-1 form of ``select``, or None when a condition of the module docstring fails."""

    if any(value for key, value in select.args.items() if key not in _SELECT_ARGS):
        return None
    windows = _own_windows(select)
    if not windows or (select.args.get("from_") or select.args.get("from")) is None:
        return None
    if any(isinstance(n, _UNSTABLE) for n in select.walk()):
        return None
    declared = {n.alias_or_name.lower() for n in select.walk() if isinstance(n, (exp.Table, exp.Subquery)) and n.alias_or_name}
    for column in select.find_all(exp.Column):
        if isinstance(column.this, exp.Star) or column.args.get("db") or (column.table and column.table.lower() not in declared):
            return None
    names = [(item.alias_or_name or "").lower() for item in select.expressions]
    if "" in names or len(set(names)) != len(names) or any(isinstance(item, exp.Star) for item in select.expressions):
        return None
    rank = _reader_rank(select)
    order = sorted(range(len(names)), key=lambda i: (rank.get(names[i], len(rank)), i))
    copy = select.copy()
    items = list(copy.expressions)
    atoms: dict[str, int] = {}
    values: list[exp.Expression] = []

    def atom(node: exp.Expression) -> exp.Column | None:
        if any(isinstance(n, (exp.Window, exp.Query, exp.Subquery, exp.Exists)) for n in node.walk()):
            return None
        text = node.sql()
        if text not in atoms:
            atoms[text] = len(values)
            values.append(node.copy())
        return exp.column(f"kqi{atoms[text]}", table=INPUT_ALIAS)

    def replace(node: exp.Expression) -> bool:
        column = atom(node)
        if column is None:
            return False
        node.replace(column)
        return True

    new_items: dict[int, exp.Expression] = {}
    for index in order:
        item = items[index]
        value = item.this if isinstance(item, exp.Alias) else item
        own = [w for w in value.find_all(exp.Window)]
        if not own:
            column = atom(value)
            if column is None:
                return None
            new_items[index] = exp.alias_(column, item.alias_or_name, quoted=False)
            continue
        holder = exp.Paren(this=value.copy())
        for window in list(holder.find_all(exp.Window)):
            if any(value for key, value in window.args.items() if key not in ("this", "partition_by", "order", "spec", "over")):
                return None
            spec = window.args.get("spec")
            if spec is not None and _reads_rows(spec):
                return None
            call = window.this
            while isinstance(call, (exp.IgnoreNulls, exp.RespectNulls)):
                call = call.this
            if not isinstance(call, exp.Func) or isinstance(call, _NOT_A_VALUE):
                return None
            arguments: list[exp.Expression] = []
            for argument in call.args.values():
                for part in argument if isinstance(argument, list) else [argument]:
                    if not isinstance(part, exp.Expression):
                        continue
                    if isinstance(part, exp.Distinct):
                        if part.args.get("on") is not None:
                            return None
                        arguments.extend(part.expressions)
                    else:
                        arguments.append(part)
            order_by = window.args.get("order")
            if order_by is not None and (type(order_by) is not exp.Order or any(type(o) is not exp.Ordered for o in order_by.expressions)):
                return None
            keys = list(window.args.get("partition_by") or []) + [o.this for o in (order_by.expressions if order_by is not None else [])]
            for part in arguments + keys:
                if isinstance(part, _NOT_A_VALUE) and not (isinstance(part, exp.Star) and part in arguments):
                    return None
                if _reads_rows(part) and not replace(part):
                    return None
        if any(isinstance(n, exp.Column) and n.table != INPUT_ALIAS for n in holder.find_all(exp.Column)) or any(
            isinstance(n, exp.AggFunc) and not isinstance(n.parent, (exp.Window, exp.IgnoreNulls, exp.RespectNulls)) for n in holder.walk()
        ):
            return None  # the item reads a column or an aggregate outside its windows
        new_items[index] = exp.alias_(holder.this, item.alias_or_name, quoted=False)
    if not values:
        return None
    input_select = exp.Select(expressions=[exp.alias_(value, f"kqi{position}", quoted=False) for position, value in enumerate(values)])
    for key in _INPUT_ARGS:
        if copy.args.get(key) is not None:
            input_select.set(key, copy.args[key].copy() if not isinstance(copy.args[key], list) else [j.copy() for j in copy.args[key]])
    outer = exp.Select(expressions=[new_items[index] for index in range(len(items))])
    outer.set("from_" if select.args.get("from_") is not None else "from", exp.From(this=exp.Subquery(this=input_select.copy(), alias=exp.TableAlias(this=exp.to_identifier(INPUT_ALIAS)))))
    return _Split(select, input_select, outer)


def _candidates(tree: exp.Expression) -> list[exp.Select]:
    found = [s for s in tree.find_all(exp.Select) if _own_windows(s) and _reachable_through_derived_tables(s)]
    return sorted(found, key=lambda s: -s.depth)  # innermost first: an outer input then holds the unified inner one


def _installed(part: _Split, input_select: exp.Select) -> exp.Select:
    outer = part.outer.copy()
    outer.find(exp.Subquery).set("this", input_select.copy())
    return outer


def unify(left_sql: str, right_sql: str, *, dialect: str, prove) -> tuple[str, str] | None:
    """The two queries with provably equal window inputs written as one text, or None when nothing was unified.

    ``prove(a, b)`` says whether two queries provably return the same bag of rows, column by column.
    """

    try:
        left, right = sqlglot.parse_one(left_sql, read=dialect), sqlglot.parse_one(right_sql, read=dialect)
    except sqlglot.errors.SqlglotError:
        return None
    if any(tree.find(exp.With) is not None or tree.find(exp.CTE) is not None for tree in (left, right)):
        return None
    if not any(left.find_all(exp.Window)) or not any(right.find_all(exp.Window)):
        return None
    proofs = 0
    changed = False
    done: dict[int, _Split] = {}  # left selects already written in the step-1 form, by the select that replaced them
    for right_select in _candidates(right):
        right_part = split(right_select)
        if right_part is None:
            continue
        right_text = None
        for left_select in _candidates(left):
            left_part = done.get(id(left_select)) or split(left_select)
            if left_part is None or left_part.skeleton() != right_part.skeleton() or left_part.tables() != right_part.tables():
                continue
            try:
                left_text = faithful_sql(left_part.input, dialect)
                right_text = right_text or faithful_sql(right_part.input, dialect)
            except Exception:  # noqa: BLE001 - an input that cannot be printed faithfully is not compared
                continue
            if max(len(left_text), len(right_text)) > MAX_CHARS:
                continue
            if left_text != right_text:
                if proofs >= MAX_PROOFS:
                    continue
                proofs += 1
                if not prove(left_text, right_text):
                    continue
            if id(left_select) not in done:
                new_left = _installed(left_part, left_part.input)
                if left_select.parent is None:
                    left = new_left
                else:
                    left_select.replace(new_left)
                done[id(new_left)] = left_part
            new_right = _installed(right_part, left_part.input)
            if right_select.parent is None:
                right = new_right
            else:
                right_select.replace(new_right)
            changed = True
            break
    if not changed:
        return None
    try:
        return faithful_sql(left, dialect), faithful_sql(right, dialect)
    except Exception:  # noqa: BLE001
        return None
