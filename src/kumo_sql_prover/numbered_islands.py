"""Runs of consecutive numbers found with a row number ("gaps and islands"), read as "a neighbour exists".

"Which free seats have a free seat next to them" is often written by numbering the free seats in order: along a run
of consecutive seats the seat number and the row number grow together, so their difference names the run.

    SELECT b.seat
    FROM (SELECT a.seat, COUNT(*) OVER (PARTITION BY a.seat - a.rn) AS run
          FROM (SELECT seat, ROW_NUMBER() OVER (ORDER BY seat) AS rn FROM seats WHERE free = 1) AS a) AS b
    WHERE b.run >= 2

The same rows are the free seats whose number minus one or plus one is a free seat too:

    SELECT b.seat
    FROM (SELECT x.seat FROM seats AS x
          WHERE x.free = 1
            AND EXISTS (SELECT 1 FROM seats AS y WHERE y.free = 1 AND (y.seat = x.seat - 1 OR y.seat = x.seat + 1))) AS b

This pre-pass (hardest-cases batch 31, LeetCode 603) writes the first as the second, which the prover's subquery
rules read.

**Why the two are equal.** Let ``k`` be a single-column key of the table that is declared NOT NULL and a whole
number, and take the rows that pass the filter in the order of ``k``: ``k1 < k2 < ..`` (no two are equal). The
``i``-th has row number ``i``, so its difference is ``ki - i``. From one row to the next ``k`` grows by at least 1
and the row number by exactly 1, so the difference never shrinks, and it stays the same exactly when ``k`` grew by 1.
Two rows therefore have the same difference exactly when every step between them is 1: the difference names the
maximal run of consecutive values the row lies in. A run holds two rows or more exactly when the row has a
neighbour: a row that passes the filter whose ``k`` is one less or one more. The key makes that neighbour at most one
row, and the filter is the same test on the neighbour's own columns.

With decimals the argument fails (1, 1.5 and 3 give the differences 0, -0.5 and 0), so the key must be a whole
number; with a repeated value the row numbers of the tied rows are arbitrary, so it must be a key.

**What is read.** The numbered select reads one declared table with no join, grouping or ``LIMIT``; its ``WHERE``
is made of comparisons, ``AND``, ``OR``, ``NOT``, ``IS NULL``, ``IN`` lists and arithmetic over the table's own
columns and literals. The numbering is ``ROW_NUMBER()``, ``RANK()`` or ``DENSE_RANK()`` (the three agree when no two
rows tie) ``OVER (ORDER BY k)`` ascending with no ``PARTITION BY``; terms after ``k`` change nothing, and a term
before it is allowed when the filter fixes that column to one whole number (``WHERE free = 1 .. ORDER BY free, k``).
The run is named by ``k - numbering``, written in the numbered select or in the select around it. Then one of:

* the window form above: ``COUNT(*)``, ``COUNT(1)`` or ``COUNT(k)`` ``OVER (PARTITION BY difference)`` in a select
  that only lists columns, tested in the select around it by a conjunct of its ``WHERE``;
* the membership form: ``WHERE difference IN (SELECT difference FROM <the same numbered select> GROUP BY difference
  HAVING COUNT(*) >= 2)``, or the same list read from a grouped derived table with the count tested outside it;
* the join form: the numbered select joined to that grouped list on the difference.

The test is ``>= 2``, ``> 1`` or ``<> 1`` (a count over a row's own partition or group is at least 1).

**A query that fails.** In MySQL a row number is unsigned, and subtracting it from a smaller key fails ("BIGINT
UNSIGNED value is out of range"; seen on MySQL 8.0 with a key of -5): the query as written has no rows on such a
table, and the reading has. A proof says the two queries are equal on the databases where neither fails, here as
everywhere.

**Left alone.** Everything else: a numbering with ``PARTITION BY``, a frame, descending order or no ``ORDER BY``
(``ROW_NUMBER() OVER ()`` numbers the rows in no promised order); a key that is nullable, unsigned, not a whole number
or of several columns; a threshold above two (a run of three needs more than one neighbour); a count or a row number
that is read anywhere else; a star in the select that tests the count; any other subquery or source in that select.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _names
from .ast_utils import FROM_KEY
from .single_row_aggregates import Keyed, column_kind

NAME = "numbered_islands"

_PREFIX = "kqi"
_NUMBERINGS = (exp.RowNumber, exp.Rank, exp.DenseRank)
# what a filter may be made of: each is a function of the row's own columns, so it can be asked of the neighbour
_FILTER_NODES = (
    exp.Column, exp.Identifier, exp.Literal, exp.Null, exp.Boolean, exp.Paren, exp.And, exp.Or, exp.Not, exp.EQ, exp.NEQ, exp.LT, exp.LTE,
    exp.GT, exp.GTE, exp.Is, exp.In, exp.Between, exp.Add, exp.Sub, exp.Mul, exp.Neg,
)


def _lower(name: str | None) -> str:
    return (name or "").lower()


def _unwrap(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _conjuncts(node: exp.Expression | None) -> list[exp.Expression]:
    node = _unwrap(node)
    if node is None:
        return []
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _set_args(node: exp.Expression) -> set[str]:
    return {key for key, value in node.args.items() if value not in (None, [], False)}


def _inside(node: exp.Expression | None, ancestor: exp.Expression) -> bool:
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False


def _fresh(taken: set[str]) -> str:
    number = 0
    while f"{_PREFIX}{number}" in taken:
        number += 1
    taken.add(f"{_PREFIX}{number}")
    return f"{_PREFIX}{number}"


def _whole(node: exp.Expression | None) -> int | None:
    node = _unwrap(node)
    if isinstance(node, exp.Literal) and not node.is_string and node.name.isdigit():
        return int(node.name)
    return None


def _derived(source: exp.Expression | None) -> tuple[exp.Select, str] | None:
    """The select and the name of a plain derived table."""

    if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select) or _set_args(source) - {"this", "alias"}:
        return None
    alias = source.args.get("alias")
    if not isinstance(alias, exp.TableAlias) or alias.args.get("columns") or not isinstance(alias.this, exp.Identifier):
        return None
    return source.this, _lower(source.alias)


def _only_source(select: exp.Select) -> exp.Expression | None:
    from_ = select.args.get(FROM_KEY)
    if from_ is None or select.args.get("joins"):
        return None
    return from_.this


def _reference(node: exp.Expression | None, alias: str) -> str | None:
    """The output of the derived table ``alias`` that ``node`` names (bare, or with that name), lower case."""

    node = _unwrap(node)
    if not isinstance(node, exp.Column) or node.args.get("db") or node.args.get("catalog") or not isinstance(node.this, exp.Identifier):
        return None
    if node.table and _lower(node.table) != alias:
        return None
    return _lower(node.name)


class _Numbered:
    """A numbered select: the table, the key the rows are numbered by, the filter, and what each output is.

    ``outputs`` maps an output name to ``("col", column)`` for a column of the table, ``("rn",)`` for the row number
    and ``("diff",)`` for the key minus the row number."""

    def __init__(self, select: exp.Select, table: Keyed, key: str, outputs: dict[str, tuple]) -> None:
        self.select, self.table, self.key, self.outputs = select, table, key, outputs

    @property
    def where(self) -> exp.Expression | None:
        where = self.select.args.get("where")
        return where.this if where is not None else None


def _filter_reads_own_row(condition: exp.Expression | None, table: Keyed) -> bool:
    if condition is None:
        return True
    for node in condition.walk():
        if not isinstance(node, _FILTER_NODES):
            return False
        if isinstance(node, exp.In) and (_set_args(node) - {"this", "expressions"} or not all(isinstance(_unwrap(e), exp.Literal) for e in node.expressions)):
            return False
        if isinstance(node, exp.Column) and table.column(node) is None:
            return False
    return True


def _fixed_columns(condition: exp.Expression | None, table: Keyed) -> set[str]:
    """The columns the filter holds to one whole number: ``c = 1`` for a column declared a whole number or a flag."""

    fixed = set()
    for part in _conjuncts(condition):
        if not isinstance(part, exp.EQ):
            continue
        for left, right in ((part.this, part.expression), (part.expression, part.this)):
            name = table.column(left)
            if name is None or _whole(right) is None:
                continue
            declared = str(table.types.get(name) or "").upper().split("(")[0].strip()
            if column_kind(table.types.get(name)) == "int" or declared in ("BOOL", "BOOLEAN"):
                fixed.add(name)
    return fixed


def _numbering_key(window: exp.Expression | None, table: Keyed, fixed: set[str]) -> str | None:
    """The key column a numbering window numbers the rows by, or None when it is not one this module reads."""

    window = _unwrap(window)
    if not isinstance(window, exp.Window) or not isinstance(window.this, _NUMBERINGS) or _set_args(window.this):
        return None
    if _set_args(window) - {"this", "order", "over"}:
        return None
    order = window.args.get("order")
    if not isinstance(order, exp.Order):
        return None
    for term in order.expressions:
        if not isinstance(term, exp.Ordered) or _set_args(term) - {"this", "desc", "nulls_first"}:
            return None
        name = table.column(term.this)
        if name is None:
            return None
        if term.args.get("desc"):
            return None
        if name in fixed:
            continue  # one value on every row that passes the filter: the term orders nothing
        if (name,) in table.keys and column_kind(table.types.get(name)) == "int":
            return name  # no two rows tie on a key, so later terms order nothing either
        return None
    return None


def _numbered(select: exp.Select, schema, keys, not_null, types) -> _Numbered | None:
    if _set_args(select) - {"expressions", FROM_KEY, "where"}:
        return None
    table = Keyed.read(_only_source(select), schema, keys, not_null, types)
    if table is None:
        return None
    where = select.args.get("where")
    condition = where.this if where is not None else None
    if not _filter_reads_own_row(condition, table):
        return None
    fixed = _fixed_columns(condition, table)
    declared = next((list(columns) for name, columns in (schema or {}).items() if _lower(name) == table.name), [])
    outputs: dict[str, tuple] = {}
    found: set[str] = set()

    def add(name: str, value: tuple) -> bool:
        if not name or name in outputs:
            return False
        outputs[name] = value
        return True

    for item in select.expressions:
        alias = _lower(item.alias) if isinstance(item, exp.Alias) else ""
        value = _unwrap(item.this if isinstance(item, exp.Alias) else item)
        if isinstance(value, exp.Star) or (isinstance(value, exp.Column) and isinstance(value.this, exp.Star)):
            if alias or (isinstance(value, exp.Column) and _lower(value.table) != table.alias) or _set_args(value) - {"this", "table"}:
                return None
            if not all(add(_lower(str(column)), ("col", _lower(str(column)))) for column in declared):
                return None
            continue
        column = table.column(value)
        if column is not None:
            if not add(alias or column, ("col", column)):
                return None
            continue
        if isinstance(value, exp.Sub):
            key, numbering = table.column(value.this), _numbering_key(value.expression, table, fixed)
            if key is None or numbering != key or not add(alias, ("diff",)):
                return None
            found.add(key)
            continue
        numbering = _numbering_key(value, table, fixed)
        if numbering is None or not add(alias, ("rn",)):
            return None
        found.add(numbering)
    if len(found) != 1:
        return None
    return _Numbered(select, table, next(iter(found)), outputs)


def _is_difference(node: exp.Expression | None, numbered: _Numbered, alias: str) -> bool:
    """Whether ``node``, read over the numbered select named ``alias``, is the key minus the row number."""

    node = _unwrap(node)
    name = _reference(node, alias)
    if name is not None:
        return numbered.outputs.get(name) == ("diff",)
    if isinstance(node, exp.Sub):
        left, right = _reference(node.this, alias), _reference(node.expression, alias)
        return left is not None and right is not None and numbered.outputs.get(left) == ("col", numbered.key) and numbered.outputs.get(right) == ("rn",)
    return False


def _counts_rows(node: exp.Expression | None, counted) -> bool:
    """Whether ``node`` is ``COUNT(*)``, ``COUNT(<number>)`` or ``COUNT`` of something ``counted`` says is never NULL."""

    node = _unwrap(node)
    if not isinstance(node, exp.Count) or _set_args(node) - {"this", "big_int"}:
        return False
    argument = _unwrap(node.this)
    return isinstance(argument, exp.Star) or _whole(argument) is not None or bool(counted(argument))


def _two_or_more(node: exp.Expression, is_count) -> bool:
    """Whether ``node`` says a count (as ``is_count`` knows it) is at least two."""

    node = _unwrap(node)
    if not isinstance(node, (exp.GTE, exp.GT, exp.LTE, exp.LT, exp.NEQ)) or _set_args(node) - {"this", "expression"}:
        return False
    left, right, kind = node.this, node.expression, type(node)
    if kind in (exp.LTE, exp.LT):  # 2 <= n is n >= 2
        left, right, kind = right, left, exp.GTE if kind is exp.LTE else exp.GT
    elif kind is exp.NEQ and _whole(left) is not None:
        left, right = right, left
    if not is_count(left):
        return False
    bound = _whole(right)
    return (kind is exp.GTE and bound == 2) or (kind in (exp.GT, exp.NEQ) and bound == 1)


def _requalified(condition: exp.Expression, name: str) -> exp.Expression:
    condition = condition.copy()
    for column in [condition] if isinstance(condition, exp.Column) else list(condition.find_all(exp.Column)):
        column.set("table", exp.to_identifier(name))
    return condition


def _rows_with_a_neighbour(numbered: _Numbered, names: dict[str, str], alias: exp.Expression, taken: set[str]) -> exp.Subquery:
    """The derived table ``alias`` of the rows that pass the filter and have a neighbour that passes it, listing the
    table's columns ``names`` (output name to column)."""

    table, key = numbered.table, numbered.key
    row = _fresh(taken)

    def read(name: str) -> exp.Table:
        source = exp.Table(this=table.table.this.copy())
        source.set("alias", exp.TableAlias(this=exp.to_identifier(name)))
        return source

    # one EXISTS for both neighbours: the form the prover's own rules give two EXISTS of one table joined by OR
    other = _fresh(taken)
    near = exp.Paren(this=exp.Or(
        this=exp.EQ(this=exp.column(key, other), expression=exp.Sub(this=exp.column(key, row), expression=exp.Literal.number(1))),
        expression=exp.EQ(this=exp.column(key, other), expression=exp.Add(this=exp.column(key, row), expression=exp.Literal.number(1))),
    ))
    inner = near if numbered.where is None else exp.And(this=exp.Paren(this=_requalified(numbered.where, other)), expression=near)
    either = exp.Exists(this=exp.Select(expressions=[exp.Literal.number(1)]).from_(read(other)).where(inner))
    test = either if numbered.where is None else exp.And(this=exp.Paren(this=_requalified(numbered.where, row)), expression=either)
    items = [exp.alias_(exp.column(column, row), output) if output != column else exp.column(column, row) for output, column in names.items()]
    select = exp.Select(expressions=items).from_(read(row)).where(test)
    return exp.Subquery(this=select, alias=alias.copy())


def _drop_conjunct(select: exp.Select, conjunct: exp.Expression) -> None:
    kept = [part for part in _conjuncts(select.args["where"].this) if part is not conjunct]
    if not kept:
        select.set("where", None)
        return
    condition = kept[0].copy()
    for part in kept[1:]:
        condition = exp.And(this=condition, expression=part.copy())
    select.args["where"].set("this", condition)


def _reads_only(select: exp.Select, alias: str, allowed: set[str], skip: list[exp.Expression]) -> bool:
    """Whether everything in ``select`` outside the nodes of ``skip`` reads the derived table ``alias`` through the
    outputs ``allowed`` only, with no other select, window or star."""

    for node in select.walk():
        if node is select or any(_inside(node, part) for part in skip):
            continue
        if isinstance(node, (exp.Select, exp.Subquery, exp.SetOperation, exp.Window)):
            return False
        if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
            return False
        if isinstance(node, exp.Column):
            if node.args.get("db") or node.args.get("catalog") or not isinstance(node.this, exp.Identifier):
                return False
            if node.table and _lower(node.table) != alias:
                return False
            if _lower(node.name) not in allowed:
                return False
    return True


def _columns_of(outputs: dict[str, tuple]) -> dict[str, str]:
    return {name: value[1] for name, value in outputs.items() if value[0] == "col"}


def _window_form(select: exp.Select, schema, keys, not_null, types, taken: set[str]) -> bool:
    """``SELECT .. FROM (SELECT .., COUNT(*) OVER (PARTITION BY difference) AS n FROM (numbered) AS a) AS b WHERE n >= 2``."""

    source = _only_source(select)
    middle = _derived(source)
    if middle is None or select.args.get("where") is None:
        return False
    counted, alias = middle
    if _set_args(counted) - {"expressions", FROM_KEY}:
        return False
    inner = _derived(_only_source(counted))
    if inner is None:
        return False
    numbered = _numbered(inner[0], schema, keys, not_null, types)
    if numbered is None:
        return False
    inner_alias = inner[1]
    outputs: dict[str, tuple] = {}
    for item in counted.expressions:
        name = _lower(item.alias) if isinstance(item, exp.Alias) else ""
        value = _unwrap(item.this if isinstance(item, exp.Alias) else item)
        if isinstance(value, exp.Star) or (isinstance(value, exp.Column) and isinstance(value.this, exp.Star)):
            if name or (isinstance(value, exp.Column) and _lower(value.table) != inner_alias) or any(output in outputs for output in numbered.outputs):
                return False
            outputs.update(numbered.outputs)
            continue
        read = _reference(value, inner_alias)
        if read is not None:
            if read not in numbered.outputs or (name or read) in outputs:
                return False
            outputs[name or read] = numbered.outputs[read]
            continue
        if not isinstance(value, exp.Window) or not name or name in outputs or _set_args(value) - {"this", "partition_by", "over"}:
            return False
        parts = value.args.get("partition_by") or []
        if len(parts) != 1 or not _is_difference(parts[0], numbered, inner_alias):
            return False
        if not _counts_rows(value.this, lambda node: numbered.outputs.get(_reference(node, inner_alias) or "") == ("col", numbered.key)):
            return False
        outputs[name] = ("count",)
    counts = {name for name, value in outputs.items() if value == ("count",)}
    tests = [part for part in _conjuncts(select.args["where"].this) if _two_or_more(part, lambda node: (_reference(node, alias) or "") in counts)]
    if len(tests) != 1:
        return False
    names = _columns_of(outputs)
    if not names or not _reads_only(select, alias, set(names), [source, tests[0]]):
        return False
    source.replace(_rows_with_a_neighbour(numbered, names, source.args["alias"], taken))
    _drop_conjunct(select, tests[0])
    return True


def _same_numbered(node: exp.Expression | None, numbered: _Numbered, schema, keys, not_null, types) -> tuple[_Numbered, str] | None:
    """A second copy of the numbered select, read as a derived table: the copy and its name."""

    derived = _derived(node)
    if derived is None or derived[0].sql() != numbered.select.sql():
        return None
    again = _numbered(derived[0], schema, keys, not_null, types)
    return (again, derived[1]) if again is not None else None


def _long_runs(select: exp.Select, numbered: _Numbered, schema, keys, not_null, types) -> str | None:
    """The output name under which ``select`` lists the differences that two rows or more share, read from a copy of
    the numbered select; None when it is anything else."""

    if len(select.expressions) != 1:
        return None
    item = select.expressions[0]
    output = _unwrap(item.this if isinstance(item, exp.Alias) else item)
    source = _only_source(select)
    again = _same_numbered(source, numbered, schema, keys, not_null, types)
    if again is not None:
        # SELECT difference FROM (numbered) GROUP BY difference HAVING COUNT(*) >= 2
        copy, alias = again
        if _set_args(select) - {"expressions", FROM_KEY, "group", "having"} or select.args.get("group") is None or select.args.get("having") is None:
            return None
        group = select.args["group"]
        if _set_args(group) - {"expressions"} or len(group.expressions) != 1 or not _is_difference(group.expressions[0], copy, alias):
            return None
        if not _is_difference(output, copy, alias):
            return None
        never_null = lambda node: _is_difference(node, copy, alias) or copy.outputs.get(_reference(node, alias) or "") == ("col", copy.key)  # noqa: E731
        if not _two_or_more(select.args["having"].this, lambda node: _counts_rows(node, never_null)):
            return None
        return _lower(item.alias_or_name) or None
    grouped = _derived(source)
    if grouped is None:
        return None
    # SELECT difference FROM (SELECT difference, COUNT(*) AS n FROM (numbered) GROUP BY difference) AS g WHERE n > 1
    inner, alias = grouped
    again = _same_numbered(_only_source(inner), numbered, schema, keys, not_null, types)
    if again is None or _set_args(select) - {"expressions", FROM_KEY, "where"} or select.args.get("where") is None:
        return None
    copy, inner_alias = again
    if _set_args(inner) - {"expressions", FROM_KEY, "group"} or inner.args.get("group") is None:
        return None
    group = inner.args["group"]
    if _set_args(group) - {"expressions"} or len(group.expressions) != 1 or not _is_difference(group.expressions[0], copy, inner_alias):
        return None
    never_null = lambda node: _is_difference(node, copy, inner_alias) or copy.outputs.get(_reference(node, inner_alias) or "") == ("col", copy.key)  # noqa: E731
    kinds: dict[str, str] = {}
    for part in inner.expressions:
        name = _lower(part.alias_or_name)
        value = _unwrap(part.this if isinstance(part, exp.Alias) else part)
        if not name or name in kinds:
            return None
        if _is_difference(value, copy, inner_alias):
            kinds[name] = "diff"
        elif _counts_rows(value, never_null):
            kinds[name] = "count"
        else:
            return None
    if kinds.get(_reference(output, alias) or "") != "diff":
        return None
    if not _two_or_more(select.args["where"].this, lambda node: kinds.get(_reference(node, alias) or "") == "count"):
        return None
    return _lower(item.alias_or_name) or None


def _membership_form(select: exp.Select, schema, keys, not_null, types, taken: set[str]) -> bool:
    """``SELECT .. FROM (numbered) AS a WHERE a.difference IN (SELECT difference FROM (numbered) .. two rows or more)``."""

    source = _only_source(select)
    derived = _derived(source)
    if derived is None or select.args.get("where") is None:
        return False
    numbered = _numbered(derived[0], schema, keys, not_null, types)
    if numbered is None:
        return False
    alias = derived[1]
    tests = []
    for part in _conjuncts(select.args["where"].this):
        if not isinstance(part, exp.In) or _set_args(part) - {"this", "query"} or not isinstance(part.args.get("query"), exp.Subquery):
            continue
        query = part.args["query"]
        if _set_args(query) - {"this"} or not isinstance(query.this, exp.Select) or not _is_difference(part.this, numbered, alias):
            continue
        if _long_runs(query.this, numbered, schema, keys, not_null, types) is not None:
            tests.append(part)
    if len(tests) != 1:
        return False
    names = _columns_of(numbered.outputs)
    if not names or not _reads_only(select, alias, set(names), [source, tests[0]]):
        return False
    source.replace(_rows_with_a_neighbour(numbered, names, source.args["alias"], taken))
    _drop_conjunct(select, tests[0])
    return True


def _join_form(select: exp.Select, schema, keys, not_null, types, taken: set[str]) -> bool:
    """``SELECT .. FROM (numbered) AS a JOIN (the differences two rows or more share) AS g ON a.difference = g.difference``."""

    from_ = select.args.get(FROM_KEY)
    joins = select.args.get("joins") or []
    if from_ is None or len(joins) != 1:
        return False
    join = joins[0]
    if _set_args(join) - {"this", "on", "kind"} or _lower(join.args.get("kind")) not in ("", "inner") or join.args.get("on") is None:
        return False
    for first, second in ((from_.this, join.this), (join.this, from_.this)):
        derived, listed = _derived(first), _derived(second)
        if derived is None or listed is None or derived[1] == listed[1]:
            continue
        numbered = _numbered(derived[0], schema, keys, not_null, types)
        if numbered is None:
            continue
        alias, list_alias = derived[1], listed[1]
        output = _long_runs(listed[0], numbered, schema, keys, not_null, types)
        if output is None:
            continue
        on = _unwrap(join.args["on"])
        if not isinstance(on, exp.EQ):
            continue
        sides = [on.this, on.expression]
        mine = [side for side in sides if isinstance(_unwrap(side), exp.Column) and _lower(_unwrap(side).table) == alias and _is_difference(side, numbered, alias)]
        theirs = [side for side in sides if isinstance(_unwrap(side), exp.Column) and _lower(_unwrap(side).table) == list_alias and _lower(_unwrap(side).name) == output]
        if len(mine) != 1 or len(theirs) != 1:
            continue
        names = _columns_of(numbered.outputs)
        # outside the two sources and the join condition, only the numbered select's own columns, named or bare
        if not names or output in names or not _reads_only(select, alias, set(names), [first, second, join.args["on"]]):
            continue
        select.set(FROM_KEY, exp.From(this=_rows_with_a_neighbour(numbered, names, first.args["alias"], taken)))
        select.set("joins", None)
        return True
    return False


def read_numbered_islands(tree: exp.Expression, schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None) -> tuple[exp.Expression, bool]:
    """``(tree with each select of the module docstring rewritten, whether one was)``. The input is not changed."""

    if not schema or not keys or not types or not any(isinstance(window.this, _NUMBERINGS) for window in tree.find_all(exp.Window)):
        return tree, False
    copy = tree.copy()
    taken = _names(copy)
    wrote = False
    for select in list(copy.find_all(exp.Select)):
        if select.parent is not None and not _inside(select, copy):
            continue  # a select of a part already replaced
        for form in (_window_form, _membership_form, _join_form):
            if form(select, schema, keys, not_null, types, taken):
                wrote = True
                break
    return (copy, True) if wrote else (tree, False)
