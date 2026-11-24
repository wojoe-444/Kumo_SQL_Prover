"""A grouping with no aggregate whose outputs the group fixes, read as ``SELECT DISTINCT``.

    SELECT p.product_id, p.product_name FROM product AS p JOIN sales AS s ON p.product_id = s.product_id
    WHERE p.product_id NOT IN (SELECT ..) GROUP BY s.product_id

groups the joined rows by the sale's product and shows the product's own columns. ``product_id`` is the key of
``product`` and the join makes it equal to the group's key, so every row of a group is the same product: the
select shows one row per product that passes, which is

    SELECT DISTINCT p.product_id, p.product_name FROM product AS p JOIN sales AS s ON p.product_id = s.product_id
    WHERE p.product_id NOT IN (SELECT ..)

The provers read the ``DISTINCT`` spelling (as an existence test on ``sales``); the grouped one they met only when
its outputs were, by their text, its keys (hardest-cases batch 30, "sold only in spring", LeetCode 1084).

**Why it is the same rows.** Call a column *fixed* when all rows of a group agree on it:

* a group key is fixed;
* a column the ``WHERE`` or the ``ON`` of an inner join equates to a fixed column (a top-level ``AND`` part
  ``a = b``) is fixed: the test is TRUE on every row that is grouped, so both are not NULL and equal. Texts can be
  equal under one collation and distinct under another, so such a step is taken only between two columns declared
  with a whole-number type;
* when the columns of a NOT NULL key of a declared table are fixed (and whole numbers), every row of the group
  holds the same row of that table, so all its columns are fixed.

If every output is fixed, a group shows the values any of its rows has, whichever row the engine picks. If also
every group key is determined by the outputs in the same way (it is an output, is equated to one, or is a column of
a table whose key the outputs fix), two rows are in one group exactly when they show the same outputs, so the groups
are the distinct output rows. Under a ``DISTINCT`` that second condition is not needed: the distinct values of
fixed outputs are the same over the groups and over the rows.

**What is read.** A select with a plain ``GROUP BY`` (columns or positions), no aggregate, window, ``HAVING``,
``ORDER BY`` or ``LIMIT``, whose sources are declared tables and named derived tables with written-out output
names, joined by inner joins with ``ON`` or by commas; every output a plain column. MySQL, PostgreSQL and SQLite,
which accept a select item outside the ``GROUP BY`` (PostgreSQL only one its key fixes).

**Left alone.** An outer join (its ``ON`` does not hold on the padded rows), ``USING`` and ``NATURAL``, a computed
output or key, a key that is nullable or not a whole number, a bare ``GROUP BY`` name that two sources have, or
that is a column and the alias of another output the rows do not equate with it, a name that is only an output's
alias in a select an enclosing query could give names to (a subquery of a ``WHERE``, a ``LATERAL``), a name the sources
do not place.

It runs in the last step of batch 27 only, so a pair an earlier step decides is decided as it was.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, _output_names, conjuncts, extended_grouping
from .group_alias_names import _closed
from .single_row_aggregates import Keyed, column_kind

NAME = "key_fixed_groupings"
DIALECTS = frozenset({"mysql", "postgres", "sqlite"})
_SELECT_ARGS = frozenset({"expressions", FROM_KEY, "joins", "where", "group", "distinct"})
_JOIN_KINDS = frozenset({"", "INNER", "CROSS"})


def _set_args(node: exp.Expression) -> set[str]:
    return {key for key, value in node.args.items() if value not in (None, [], False)}


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


class _Sources:
    """The sources of one select by the name the select calls them: their columns, and the keyed tables among them."""

    def __init__(self) -> None:
        self.columns: dict[str, set[str]] = {}
        self.keyed: dict[str, Keyed] = {}

    def add(self, source: exp.Expression, schema, keys, not_null, types) -> bool:
        table = Keyed.read(source, schema, keys, not_null, types)
        if table is not None:
            name, columns = table.alias, set(table.columns)
            self.keyed[name] = table
        elif isinstance(source, exp.Subquery) and source.alias and not _set_args(source) - {"this", "alias"}:
            alias = source.args["alias"]
            names = _output_names(source.this)
            if alias.args.get("columns") or not isinstance(alias.this, exp.Identifier) or names is None:
                return False
            name, columns = source.alias.lower(), {n.lower() for n in names}
        else:
            return False
        if name in self.columns:
            return False
        self.columns[name] = columns
        return True

    def column(self, node: exp.Expression) -> tuple[str, str] | None:
        """``(source, column)`` for a plain column of one of the sources."""

        node = _unwrap(node)
        if not isinstance(node, exp.Column) or not isinstance(node.this, exp.Identifier) or node.args.get("db") or node.args.get("catalog"):
            return None
        name = node.name.lower()
        if node.table:
            source = node.table.lower()
            return (source, name) if name in self.columns.get(source, ()) else None
        owners = [source for source, columns in self.columns.items() if name in columns]
        return (owners[0], name) if len(owners) == 1 else None

    def whole(self, column: tuple[str, str]) -> bool:
        table = self.keyed.get(column[0])
        return table is not None and column_kind(table.types.get(column[1])) == "int"


def read_key_fixed_grouping(select: exp.Select, schema, keys, not_null, types, dialect: str | None) -> exp.Select | None:
    """``select`` as ``SELECT DISTINCT`` when it is such a grouping (module doc), else ``None``. The input is not changed."""

    if (dialect or "").lower() not in DIALECTS or not schema or not keys:
        return None
    group = select.args.get("group")
    if group is None or not group.expressions or extended_grouping(group) or _set_args(group) - {"expressions"}:
        return None
    if _set_args(select) - _SELECT_ARGS or select.args.get(FROM_KEY) is None:
        return None
    distinct = select.args.get("distinct")
    if distinct is not None and _set_args(distinct):
        return None  # DISTINCT ON
    if any(isinstance(node, (exp.AggFunc, exp.Window)) and node.find_ancestor(exp.Select) is select for node in select.walk()):
        return None
    sources = _Sources()
    joins = select.args.get("joins") or []
    if not sources.add(select.args[FROM_KEY].this, schema, keys, not_null, types):
        return None
    for join in joins:
        if _set_args(join) - {"this", "on", "kind"} or (join.args.get("kind") or "").upper() not in _JOIN_KINDS:
            return None
        if not sources.add(join.this, schema, keys, not_null, types):
            return None
    outputs: list[tuple[str, str]] = []
    aliases: dict[str, list[tuple[str, str] | None]] = {}
    for item in select.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        column = sources.column(value)
        if column is None:
            return None
        outputs.append(column)
        if isinstance(item, exp.Alias):
            aliases.setdefault(item.alias.lower(), []).append(column)
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(column: tuple[str, str]) -> tuple[str, str]:
        while parent.get(column, column) != column:
            column = parent[column]
        return column

    tests = [part for join in joins if join.args.get("on") is not None for part in conjuncts(join.args["on"])]
    if select.args.get("where") is not None:
        tests += conjuncts(select.args["where"].this)
    for test in tests:
        if not isinstance(test, exp.EQ):
            continue
        left, right = sources.column(test.this), sources.column(test.expression)
        if left is not None and right is not None and sources.whole(left) and sources.whole(right):
            parent[find(left)] = find(right)

    group_keys: list[tuple[str, str]] = []
    for key in group.expressions:
        key = _unwrap(key)
        if isinstance(key, exp.Literal):
            if key.is_string or not key.this.isdigit() or not 1 <= int(key.this) <= len(outputs):
                return None
            group_keys.append(outputs[int(key.this) - 1])
            continue
        if not isinstance(key, exp.Column) or not isinstance(key.this, exp.Identifier) or key.args.get("db") or key.args.get("catalog"):
            return None
        if key.table:
            meant = [sources.column(key)]
        else:
            # a bare name is a column of one source or an output of the select list, and the engines do not agree on
            # which: it is read only when the rows equate every column it could mean
            name = key.name.lower()
            owners = [(source, name) for source, columns in sources.columns.items() if name in columns]
            if len(owners) > 1:
                return None
            if not owners and not _closed(select):
                return None  # only an output's alias here: an enclosing query could give the name another meaning
            meant = owners + aliases.get(name, [])
        if not meant or None in meant or len({find(column) for column in meant}) != 1:
            return None
        group_keys.append(meant[0])

    def fixes(given: list[tuple[str, str]], column: tuple[str, str]) -> bool:
        """Whether rows that agree on ``given`` agree on ``column``."""

        classes = {find(one) for one in given}
        if find(column) in classes:
            return True
        table = sources.keyed.get(column[0])
        if table is None:
            return False
        return any(all(sources.whole((column[0], part)) and find((column[0], part)) in classes for part in key) for key in table.keys)

    if not all(fixes(group_keys, column) for column in outputs):
        return None
    if distinct is None and not all(fixes(outputs, column) for column in group_keys):
        return None
    copy = select.copy()
    copy.set("group", None)
    if distinct is None:
        copy.set("distinct", exp.Distinct())
    return copy
