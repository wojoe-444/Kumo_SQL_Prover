"""A ``UNION`` whose branches cannot return the same row twice is a ``UNION ALL``.

    SELECT g.day, 'first' AS slot, g.first AS who FROM games AS g
    UNION
    SELECT g.day, 'second', g.second FROM games AS g

removes duplicate rows, and has none to remove: ``day`` is the key of ``games``, so no branch returns a row
twice, and a row of the first branch says ``'first'`` where every row of the second says ``'second'``. It is the
same query with ``UNION ALL``, which is the shape :mod:`kumo_sql_prover.union_all_counts` reads.

**The rule.** A derived table of the select that is a tree of ``UNION`` / ``UNION ALL`` (at least one ``UNION``)
has every ``UNION`` in it written as ``UNION ALL`` when

* every branch is a plain select over base tables (inner or cross joins, an optional ``WHERE``) whose outputs
  include, as plain columns, every column of a declared key of each of its tables, each of those columns declared
  NOT NULL; and
* every two branches differ at an output position where every branch of the tree writes a string literal, or
  every branch writes a whole number: two strings that still differ when case and trailing spaces are ignored,
  or two different numbers.

**Why it is sound.** ``UNION`` keeps one copy of each row, where two rows are the same when they agree at every
position (two NULLs agree). Two different rows of a base table differ in a column of its key, and a key column
that is NOT NULL holds no NULL, so the two rows differ where the branch returns that column: a branch over one
table returns no row twice, and a branch over several tables returns one row per combination of their rows that
passes its filters, each combination told apart by the keys of the tables. Rows of two different branches differ
at the position of their two literals. So no two rows of the whole tree agree, every ``UNION`` inside it has
nothing to remove whatever the nesting, and the tree returns the rows of its branches once each, as ``UNION ALL``
does. The strings are compared more strictly than the engines compare them (case and trailing spaces ignored,
plain ASCII only), so two literals an engine's collation treats as the same never count as different. A
position counts only when all branches write literals of one kind there, because a column or a literal of
another kind in some branch can change the type the engine gives the position, and with it what ``'1'`` and
``'01'``, or two spellings of a date, are converted to before the rows are compared. For the key columns the
rule relies on what the prover assumes of every comparison, that two different values stay different when an
engine brings them to the common type of their position.

**What is refused.** A branch with ``DISTINCT``, grouping, aggregates, windows, ``LIMIT``, an outer join, a
derived table or a nested query, a star, or a table without a declared key of NOT NULL columns among the plain
output columns; two branches that differ at no position where every branch writes a literal of the same kind
(the same literal, a column or a string against a number in some branch, a string outside plain ASCII);
``EXCEPT`` and ``INTERSECT``; a ``UNION`` with
its own ``ORDER BY`` or ``LIMIT``; a tree that is already all ``UNION ALL``.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key
from .ast_utils import select_sources
from .union_all_counts import _BRANCH_ARGS, _owner, _plain_joins, _row_value


def union_of_distinct_rows(
    select: exp.Expression,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression | None:
    """The select with each such ``UNION`` source written as ``UNION ALL`` (see the module docstring), or ``None``."""

    if not schema or not keys or not not_null or not isinstance(select, exp.Select):
        return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    declared = {name.lower(): {c.lower() for c in listed} for name, listed in not_null.items()}
    declared_keys = {name.lower(): [tuple(c.lower() for c in key) for key in listed] for name, listed in keys.items()}
    positions = [
        position
        for position, source in enumerate(select_sources(select))
        if isinstance(source, exp.Subquery) and _rows_never_repeat(source.this, known, declared_keys, declared)
    ]
    if not positions:
        return None
    copy = select.copy()
    sources = select_sources(copy)
    for position in positions:
        for union in sources[position].this.find_all(exp.Union):
            union.set("distinct", False)
    return copy


def _rows_never_repeat(query: exp.Expression, known, declared_keys, declared) -> bool:
    """Whether ``query`` is a tree of unions, one of them a ``UNION``, whose branches never return a row twice."""

    unions: list[exp.Union] = []
    branches = _branches(query, unions)
    if branches is None or len(branches) < 2 or not any(union.args.get("distinct") for union in unions):
        return False
    width = len(branches[0].expressions)
    literals = []
    for branch in branches:
        if len(branch.expressions) != width or not _keyed(branch, known, declared_keys, declared):
            return False
        literals.append([_literal(item.unalias()) for item in branch.expressions])
    # a position tells branches apart only when every branch writes a literal of one kind there: the column then
    # has that kind in every dialect, and no literal is converted to another type before the rows are compared
    usable = [
        position
        for position in range(width)
        if all(row[position] is not None for row in literals) and len({row[position][0] for row in literals}) == 1
    ]
    return all(
        any(literals[i][position][1] != literals[j][position][1] for position in usable)
        for i in range(len(branches))
        for j in range(i)
    )


def _branches(query: exp.Expression, unions: list[exp.Union]) -> list[exp.Select] | None:
    """The selects of a tree of unions, left to right, and its union nodes; ``None`` for anything else."""

    while isinstance(query, exp.Subquery) and not any(value for key, value in query.args.items() if key != "this"):
        query = query.this
    if isinstance(query, exp.Select):
        return [query]
    if type(query) is not exp.Union or any(value for key, value in query.args.items() if key not in ("this", "expression", "distinct")):
        return None
    unions.append(query)
    left, right = _branches(query.this, unions), _branches(query.expression, unions)
    return None if left is None or right is None else left + right


def _keyed(branch: exp.Select, known, declared_keys, declared) -> bool:
    """A plain select over base tables that returns a NOT NULL key of each of them."""

    if any(value for key, value in branch.args.items() if key not in _BRANCH_ARGS) or not _plain_joins(branch):
        return False
    if not branch.expressions or not all(_row_value(item) for item in branch.expressions):
        return False
    if any(node is not branch for node in branch.find_all(exp.Select, exp.Subquery, exp.Exists, exp.Window, exp.AggFunc)):
        return False
    columns: dict[str, list[str]] = {}
    tables: dict[str, str] = {}
    for source in select_sources(branch):
        listed = _base_table(source, known)
        alias = source.alias_or_name.lower()
        if listed is None or not alias or alias in columns:
            return False
        columns[alias] = listed
        tables[alias] = _table_key(source)
    if not columns:
        return False
    returned: dict[str, set[str]] = {alias: set() for alias in columns}
    for item in branch.expressions:
        value = item.unalias()
        if isinstance(value, exp.Column):
            owner = _owner(value, columns)
            if owner is not None:
                returned[owner].add(value.name.lower())
    return all(
        any(key and set(key) <= returned[alias] and set(key) <= declared.get(table, set()) for key in declared_keys.get(table, ()))
        for alias, table in tables.items()
    )


def _literal(node: exp.Expression) -> tuple[str, object] | None:
    """A literal as (kind, value to compare); ``None`` for anything that is not a plain string or whole number."""

    if not isinstance(node, exp.Literal):
        return None
    text = node.this
    if node.is_string:
        return ("string", text.rstrip(" ").lower()) if text.isascii() and text.isprintable() else None
    return ("number", int(text)) if text.isdigit() else None
