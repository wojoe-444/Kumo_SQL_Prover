"""Under ``DISTINCT`` or in an ``IN`` list, a derived table that only joins and projects is read as its join.

    SELECT DISTINCT x.item_id, x.label
    FROM (SELECT s.item_id, i.label FROM sale AS s JOIN item AS i ON s.item_id = i.item_id) AS x
    WHERE x.item_id NOT IN (SELECT k.item_id FROM sale AS k WHERE k.paid > 5)

reads the rows of a join through a derived table that lists two of its columns. It is

    SELECT DISTINCT s.item_id AS item_id, i.label AS label
    FROM sale AS s JOIN item AS i ON s.item_id = i.item_id
    WHERE s.item_id NOT IN (SELECT k.item_id FROM sale AS k WHERE k.paid > 5)

**Why it holds.** The derived table has no ``DISTINCT``, grouping, aggregate, window, ``ORDER BY`` or row limit,
so it returns one row per row of its join that passes its conditions, and each of its outputs is a column of that
row. Reading ``x.c`` is reading that column, at any depth. The outer select therefore sees the same rows either
way, repeats included; the rule fires only under ``DISTINCT`` or for the select of an ``IN`` list because the
solver reads the other cases by itself, and these are the ones the rules for joins under ``DISTINCT``
(``keyed_semijoin``, ``distinct_match_joins``) need to see.

**What is read.** A select whose only source is an aliased derived table (no join beside it), that is ``SELECT
DISTINCT`` (no ``ON``) or the select of an ``IN`` / ``NOT IN`` list, has a nested query in its ``WHERE`` or is
such a list, and has no grouping, aggregate, window, ``ORDER BY`` or row limit. The derived table: a plain select
over base tables under inner, cross or comma joins (``ON`` only), each under its own name, whose outputs are
plain columns that name their source, under distinct output names; every column in it names its source and it
holds no nested query. Outside it, a read of the derived table is ``x.c`` at any depth or a bare ``c`` directly
in the outer select; bare columns of nested one-table queries are first given their table (when the caller
supplied its columns) and nested sources named like the derived table or one of its tables get names of their own
(``keyed_semijoin.rename_nested_sources``). A bare name left in a nested query, a ``*``, a quoted name or a
volatile call leaves the select as it is.

A narrow retry rule (``shape_fallback``). ``tests/test_distinct_derived_joins.py`` replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, conjuncts
from .own_group_membership import _plain_table

_OUTER_BANNED = ("joins", "group", "having", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample", "cluster", "distribute", "sort", "connect", "match")
_INNER_BANNED = ("distinct", "group", "having", "qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample", "cluster", "distribute", "sort", "connect", "match")
_JOIN_EXTRAS = ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")


def _plain_column(node: exp.Expression | None) -> bool:
    return isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier) and not node.this.args.get("quoted") and bool(node.table) and not node.args.get("db")


def read_derived_join(select: exp.Select, schema: dict | None = None) -> exp.Select | None:
    """``select`` over the join its derived table shows, or None."""

    from .counted_membership import _unstable
    from .keyed_semijoin import name_nested_sources, rename_nested_sources

    distinct = select.args.get("distinct")
    holder = select.parent
    listing = isinstance(holder, exp.Subquery) and not holder.alias and isinstance(holder.parent, exp.In) and holder.arg_key == "query"
    from_ = select.args.get(FROM_KEY)
    if from_ is None or any(select.args.get(key) for key in _OUTER_BANNED) or (distinct is None and not listing) or (distinct is not None and distinct.args.get("on")):
        return None
    source = from_.this
    if not isinstance(source, exp.Subquery) or not source.alias or not isinstance(source.this, exp.Select) or any(v for k, v in source.args.items() if k not in ("this", "alias", "comments")):
        return None
    if source.args["alias"].args.get("columns") or source.args["alias"].this.args.get("quoted"):
        return None
    inner = source.this
    inner_from = inner.args.get(FROM_KEY)
    if inner_from is None or any(inner.args.get(key) for key in _INNER_BANNED):
        return None
    inner_joins = inner.args.get("joins") or []
    for join in inner_joins:
        if join.side or (join.kind or "").upper() not in ("", "INNER", "CROSS") or any(join.args.get(key) for key in _JOIN_EXTRAS):
            return None
    tables = [inner_from.this] + [join.this for join in inner_joins]
    if not inner_joins or any(not _plain_table(table) or not isinstance(table.this, exp.Identifier) or table.this.args.get("quoted") for table in tables):
        return None
    aliases = [(table.alias_or_name or "").lower() for table in tables]
    if not all(aliases) or len(set(aliases)) != len(aliases):
        return None
    if any(isinstance(node, (exp.Query, exp.Subquery, exp.AggFunc, exp.Window, exp.Star, exp.Exists)) for node in inner.walk() if node is not inner):
        return None
    if any(not _plain_column(column) or column.table.lower() not in aliases for column in inner.find_all(exp.Column)):
        return None
    shown: dict[str, exp.Column] = {}
    for item in inner.expressions:
        value = item.unalias()
        name = (item.alias_or_name or "").lower()
        if not _plain_column(value) or not name or name in shown or (isinstance(item, exp.Alias) and item.args["alias"].args.get("quoted")):
            return None
        shown[name] = value
    alias = source.alias.lower()
    if not listing and (select.args.get("where") is None or select.args["where"].find(exp.Select) is None):
        return None  # left to the solver, which reads it by itself
    # The other nested queries are renamed and named with the derived table set aside (its own sources keep their names)
    select = select.copy()
    aside = select.args[FROM_KEY].this
    aside.replace(exp.Table(this=exp.to_identifier("kdj_aside"), alias=exp.TableAlias(this=exp.to_identifier(aside.alias))))
    select = name_nested_sources(rename_nested_sources(select, {alias, *aliases}), schema)
    select = select.copy()
    select.args[FROM_KEY].this.replace(aside)
    source = select.args[FROM_KEY].this
    inner = source.this
    if select.find(exp.Star) is not None or any(node.find_ancestor(exp.Select) is select for node in select.find_all(exp.AggFunc, exp.Window)):
        return None
    nested = [query for query in select.find_all(exp.Select) if query is not select and query is not inner]
    for query in nested:
        first = query.args.get(FROM_KEY)
        named = ([first.this] if first is not None else []) + [join.this for join in query.args.get("joins") or []]
        if any((entry.alias_or_name or "").lower() in {alias, *aliases} for entry in named):
            return None  # a nested source still carries one of the names
    reads: list[exp.Column] = []
    for column in select.find_all(exp.Column):
        if column.find_ancestor(exp.Select) is inner:
            continue
        own = column.find_ancestor(exp.Select) is select
        if not isinstance(column.this, exp.Identifier) or column.this.args.get("quoted") or column.args.get("db"):
            return None
        if column.table:
            if column.table.lower() == alias:
                reads.append(column)
            elif column.table.lower() in aliases:
                return None  # a name from further out that the join's tables would capture
        elif own:
            reads.append(column)
        else:
            return None  # a bare name in a nested query: its owner is not certain
    if any(column.name.lower() not in shown for column in reads) or _unstable(select):
        return None
    result = select.copy()
    source = result.args[FROM_KEY].this
    inner = source.this
    outputs = set(map(id, result.expressions))
    for column in list(result.find_all(exp.Column)):
        if column.find_ancestor(exp.Select) is inner or isinstance(column.this, exp.Star):
            continue
        if not (column.table.lower() == alias or (not column.table and column.find_ancestor(exp.Select) is result)):
            continue
        value = shown[column.name.lower()].copy()
        if id(column) in outputs:
            column.replace(exp.alias_(value, column.name, quoted=False))  # an output keeps its name
        else:
            column.replace(value)
    tests = [part.copy() for part in conjuncts(inner.args["where"].this)] if inner.args.get("where") is not None else []
    if result.args.get("where") is not None:
        tests += [part.copy() for part in conjuncts(result.args["where"].this)]
    result.set(FROM_KEY, inner.args[FROM_KEY].copy())
    result.set("joins", [join.copy() for join in inner.args.get("joins") or []])
    result.set("where", exp.Where(this=exp.and_(*tests, copy=False)) if tests else None)
    return result
