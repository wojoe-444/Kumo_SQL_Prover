"""An output of a derived table that nothing reads is dropped, also when the derived table computes a window.

    SELECT d.id FROM (SELECT p.id, COUNT(c.id) AS n, ROW_NUMBER() OVER (ORDER BY p.day) AS rn
                      FROM p LEFT JOIN c ON .. GROUP BY p.id, p.day) AS d WHERE d.rn <= 10

never reads ``d.n``, so it is the same query with ``COUNT(c.id) AS n`` left out.
``algebraic_equivalence._prune_derived`` drops unread outputs of plain derived tables; it stays away from any select
that holds a window function, at any depth. A rewriter that prunes such a column (Calcite's projection pruning)
then leaves a windowed select with fewer outputs than the original's, and the two are compared as different
relations.

Why it holds. A select without ``DISTINCT`` returns one row per row left after its ``FROM``, ``WHERE``, ``GROUP BY``
and ``HAVING``, and each select item is computed from that row (a window function from the row and the other rows,
which are the same rows with or without the dropped item). Leaving out an item nothing reads changes no other
value and no row count, with NULLs, duplicates or no rows alike. The one exception is the item that makes a select
without ``GROUP BY`` an aggregate (one row for any input): such a select keeps one of its aggregates.

Conditions checked (the derived table is left alone otherwise):

* it is an aliased derived table (no column list) in the ``FROM`` or a join of the select, its body is one ``SELECT``
  that holds a window function at some depth (the others are ``_prune_derived``'s), with no ``DISTINCT`` (dropping
  a column would merge rows), ``ORDER BY``, ``LIMIT``, ``OFFSET``, ``QUALIFY``, named ``WINDOW`` or ``WITH``, and no star;
* the reading select has no star other than ``COUNT(*)`` and no ``USING`` or ``NATURAL`` join (they read columns by
  name without naming them);
* an output is kept when a column anywhere in the reading select, outside the derived table itself, has its name
  and is unqualified or qualified by the derived table's alias (a superset of the real reads: a subquery or another
  scope that shadows the alias only makes it keep more), or when the body itself has an unqualified column of that
  name outside its select list (a ``GROUP BY`` or ``HAVING`` that names an output);
* an unread output is dropped only when every column it reads is found: a qualified column in the source of the
  body that has that alias, an unqualified one in some source of the body, where a base table's columns are the
  declared ones (``schema``) and a derived table's are its output names. An output that reads a column nothing
  provides is an error when the query runs, and dropping it would turn a query that fails into one that answers;
  so would dropping one whose columns cannot be checked (no schema for its table, a star in a source, a subquery
  in the output), and those are kept too;
* outputs without a name are kept, at least one output is kept, and a select that aggregates without ``GROUP BY``
  keeps an output with one of its aggregates.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources
from .window_calls import is_window_function

_BODY_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having", "kind"})


def _inside(node: exp.Expression, root: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is root:
            return True
        parent = parent.parent
    return False


def _aggregates(select: exp.Select, item: exp.Expression | None = None) -> bool:
    """Whether ``select`` (or one item of it) evaluates an aggregate of its own, outside any window function call."""

    for call in (item or select).find_all(exp.AggFunc):
        if call.find_ancestor(exp.Select) is select and not is_window_function(call):
            return True
    return False


def _known_columns(source: exp.Expression, schema: dict[str, list[str]]) -> set[str] | None:
    """The column names ``source`` provides, or None when they cannot be listed."""

    if isinstance(source, exp.Table):
        if any(value for key, value in source.args.items() if key not in ("this", "alias", "db", "catalog")):
            return None
        alias = source.args.get("alias")
        if alias is not None and alias.args.get("columns"):
            return None
        return schema.get(".".join(part.name for part in source.parts).lower())
    if isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select) and not (source.args.get("alias") is not None and source.args["alias"].args.get("columns")):
        names = set()
        for item in source.this.expressions:
            if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) or not item.alias_or_name:
                return None
            names.add(item.alias_or_name.lower())
        return names
    return None


def _reads_only_known_columns(item: exp.Expression, body: exp.Select, schema: dict[str, list[str]]) -> bool:
    """Whether every column ``item`` reads is provided by a source of ``body`` (see the module docstring)."""

    if any(isinstance(node, (exp.Query, exp.Subquery)) for node in item.walk()):
        return False
    provided = [(source.alias_or_name.lower(), _known_columns(source, schema)) for source in select_sources(body)]
    for column in item.find_all(exp.Column):
        if isinstance(column.this, exp.Star):
            return False
        name, qualifier = column.name.lower(), column.table.lower()
        if qualifier:
            hits = [columns for alias, columns in provided if alias == qualifier]
            if len(hits) != 1 or hits[0] is None or name not in hits[0]:
                return False
        elif not any(columns is not None and name in columns for _, columns in provided):
            return False
    return True


def prune_windowed_outputs(select: exp.Select, schema: dict[str, list[str]] | None = None) -> exp.Expression | None:
    """``select`` with the unread outputs of its windowed derived tables dropped, or None (see the module docstring)."""

    schema = {name.lower(): {column.lower() for column in columns} for name, columns in (schema or {}).items()}

    if any(isinstance(star, exp.Star) and not isinstance(star.parent, exp.Count) for star in select.find_all(exp.Star)):
        return None
    if any(join.args.get("using") or join.args.get("method") for join in select.args.get("joins") or []):
        return None
    plans = []
    for position, source in enumerate(select_sources(select)):
        if not isinstance(source, exp.Subquery) or not source.alias or not isinstance(source.this, exp.Select):
            continue
        if source.args["alias"].args.get("columns") or any(value for key, value in source.args.items() if key not in ("this", "alias")):
            continue
        body = source.this
        if any(value for key, value in body.args.items() if key not in _BODY_ARGS) or not any(body.find_all(exp.Window)):
            continue
        if any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in body.expressions):
            continue
        alias = source.alias.lower()
        used = {c.name.lower() for c in select.find_all(exp.Column) if not _inside(c, source) and (not c.table or c.table.lower() == alias)}
        in_list = {id(c) for item in body.expressions for c in item.find_all(exp.Column)}
        used |= {c.name.lower() for c in body.find_all(exp.Column) if not c.table and id(c) not in in_list and c.find_ancestor(exp.Select) is body}
        keep = [
            index
            for index, item in enumerate(body.expressions)
            if not item.alias_or_name or item.alias_or_name.lower() in used or not _reads_only_known_columns(item, body, schema)
        ]
        if not keep:
            keep = [0]
        if _aggregates(body) and not body.args.get("group") and not any(_aggregates(body, body.expressions[index]) for index in keep):
            witness = next((index for index, item in enumerate(body.expressions) if _aggregates(body, item)), None)
            if witness is None:
                continue  # the aggregate is in the HAVING alone: nothing in the select list holds the select's one row
            keep = sorted(keep + [witness])
        if len(keep) < len(body.expressions):
            plans.append((position, keep))
    if not plans:
        return None
    copy = select.copy()
    sources = select_sources(copy)
    for position, keep in plans:
        body = sources[position].this
        body.set("expressions", [body.expressions[index] for index in keep])
    return copy
