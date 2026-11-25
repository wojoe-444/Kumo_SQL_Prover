"""One table read under two spellings: ``c.customer_id`` beside ``customer_id``.

    SELECT DISTINCT customer_id FROM customer AS c GROUP BY c.customer_id

names one column twice, once bare and once with the table's name. The solver reads both as the same column, but
the rules that compare a select list with a grouping (a ``DISTINCT`` over the grouped columns, a grouping by a
key) compare what is written, so the pair of spellings stopped them and a query a person would call trivially
equal stayed unproved. This pre-pass (hardest-cases batch 29) writes the qualified read bare when a select with
one source names a column both ways.

**Why it is the same query.** The select has exactly one source, a table or a named derived table, called ``c``.
``c.x`` is its column ``x``. A bare ``x`` written directly in that select is looked up in the select's own source
before any enclosing query, and the source has an ``x`` (``c.x`` is valid), so it is the same column. Two places
can give a bare name another meaning, and both are left alone:

* ``GROUP BY``, ``HAVING`` and ``ORDER BY`` may read a name as an output of the select list, and the engines do
  not agree on when. So a column whose name is also the alias of a select item stays qualified, unless every item
  of that alias is that very column (``c.x AS x``), when both meanings are one.
* a column named like the source itself (``c.c``) stays qualified: a bare ``c`` can be the whole row.

**What is read.** A select whose one source is a table or an aliased derived table (no join, ``LATERAL``, pivot or
table function), and a column it names both ways, ``x`` and ``c.x``, directly in the select (its select list,
``WHERE``, ``GROUP BY``, ``HAVING``, ``QUALIFY``, ``ORDER BY``, window definitions). Only such a column is rewritten:
its bare spelling is already in the query, so no new name is introduced that a keyword or an outer query could
claim. Reads inside a nested query are that query's own and stay as written.

It runs in the last step of batch 27 only, so a pair an earlier step decides is decided as it was.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources

NAME = "one_source_names"
_NESTED = (exp.Select, exp.SetOperation, exp.Subquery)
_SOURCE_ARGS = ("from", "from_", "joins", "laterals", "with", "with_")


def read_one_source_names(tree: exp.Expression) -> tuple[exp.Expression, bool]:
    """``(tree with those qualified reads written bare, whether one was)``. The input is not changed."""

    if not any(_readings(select) for select in tree.find_all(exp.Select)):
        return tree, False
    copy = tree.copy()
    for select in list(copy.find_all(exp.Select)):
        for column in _readings(select):
            column.set("table", None)
    return copy, True


def _readings(select: exp.Select) -> list[exp.Column]:
    """The qualified reads of the one source of ``select`` that can be written bare."""

    if select.args.get("joins") or select.args.get("laterals"):
        return []
    sources = select_sources(select)
    if len(sources) != 1:
        return []
    source = sources[0]
    if isinstance(source, exp.Table):
        if not isinstance(source.this, exp.Identifier) or source.args.get("pivots") or source.args.get("joins"):
            return []
    elif not (isinstance(source, exp.Subquery) and source.alias and not source.args.get("pivots") and not source.args.get("joins")):
        return []
    name = (source.alias_or_name or "").lower()
    if not name:
        return []
    direct = [column for key, value in select.args.items() if key not in _SOURCE_ARGS for column in _direct(value)]
    plain = [column for column in direct if isinstance(column.this, exp.Identifier) and not column.args.get("db") and not column.args.get("catalog")]
    bare = {column.name.lower() for column in plain if not column.table}
    if not bare:
        return []
    aliases: dict[str, list[exp.Expression]] = {}
    for item in select.expressions:
        if isinstance(item, exp.Alias):
            aliases.setdefault(item.alias.lower(), []).append(item.this)
    found = []
    for column in plain:
        if not column.table or column.table.lower() != name or column.name.lower() == name or column.name.lower() not in bare:
            continue
        if not all(_same(value, column, name) for value in aliases.get(column.name.lower(), ())):
            continue  # GROUP BY, HAVING or ORDER BY could read the bare name as that output
        found.append(column)
    return found


def _same(value: exp.Expression, column: exp.Column, source: str) -> bool:
    """Whether a select item's expression is the column itself, bare or under the source's name."""

    while isinstance(value, exp.Paren):
        value = value.this
    return (
        isinstance(value, exp.Column) and isinstance(value.this, exp.Identifier) and value.name.lower() == column.name.lower()
        and (value.table or source).lower() == source and not value.args.get("db") and not value.args.get("catalog")
    )


def _direct(value) -> list[exp.Column]:
    """The columns written under ``value`` outside every nested query."""

    found: list[exp.Column] = []
    stack = list(value) if isinstance(value, list) else [value]
    while stack:
        node = stack.pop()
        if not isinstance(node, exp.Expression) or isinstance(node, _NESTED):
            continue
        if isinstance(node, exp.Column):
            found.append(node)
            continue
        stack.extend(node.iter_expressions())
    return found
