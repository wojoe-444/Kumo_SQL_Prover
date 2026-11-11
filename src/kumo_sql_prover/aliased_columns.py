"""A bare column in a select list, a ``WHERE`` or an ``ON`` is a source column, whatever the select names its outputs.

    SELECT s.id, ROUND(CASE WHEN x.top > 100 THEN pay * 0.5 ELSE pay END) AS pay
    FROM staff AS s JOIN (SELECT team, MAX(pay) AS top FROM staff GROUP BY team) AS x ON s.team = x.team

reads ``staff.pay`` inside the ``CASE``: the output named ``pay`` does not exist yet where the select list is
evaluated. The normalizer's qualification pass for joined selects leaves every bare name that is also an output
alias alone, because ``GROUP BY``, ``HAVING`` and ``ORDER BY`` may read the output first (and which one wins
differs by dialect). That caution is not needed in the three places where an output alias cannot be read at all,
and a bare column left there keeps the rules that need every column to name its source (pulling a grouped
derived table up, among others) from reading the select, so that ``s.pay`` and ``pay`` spellings of one query
end in different normal forms.

**The rule.** In a select with at least one join, a bare column that is read in the select's own scope inside

* a select-list expression,
* the ``WHERE`` clause, or
* the ``ON`` condition of one of its joins

gets the qualifier of the one source that has a column of that name.

**Why it is sound.** SQL resolves a bare name in a select list, ``WHERE`` or ``ON`` against the sources of the
innermost query that has a column of that name; an output alias of the same select is not in scope there (MySQL
and BigQuery reject ``SELECT a AS b, b + 1``, and where an engine does allow it, a source column of that name
comes first). So when every source of the select is known and exactly one has the name, that source is the
owner, and writing its qualifier does not change what is read. Only qualifiers are added.

**What is read, and what is left alone.**

* Every source of the select is a plain base table whose columns the caller supplied (and no ``WITH`` table of
  that name is left in the statement), or a derived table that names every output (no star, no unnamed
  expression, no repeated name). The source names are distinct. With any other source nothing is qualified.
* No join of the select uses ``USING`` or ``NATURAL``: the merged column belongs to no single source.
* The name has exactly one owner among the sources. A name nobody has (a reference to an outer query) and a
  name two sources have (the engine rejects it) stay bare.
* ``GROUP BY``, ``HAVING``, ``ORDER BY`` and window definitions named in a ``WINDOW`` clause are not touched,
  and neither is anything below a nested query, set operation or lambda (their own names come first).
* The name is not the head of a field path and not a date part given to a function.

On BigQuery the added qualifiers are re-derived from the text by ``proof_columns``, as for the normalizer's other
qualification passes.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources
from .single_source_columns import _DATE_PARTS, _source_columns


def qualify_aliased_columns(tree: exp.Expression, schema: dict[str, list[str]] | None, dialect: str = "bigquery") -> exp.Expression:
    """Qualify, in place, the bare columns described in the module docstring."""

    if not schema or not any(True for _ in tree.find_all(exp.Join)):
        return tree
    from .algebraic_equivalence import _checked_qualification

    known = {key.lower(): {c.lower() for c in columns} for key, columns in schema.items()}
    return _checked_qualification(tree, lambda t: _qualify(t, known), schema, dialect, "aliased_column_qualification")


def _owners(select: exp.Select, known: dict[str, set[str]], with_names: set[str]) -> dict[str, list[exp.Identifier]] | None:
    """Each column name the select's sources have, with the qualifiers of the sources that have it."""

    joins = select.args.get("joins") or []
    sources = select_sources(select)
    if not joins or len(sources) != len(joins) + 1 or any(select.args.get(k) for k in ("laterals", "with", "with_")):
        return None
    if any(join.args.get("using") or join.method or (join.kind or "").upper() not in ("", "INNER", "CROSS", "OUTER") for join in joins):
        return None
    owners: dict[str, list[exp.Identifier]] = {}
    names = set()
    for source in sources:
        columns = _source_columns(source, known, with_names)
        if not columns:
            return None
        alias = source.args.get("alias")
        qualifier = alias.this if isinstance(alias, exp.TableAlias) and alias.this is not None else source.this
        if not isinstance(qualifier, exp.Identifier) or not qualifier.name or qualifier.name.lower() in names:
            return None
        names.add(qualifier.name.lower())
        for column in columns:
            owners.setdefault(column, []).append(qualifier)
    return owners


def _own_part(column: exp.Column, select: exp.Select) -> bool:
    """Whether ``column`` is read by ``select`` itself in its select list, its ``WHERE`` or an ``ON`` of its joins."""

    child, node = column, column.parent
    while node is not None and node is not select:
        if isinstance(node, (exp.Query, exp.Subquery, exp.From, exp.Lambda, exp.Values, exp.Table)):
            return False
        if isinstance(node, exp.Join) and child is not node.args.get("on"):
            return False
        child, node = node, node.parent
    if node is not select:
        return False
    if isinstance(child, exp.Join):
        return child in (select.args.get("joins") or [])
    return child is select.args.get("where") or any(child is item for item in select.expressions)


def _qualify(tree: exp.Expression, known: dict[str, set[str]]) -> exp.Expression:
    with_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for select in list(tree.find_all(exp.Select)):
        owners = _owners(select, known, with_names)
        if owners is None:
            continue
        for column in list(select.find_all(exp.Column)):
            if column.args.get("table") or not isinstance(column.this, exp.Identifier):
                continue
            name = column.name.lower()
            if len(owners.get(name, [])) != 1:
                continue
            parent = column.parent
            if isinstance(parent, (exp.Dot, exp.PropertyEQ, exp.Kwarg)):
                continue
            if isinstance(parent, exp.Func) and name.upper() in _DATE_PARTS:
                continue
            if not _own_part(column, select):
                continue
            column.set("table", owners[name][0].copy())
    return tree
