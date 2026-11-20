"""A ``GROUP BY`` name that two joined tables share is the column the select list names: MySQL only.

    SELECT p.product_id, SUM(u.units)
    FROM prices AS p JOIN units_sold AS u ON p.product_id = u.product_id
    GROUP BY product_id                                   -- GROUP BY p.product_id

Both tables have a ``product_id``, so the bare name in ``GROUP BY`` does not say which one is meant, and the prover
declined the query ("cannot resolve column product_id"). MySQL accepts it and has a rule for it.

**The rule MySQL follows** (its manual, "Problems with Column Aliases" and the ``SELECT`` page): a bare name in
``GROUP BY`` or ``HAVING`` is looked up among the columns of the ``FROM`` tables first, and among the outputs of the
select list when that does not give one column. So when the name is a column of one table only, it means that
column; when two tables have it, it means the select-list output of that name, and the statement is an error when
no output or more than one output has the name.

This module reads the case in which both lookups agree on what is meant: exactly one output of the select list has
the name, and that output is a plain qualified column ``t.name`` of the same name (with or without ``AS name``),
where ``t`` is a table or derived table of this select. If ``name`` is a column of ``t`` alone, the first lookup
finds ``t.name``; if another source has it too, the second lookup finds the output, which is ``t.name``. Either
way the bare name is ``t.name``, and the module writes it so.

Checked on MySQL 8.0.46 (the note in ``tests/test_shared_group_names.py`` lists the statements): with ``p`` left
joined to ``u``, ``SELECT u.pid .. GROUP BY pid`` groups by ``u.pid`` (the padded rows fall into one NULL group)
and ``SELECT p.pid .. GROUP BY pid`` by ``p.pid``; ``HAVING pid IS NULL`` follows the same output; an output
``u.k AS pid`` makes ``GROUP BY pid`` group by ``u.k`` when two tables have a ``pid`` (that case is not read here);
two outputs named ``pid``, or none, is error 1052 ("ambiguous"); and a bare shared name inside an aggregate
(``HAVING COUNT(pid) > 0``) is error 1052 too, so only a bare column that is a grouping expression, or stands in
``HAVING`` outside any aggregate, is rewritten.

**What is read.** In a MySQL select with a ``FROM``: each ``GROUP BY`` expression that is a bare column name, and
each bare column in ``HAVING`` that is not inside an aggregate, a subquery or a window, when the select list has
exactly one output of that name and it is ``t.name`` as described. Everything else is left as it is written,
including every other dialect (PostgreSQL and SQLite look names up in another order).

A pre-pass of the eleventh hardest-cases batch (``shape_fallback.ELEVENTH_BATCH``).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources

DIALECTS = frozenset({"mysql"})


def _bare(node: exp.Expression) -> bool:
    return isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier) and not node.table and not node.args.get("db") and not node.args.get("catalog")


def _directly_under(node: exp.Expression, having: exp.Expression) -> bool:
    """Whether ``node`` stands in ``having`` outside every aggregate, window and subquery."""

    parent = node.parent
    while parent is not None and parent is not having:
        if isinstance(parent, (exp.AggFunc, exp.Window, exp.Subquery, exp.Query)):
            return False
        parent = parent.parent
    return parent is having


def _named_outputs(select: exp.Select) -> dict[str, exp.Column] | None:
    """``name -> t.name`` for the outputs that are a qualified column of their own name and the only output of that name."""

    names = {source.alias_or_name.lower() for source in select_sources(select) if isinstance(source, (exp.Table, exp.Subquery)) and source.alias_or_name}
    seen: dict[str, int] = {}
    found: dict[str, exp.Column] = {}
    for item in select.expressions:
        if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            return None  # a star lists outputs this module cannot name
        name = (item.alias_or_name or "").lower()
        if not name:
            continue
        seen[name] = seen.get(name, 0) + 1
        column = item.this if isinstance(item, exp.Alias) else item
        if (
            isinstance(column, exp.Column)
            and isinstance(column.this, exp.Identifier)
            and column.table
            and not column.args.get("db")
            and column.table.lower() in names
            and column.name.lower() == name
        ):
            found[name] = column
    return {name: column for name, column in found.items() if seen[name] == 1}


def qualify_shared_group_names(tree: exp.Expression, dialect: str) -> tuple[exp.Expression, bool]:
    """``tree`` with each bare ``GROUP BY`` or ``HAVING`` name written as the select-list column it means (module doc)."""

    if dialect not in DIALECTS or tree.find(exp.Group) is None:
        return tree, False

    def targets(select: exp.Select) -> list[tuple[exp.Column, exp.Column]]:
        group = select.args.get("group")
        if group is None or not select_sources(select) or any(group.args.get(k) for k in ("grouping_sets", "cube", "rollup", "totals", "all")):
            return []
        outputs = _named_outputs(select)
        if not outputs:
            return []
        found = [(node, outputs[node.name.lower()]) for node in group.expressions if _bare(node) and node.name.lower() in outputs]
        having = select.args.get("having")
        if having is not None:
            for node in having.find_all(exp.Column):
                if not _bare(node) or node.name.lower() not in outputs:
                    continue
                if not _directly_under(node, having):
                    continue
                found.append((node, outputs[node.name.lower()]))
        return found

    if not any(targets(select) for select in tree.find_all(exp.Select)):
        return tree, False
    tree = tree.copy()
    changed = False
    for select in list(tree.find_all(exp.Select)):
        for node, column in targets(select):
            node.replace(exp.Column(this=column.this.copy(), table=column.args["table"].copy()))
            changed = True
    return tree, changed
