"""A ``HAVING`` condition over grouping expressions alone filters rows before grouping.

    SELECT 2 * a AS d, COUNT(*) AS n FROM t GROUP BY 2 * a HAVING 2 * a IS NOT NULL AND COUNT(*) > 1
    SELECT 2 * a AS d, COUNT(*) AS n FROM t WHERE 2 * a IS NOT NULL GROUP BY 2 * a HAVING COUNT(*) > 1

Every row of a group has the same value for each ``GROUP BY`` expression (rows are grouped by those values, NULLs
together). A condition that reads the input rows only through whole ``GROUP BY`` expressions therefore has one
truth value for all rows of a group: ``HAVING`` keeps the group when it is TRUE, and ``WHERE`` keeps exactly the
rows of the groups on which it is TRUE and none of the others (FALSE and NULL both drop). A kept group has all
of its rows either way, so every aggregate sees the same rows, and a dropped group leaves no row to form it
again. With no input rows there is no group in both forms (the list has at least one key, so this is not the
one-group ``GROUP BY ()``).

``having_rules.key_having_to_where`` already does this for conjuncts that read key *columns*. Calcite groups by
expressions (``GROUP BY 2 * a``, ``GROUP BY a + 5``) and writes ``HAVING 2 * a IS NOT NULL`` on them when it turns a
semi-join into a join, and that condition stayed above the grouping.

Conditions, all checked:

* a plain ``GROUP BY`` list with at least one key: no ``ROLLUP``, ``CUBE``, ``GROUPING SETS``, ``GROUP BY ALL``,
  ``WITH TOTALS`` or ``()`` (under those a key reads as NULL in some sets), and no literal key (a position);
* no window function, ``QUALIFY`` or named ``WINDOW`` in the select;
* in the moved conjunct every column lies inside a sub-expression that is, by its text, one of the ``GROUP BY``
  expressions, and at least one such sub-expression is not a bare column (those conjuncts are the older rule's);
* the conjunct holds no aggregate, subquery, window, random or unknown function;
* no column in it bears the name of a select-list alias that renames something else (in dialects where ``HAVING``
  reads aliases the name would mean the alias).

Other conjuncts stay in ``HAVING``.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping

_UNSAFE = (exp.AggFunc, exp.Subquery, exp.Select, exp.Exists, exp.Window, exp.Rand, exp.Anonymous)


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _reads_only_keys(node: exp.Expression, keys: set[str], found: list[bool]) -> bool:
    """Whether every column under ``node`` lies inside an occurrence of a key; ``found`` notes a non-column key."""

    if node.sql() in keys:
        if not isinstance(node, exp.Column):
            found[0] = True
        return True
    if isinstance(node, exp.Column):
        return False
    return all(_reads_only_keys(child, keys, found) for child in node.iter_expressions())


def key_expression_having_to_where(select: exp.Select) -> exp.Select | None:
    """``select`` with each ``HAVING`` conjunct that reads only grouping expressions moved to ``WHERE``, or None."""

    group, having = select.args.get("group"), select.args.get("having")
    if group is None or having is None or not group.expressions:
        return None
    if extended_grouping(group) or any(group.args.get(k) for k in ("all", "totals")):
        return None
    if any(select.args.get(k) for k in ("qualify", "windows", "with_", "with")) or select.find(exp.Window):
        return None
    if any(isinstance(key, (exp.Literal, exp.Tuple)) or key.find(*_UNSAFE) for key in group.expressions):
        return None
    keys = {key.sql() for key in group.expressions}
    shadowed = {
        item.alias.lower() for item in select.expressions
        if isinstance(item, exp.Alias) and not (isinstance(item.this, exp.Column) and item.this.name.lower() == item.alias.lower())
    }
    moved, kept = [], []
    for conjunct in _conjuncts(having.this):
        found = [False]
        if (
            not conjunct.find(*_UNSAFE)
            and _reads_only_keys(conjunct, keys, found)
            and found[0]
            and not any(column.name.lower() in shadowed for column in conjunct.find_all(exp.Column))
        ):
            moved.append(conjunct.copy())
        else:
            kept.append(conjunct.copy())
    if not moved:
        return None
    result = select.copy()
    where = result.args.get("where")
    parts = ([where.this] if where is not None else []) + moved
    parts = [exp.Paren(this=p) if isinstance(p, (exp.Or, exp.Xor)) else p for p in parts]
    condition = parts[0]
    for part in parts[1:]:
        condition = exp.And(this=condition, expression=part)
    result.set("where", exp.Where(this=condition))
    result.set("having", exp.Having(this=exp.and_(*kept)) if kept else None)
    return result
