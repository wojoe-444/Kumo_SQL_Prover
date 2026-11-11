"""A conditional count of a value that cannot be NULL where the condition holds counts the rows that pass.

    SELECT COUNT(CASE WHEN a < 5 THEN a > 10 END) FROM t                       -- COUNT(CASE WHEN a < 5 THEN 1 END)
    SELECT COUNT(CASE WHEN d.p THEN d.e END)                                   -- COUNT(CASE WHEN d.p THEN 1 END)
    FROM (SELECT a < 5 AS p, a > 10 AS e FROM t GROUP BY a < 5, a > 10) AS d

``COUNT(CASE WHEN p THEN e END)`` counts the rows on which ``p`` is TRUE and ``e`` is not NULL; ``COUNT(CASE WHEN p
THEN 1 END)`` counts the rows on which ``p`` is TRUE. They are the same number when ``e`` is never NULL on a row
where ``p`` is TRUE, whatever the rows are (duplicates count alike on both sides, no row gives 0 on both).

That is known here from the text alone, without any schema constraint: ``p`` is a comparison (``<``, ``<=``, ``>``,
``>=``, ``=``, ``<>``) of terms built from columns, literals, ``+``, ``-`` and ``*``. Such a comparison is NULL as soon
as one of its columns is NULL, so on a row where it is TRUE every column it reads is not NULL. ``e`` is a term or a
comparison of the same kind that reads only columns ``p`` reads, so it has a value there (these operators return
NULL only for a NULL operand; an overflow is an error in both spellings).

``p`` and ``e`` are read where they are written, or, when both are plain columns of the select's one derived table,
in that table's select: each of its rows carries ``p`` and ``e`` computed from the same underlying row, or from the
same group, where each is a grouping key or reads only grouping keys.

Calcite's expansion of distinct aggregates writes ``COUNT(c) FILTER (WHERE flag)`` over such a derived table, and
its later rules write ``COUNT(*) FILTER (WHERE flag)``; the prover spells both filters as the ``CASE`` above.

Conditions, all checked:

* the call is ``COUNT`` of a searched ``CASE`` with one ``WHEN`` and no ``ELSE`` (or ``ELSE NULL``), without ``DISTINCT``
  (the distinct count of ``e`` is not the number of rows) and it belongs to the select being read;
* ``p`` is one of the six comparisons, ``e`` a comparison or a term, both built only from columns, number or string
  literals, ``+``, ``-``, ``*``, unary minus and parentheses: no division (NULL or an error on zero, by dialect), no
  function, no ``CASE``, no subquery, no NULL literal;
* every column of ``e`` is, by its text (qualifier included), a column of ``p``, and ``e`` is not already a literal;
* through a derived table: the select reads exactly one source, a derived table with an alias and no column
  list, that is one ``SELECT`` (not a set operation) without a star and with distinct output names; ``p`` and ``e``
  are plain columns naming two of its outputs; when it is grouped, the grouping is a plain ``GROUP BY`` list (under
  ``ROLLUP``, ``CUBE`` or ``GROUPING SETS`` a key is NULL in the groups that leave it out while another key is not) and
  each definition is a key or reads only columns that are keys.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping

_COMPARISONS = (exp.LT, exp.LTE, exp.GT, exp.GTE, exp.EQ, exp.NEQ)
_ARITHMETIC = (exp.Add, exp.Sub, exp.Mul)


def _bare(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _term_columns(node: exp.Expression | None) -> set[tuple[str, ...]] | None:
    """The columns of a term (columns, literals, ``+ - *``, unary minus), or None when it is anything else."""

    node = _bare(node)
    if isinstance(node, exp.Column):
        if isinstance(node.this, exp.Star) or not isinstance(node.this, exp.Identifier):
            return None
        return {tuple(part.sql() for part in node.parts)}
    if isinstance(node, exp.Literal):
        return set()
    if isinstance(node, exp.Neg):
        return _term_columns(node.this)
    if type(node) in _ARITHMETIC:
        left, right = _term_columns(node.this), _term_columns(node.expression)
        return None if left is None or right is None else left | right
    return None


def _comparison_columns(node: exp.Expression | None) -> set[tuple[str, ...]] | None:
    node = _bare(node)
    if type(node) not in _COMPARISONS:
        return None
    left, right = _term_columns(node.this), _term_columns(node.expression)
    return None if left is None or right is None else left | right


def _known_when(condition: exp.Expression, value: exp.Expression) -> bool:
    """Whether ``value`` cannot be NULL on a row where ``condition`` is TRUE (both read in one scope)."""

    needed = _comparison_columns(condition)
    if needed is None:
        return False
    read = _comparison_columns(value)
    if read is None:
        read = _term_columns(value)
    return read is not None and read <= needed


def _derived_outputs(select: exp.Select) -> tuple[str, dict[str, exp.Expression]] | None:
    """The alias and the output definitions of the one derived table ``select`` reads, or None."""

    source = select.args.get("from_") or select.args.get("from")
    if source is None or select.args.get("joins") or select.args.get("laterals"):
        return None
    derived = source.this
    if not isinstance(derived, exp.Subquery) or not isinstance(derived.this, exp.Select) or not derived.alias:
        return None
    alias = derived.args.get("alias")
    if alias is None or alias.args.get("columns") or derived.args.get("pivots") or derived.args.get("sample"):
        return None
    inner = derived.this
    group = inner.args.get("group")
    keys: list[exp.Expression] = []
    if group is not None:
        if not group.expressions or extended_grouping(group) or any(group.args.get(k) for k in ("all", "totals")):
            return None
        keys = [_bare(key) for key in group.expressions]
    key_texts = {key.sql() for key in keys}
    key_columns = {tuple(part.sql() for part in key.parts) for key in keys if isinstance(key, exp.Column)}
    outputs: dict[str, exp.Expression] = {}
    for item in inner.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        name = item.alias_or_name
        if not name or name in outputs or isinstance(value, exp.Star) or (isinstance(value, exp.Column) and isinstance(value.this, exp.Star)):
            return None
        outputs[name] = value
    if group is not None:
        # under a GROUP BY only a key, or a definition over key columns, is known to be read on the group's keys
        for name in list(outputs):
            value = _bare(outputs[name])
            columns = {tuple(part.sql() for part in column.parts) for column in value.find_all(exp.Column)}
            if value.sql() not in key_texts and not columns <= key_columns:
                del outputs[name]
    return derived.alias, outputs


def _definition(node: exp.Expression, derived: tuple[str, dict[str, exp.Expression]] | None) -> exp.Expression | None:
    node = _bare(node)
    if derived is None or not isinstance(node, exp.Column) or not isinstance(node.this, exp.Identifier):
        return None
    alias, outputs = derived
    if len(node.parts) > 2 or (node.table and node.table != alias):
        return None
    return outputs.get(node.name)


def count_known_values(select: exp.Select) -> exp.Select | None:
    """``select`` with ``COUNT(CASE WHEN p THEN e END)`` written ``COUNT(CASE WHEN p THEN 1 END)`` where ``e`` is known
    to have a value whenever ``p`` holds (module doc), or None."""

    derived = None
    resolved = False
    targets = []
    for call in select.find_all(exp.Count):
        if call.find_ancestor(exp.Select) is not select or any(v for k, v in call.args.items() if k not in ("this", "big_int")):
            continue
        case = _bare(call.this)
        if not isinstance(case, exp.Case) or case.args.get("this") is not None or len(case.args.get("ifs") or []) != 1:
            continue
        default = case.args.get("default")
        if default is not None and not isinstance(default, exp.Null):
            continue
        arm = case.args["ifs"][0]
        condition, value = arm.this, arm.args.get("true")
        if value is None or isinstance(_bare(value), (exp.Literal, exp.Null)):
            continue
        if _known_when(condition, value):
            targets.append(arm)
            continue
        if not resolved:
            derived, resolved = _derived_outputs(select), True
        inner_condition, inner_value = _definition(condition, derived), _definition(value, derived)
        if inner_condition is not None and inner_value is not None and _known_when(inner_condition, inner_value):
            targets.append(arm)
    if not targets:
        return None
    marks = {id(arm) for arm in targets}
    positions = [index for index, arm in enumerate(select.find_all(exp.If)) if id(arm) in marks]
    copy = select.copy()
    for index, arm in enumerate(list(copy.find_all(exp.If))):
        if index in positions:
            arm.set("true", exp.Literal.number(1))
    return copy
