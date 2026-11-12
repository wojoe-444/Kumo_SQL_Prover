"""One spelling for the input of a windowed select, which the prover compares by its text.

A select that computes window functions is kept whole (``algebraic_equivalence._isolate_windows``): two queries
agree on it when the select reads alike on both sides, source by source and condition by condition. So

    SELECT e.sal, SUM(b.sal + b.sal + 100) OVER (PARTITION BY b.job) FROM emp AS e JOIN bonus AS b ON e.name = b.name AND e.dept = 10
    SELECT e.sal, SUM(2 * b.sal + 100) OVER (PARTITION BY b.job) FROM bonus AS b, emp AS e WHERE e.dept = 10 AND b.name = e.name

never met, although they compute the same windows over the same rows. ``spell_windowed_input`` writes both as

    SELECT e.sal, SUM(2 * b.sal + 100) OVER (PARTITION BY b.job) FROM bonus AS b, emp AS e WHERE b.name = e.name AND e.dept = 10

Each step keeps the bag of rows the windows read and every value they compute:

* **``ON`` conditions of inner joins move to ``WHERE``.** ``a JOIN b ON p JOIN c ON q WHERE w`` is the rows of
  ``a x b x c`` on which ``p``, ``q`` and ``w`` are all TRUE, which is ``FROM a, b, c WHERE p AND q AND w``: a row is
  dropped when any condition is FALSE or NULL either way, and duplicates are kept either way. Window functions are
  computed after ``WHERE``, so they see the same rows.
* **A derived table that only filters and projects one table is read as that table.** ``(SELECT name, sal * 2 AS
  d FROM emp WHERE dept = 10) AS e`` has one row per row of ``emp`` that passes its filter, so the cross product
  with it is the cross product with ``emp AS e`` under the added condition ``e.dept = 10``, with ``e.d`` read as
  ``e.sal * 2``. Derived tables nested this way are read through all their levels. (The prover already reads
  such a table this way when it has no filter; a rewrite that pushes a filter into it was left unmatched.)
* **Base tables are listed in the order of their names** (two reads of one table keep their written order). A
  cross product does not depend on the order of its inputs.
* **The conditions are listed in one order**, each ``=`` with its operands in one order, and a condition written
  twice is kept once. ``AND`` and ``=`` are commutative in three-valued logic, and ``p AND p`` is ``p``. The order
  is that of the text with each qualifier replaced by the position of its source, so it does not depend on aliases.
  A condition ``p = TRUE`` or ``p IS TRUE`` is written ``p``, and ``CASE WHEN p THEN 1 ELSE 0 END = 1`` (any two
  different integer literals, compared with the first) is written ``p``: each is TRUE exactly when ``p`` is, and a
  ``WHERE`` keeps a row only when its condition is TRUE.
* **A sum of integer columns and integer literals inside a window is written as its linear form**: ``c1 * x1 +
  c2 * x2 + .. + c0``, columns in source order. Over the integers ``x + x + 100`` and ``2 * x + 100`` are the same
  number, and both are NULL exactly when a column in them is NULL (a column whose coefficients cancel is kept as
  ``0 * x`` for that reason). As elsewhere in the prover, integer arithmetic is read as exact: a spelling that
  overflows where the other does not is not told apart.

Declined (the select is left as written):

* a select without a window function of its own, or with ``GROUP BY``, ``HAVING``, ``DISTINCT``, ``QUALIFY``,
  ``ORDER BY``, ``LIMIT``, ``OFFSET``, a named ``WINDOW`` clause, or an aggregate outside a window;
* any join that is not a plain inner join: ``LEFT``, ``RIGHT``, ``FULL``, ``SEMI``, ``ANTI``, ``NATURAL``, ``USING``,
  ``LATERAL``, ``ASOF``, a hint or any other decoration (moving a condition across an outer join changes rows);
* any source that is not a table or a derived table with an alias (``UNNEST``, a table function, a parenthesized
  join), or whose alias lists column names. A derived table is read as its table only when it is a select over one table (or one such derived table)
  with nothing but a select list and a ``WHERE``, both built from columns, literals, comparisons, ``AND``, ``OR``,
  ``NOT``, ``IS``, ``+ - *``, ``CASE`` and ``CAST`` alone (no function call, subquery, star or column list on the
  alias), when every bare column in it is a declared column of its table, and when no nested query of the
  windowed select uses its alias. With any other derived table among the
  sources the conditions still move, but the sources keep their order;
* for the linear form: a column that is not of a declared integer type, a decimal or string literal, a product of
  two columns, division, or any function call. Other parts of the window are not touched.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .window_calls import is_window_function

_BANNED = ("group", "having", "distinct", "qualify", "order", "limit", "offset", "windows", "with_", "with")
_INTEGER_TYPES = {"INT", "INTEGER", "BIGINT", "SMALLINT", "INT64", "INT32", "INT16", "INT8", "INT4", "INT2"}
_SOURCE_ARGS = frozenset({"this", "alias"})
_TABLE_ARGS = frozenset({"this", "alias", "db", "catalog"})


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _and_all(parts: list[exp.Expression]) -> exp.Expression:
    parts = [exp.Paren(this=p) if isinstance(p, (exp.Or, exp.Xor)) else p for p in parts]  # they bind looser than AND
    result = parts[0]
    for part in parts[1:]:
        result = exp.And(this=result, expression=part)
    return result


def _plain_source(node: exp.Expression) -> bool:
    alias = node.args.get("alias")
    if isinstance(alias, exp.TableAlias) and alias.args.get("columns"):
        return False  # a column list renames the columns
    if isinstance(node, exp.Table):
        return isinstance(node.this, exp.Identifier) and not any(v for k, v in node.args.items() if k not in _TABLE_ARGS)
    if isinstance(node, exp.Subquery):
        return bool(node.alias) and not any(v for k, v in node.args.items() if k not in _SOURCE_ARGS)
    return False


def _plain_inner(join: exp.Join) -> bool:
    kind = (join.args.get("kind") or "").upper()
    return kind in ("", "INNER", "CROSS") and not any(v for k, v in join.args.items() if k not in ("this", "kind", "on"))


def _table_name(table: exp.Table) -> str:
    return ".".join(part.name.lower() for part in table.parts)


def _own(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select


_SCALAR = (
    exp.Column, exp.Identifier, exp.Literal, exp.Null, exp.Boolean, exp.Paren, exp.And, exp.Or, exp.Not, exp.EQ, exp.NEQ,
    exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Is, exp.Add, exp.Sub, exp.Mul, exp.Neg, exp.Case, exp.If, exp.Cast, exp.DataType,
    exp.DataTypeParam,
)
_MARK = "kqsimple"


def _scalar(node: exp.Expression) -> bool:
    return all(type(n) in _SCALAR for n in node.walk())


def _simple_body(body: exp.Expression, types: dict) -> tuple[exp.Table, dict[str, exp.Expression], list[exp.Expression]] | None:
    """A select that only filters and projects one table: the table, each output over the table's columns
    (qualified ``_MARK``) and the filter's conjuncts; None for any other body."""

    if not isinstance(body, exp.Select) or any(v for k, v in body.args.items() if k not in ("expressions", FROM_KEY, "where")):
        return None
    from_ = body.args.get(FROM_KEY)
    if from_ is None or any(v for k, v in from_.args.items() if k != "this") or not _plain_source(from_.this):
        return None
    source = from_.this
    alias = (source.alias_or_name or "").lower()
    nested = None
    if isinstance(source, exp.Subquery):
        nested = _simple_body(source.this, types)
        if nested is None:
            return None

    def read(node: exp.Expression) -> exp.Expression | None:
        if not _scalar(node):
            return None
        node = exp.Paren(this=node.copy())  # a root to replace under
        for column in list(node.find_all(exp.Column)):
            if column.args.get("db") or column.args.get("catalog") or (column.table and column.table.lower() != alias):
                return None
            if nested is None:
                if not column.table and column.name.lower() not in (types.get(_table_name(source)) or {}):
                    return None  # a bare name the table is not known to have could belong to an enclosing query
                column.set("table", exp.to_identifier(_MARK))
            else:
                value = nested[1].get(column.name.lower())
                if value is None:
                    return None
                column.replace(value.copy() if isinstance(value, exp.Column) else exp.Paren(this=value.copy()))
        return node.this

    outputs: dict[str, exp.Expression] = {}
    for item in body.expressions:
        name = item.alias_or_name.lower()
        value = read(item.this if isinstance(item, exp.Alias) else item)
        if not name or name in outputs or value is None or not isinstance(item, (exp.Alias, exp.Column)):
            return None
        outputs[name] = value
    conditions = list(nested[2]) if nested is not None else []
    if body.args.get("where") is not None:
        for conjunct in _conjuncts(body.args["where"].this):
            value = read(conjunct)
            if value is None:
                return None
            conditions.append(value)
    return (source if nested is None else nested[0]), outputs, conditions


def _requalified(node: exp.Expression, alias: str) -> exp.Expression:
    node = exp.Paren(this=node.copy())
    for column in node.find_all(exp.Column):
        if column.table == _MARK:
            column.set("table", exp.to_identifier(alias))
    return node.this


def _read_simple_sources(select: exp.Select, sources: list[exp.Expression], conditions: list[exp.Expression], types: dict) -> list[exp.Expression] | None:
    """``sources`` with each simple derived table replaced by its table, the reads of its outputs in ``select``
    and ``conditions`` rewritten and its filter added to ``conditions``; None when a read cannot be resolved."""

    result = []
    for source in sources:
        simple = _simple_body(source.this, types) if isinstance(source, exp.Subquery) else None
        if simple is None:
            result.append(source)
            continue
        table, outputs, filters = simple
        alias = source.alias
        declared_inside = any(
            (n.alias_or_name or "").lower() == alias.lower()
            for part in [select] + conditions
            for n in part.find_all(exp.Table, exp.Subquery)
            if n is not source and not _within(n, sources)
        )
        if declared_inside:
            result.append(source)
            continue
        roots = [select] + conditions
        reads = [c for root in roots for c in root.find_all(exp.Column) if not _within(c, sources)]
        if any(not c.table for c in reads):
            return None  # a bare column could be an output of the derived table
        mine = [c for c in reads if c.table.lower() == alias.lower()]
        if any(c.name.lower() not in outputs or c.args.get("db") for c in mine):
            return None
        for column in mine:
            value = _requalified(outputs[column.name.lower()], alias)
            if isinstance(column.parent, exp.Select) and column.arg_key == "expressions" and not (isinstance(value, exp.Column) and value.name == column.name):
                value = exp.alias_(value, column.name)
            elif not isinstance(value, exp.Column) and isinstance(column.parent, (exp.Binary, exp.Unary, exp.Between, exp.In)) and not isinstance(column.parent, exp.Paren):
                value = exp.Paren(this=value)
            if any(column is root for root in conditions):
                conditions[[column is root for root in conditions].index(True)] = value
            else:
                column.replace(value)
        conditions.extend(_requalified(f, alias) for f in filters)
        replacement = table.copy()
        replacement.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
        result.append(replacement)
    return result


def _within(node: exp.Expression, sources: list[exp.Expression]) -> bool:
    ancestor = node
    while ancestor is not None:
        if any(ancestor is source for source in sources):
            return True
        ancestor = ancestor.parent
    return False


def _filter_reading(condition: exp.Expression) -> exp.Expression:
    """``condition`` without a wrapper that is TRUE exactly when its operand is (module doc)."""

    while True:
        inner = condition
        while isinstance(inner, exp.Paren):
            inner = inner.this
        if type(inner) is exp.Is and isinstance(inner.expression, exp.Boolean) and inner.expression.this is True:
            condition = inner.this
        elif type(inner) is exp.EQ and isinstance(inner.expression, exp.Boolean) and inner.expression.this is True:
            condition = inner.this
        elif type(inner) is exp.EQ and isinstance(inner.this, exp.Boolean) and inner.this.this is True:
            condition = inner.expression
        elif type(inner) is exp.EQ and (_flag_case(inner.this, inner.expression) or _flag_case(inner.expression, inner.this)) is not None:
            condition = _flag_case(inner.this, inner.expression) or _flag_case(inner.expression, inner.this)
        else:
            return inner if isinstance(condition, exp.Paren) and not isinstance(inner, (exp.Or, exp.Xor, exp.And)) else condition


def _integer(node: exp.Expression) -> int | None:
    return int(node.name) if isinstance(node, exp.Literal) and not node.is_string and node.name.isdigit() else None


def _flag_case(case: exp.Expression, wanted: exp.Expression) -> exp.Expression | None:
    """``p`` for ``CASE WHEN p THEN a ELSE b END`` compared with ``a``, two different integer literals."""

    while isinstance(case, exp.Paren):
        case = case.this
    if not isinstance(case, exp.Case) or case.args.get("this") is not None or len(case.args.get("ifs") or []) != 1:
        return None
    arm = case.args["ifs"][0]
    taken, other, target = _integer(arm.args.get("true")), _integer(case.args.get("default")) if case.args.get("default") is not None else None, _integer(wanted)
    if taken is None or other is None or target is None or taken != target or other == target:
        return None
    return arm.this


class _Positions:
    """Alias-independent texts: each qualifier that names a source of the select reads as that source's position."""

    def __init__(self, sources: list[exp.Expression]):
        self.index = {(source.alias_or_name or "").lower(): position for position, source in enumerate(sources)}

    def key(self, node: exp.Expression) -> str:
        copy = node.copy()
        for column in [copy] if isinstance(copy, exp.Column) else list(copy.find_all(exp.Column)):
            position = self.index.get(column.table.lower()) if column.table else None
            if position is not None:
                column.set("table", exp.to_identifier(f"kqsrc{position:03d}"))
        return copy.sql()

    def operand(self, node: exp.Expression) -> tuple[bool, str]:
        """The order of the operands of ``=``: one that reads a column comes before a constant."""

        return node.find(exp.Column) is None, self.key(node)


def _linear(node: exp.Expression, integer_column) -> tuple[dict[str, tuple[int, exp.Column]], int] | None:
    """``node`` as ``sum(coefficient * column) + constant`` over integer columns, or None."""

    if isinstance(node, exp.Paren):
        return _linear(node.this, integer_column)
    if isinstance(node, exp.Literal):
        if node.is_string or not node.name.isdigit():
            return None
        return {}, int(node.name)
    if isinstance(node, exp.Column):
        return ({node.sql(): (1, node)}, 0) if integer_column(node) else None
    if isinstance(node, exp.Neg):
        inner = _linear(node.this, integer_column)
        return None if inner is None else ({k: (-c, col) for k, (c, col) in inner[0].items()}, -inner[1])
    if type(node) in (exp.Add, exp.Sub):
        left, right = _linear(node.this, integer_column), _linear(node.expression, integer_column)
        if left is None or right is None:
            return None
        sign = 1 if type(node) is exp.Add else -1
        terms = dict(left[0])
        for key, (coefficient, column) in right[0].items():
            terms[key] = (terms[key][0] + sign * coefficient if key in terms else sign * coefficient, column)
        return terms, left[1] + sign * right[1]
    if type(node) is exp.Mul:
        left, right = _linear(node.this, integer_column), _linear(node.expression, integer_column)
        if left is None or right is None:
            return None
        if left[0] and right[0]:
            return None  # a product of columns
        (terms, constant), factor = (left, right[1]) if left[0] else (right, left[1])
        if not terms:
            return {}, left[1] * right[1]
        return {k: (c * factor, col) for k, (c, col) in terms.items()}, constant * factor
    return None


def _number(value: int) -> exp.Expression:
    return exp.Literal.number(value) if value >= 0 else exp.Neg(this=exp.Literal.number(-value))


def _linear_form(terms: dict[str, tuple[int, exp.Column]], constant: int, positions: _Positions) -> exp.Expression:
    result: exp.Expression | None = None
    for _, (coefficient, column) in sorted(terms.items(), key=lambda item: positions.key(item[1][1])):
        magnitude = abs(coefficient) if result is not None else coefficient
        term = column.copy() if magnitude == 1 else exp.Mul(this=_number(magnitude), expression=column.copy())
        if result is None:
            result = term
        else:
            result = (exp.Sub if coefficient < 0 else exp.Add)(this=result, expression=term)
    if result is None:
        return _number(constant)
    if constant:
        result = (exp.Sub if constant < 0 else exp.Add)(this=result, expression=exp.Literal.number(abs(constant)))
    return result


def _spell_sums(select: exp.Select, sources: list[exp.Expression], positions: _Positions, types: dict) -> None:
    tables = {(s.alias_or_name or "").lower(): _table_name(s) for s in sources if isinstance(s, exp.Table)}

    def integer_column(column: exp.Column) -> bool:
        if not column.table or column.args.get("db") or column.args.get("catalog") or not isinstance(column.this, exp.Identifier):
            return False
        table = tables.get(column.table.lower())
        declared = (types.get(table) or {}).get(column.name.lower()) if table is not None else None
        return declared is not None and str(declared).upper().split("(")[0].strip() in _INTEGER_TYPES

    def visit(node: exp.Expression) -> None:
        if isinstance(node, (exp.Query, exp.Subquery)):
            return
        inner = node
        while isinstance(inner, exp.Paren):
            inner = inner.this
        if type(inner) in (exp.Add, exp.Sub, exp.Mul, exp.Neg):
            linear = _linear(inner, integer_column)
            if linear is not None:
                spelled = _linear_form(linear[0], linear[1], positions)
                if isinstance(node.parent, (exp.Binary, exp.Unary, exp.Between, exp.In)) and not isinstance(node.parent, exp.Paren):
                    spelled = exp.Paren(this=spelled)  # an operator around it: keep it one operand
                if spelled.sql() != node.sql():
                    node.replace(spelled)
                return
        for child in list(node.iter_expressions()):
            visit(child)

    for window in [w for w in select.find_all(exp.Window) if _own(w, select)]:
        for child in list(window.iter_expressions()):
            visit(child)


def spell_windowed_input(select: exp.Select, types: dict | None = None) -> exp.Select | None:
    """``select`` with its sources, conditions and windowed sums in the one spelling of the module doc, or None."""

    if any(select.args.get(key) for key in _BANNED):
        return None
    if not any(_own(w, select) for w in select.find_all(exp.Window)):
        return None
    if any(_own(a, select) and not is_window_function(a) for a in select.find_all(exp.AggFunc)):
        return None
    from_ = select.args.get(FROM_KEY)
    if from_ is None or any(v for k, v in from_.args.items() if k != "this"):
        return None
    joins = list(select.args.get("joins") or [])
    sources = [from_.this] + [join.this for join in joins]
    if not all(_plain_inner(join) for join in joins) or not all(_plain_source(source) for source in sources):
        return None
    aliases = [(source.alias_or_name or "").lower() for source in sources]
    if "" in aliases or len(set(aliases)) != len(aliases):
        return None
    before = select.sql()
    copy = select.copy()
    from_ = copy.args[FROM_KEY]
    joins = list(copy.args.get("joins") or [])
    sources = [from_.this] + [join.this for join in joins]
    conditions = [c for join in joins if join.args.get("on") is not None for c in _conjuncts(join.args["on"])]
    if copy.args.get("where") is not None:
        conditions += _conjuncts(copy.args["where"].this)
    types = {str(t).lower(): {str(c).lower(): v for c, v in cols.items()} for t, cols in (types or {}).items()}
    sources = _read_simple_sources(copy, sources, conditions, types)
    if sources is None:
        return None
    conditions = [c for condition in conditions for c in _conjuncts(_filter_reading(condition))]
    if all(isinstance(source, exp.Table) for source in sources):
        sources.sort(key=_table_name)  # stable: two reads of one table keep their order
    positions = _Positions(sources)
    spelled: dict[str, exp.Expression] = {}
    for condition in conditions:
        condition = condition.copy()
        if type(condition) is exp.EQ and positions.operand(condition.expression) < positions.operand(condition.this):
            condition = exp.EQ(this=condition.expression, expression=condition.this)
        spelled.setdefault(positions.key(condition), condition)
    copy.set(FROM_KEY, exp.From(this=sources[0].copy()))
    copy.set("joins", [exp.Join(this=source.copy()) for source in sources[1:]] or None)
    copy.set("where", exp.Where(this=_and_all([spelled[key] for key in sorted(spelled)])) if spelled else None)
    _spell_sums(copy, sources, positions, types)
    return copy if copy.sql() != before else None
