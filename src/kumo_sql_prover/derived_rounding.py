"""MySQL rounds a quotient or ``AVG`` when a derived table or CTE it materialises stores it.

A division of exact numbers has a declared scale four digits above its dividend's, but the value MySQL computes carries
more digits than that (nine fractional digits for whole-number operands, cut off, not rounded). Used directly, those
digits take part in what follows: ``1 / 3 * 100`` is ``0.333333333 * 100``. When the quotient is an output of a derived
table that MySQL has to materialise, it is stored in a ``DECIMAL`` column of the declared scale and rounded to it:
``0.3333``, and ``0.3333 * 100`` is ``33.3300``. ``AVG`` of exact numbers does the same. The provers read a derived table's
output where it is used, as if the table were merged, so ``SELECT t.q * 100 FROM (SELECT SUM(a) / COUNT(*) AS q FROM r) t``
was proved equal to ``SELECT SUM(a) / COUNT(*) * 100 FROM r``; MySQL 8.0.46 returns ``33.3300`` and ``33.3333`` for them on
three rows (Kumo_SQL_Prover issue 45).

What rounds (measured on MySQL 8.0.46): a derived table or CTE with an aggregate, ``GROUP BY``, ``HAVING``, ``DISTINCT``,
``LIMIT``, a window function or a set operation, and a recursive CTE. What does not: a derived table or CTE MySQL merges
(a plain select-project-join, also when it is read twice or has a subquery in its select list), a scalar subquery, and the
subqueries of ``IN``, ``ANY`` and ``EXISTS``.

How it is done. Before the provers see a pair in the ``mysql`` dialect, each output of a materialising derived table that
holds such a value (a division of numbers not known to be floating point, ``AVG``, and any arithmetic, ``SUM``, ``CASE`` or
function built on one; see :func:`_Reading.hidden`) is wrapped in a call to ``KQ_STORED_QUOTIENT(..)``, a function the
provers have no definition for. They read an unknown function as "equal arguments, equal result", which is what the
storing is: a function of the value it stores. So the wrapped output matches only the same wrapped output, and a proof that
needed the table to be merged is lost. The wrapper is left off where it cannot be seen: an output the statement only
returns, as a plain item of its outermost ``SELECT`` (the quotient is rounded to the same four decimals directly and through
a derived table). It is also off for the columns of a merged derived table, for a value already rounded to at most four
decimals (``ROUND(.., 2)``) or cast to a decimal, and for a division of floating-point operands.

What is returned. A proof of the wrapped pair holds for the pair as written, because the wrapper only forgets what the
function does. A pair the wrapped pair does not prove is run as written: a refutation of it is kept as it was; a proof of it
becomes ``NOT_PROVEN`` with the reason, because it is the one this module exists to remove; anything else is returned as
it was.

Limits. ``ROUND(t.q * 100, 2)`` over a derived quotient is exactly the direct ``ROUND(S / C * 100, 2)`` (rounding a ratio to
four decimals and multiplying by 100 is rounding the percentage to two); the wrapper does not know the declared scale, so it
declines that pair too. A view, or a table function, is not read. Other dialects were not checked: DuckDB divides into a
floating-point number.
"""

from __future__ import annotations

import dataclasses
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation, localcontext
from typing import Callable

import sqlglot
from sqlglot import exp

from . import declared_scale as scales

DIALECTS = frozenset({"mysql"})
WRAPPER = "KQ_STORED_QUOTIENT"
DECLINED = (
    "unsupported: MySQL stores a quotient or AVG of exact numbers rounded to its declared scale when a derived table or CTE "
    "it materialises holds it, and one query reads it from there in arithmetic, a comparison or ROUND while the other does "
    "not (Kumo_SQL_Prover issue 45)"
)

DECLINED_SCALE = (
    "unsupported: MySQL rounds a quotient or AVG of exact numbers to the number of decimals its expression declares, and "
    "the two queries return one at a different declared scale, or one this check cannot work out (a decimal literal such "
    "as 1.0 in a factor adds a decimal; Kumo_SQL_Prover issue 47)"
)

_ACTIVE: ContextVar[bool] = ContextVar("kumosql_derived_rounding_active", default=False)
_SET_OPERATIONS = (exp.Union, exp.Except, exp.Intersect)
_FLOAT_TYPES = ("DOUBLE", "FLOAT", "REAL")
_NUMBER_TARGETS = {
    exp.DataType.Type.DECIMAL, exp.DataType.Type.DOUBLE, exp.DataType.Type.FLOAT, exp.DataType.Type.INT, exp.DataType.Type.BIGINT,
    exp.DataType.Type.SMALLINT, exp.DataType.Type.TINYINT, exp.DataType.Type.UBIGINT, exp.DataType.Type.UINT, exp.DataType.Type.BIT,
}
_FLOAT_FUNCTIONS = tuple(c for c in (getattr(exp, n, None) for n in ("Sqrt", "Exp", "Ln", "Log", "Pow", "Rand", "Sin", "Cos", "Tan", "Atan", "Asin", "Acos", "Stddev", "StddevPop", "StddevSamp", "Variance", "VariancePop")) if c)
_MAX_DEPTH = 8
_SCALE_DEPTH = 40  # a scale is worked out through a whole expression: a TPC-H revenue sum is eight levels deep


def _from(select: exp.Select):
    return select.args.get("from_") or select.args.get("from")


def _body(node):
    while isinstance(node, exp.Subquery):
        node = node.this
    return node


def _leaves(body) -> list[exp.Select]:
    body = _body(body)
    if isinstance(body, exp.Select):
        return [body]
    if isinstance(body, _SET_OPERATIONS):
        return _leaves(body.this) + _leaves(body.expression)
    return []


def _sources(select: exp.Select) -> list[exp.Expression]:
    out = []
    if _from(select) is not None:
        out.append(_from(select).this)
    out.extend(join.this for join in select.args.get("joins") or [])
    return [s.this if isinstance(s, exp.Lateral) else s for s in out]


def _own_nodes(select: exp.Select):
    """The nodes of ``select`` outside any nested select."""

    stack = [child for child in select.iter_expressions()]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (exp.Select, exp.Subquery, *_SET_OPERATIONS)):
            continue
        stack.extend(node.iter_expressions())


def _materialises(body, recursive: bool = False) -> bool:
    body = _body(body)
    if isinstance(body, _SET_OPERATIONS) or recursive:
        return True
    if not isinstance(body, exp.Select):
        return False
    for key in ("group", "having", "limit", "offset", "distinct", "qualify", "windows"):
        if body.args.get(key):
            return True
    return any(isinstance(node, (exp.AggFunc, exp.Window)) for node in _own_nodes(body))


def _output_name(item: exp.Expression) -> str:
    return (item.alias_or_name or "").lower()


def _exact_literal_quotient(node: exp.Div) -> bool:
    """True for a quotient of two number literals whose value has no digit past the four MySQL declares (``24 / 100``)."""

    parts = []
    for side in (node.this, node.expression):
        while isinstance(side, exp.Paren):
            side = side.this
        if not isinstance(side, exp.Literal) or side.is_string:
            return False
        try:
            parts.append(Decimal(side.this))
        except InvalidOperation:
            return False
    dividend, divisor = parts
    if divisor == 0 or "e" in node.this.sql().lower() or "e" in node.expression.sql().lower():
        return False
    scale = max(0, -dividend.as_tuple().exponent)
    with localcontext() as context:
        context.prec = 100
        quotient = dividend / divisor
        return quotient * divisor == dividend and -quotient.as_tuple().exponent <= scale + 4


class _Reading:
    """One statement: its derived tables and CTEs, which of them materialise, and which outputs hide digits."""

    def __init__(self, tree: exp.Expression, types: dict | None, schema: dict | None = None):
        self.tree = tree
        self.schema = {str(t).lower(): {str(c).lower() for c in cols} for t, cols in (schema or {}).items() if isinstance(cols, (list, tuple, set, frozenset, dict))}
        self.through = False  # True: a column of a materialising derived table is read as the value it stores
        self.types = {str(t).lower(): {str(c).lower(): str(v).upper() for c, v in cols.items()} for t, cols in (types or {}).items() if isinstance(cols, dict)}
        self.ctes: dict[str, tuple[exp.CTE, bool]] = {}
        for cte in tree.find_all(exp.CTE):
            with_ = cte.parent
            self.ctes[cte.alias.lower()] = (cte, bool(with_ is not None and with_.args.get("recursive")))

    # -- what a source is -------------------------------------------------------------------------------------

    def derived(self, source: exp.Expression):
        """``(body, materialises)`` of a derived table or CTE reference, or None for a base table and anything else."""

        if isinstance(source, exp.Subquery):
            return _body(source), _materialises(source.this)
        if isinstance(source, exp.Table) and not source.args.get("db") and source.name.lower() in self.ctes:
            cte, recursive = self.ctes[source.name.lower()]
            return _body(cte.this), _materialises(cte.this, recursive)
        return None

    def _enclosing(self, select: exp.Select):
        parent = select.parent
        while parent is not None and not isinstance(parent, exp.Select):
            parent = parent.parent
        return parent

    def _candidates(self, column: exp.Column, select: exp.Select):
        """``(source, body-or-None)`` pairs a column read may refer to, innermost scope first that has any."""

        scope = select
        while scope is not None:
            found = []
            for source in _sources(scope):
                alias = (source.alias_or_name or "").lower()
                if column.table and alias != column.table.lower():
                    continue
                found.append(source)
            if found:
                return found
            scope = self._enclosing(scope)
        return []

    # -- types ------------------------------------------------------------------------------------------------

    def floating(self, node: exp.Expression, select: exp.Select, depth: int = 0) -> bool:
        """True when ``node`` is known to be a floating-point value (a division by it gives a floating-point number)."""

        if depth > _MAX_DEPTH:
            return False
        if isinstance(node, exp.Literal):
            return not node.is_string and "e" in node.this.lower()
        if isinstance(node, (exp.Cast, exp.TryCast)):
            return node.to.this.name in _FLOAT_TYPES
        if isinstance(node, _FLOAT_FUNCTIONS):
            return True
        if isinstance(node, exp.Column):
            if isinstance(node.this, exp.Star):
                return False
            for source in self._candidates(node, select):
                if isinstance(source, exp.Table) and source.name.lower() in self.types and source.name.lower() not in self.ctes:
                    declared = self.types[source.name.lower()].get(node.name.lower())
                    if declared and declared.startswith(_FLOAT_TYPES):
                        return True
                    continue
                found = self.derived(source)
                if found is None:
                    continue
                for leaf in _leaves(found[0]):
                    for item in leaf.expressions:
                        if _output_name(item) == node.name.lower() and self.floating(item.unalias(), leaf, depth + 1):
                            return True
            return False
        if isinstance(node, (exp.Subquery, exp.Select)):
            return False
        return any(self.floating(child, select, depth + 1) for child in node.iter_expressions())

    # -- hidden digits ----------------------------------------------------------------------------------------

    def hidden(self, node: exp.Expression, select: exp.Select, depth: int = 0) -> bool:
        """True when ``node`` may carry more digits than its declared scale (a division, ``AVG``, or a value built on one).

        A column of a materialising derived table or CTE does not: the table stored it. A column of a merged one is the
        expression it names. A value rounded to at most four decimals (``ROUND(x, 2)``) or cast to a number has none,
        nor has a comparison, a count, or a floating-point division."""

        if depth > _MAX_DEPTH:
            return True
        if isinstance(node, exp.Anonymous) and node.name.upper() == WRAPPER:
            return False
        if isinstance(node, exp.Div):
            return not (self.floating(node.this, select) or self.floating(node.expression, select) or _exact_literal_quotient(node))
        if isinstance(node, exp.Avg):
            return not self.floating(node.this, select)
        if isinstance(node, (exp.Literal, exp.Null, exp.Boolean, exp.Star, exp.Predicate, exp.Connector, exp.Not, exp.Count, exp.IntDiv)):
            return False
        if isinstance(node, (exp.Cast, exp.TryCast)):
            return False if node.to.this in _NUMBER_TARGETS else self.hidden(node.this, select, depth + 1)
        if isinstance(node, (exp.Round, getattr(exp, "Truncate", exp.Round))):
            digits = node.args.get("decimals") or node.args.get("expression")
            if digits is None:  # ROUND(x) rounds to a whole number
                return False
            if isinstance(digits, exp.Literal) and not digits.is_string and digits.this.isdigit() and int(digits.this) <= 4:
                return False
            return self.hidden(node.this, select, depth + 1)
        if isinstance(node, exp.Column):
            return self._column_hidden(node, select, depth)
        if isinstance(node, exp.Window):
            return self.hidden(node.this, select, depth + 1)
        if isinstance(node, exp.Case):
            arms = [branch.args.get("true") for branch in node.args.get("ifs") or []] + [node.args.get("default")]
            return any(arm is not None and self.hidden(arm, select, depth + 1) for arm in arms)
        if isinstance(node, exp.If):
            return any(arm is not None and self.hidden(arm, select, depth + 1) for arm in (node.args.get("true"), node.args.get("false")))
        if isinstance(node, (exp.Subquery, exp.Select)):
            inner = _body(node)
            return any(self.hidden(item.unalias(), leaf, depth + 1) for leaf in _leaves(inner) for item in leaf.expressions)
        return any(self.hidden(child, select, depth + 1) for child in node.iter_expressions())

    def _column_hidden(self, column: exp.Column, select: exp.Select, depth: int) -> bool:
        if isinstance(column.this, exp.Star):
            return False
        for source in self._candidates(column, select):
            found = self.derived(source)
            if found is None:
                continue
            body, materialises = found
            if materialises and not self.through:
                continue
            for leaf in _leaves(body):
                named = [item for item in leaf.expressions if _output_name(item) == column.name.lower()]
                if named:
                    if self.hidden(named[0].unalias(), leaf, depth + 1):
                        return True
                elif any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in leaf.expressions):
                    if self.hidden(exp.column(column.name), leaf, depth + 1):
                        return True
        return False

    # -- declared scales (issue 47) ---------------------------------------------------------------------------

    def hidden_through(self, node: exp.Expression, select: exp.Select) -> bool:
        """:meth:`hidden`, following a column of a materialising derived table to the expression it stores."""

        self.through = True
        try:
            return self.hidden(node, select)
        finally:
            self.through = False

    def scale(self, node: exp.Expression, select: exp.Select, depth: int = 0):
        """The declared scale of ``node`` (see :mod:`kumo_sql_prover.declared_scale`), or None when it is not known."""

        if depth > _SCALE_DEPTH:
            return None
        deeper = depth + 1
        arg = lambda child: self.scale(child, select, deeper)  # noqa: E731
        if isinstance(node, exp.Distinct):  # AVG(DISTINCT x): the one expression it is over
            return arg(node.expressions[0]) if len(node.expressions) == 1 else None
        if isinstance(node, (exp.Paren, exp.Neg, exp.Abs, exp.Min, exp.Max, exp.Sum, exp.Window, exp.Filter)):
            return arg(node.this) if node.this is not None else None
        if isinstance(node, exp.Literal):
            return None if node.is_string else scales.of_literal(node.this)
        if isinstance(node, exp.Null):
            return scales.largest([])  # None: only a COALESCE or CASE arm gives it a meaning, handled there
        if isinstance(node, (exp.Predicate, exp.Connector, exp.Not, exp.Boolean, exp.Count, exp.IntDiv)):
            return scales.constant(0)
        if isinstance(node, (exp.Add, exp.Sub, exp.Mod)):
            return scales.largest([arg(node.this), arg(node.expression)])
        if isinstance(node, exp.Mul):
            return scales.plus(arg(node.this), arg(node.expression))
        if isinstance(node, exp.Div):
            dividend, divisor = arg(node.this), arg(node.expression)
            if divisor is None or dividend is None:
                return None
            if scales.FLOAT in (dividend, divisor):
                return scales.FLOAT
            return scales.add_digits(dividend, scales.DIVISION_INCREMENT)
        if isinstance(node, exp.Avg):
            return scales.add_digits(arg(node.this), scales.DIVISION_INCREMENT)
        if isinstance(node, (exp.Round, exp.Trunc)):
            inner = arg(node.this)
            digits = node.args.get("decimals") or node.args.get("expression")
            if digits is None:
                return scales.rounded(inner, 0)
            if isinstance(digits, exp.Neg) and isinstance(digits.this, exp.Literal):
                return scales.rounded(inner, 0)
            if isinstance(digits, exp.Literal) and not digits.is_string and digits.this.isdigit():
                return scales.rounded(inner, int(digits.this))
            return None
        if isinstance(node, (exp.Floor, exp.Ceil)):
            inner = arg(node.this)
            return inner if inner in (None, scales.FLOAT) else scales.constant(0)
        if isinstance(node, (exp.Cast, exp.TryCast)):
            return self._cast_scale(node)
        if isinstance(node, exp.Case):
            arms = [branch.args.get("true") for branch in node.args.get("ifs") or []] + [node.args.get("default")]
            return scales.largest([arg(a) for a in arms if a is not None and not isinstance(a, exp.Null)])
        if isinstance(node, exp.If):
            return scales.largest([arg(a) for a in (node.args.get("true"), node.args.get("false")) if a is not None and not isinstance(a, exp.Null)])
        if isinstance(node, (exp.Coalesce, exp.Greatest, exp.Least)):
            operands = [node.this, *node.expressions]
            return scales.largest([arg(a) for a in operands if a is not None and not isinstance(a, exp.Null)])
        if isinstance(node, exp.Nullif):
            return arg(node.this)
        if isinstance(node, exp.Column):
            return self._column_scale(node, select, depth)
        if isinstance(node, (exp.Subquery, exp.Select)):
            return self._item_scale(_body(node), 0, depth)
        return None

    def _cast_scale(self, node):
        target = node.to
        if target.this == exp.DataType.Type.DECIMAL:
            params = [p.this for p in target.expressions if isinstance(p, exp.DataTypeParam)]
            if len(params) >= 2 and isinstance(params[1], exp.Literal) and params[1].this.isdigit():
                return scales.constant(int(params[1].this))
            return scales.constant(0)
        if target.this in (exp.DataType.Type.DOUBLE, exp.DataType.Type.FLOAT):
            return scales.FLOAT
        if target.this in _NUMBER_TARGETS:
            return scales.constant(0)
        return None

    def _item_scale(self, body, position: int, depth: int):
        """The scale of output ``position`` of a select or set operation: the largest over its branches."""

        found = []
        for leaf in _leaves(body):
            if position >= len(leaf.expressions):
                return None
            item = leaf.expressions[position]
            if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
                return None
            found.append(self.scale(item.unalias(), leaf, depth + 1))
        return scales.largest(found)

    def _derived_column_scale(self, body, name: str, depth: int):
        """The scale of column ``name`` of a derived table: the largest over its branches; None where a branch lacks it."""

        found = []
        for leaf in _leaves(body):
            named = [item for item in leaf.expressions if _output_name(item) == name]
            if named:
                found.append(self.scale(named[0].unalias(), leaf, depth + 1))
            elif any(isinstance(i, exp.Star) or (isinstance(i, exp.Column) and isinstance(i.this, exp.Star)) for i in leaf.expressions):
                found.append(self.scale(exp.column(name), leaf, depth + 1))
            else:
                return None
        return scales.largest(found)

    def _column_scale(self, node: exp.Column, select: exp.Select, depth: int):
        if isinstance(node.this, exp.Star):
            return None
        name = node.name.lower()
        options = []
        for source in self._candidates(node, select):
            found = self.derived(source)
            if found is not None:
                leaves = _leaves(found[0])
                first = leaves[0].expressions if leaves else []
                if any(_output_name(i) == name or isinstance(i, exp.Star) or (isinstance(i, exp.Column) and isinstance(i.this, exp.Star)) for i in first):
                    options.append(self._derived_column_scale(found[0], name, depth))
                continue
            if isinstance(source, exp.Table):
                columns = self.types.get(source.name.lower())
                if columns is not None:
                    if name in columns:  # a typed table that lacks the column is not the one it names
                        options.append(scales.of_type(columns[name]))
                    continue
                known = self.schema.get(source.name.lower())
                if known is not None and name not in known:  # the schema says this table has no such column
                    continue
                # An untyped table: the column is a symbol known by its name, the same in both queries whatever the tables are
                # called or whether the read says which one (the schema is the same for both).
                options.append(scales.column(name))
        distinct = list(dict.fromkeys(options))
        return distinct[0] if len(distinct) == 1 else None

    def expanded_outputs(self, leaf: exp.Select):
        """The output expressions of a select with each star replaced by the columns of the derived tables it covers.

        None when a star covers a base table, a ``USING`` or ``NATURAL`` join, or a derived table whose first branch has
        a star of its own."""

        outputs = []
        for item in leaf.expressions:
            if not (isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star))):
                outputs.append(item.unalias())
                continue
            if any(j.args.get("using") or str(j.args.get("method") or "").upper() == "NATURAL" for j in leaf.args.get("joins") or []):
                return None
            wanted = item.table.lower() if isinstance(item, exp.Column) else ""
            covered = [s for s in _sources(leaf) if not wanted or (s.alias_or_name or "").lower() == wanted]
            if not covered:
                return None
            for source in covered:
                found = self.derived(source)
                leaves = _leaves(found[0]) if found else []
                if not leaves or any(isinstance(i, exp.Star) or (isinstance(i, exp.Column) and isinstance(i.this, exp.Star)) for i in leaves[0].expressions):
                    return None
                alias = source.alias_or_name
                outputs.extend(exp.column(_output_name(i), table=alias or None) for i in leaves[0].expressions)
        return outputs

    # -- reads of a derived table's columns -------------------------------------------------------------------

    def _plain(self, node: exp.Expression) -> bool:
        """True for an item of the outermost ``SELECT`` that is just the column: it is returned, not computed on."""

        root = self.tree
        if not isinstance(root, exp.Select) or root.args.get("distinct") or root.args.get("group"):
            return False
        order = root.args.get("order")
        if order is not None and any(isinstance(n, exp.Literal) and not n.is_string for n in order.find_all(exp.Literal)):
            return False  # ORDER BY 1 sorts on the stored value
        item = node.parent if isinstance(node.parent, exp.Alias) else node
        return item.parent is root and item.arg_key == "expressions"

    def needs_wrapper(self, aliases: set[str], name: str) -> bool:
        """True when the column ``name`` of the relation known as any of ``aliases`` is read other than as a plain output.

        An output with no name of its own (``SELECT a / b``) is called by its text, which is not known here: any column
        read counts."""

        for node in self.tree.find_all(exp.Column):
            if isinstance(node.this, exp.Star):
                if node.table.lower() in aliases and not self._plain(node):
                    return True
                continue
            if (name and node.name.lower() != name) or (node.table and node.table.lower() not in aliases):
                continue
            if not self._plain(node):
                return True
        for star in self.tree.find_all(exp.Star):
            if isinstance(star.parent, exp.Column):
                continue
            select = star.parent
            if isinstance(select, exp.Select) and any(((s.alias_or_name or "").lower() in aliases) for s in _sources(select)):
                if not self._plain(star):
                    return True
        for join in self.tree.find_all(exp.Join):
            if join.args.get("using") and any(not name or i.name.lower() == name for i in join.args["using"]):
                return True
            if str(join.args.get("method") or "").upper() == "NATURAL":
                return True
        return False

    # -- the rewrite ------------------------------------------------------------------------------------------

    def wrap(self) -> int:
        """Wrap, in place, the hidden outputs of materialising derived tables and CTEs that are read other than plainly."""

        jobs: list[tuple[exp.Expression, exp.Expression, int]] = []
        bodies: list[tuple[exp.Expression, set[str]]] = []
        for sub in self.tree.find_all(exp.Subquery):
            if isinstance(sub.parent, (exp.From, exp.Join, exp.Lateral)) and sub.alias:
                bodies.append((sub.this, {sub.alias.lower()}))
        for name, (cte, recursive) in self.ctes.items():
            aliases = {name} | {t.alias.lower() for t in self.tree.find_all(exp.Table) if t.name.lower() == name and t.alias}
            bodies.append((cte.this, aliases))
        for body, aliases in bodies:
            if not _materialises(body, any(self.ctes.get(a, (None, False))[1] for a in aliases)):
                continue
            leaves = _leaves(body)
            first = leaves[0].expressions if leaves else []
            for leaf in leaves:
                for position, item in enumerate(leaf.expressions):
                    if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
                        continue
                    name = _output_name(first[position]) if position < len(first) else _output_name(item)
                    if self.hidden(item.unalias(), leaf) and self.needs_wrapper(aliases, name):
                        jobs.append((item, body, position))
        # The second argument is the scale the table stores the value at: two scales that differ are two functions. All are
        # read before the first wrapper goes in, which would hide the value it wraps.
        stored_scale = {}
        for item, body, position in jobs:
            inner = item.this if isinstance(item, exp.Alias) else item
            stored_scale.setdefault(id(item), exp.Literal.string(scales.tag(self._item_scale(body, position, 0), inner.sql(dialect="mysql"))))
        done = set()
        for item, _body_, _position in jobs:
            if id(item) in done:
                continue
            done.add(id(item))
            inner = item.this if isinstance(item, exp.Alias) else item
            stored = stored_scale[id(item)]
            wrapped = exp.Anonymous(this=WRAPPER, expressions=[inner.copy(), stored])
            if isinstance(item, exp.Alias):
                item.set("this", wrapped)
            else:
                label = item.alias_or_name or item.sql(dialect="mysql")
                item.replace(exp.alias_(wrapped, label, quoted=not item.alias_or_name))
        return len(done)


@dataclasses.dataclass(frozen=True)
class Found:
    left_sql: str
    right_sql: str
    wrapped: int


def reading(left_sql: str, right_sql: str, dialect: str | None, types: dict | None = None) -> Found | None:
    """The pair with the hidden outputs of materialising derived tables wrapped, or None when there are none."""

    if (dialect or "bigquery").lower() not in DIALECTS:
        return None
    texts, total = [], 0
    for sql in (left_sql, right_sql):
        low = sql.lower()
        if "/" not in sql and "avg" not in low:
            texts.append(sql)
            continue
        try:
            tree = sqlglot.parse_one(sql, read="mysql")
        except Exception:  # noqa: BLE001 - the prover reports the parse error itself
            return None
        count = _Reading(tree, types).wrap() if isinstance(tree, (exp.Select, *_SET_OPERATIONS)) else 0
        total += count
        texts.append(tree.sql(dialect="mysql") if count else sql)
    return Found(texts[0], texts[1], total) if total else None


def _output_scales(sql: str, types: dict | None, schema: dict | None = None):
    """``(hidden, scales)`` per output column of a statement: does it return a value with digits past its declared scale,
    and at which declared scale (an unknown one is written ``?`` and the expression's text). None when the outputs cannot
    be listed (a star, a statement that is not a query)."""

    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except Exception:  # noqa: BLE001 - the prover reports the parse error itself
        return None
    if not isinstance(tree, (exp.Select, *_SET_OPERATIONS)):
        return None
    reading = _Reading(tree, types)
    leaves = _leaves(tree)
    if not leaves or len({len(leaf.expressions) for leaf in leaves}) != 1:
        return None
    expanded = [reading.expanded_outputs(leaf) for leaf in leaves]
    if any(items is None for items in expanded) or len({len(items) for items in expanded}) != 1:
        return None
    columns = []
    for position in range(len(expanded[0])):
        items = [(leaf, outputs[position]) for leaf, outputs in zip(leaves, expanded)]
        hidden = any(reading.hidden_through(item, leaf) for leaf, item in items)
        scale = scales.largest([reading.scale(item, leaf) for leaf, item in items])
        columns.append((hidden, scale, scales.tag(scale, " UNION ".join(item.sql(dialect="mysql") for _, item in items))))
    return columns


def scale_differs(left_sql: str, right_sql: str, dialect: str | None, types: dict | None = None, schema: dict | None = None) -> bool:
    """True when a column one query returns with digits past its declared scale is not at the other's declared scale.

    ``paid * 1.0 / due`` and ``paid / due`` return the same quotient at five and at four decimals (issue 47). A column is
    compared when either query's value carries such digits; a scale this module cannot work out matches only the same
    expression text. When only one side carries them and its declared scale is the larger, the other side's value is a
    number that scale holds exactly (``AVG(x)`` over a group of one row against ``x``), and the pair is not a difference."""

    if (dialect or "bigquery").lower() not in DIALECTS:
        return False
    if all("/" not in sql and "avg" not in sql.lower() for sql in (left_sql, right_sql)):
        return False
    left, right = _output_scales(left_sql, types, schema), _output_scales(right_sql, types, schema)
    if left is None or right is None or len(left) != len(right):
        return False
    return any(_loses_digits(a, b) for a, b in zip(left, right))


def _loses_digits(first, second) -> bool:
    """Whether one column of a pair proved equal could be returned with different digits (see :func:`scale_differs`)."""

    (first_hidden, first_scale, first_tag), (second_hidden, second_scale, second_tag) = first, second
    if not (first_hidden or second_hidden) or first_tag == second_tag:
        return False
    if first_hidden and second_hidden:
        return True
    # One value is exact at its declared scale. Proved equal to it, the other is that number, which the other's declared
    # scale (the larger one) holds without rounding: AVG(x) over a group of one row is x, 5 is 5.0000.
    hidden, exact = (first_scale, second_scale) if first_hidden else (second_scale, first_scale)
    return not scales.dominates(hidden, exact)


def proved(prove: Callable, left_sql: str, right_sql: str, kwargs: dict):
    """The proof of the pair as MySQL reads it: see :func:`_proved_as_stored` and :func:`scale_differs`.

    A proof of a pair that returns a quotient at two different declared scales is declined."""

    result = _proved_as_stored(prove, left_sql, right_sql, kwargs)
    if _ACTIVE.get() or result.status not in (_proof_statuses()):
        return result
    if not scale_differs(left_sql, right_sql, kwargs.get("dialect"), kwargs.get("types"), kwargs.get("schema")):
        return result
    from .smt_equivalence import SmtEquivalenceResult, SmtStatus

    return SmtEquivalenceResult(SmtStatus.NOT_PROVEN, DECLINED_SCALE, assumptions=tuple(result.assumptions))


def _proof_statuses():
    from .smt_equivalence import SmtStatus

    return (SmtStatus.PROVEN_EQUIVALENT, SmtStatus.PROVEN_CONDITIONALLY)


def _proved_as_stored(prove: Callable, left_sql: str, right_sql: str, kwargs: dict):
    """``prove(left_sql, right_sql, **kwargs)`` for a pair whose derived-table quotients are read as MySQL stores them.

    A pair with nothing to wrap is proved as it is. Otherwise see the module docstring: a proof of the wrapped pair is
    returned; a proof of the pair as written, which the wrapped one does not give, is declined."""

    if _ACTIVE.get():
        return prove(left_sql, right_sql, **kwargs)
    found = reading(left_sql, right_sql, kwargs.get("dialect"), kwargs.get("types"))
    if found is None:
        return prove(left_sql, right_sql, **kwargs)
    from .smt_equivalence import SmtEquivalenceResult, SmtStatus

    proofs = (SmtStatus.PROVEN_EQUIVALENT, SmtStatus.PROVEN_CONDITIONALLY)
    token = _ACTIVE.set(True)
    try:
        result = prove(found.left_sql, found.right_sql, **kwargs)
        if result.status in proofs:
            return result
        plain = prove(left_sql, right_sql, **kwargs)
    finally:
        _ACTIVE.reset(token)
    if plain.status in proofs:
        return SmtEquivalenceResult(SmtStatus.NOT_PROVEN, DECLINED, assumptions=tuple(plain.assumptions))
    return plain
