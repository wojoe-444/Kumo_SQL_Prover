"""A MySQL sum of a truth test is a sum of ones: ``SUM(p)`` written ``SUM(CASE WHEN p THEN 1 ELSE 0 END)``.

    SELECT s.buyer_id
    FROM sales AS s JOIN product AS p ON s.product_id = p.product_id
    GROUP BY s.buyer_id
    HAVING SUM(p.name = 'S8') > 0 AND SUM(p.name = 'iPhone') = 0

In MySQL a comparison is a number: 1 when it is TRUE, 0 when it is FALSE, NULL when it is unknown. So people count
the rows that meet a condition by summing the condition itself. The rules that recognise a count of matching rows
(``counted_presence``, ``conditional_count_sums``, ``counted_sum_ratios``) read the ``CASE`` spelling only, and the
query above did not meet the same question asked with ``CASE``, with a filter and ``NOT IN``, or with ``COUNT``.

**Why it holds.** ``SUM`` skips NULL. Over the rows of a group, ``SUM(p)`` is the number of rows on which ``p`` is TRUE
when ``p`` is known on at least one row, and NULL when ``p`` is unknown on every row (or there is no row).
``SUM(CASE WHEN p THEN 1 ELSE 0 END)`` is the number of rows on which ``p`` is TRUE whenever there is a row. The two
differ only when there are rows and ``p`` is unknown on all of them: NULL against 0. Two readings rule that out.

* **A test that is never unknown.** When ``p`` is built from comparisons whose operands are literals and columns
  declared NOT NULL of a table that no outer join pads, ``p`` is TRUE or FALSE on every row, and the two sums are the
  same number (both NULL over no rows). ``c IS NULL`` and ``c IS NOT NULL`` are never unknown for any column. The
  rewrite is made wherever the sum stands, also as a term of an addition inside the sum:
  ``SUM((a = b) + (a = c))`` is ``SUM(CASE WHEN a = b THEN 1 ELSE 0 END + CASE WHEN a = c THEN 1 ELSE 0 END)``.
* **A test for "at least" in ``HAVING``.** ``SUM(p) > 0`` keeps a group exactly when ``p`` is TRUE on one of its
  rows: when ``p`` is unknown on every row the comparison is unknown and the group is dropped, and with the ``CASE``
  spelling it is FALSE and the group is dropped. The same holds for ``>= n`` and ``= n`` with ``n >= 1``, ``> n`` with
  ``n >= 0`` and ``<> 0``: each is TRUE only for a count of at least one, which both spellings give alike, and
  otherwise the group goes either way. This reading needs no NOT NULL fact. It is made only for a comparison that is
  the whole ``HAVING`` or one of the tests its ``AND`` joins, where unknown and FALSE both drop the group. ``= 0``,
  ``< 1`` and a sum that is an output are not read this way: there NULL and 0 differ.

* **A test for "none" beside a test for "at least" that makes it known.** In
  ``HAVING SUM(name = 'S8') > 0 AND SUM(name = 'iPhone') = 0`` the first test keeps only groups with a row on which
  ``name = 'S8'`` is TRUE, and on that row ``name`` is not NULL, so ``name = 'iPhone'`` is known there and the second
  sum is not NULL: it is the number of rows on which its test is TRUE, as the ``CASE`` spelling is. A group the first
  test drops is dropped whatever the second gives. So a sum compared with ``= 0``, ``< 1`` or ``<= 0`` is read when
  another test of the same ``AND`` is an "at least" test of a sum whose condition, when TRUE, leaves no column of this
  sum's condition NULL: each of its columns (outside ``IS NULL``) is an operand of a comparison that the other
  condition requires (through ``AND``; a column that only one side of an ``OR`` compares is not required, and
  nothing under ``NOT`` is counted). This reading needs no NOT NULL fact either, and holds on the padded side of
  an outer join.

Both sums have the same type: MySQL sums whole numbers to a DECIMAL, and the test and the ``CASE`` are both whole
numbers. ``tests/test_boolean_sums.py`` replays both readings on generated tables (SQLite, which also reads a
comparison as 1, 0 or NULL), and the note there names the check made on a MySQL 8.0 server.

**What is read.** ``SUM(p)`` with no ``DISTINCT``, outside a window, in a MySQL query, where ``p`` is a comparison
(``=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``), ``BETWEEN``, ``IN`` over a list of literals, ``LIKE`` with a literal
pattern or ``IS [NOT] NULL``, over plain columns and literals, or such tests joined by ``AND``, ``OR`` and ``NOT``.
For the first reading the select reads tables and derived tables joined by inner, cross and left joins, and every
column of the test (outside ``IS NULL``) is a column, declared NOT NULL, of one of its tables that is not the padded
side of a left join. A column that cannot be placed (no qualifier and two sources that may hold it, or a derived
table's column) leaves the sum alone. The other two readings look at the ``HAVING`` only. Every other dialect is left alone: PostgreSQL has no sum of a truth value.

A pre-pass of the eleventh hardest-cases batch (``shape_fallback.ELEVENTH_BATCH``).
"""

from __future__ import annotations

from sqlglot import exp

from .conditional_count_sums import _own_aggregate
from .counted_presence import _Facts, _sources

DIALECTS = frozenset({"mysql"})
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _value(node: exp.Expression | None) -> bool:
    """A plain column, or a literal that is not NULL (a number may carry a sign)."""

    node = _unparen(node)
    if isinstance(node, exp.Neg):
        node = _unparen(node.this)
        return isinstance(node, exp.Literal) and not node.is_string
    if isinstance(node, exp.Column):
        return isinstance(node.this, exp.Identifier) and not node.args.get("db") and not node.args.get("catalog")
    return isinstance(node, exp.Literal)


def _literal(node: exp.Expression | None) -> bool:
    node = _unparen(node)
    return isinstance(node, exp.Literal) or (isinstance(node, exp.Neg) and isinstance(_unparen(node.this), exp.Literal) and not _unparen(node.this).is_string)


def _test(node: exp.Expression | None) -> bool:
    """Whether ``node`` is a truth test of the grammar the module reads."""

    node = _unparen(node)
    if isinstance(node, (exp.And, exp.Or)):
        return _test(node.this) and _test(node.expression)
    if isinstance(node, exp.Not):
        return _test(node.this)
    if isinstance(node, _COMPARISONS):
        return _value(node.this) and _value(node.expression)
    if type(node) is exp.Between:
        return _value(node.this) and _value(node.args.get("low")) and _value(node.args.get("high")) and not node.args.get("symmetric")
    if type(node) is exp.In:
        listed = node.expressions
        return _value(node.this) and bool(listed) and all(_literal(item) for item in listed) and not any(node.args.get(k) for k in ("query", "unnest", "field", "is_global"))
    if type(node) is exp.Like:
        pattern = _unparen(node.expression)
        return _value(node.this) and isinstance(pattern, exp.Literal) and pattern.is_string
    if type(node) is exp.Is:
        return _value(node.this) and isinstance(_unparen(node.expression), exp.Null)
    return False


def _never_unknown(test: exp.Expression, select: exp.Select, facts: _Facts) -> bool:
    """Whether every column of ``test`` outside ``IS NULL`` is a NOT NULL column of a table of ``select`` no outer join pads."""

    found = _sources(select)
    if found is None:
        return False
    sources, real = found
    for column in test.find_all(exp.Column):
        if type(column.parent) is exp.Is and column.parent.this is column:
            if facts.owner(column, sources) is None:
                return False  # it must still be a column of this select (an outer row's column makes the sum another query's)
            continue
        owner = facts.owner(column, sources)
        if owner is None:
            return False
        name, field = owner
        source = sources[name]
        if name not in real or not isinstance(source, exp.Table) or field not in facts.declared.get(source.name.lower(), ()):
            return False
    return True


def _at_least(call: exp.Sum, select: exp.Select) -> bool:
    """Whether ``call`` stands in a test of ``select``'s ``HAVING`` that only a count of at least one passes (module doc)."""

    node: exp.Expression = call
    while isinstance(node.parent, exp.Paren):
        node = node.parent
    comparison = node.parent
    if not isinstance(comparison, _COMPARISONS):
        return False
    other = _unparen(comparison.expression if comparison.this is node else comparison.this)
    if not isinstance(other, exp.Literal) or other.is_string or not other.this.isdigit():
        return False
    bound = int(other.this)
    kind = type(comparison)
    if comparison.this is not node:  # n < SUM(p) is SUM(p) > n
        kind = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE}.get(kind, kind)
    if not ((kind is exp.GT and bound >= 0) or (kind in (exp.GTE, exp.EQ) and bound >= 1) or (kind is exp.NEQ and bound == 0)):
        return False
    top: exp.Expression = comparison
    while isinstance(top.parent, (exp.Paren, exp.And)):
        top = top.parent
    having = select.args.get("having")
    return having is not None and top.parent is having


def _key(column: exp.Column) -> tuple[str, str]:
    return (column.table.lower(), column.name.lower())


def _required(test: exp.Expression | None) -> set[tuple[str, str]]:
    """The columns that are not NULL on every row on which ``test`` is TRUE (a subset: what the comparisons it requires read)."""

    test = _unparen(test)
    if isinstance(test, exp.And):
        return _required(test.this) | _required(test.expression)
    if isinstance(test, exp.Or):
        return _required(test.this) & _required(test.expression)
    if isinstance(test, _COMPARISONS):
        return {_key(side) for side in (_unparen(test.this), _unparen(test.expression)) if isinstance(side, exp.Column)}
    if type(test) in (exp.Between, exp.In, exp.Like) and isinstance(_unparen(test.this), exp.Column):
        return {_key(_unparen(test.this))}
    return set()


def _conjuncts(node: exp.Expression | None) -> list[exp.Expression]:
    node = _unparen(node)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node] if node is not None else []


def _compared_sum(comparison: exp.Expression | None) -> tuple[exp.Sum, type, int] | None:
    """``(the sum, the comparison read with the sum on the left, the whole number)`` for ``SUM(..) <op> n`` or ``n <op> SUM(..)``."""

    if not isinstance(comparison, _COMPARISONS):
        return None
    left, right = _unparen(comparison.this), _unparen(comparison.expression)
    kind = type(comparison)
    if type(left) is not exp.Sum:
        left, right = right, left
        kind = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE}.get(kind, kind)
    if type(left) is not exp.Sum or not isinstance(right, exp.Literal) or right.is_string or not right.this.isdigit():
        return None
    return left, kind, int(right.this)


def _is_at_least(kind: type, bound: int) -> bool:
    return (kind is exp.GT and bound >= 0) or (kind in (exp.GTE, exp.EQ) and bound >= 1) or (kind is exp.NEQ and bound == 0)


def _is_none(kind: type, bound: int) -> bool:
    return (kind is exp.EQ and bound == 0) or (kind is exp.LT and bound == 1) or (kind is exp.LTE and bound == 0)


def _flag_condition(node: exp.Expression | None) -> exp.Expression | None:
    """``p`` for ``CASE WHEN p THEN 1 ELSE 0 END`` (the spelling this module writes, which a query may hold already)."""

    node = _unparen(node)
    if type(node) is not exp.Case or node.args.get("this") is not None or len(node.args.get("ifs") or []) != 1:
        return None
    arm, other = node.args["ifs"][0], _unparen(node.args.get("default"))
    one = _unparen(arm.args.get("true"))
    if not (isinstance(one, exp.Literal) and not one.is_string and one.this == "1" and isinstance(other, exp.Literal) and not other.is_string and other.this == "0"):
        return None
    return arm.this


def _known_beside(call: exp.Sum, select: exp.Select) -> bool:
    """Whether ``call`` is tested for "none" in ``select``'s ``HAVING`` beside an "at least" test that makes its condition known (module doc)."""

    having = select.args.get("having")
    if having is None:
        return False
    tests = _conjuncts(having.this)
    mine = [test for test in tests if (read := _compared_sum(test)) is not None and read[0] is call and _is_none(read[1], read[2])]
    if not mine:
        return False
    needed = set()
    for column in call.this.find_all(exp.Column):
        if type(column.parent) is exp.Is and column.parent.this is column:
            continue
        needed.add(_key(column))
    for test in tests:
        read = _compared_sum(test)
        if read is None or read[0] is call or not _is_at_least(read[1], read[2]):
            continue
        other = read[0]
        if other.args.get("distinct") or isinstance(other.this, exp.Distinct):
            continue
        condition = _flag_condition(other.this) or other.this
        if _test(condition) and needed <= _required(condition):
            return True
    return False


def _flag(test: exp.Expression) -> exp.Case:
    return exp.Case(ifs=[exp.If(this=test.copy(), true=exp.Literal.number(1))], default=exp.Literal.number(0))


def _terms(node: exp.Expression) -> list[exp.Expression]:
    """The terms of an addition (``node`` itself when it is not one), parentheses kept as the nodes that hold them."""

    inner = _unparen(node)
    if isinstance(inner, exp.Add):
        return _terms(inner.this) + _terms(inner.expression)
    return [node]


def write_boolean_sums(tree: exp.Expression, schema: dict | None, not_null: dict | None, types: dict | None, dialect: str) -> tuple[exp.Expression, bool]:
    """``tree`` with each MySQL sum of a truth test as a sum of ones (module doc); whether anything changed."""

    if dialect not in DIALECTS or tree.find(exp.Sum) is None:
        return tree, False
    facts = _Facts(not_null or {}, types, schema)

    def readable(call: exp.Sum) -> list[exp.Expression]:
        """The terms of the sum to rewrite, or nothing."""

        if type(call) is not exp.Sum or call.args.get("distinct") or isinstance(call.this, exp.Distinct) or call.find_ancestor(exp.Window, exp.Filter) is not None:
            return []
        select = call.find_ancestor(exp.Select)
        if select is None:
            return []
        terms = _terms(call.this)
        tests = [term for term in terms if _test(term)]
        if not tests:
            return []
        whole = len(terms) == 1
        if not all(_own_aggregate(_unparen(term), select, select) for term in tests):
            return []
        if whole and (_at_least(call, select) or _known_beside(call, select)):
            return tests
        return [term for term in tests if _never_unknown(_unparen(term), select, facts)]

    if not any(readable(call) for call in tree.find_all(exp.Sum)):
        return tree, False
    tree = tree.copy()
    changed = False
    plans = [(call, readable(call)) for call in tree.find_all(exp.Sum)]  # all read before any is written: one test reads another
    for call, terms in plans:
        for term in terms:
            flag = _flag(_unparen(term))
            if term is call.this:
                call.set("this", flag)
            else:
                term.replace(exp.Paren(this=flag) if isinstance(term, exp.Paren) or isinstance(term.parent, exp.Add) else flag)
            changed = True
    return tree, changed
