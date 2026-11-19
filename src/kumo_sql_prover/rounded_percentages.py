"""A rounded percentage of two whole numbers, however the 100 is placed: MySQL only.

    ROUND(100 * n / d, 2)        ROUND(n * 100 / d, 2)        ROUND((n / d) * 100, 2)        -- ROUND(n / d * 100, 2)

**Why it holds.** MySQL divides two exact numbers with whole-number values into a quotient of nine fractional
digits, cut off and not rounded (its declared scale is four, but the nine digits stay while the value is used
directly), and it multiplies exact numbers exactly. So with ``p = 10^j * n / d``:

* ``10^j * n / d`` and ``n * 10^j / d`` are ``p`` cut off after 9 decimals;
* ``n / d * 10^j`` and ``10^j * (n / d)`` are ``n / d`` cut off after 9 decimals with the point moved: ``p`` cut off
  after ``9 - j`` decimals.

``ROUND(v, r)`` rounds half away from zero at decimal ``r``. Cutting ``p`` off after ``m >= r + 1`` decimals does not
change that rounding: a value that reaches the tie ``t + 5 * 10^-(r+1)`` still reaches it after the cut, because the
tie itself has only ``r + 1`` decimals, and a value below it stays below. So every spelling above is ``p`` rounded half
up at ``r`` whenever ``9 - j >= r + 1``, and they are the same number. Division by zero and a NULL operand give NULL in
each of them.

The number of places matters too. A quotient of whole numbers declares four decimals, and ``ROUND(v, r)`` gives no more
decimals than ``v`` declares: ``ROUND(100 * 2 / 3, 5)`` is ``66.6667``, while ``ROUND(100.0 * 2 / 3, 5)`` is ``66.66667``, because
the literal ``100.0`` adds a declared decimal. With ``r <= 4`` every spelling declares at least ``r`` decimals and the
rounding is the one asked for, so only ``r`` from 0 to 4 is read.

The power of ten matters: ``ROUND(1 / 600 * 3, 2)`` is ``0.00`` (three times ``0.001666666``) and ``ROUND(3 * 1 / 600, 2)``
is ``0.01``. So does directness: a quotient that is an output of a derived table MySQL materialises is rounded to
four decimals there (Kumo_SQL_Prover issue 45), which is why only whole-number terms are read as ``n`` and ``d``, never a
column that might hold a quotient.

Measured on MySQL 8.0.46: 92,294 rows of whole numbers chosen next to rounding ties, 277 expressions (seven
spellings, ``n`` as ``DECIMAL(32,0)`` and ``BIGINT``, ``d`` as both, ``j`` from 0 to 6 and ``r`` from 0 to 4 with ``j + r <= 8``,
also with the factor written ``100.0``), every one equal to the exact value rounded half up; the factors 3 and 7 and
``j + r = 9`` differ, as the argument says.

**What is read.** ``ROUND(e, r)`` with ``r`` a whole-number literal from 0 to 4 (or absent), where ``e`` is built from
``*`` and exactly one ``/`` over

* one dividend term and one divisor term, each a whole number by construction: ``COUNT(..)``, ``SUM`` of a ``CASE`` or
  ``IF`` whose results are whole-number literals that are not negative (or NULL), ``SUM`` of a comparison, or a scalar subquery whose one
  output is such a term. A dividend ``SUM(CASE WHEN p THEN 100 ELSE 0 END)`` is read as ``SUM(CASE WHEN p THEN 1 ELSE 0 END)``
  with the 100 as a factor: a sum of whole numbers is exact;
* number literals that are powers of ten (``100``, ``100.0``, ``10``, ``1.0``), as factors of the dividend or of the
  quotient, never of the divisor, with ``j + r <= 8`` for their product ``10^j``.

It is written ``ROUND(n / d * 10^j, r)`` (``ROUND(n / d, r)`` when ``j`` is 0). Window aggregates, columns, other
functions around the terms, and every other dialect are left alone.

A pre-pass of the tenth hardest-cases batch (``shape_fallback.TENTH_BATCH``). ``tests/test_rounded_percentages.py``
checks the argument on exact arithmetic (the cut and the rounding modelled as MySQL does them); the MySQL replay is
recorded above.
"""

from __future__ import annotations

from sqlglot import exp

_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
MAX_SHIFT = 8  # j + r: the quotient keeps nine fractional digits, and a rounding at r needs r + 1 of them
MAX_PLACES = 4  # r: the declared scale of a quotient of whole numbers; ROUND cannot give more decimals than its argument declares


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _whole_literal(node: exp.Expression | None) -> bool:
    node = _unparen(node)  # not negative: the replay on MySQL covered whole numbers from zero up
    return isinstance(node, exp.Literal) and not node.is_string and node.this.isdigit()


def _power_of_ten(node: exp.Expression | None) -> int | None:
    """``j`` when ``node`` is a number literal equal to ``10^j`` (``100``, ``100.0``, ``1``), else None."""

    node = _unparen(node)
    if not isinstance(node, exp.Literal) or node.is_string:
        return None
    text = node.this
    whole, _, fraction = text.partition(".")
    if not whole.isdigit() or (fraction and set(fraction) != {"0"}) or len(fraction) > 4:
        return None
    if whole[0] != "1" or set(whole[1:]) - {"0"}:
        return None
    return len(whole) - 1


def _whole_result(node: exp.Expression | None) -> bool:
    """A ``CASE``/``IF`` whose every result is a whole-number literal or NULL, or a comparison."""

    node = _unparen(node)
    if isinstance(node, _COMPARISONS):
        return True
    if isinstance(node, exp.Case):
        arms = [arm.args.get("true") for arm in node.args.get("ifs") or []]
        default = node.args.get("default")
        return bool(arms) and all(_whole_literal(arm) or isinstance(_unparen(arm), exp.Null) for arm in arms) and (default is None or _whole_literal(default) or isinstance(_unparen(default), exp.Null))
    if type(node) is exp.If:
        results = [node.args.get("true"), node.args.get("false")]
        return node.args.get("true") is not None and all(arm is None or _whole_literal(arm) or isinstance(_unparen(arm), exp.Null) for arm in results)
    return False


def _whole_term(node: exp.Expression | None, depth: int = 0) -> bool:
    """Whether ``node`` is a whole number by construction (module doc)."""

    node = _unparen(node)
    if node is None or isinstance(node.parent, (exp.Window, exp.Filter)):
        return False
    if type(node) is exp.Count:
        return not any(v for key, v in node.args.items() if key not in ("this", "expressions", "big_int"))
    if type(node) is exp.Sum:
        return not any(v for key, v in node.args.items() if key != "this") and _whole_result(node.this)
    if type(node) is exp.Subquery and depth == 0 and not any(v for key, v in node.args.items() if key != "this"):
        body = node.this
        if type(body) is exp.Select and len(body.expressions) == 1 and not any(body.args.get(key) for key in ("windows", "qualify")):
            return _whole_term(body.expressions[0].unalias(), depth + 1)
    return False


def _unit_flag_sum(term: exp.Expression) -> tuple[exp.Expression, int]:
    """``(SUM(CASE WHEN p THEN 1 ELSE 0 END), k)`` for ``SUM(CASE WHEN p THEN 10^k ELSE 0 END)``, else ``(term, 0)``.

    A sum of whole numbers is exact, so the sum of ``10^k`` on the rows that pass is ``10^k`` times their number, and
    NULL in both spellings when there is no row (or, without an ``ELSE``, when no row passes)."""

    if type(term) is not exp.Sum:
        return term, 0
    case = _unparen(term.this)
    if type(case) is exp.If and not isinstance(case.parent, exp.Case) and case.args.get("true") is not None:
        arms, default, test = [case], case.args.get("false"), case.this
    elif isinstance(case, exp.Case) and case.args.get("this") is None and len(case.args.get("ifs") or []) == 1:
        arms, default, test = case.args["ifs"], case.args.get("default"), case.args["ifs"][0].this
    else:
        return term, 0
    result = _unparen(arms[0].args.get("true"))
    k = _power_of_ten(result) if _whole_literal(result) else None
    zero = _unparen(default)
    if not k or not (zero is None or isinstance(zero, exp.Null) or (_whole_literal(zero) and zero.this == "0")):
        return term, 0
    unit = exp.Case(ifs=[exp.If(this=test.copy(), true=exp.Literal.number(1))])
    if zero is not None:
        unit.set("default", zero.copy())
    return exp.Sum(this=unit), k


def _read(node: exp.Expression | None):
    """``(n, division, j)`` when ``node`` is ``10^j * n / d`` in one of the spellings of the module doc, else None."""

    node = _unparen(node)
    if isinstance(node, exp.Div):
        if any(v for key, v in node.args.items() if key not in ("this", "expression", "typed", "safe")):
            return None
        divisor = _unparen(node.expression)
        if not _whole_term(divisor):
            return None
        shift, term = 0, None
        pending = [node.this]
        while pending:  # the dividend: one whole-number term times powers of ten
            part = _unparen(pending.pop())
            if isinstance(part, exp.Mul):
                pending += [part.this, part.expression]
            elif _power_of_ten(part) is not None:
                shift += _power_of_ten(part)
            elif term is None and _whole_term(part):
                term, scale = _unit_flag_sum(part)
                shift += scale
            else:
                return None
        return (term, node, shift) if term is not None else None
    if isinstance(node, exp.Mul):
        for inner, factor in ((node.this, node.expression), (node.expression, node.this)):
            j = _power_of_ten(factor)
            if j is None:
                continue
            found = _read(inner)
            if found is not None:
                return found[0], found[1], found[2] + j
    return None


def write_rounded_percentages(tree: exp.Expression, dialect: str) -> tuple[exp.Expression, bool]:
    """``tree`` with each such ``ROUND`` written ``ROUND(n / d * 10^j, r)`` (module doc); whether anything changed."""

    if dialect != "mysql" or not any(True for _ in tree.find_all(exp.Round)):
        return tree, False
    tree = tree.copy()
    changed = False
    for call in list(tree.find_all(exp.Round)):
        if any(v for key, v in call.args.items() if key not in ("this", "decimals")):
            continue
        digits = call.args.get("decimals")
        if digits is None:
            places = 0
        else:
            digits = _unparen(digits)
            if not (isinstance(digits, exp.Literal) and not digits.is_string and digits.this.isdigit()) or int(digits.this) > MAX_PLACES:
                continue
            places = int(digits.this)
        found = _read(call.this)
        if found is None:
            continue
        term, division, shift = found
        if shift + places > MAX_SHIFT:
            continue
        quotient = division.copy()  # the dialect's own division node, with the bare term as its dividend
        quotient.set("this", term.copy())
        quotient.set("expression", _unparen(quotient.expression))
        written = quotient if shift == 0 else exp.Mul(this=quotient, expression=exp.Literal.number(10**shift))
        if written.sql(dialect="mysql") == _unparen(call.this).sql(dialect="mysql"):
            continue
        call.set("this", written)
        changed = True
    return tree, changed
