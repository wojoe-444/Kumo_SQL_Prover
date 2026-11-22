"""A test on a value looked up by a key, in a filter: read as ``EXISTS`` with the test inside.

    WHERE (SELECT b.free FROM seats AS b WHERE b.seat = a.seat - 1) IS TRUE
    WHERE EXISTS (SELECT 1 FROM seats AS b WHERE b.seat = a.seat - 1 AND b.free IS TRUE)

The prover reads a looked-up value compared with a number (``(SELECT ..) = 1``). The same lookup is also tested as
``(SELECT ..) IS TRUE``, ``(SELECT ..) <=> 1``, ``COALESCE((SELECT ..), 0) = 1``, ``NOT (SELECT ..) = 0`` or, in MySQL,
as the looked-up number alone; this rule gives each of those the form above, which the subquery rules read.

**Why the two are TRUE on the same rows.** ``seat`` is a declared key of the table and never NULL, so the lookup
finds at most one row. Call the test ``C`` and the looked-up value ``x``. If a row is found, ``C`` is the test on
that row's ``x``, and the ``EXISTS`` returns that one row exactly when the test is TRUE on it. If no row is found the
lookup is NULL and the ``EXISTS`` is FALSE, so the two agree as long as ``C`` with NULL in the lookup's place is not
TRUE. The rule works that out from the text and writes only when the answer is FALSE or NULL: ``NULL IS TRUE`` is
FALSE, ``NULL <=> 1`` is FALSE, ``COALESCE(NULL, 0) = 1`` is FALSE, ``NOT NULL = 0`` is NULL. A test that is TRUE for a
missing row (``IS NULL``, ``COALESCE(.., 1) = 1``) or that it cannot work out (the fallback is a column) is left
alone. FALSE and NULL are told apart nowhere here: the test stands under ``AND`` and ``OR`` alone, up to the
``WHERE``.

**Left alone.** A subquery that is not one qualified column of one declared table filtered by ``key = value`` alone
(a single-column key declared NOT NULL, a value that reads no column of the subquery's table); a test that holds
another subquery, an aggregate, a window, a call that changes between readings, or a column without a qualifier or
qualified by the lookup's own name; a test under ``NOT`` or inside any other expression that is not part of the
test itself; a plain comparison of the lookup with something (the prover reads those already); a number alone as a
test outside MySQL, SQLite and DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .filter_terms import _lower, _positive, _same_each_time, _unwrapped

NAME = "lookup_conditions"

_UNKNOWN = ("unknown",)
_NULL = ("null",)
_TRUTH_IS_NUMBER = ("mysql", "sqlite", "duckdb")
_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)


def _number(value: tuple, lax: bool):
    """The number a known value is, or None: a Boolean is 1 or 0 only where truth is a number."""

    if value[0] == "num":
        return value[1]
    if value[0] == "bool" and lax:
        return int(value[1])
    return None


def _truth(value: tuple, lax: bool) -> tuple:
    """A value read as a condition: ``("bool", ..)``, NULL or unknown."""

    if value in (_NULL, _UNKNOWN) or value[0] == "bool":
        return value
    if value[0] == "num" and lax:
        return ("bool", value[1] != 0)
    return _UNKNOWN


def _evaluate(node: exp.Expression, lookup: exp.Expression, lax: bool) -> tuple:
    """The value of ``node`` when ``lookup`` is NULL, as far as the text alone tells: ``("num", n)``, ``("bool", b)``,
    NULL or unknown."""

    if node is lookup:
        return _NULL
    if isinstance(node, exp.Paren):
        return _evaluate(node.this, lookup, lax)
    if isinstance(node, exp.Null):
        return _NULL
    if isinstance(node, exp.Boolean):
        return ("bool", bool(node.this))
    if isinstance(node, exp.Literal):
        return ("num", int(node.name)) if not node.is_string and node.name.isdigit() else _UNKNOWN
    if isinstance(node, exp.Coalesce):
        for part in [node.this, *node.expressions]:
            value = _evaluate(part, lookup, lax)
            if value != _NULL:
                return value  # the first that is not NULL, or unknown
        return _NULL
    if isinstance(node, _COMPARISONS):
        left, right = _evaluate(node.left, lookup, lax), _evaluate(node.right, lookup, lax)
        if _NULL in (left, right):
            return _NULL
        a, b = _number(left, lax), _number(right, lax)
        if a is None or b is None:
            return _UNKNOWN
        return ("bool", {exp.EQ: a == b, exp.NEQ: a != b, exp.LT: a < b, exp.LTE: a <= b, exp.GT: a > b, exp.GTE: a >= b}[type(node)])
    if isinstance(node, exp.NullSafeEQ):
        left, right = _evaluate(node.left, lookup, lax), _evaluate(node.right, lookup, lax)
        if _UNKNOWN in (left, right):
            return _UNKNOWN
        if _NULL in (left, right):
            return ("bool", left == right)
        a, b = _number(left, lax), _number(right, lax)
        return ("bool", a == b) if a is not None and b is not None else _UNKNOWN
    if isinstance(node, exp.Is):
        value = _evaluate(node.this, lookup, lax)
        test = node.expression
        if value == _UNKNOWN:
            return _UNKNOWN
        if isinstance(test, exp.Null):
            return ("bool", value == _NULL)
        if isinstance(test, exp.Boolean):
            truth = _truth(value, lax)
            if truth == _UNKNOWN:
                return _UNKNOWN
            return ("bool", truth != _NULL and truth[1] is bool(test.this))
        return _UNKNOWN
    if isinstance(node, exp.Not):
        truth = _truth(_evaluate(node.this, lookup, lax), lax)
        return truth if truth in (_NULL, _UNKNOWN) else ("bool", not truth[1])
    if isinstance(node, (exp.And, exp.Or)):
        left, right = _truth(_evaluate(node.left, lookup, lax), lax), _truth(_evaluate(node.right, lookup, lax), lax)
        decides = ("bool", isinstance(node, exp.Or))  # TRUE decides an OR, FALSE an AND
        if decides in (left, right):
            return decides
        if _UNKNOWN in (left, right):
            return _UNKNOWN
        if _NULL in (left, right):
            return _NULL
        return ("bool", not decides[1])
    if isinstance(node, exp.In) and not node.args.get("query") and not node.args.get("unnest"):
        return _NULL if _evaluate(node.this, lookup, lax) == _NULL else _UNKNOWN
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
        return _NULL if _NULL in (_evaluate(node.left, lookup, lax), _evaluate(node.right, lookup, lax)) else _UNKNOWN
    if isinstance(node, exp.Between):
        return _NULL if _evaluate(node.this, lookup, lax) == _NULL else _UNKNOWN
    if isinstance(node, exp.Neg):
        return _NULL if _evaluate(node.this, lookup, lax) == _NULL else _UNKNOWN
    return _UNKNOWN


def _single_key(table: exp.Table, keys: dict | None, not_null: dict | None) -> set[str]:
    """The columns that are each a declared key of the table on their own and declared NOT NULL."""

    declared = next((value for name, value in (keys or {}).items() if _lower(name) == _lower(table.name)), None) or []
    never_null = {_lower(c) for c in next((value for name, value in (not_null or {}).items() if _lower(name) == _lower(table.name)), None) or ()}
    return {_lower(key[0]) for key in declared if len(key) == 1 and _lower(key[0]) in never_null}


def _keyed_lookup(node: exp.Subquery, keys: dict | None, not_null: dict | None) -> tuple[exp.Select, str] | None:
    """The select of a lookup of one qualified column of one declared table by its key, and the name the table goes
    by there; or None."""

    body = node.this
    if node.alias or not isinstance(body, exp.Select) or len(body.expressions) != 1:
        return None
    if any(body.args.get(key) for key in ("joins", "group", "having", "limit", "offset", "qualify", "windows", "distinct", "order", "with_", "with", "laterals", "pivots", "sample")):
        return None
    if any(node.args.get(key) for key in ("limit", "offset", "order", "with_", "with")):
        return None
    from_ = body.args.get(FROM_KEY)
    where = body.args.get("where")
    table = from_.this if from_ is not None else None
    if where is None or not isinstance(table, exp.Table) or table.args.get("db") or table.args.get("catalog") or not table.alias_or_name:
        return None
    name = _lower(table.alias_or_name)
    item = body.expressions[0]
    if not isinstance(item, exp.Column) or _lower(item.table) != name or item.args.get("db"):
        return None
    test = _unwrapped(where.this)
    if not isinstance(test, exp.EQ):
        return None
    single = _single_key(table, keys, not_null)
    for column, value in ((_unwrapped(test.left), test.right), (_unwrapped(test.right), test.left)):
        if not isinstance(column, exp.Column) or _lower(column.table) != name or _lower(column.name) not in single:
            continue
        reads = list(value.find_all(exp.Column))
        if any(not part.table or _lower(part.table) == name for part in reads) or value.find(exp.Select, exp.AggFunc, exp.Window) is not None or not _same_each_time(value):
            continue
        return body, name
    return None


def lookup_conditions(select: exp.Select, keys: dict | None, not_null: dict | None, dialect: str | None) -> exp.Select | None:
    """The select with each test on a keyed lookup in its ``WHERE`` written as ``EXISTS`` with the test inside (module
    doc), or None."""

    where = select.args.get("where")
    if where is None or not keys or where.find(exp.Subquery) is None:
        return None
    lax = (dialect or "").lower() in _TRUTH_IS_NUMBER
    new = select.copy()
    where = new.args["where"]
    wrote = False
    for node in list(where.find_all(exp.Subquery)):
        if node.find_ancestor(exp.Select) is not new or isinstance(node.parent, (exp.Exists, exp.From, exp.Join)) or node.arg_key == "query":
            continue
        found = _keyed_lookup(node, keys, not_null)
        if found is None:
            continue
        body, name = found
        test: exp.Expression = node  # the whole test the lookup is part of: up to the AND, OR or WHERE above it
        while not isinstance(test.parent, (exp.And, exp.Or, exp.Where)) and test.parent is not None:
            test = test.parent
        while isinstance(test.parent, exp.Paren):
            test = test.parent
        if not _positive(test, where):
            continue
        inside = _unwrapped(test)
        if isinstance(inside, _COMPARISONS) and (_unwrapped(inside.left) is node or _unwrapped(inside.right) is node):
            continue  # a plain comparison of the lookup: read already
        if inside is node and not lax:
            continue  # a number alone is a test only where truth is a number
        others = [part for part in inside.walk() if part is not node and part.find_ancestor(exp.Subquery) is not node]
        if any(isinstance(part, (exp.Subquery, exp.Select, exp.Exists, exp.AggFunc, exp.Window, exp.Star, exp.Placeholder, exp.Parameter)) for part in others):
            continue
        if any(isinstance(part, exp.Column) and (not part.table or _lower(part.table) == name or part.args.get("db")) for part in others):
            continue
        if not _same_each_time(inside):
            continue
        missing = _truth(_evaluate(inside, node, lax), lax)
        if missing not in (_NULL, ("bool", False)):
            continue  # TRUE for a missing row, or not worked out
        column = body.expressions[0].copy()
        key_test = body.args["where"].this.copy()
        if inside is node:
            moved: exp.Expression = column
        else:
            marker = inside.copy()
            marker.find(exp.Subquery).replace(column)  # the test holds no other subquery
            moved = exp.Paren(this=marker) if isinstance(marker, (exp.And, exp.Or)) else marker
        exists_body = exp.Select().select(exp.Literal.number(1)).from_(body.args[FROM_KEY].this.copy()).where(exp.And(this=key_test, expression=moved))
        test.replace(exp.Exists(this=exists_body))
        wrote = True
    return new if wrote else None
