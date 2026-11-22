"""Three readings of a ``WHERE`` clause that keep its rows: a term every branch of an ``OR`` repeats, a term of an
``EXISTS`` that does not look at the subquery's rows, and a column the filter pins to one whole number.

They are what stands between a self join or a neighbour lookup and the form the prover's subquery rules match:

    -- "free seats next to a free seat", three ways
    SELECT a.seat FROM seats AS a WHERE a.free = 1 AND EXISTS (SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND b.free = 1)
    SELECT a.seat FROM seats AS a WHERE EXISTS (SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND a.free = 1 AND b.free = 1)
    SELECT a.seat FROM seats AS a WHERE a.free = 1 AND EXISTS (SELECT 1 FROM seats AS b WHERE b.seat = a.seat + 1 AND b.free = a.free)

A ``WHERE`` keeps the rows on which its condition is TRUE. Each rule below gives a condition that is TRUE on exactly
the same rows; where it could change FALSE into NULL or back, it only writes in a place where that cannot matter.

**Shared terms** (``shared_filter_terms``). ``(a AND b) OR (a AND c)`` is written ``a AND (b OR c)``, and
``a OR (a AND c)`` is written ``a``. Both hold for all three truth values (AND and OR are the minimum and maximum of
FALSE < NULL < TRUE, and minimum distributes over maximum), so this one keeps the value of the condition and not only
its TRUE rows. Terms are the same when their text is the same; a term with a call that gives another value each time
it is read (``RAND()``, a clock, a function the prover does not know) is never taken as shared.

**Outer terms of EXISTS** (``exists_outer_terms``). In ``EXISTS (SELECT .. FROM s WHERE p AND q)`` a term ``q`` that
names no column of the subquery's own sources has one value for all of its rows, so the subquery returns a row
exactly when ``q`` is TRUE and ``EXISTS (SELECT .. FROM s WHERE p)`` is: ``q AND EXISTS (.. WHERE p)`` is TRUE on the
same rows. It is not the same value (when ``q`` is NULL the first is FALSE and the second NULL or FALSE), so the
rule writes only where the ``EXISTS`` stands under ``AND`` and ``OR`` alone, up to the ``WHERE``: there a FALSE and a
NULL do the same. Left alone: an ``EXISTS`` under ``NOT`` or inside any other expression; a subquery with ``GROUP BY``,
``HAVING``, ``LIMIT``, ``OFFSET`` or an aggregate in its select list (it can return a row over no rows, or none over
some); a term with a column that has no qualifier (it may be the subquery's own), with a subquery, an aggregate or a
window of its own, or with a call that changes between readings.

**Pinned columns** (``pinned_columns``). When a term of the ``WHERE``, joined to the rest by ``AND``, is
``a.c = 1`` for a column declared a whole number, every row the filter keeps has the number 1 in ``a.c``. In the
other terms of that ``WHERE``, subqueries included, ``a.c`` compared with something is then the literal compared with
it: on a kept row the two are the same number, and a row the pin drops is dropped either way. That holds while the
comparison is one of numbers. Against a text, a date or a decimal an engine converts one side first, and not the same
way for a column and for a number written out (SQLite compares a whole-number column with a text column as numbers,
and the number 1 with it as texts: ``c < '05'`` is TRUE for ``c = 1`` and ``1 < '05'`` is not). So the column is
written as the literal only where the other side is plainly a whole number: digits, a column of a declared table
with a whole-number type, a sum, difference or product of those, or a scalar subquery that selects one. Left alone:
a column without a declared whole-number type (a decimal 1.0, a text '1' and a float are other values that equal 1);
a use of the column that is not an operand of a plain comparison, or is compared with anything else; a use inside a
subquery where another source has taken the name ``a``; a pin under ``OR`` or ``NOT``; a select with a ``WITH`` clause
in or around it (a name there could hide the declared table).
"""

from __future__ import annotations

from sqlglot import exp

from . import volatile_calls
from .ast_utils import FROM_KEY

SHARED = "shared_filter_terms"
OUTER = "exists_outer_terms"
PINNED = "pinned_columns"
NAMES = (SHARED, OUTER, PINNED)

_WHOLE_TYPES = ("INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT", "INT64", "BOOL", "BOOLEAN")
_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
_ATOMS = (exp.Column, exp.Literal, exp.Subquery, exp.Paren, exp.Func, exp.Case, exp.Null, exp.Boolean, exp.Predicate, exp.Not)


def _lower(name: str | None) -> str:
    return (name or "").lower()


def _unwrapped(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _terms(node: exp.Expression, kind: type) -> list[exp.Expression]:
    node = _unwrapped(node)
    if isinstance(node, kind):
        return _terms(node.left, kind) + _terms(node.right, kind)
    return [node]


def _same_each_time(node: exp.Expression) -> bool:
    return not any(isinstance(part, (exp.Rand, exp.Anonymous)) or volatile_calls.kind(part) is not None for part in node.walk())


def _joined(terms: list[exp.Expression], kind: type) -> exp.Expression:
    terms = [term if isinstance(term, _ATOMS) and not isinstance(term, (exp.And, exp.Or)) else exp.Paren(this=term) for term in terms]
    result = terms[0]
    for term in terms[1:]:
        result = kind(this=result, expression=term)
    return result


def _shared(node: exp.Expression, found: list) -> exp.Expression:
    """``node`` with the terms every branch of an ``OR`` holds written once; ``found`` gets an entry per ``OR`` read."""

    node = _unwrapped(node)
    if isinstance(node, exp.And):
        return _joined([_shared(term, found) for term in _terms(node, exp.And)], exp.And)
    if not isinstance(node, exp.Or):
        return node
    branches = [_terms(_shared(branch, found), exp.And) for branch in _terms(node, exp.Or)]
    texts = [[term.sql() for term in branch] for branch in branches]
    names: list[str] = []
    for term, text in zip(branches[0], texts[0]):
        if text not in names and all(text in other for other in texts[1:]) and _same_each_time(term):
            names.append(text)
    if not names:
        return _joined([_joined(branch, exp.And) for branch in branches], exp.Or)
    found.append(names)
    shared = [branches[0][texts[0].index(text)] for text in names]
    rest = [[term for term, text in zip(branch, branch_texts) if text not in names] for branch, branch_texts in zip(branches, texts)]
    if any(not branch for branch in rest):
        return _joined(shared, exp.And)  # a OR (a AND c) is a
    return _joined(shared + [_joined([_joined(branch, exp.And) for branch in rest], exp.Or)], exp.And)


def shared_filter_terms(select: exp.Select) -> exp.Select | None:
    """The select with the shared terms of each ``OR`` of its ``WHERE`` written once (module doc), or None."""

    where = select.args.get("where")
    if where is None or where.find(exp.Or) is None:
        return None
    found: list = []
    condition = _shared(where.this.copy(), found)
    if not found:
        return None
    new = select.copy()
    new.args["where"].set("this", condition)
    return new


def _sources(select: exp.Select) -> list[exp.Expression]:
    from_ = select.args.get(FROM_KEY)
    sources = [from_.this] if from_ is not None and from_.this is not None else []
    return sources + [join.this for join in select.args.get("joins") or []]


def _source_names(select: exp.Select) -> set[str] | None:
    """The lower-case names the select's sources go by, or None when one has no name to go by."""

    names = set()
    for source in _sources(select):
        if not isinstance(source, (exp.Table, exp.Subquery)) or not source.alias_or_name:
            return None
        names.add(_lower(source.alias_or_name))
    return names


def _positive(node: exp.Expression, where: exp.Where) -> bool:
    """Whether ``node`` stands under ``AND``, ``OR`` and parentheses alone, up to the ``WHERE``."""

    parent = node.parent
    while isinstance(parent, (exp.And, exp.Or, exp.Paren)):
        parent = parent.parent
    return parent is where


def _outer_term(term: exp.Expression, inner: set[str]) -> bool:
    columns = 0
    for part in term.walk():
        if isinstance(part, (exp.Subquery, exp.Select, exp.Exists, exp.AggFunc, exp.Window, exp.Star, exp.Placeholder, exp.Parameter)):
            return False
        if isinstance(part, exp.Column):
            if not part.table or _lower(part.table) in inner or part.args.get("db") or part.args.get("catalog"):
                return False
            columns += 1
    return columns > 0 and _same_each_time(term)


def exists_outer_terms(select: exp.Select) -> exp.Select | None:
    """The select with the terms of an ``EXISTS`` that do not read the subquery's rows moved in front of it (module
    doc), or None."""

    where = select.args.get("where")
    if where is None or where.find(exp.Exists) is None:
        return None
    new = select.copy()
    where = new.args["where"]
    wrote = False
    for exists in list(where.find_all(exp.Exists)):
        if not _positive(exists, where):
            continue
        inner = exists.this
        if not isinstance(inner, exp.Select) or inner.args.get("where") is None:
            continue
        if any(inner.args.get(key) for key in ("group", "having", "limit", "offset", "qualify", "windows", "with_", "with", "laterals", "pivots")):
            continue
        if any(item.find(exp.AggFunc) is not None or item.find(exp.Window) is not None for item in inner.expressions):
            continue
        names = _source_names(inner)
        if not names:
            continue
        terms = _terms(inner.args["where"].this, exp.And)
        moved = [term for term in terms if _outer_term(term, names)]
        if not moved:
            continue
        kept = [term for term in terms if not any(term is one for one in moved)]
        if kept:
            inner.args["where"].set("this", _joined([term.copy() for term in kept], exp.And))
        else:
            inner.set("where", None)
        replacement = exp.Paren(this=_joined([term.copy() for term in moved] + [exists.copy()], exp.And))
        exists.replace(replacement)
        wrote = True
    return new if wrote else None


def _whole_literal(node: exp.Expression) -> bool:
    return isinstance(node, exp.Literal) and not node.is_string and node.name.isdigit()


def _typed_whole(table: exp.Expression, column: str, types: dict | None) -> bool:
    if not isinstance(table, exp.Table):
        return False
    declared = next((columns for name, columns in (types or {}).items() if _lower(name) == _lower(table.name)), None) or {}
    kind = next((kind for name, kind in declared.items() if _lower(name) == _lower(column)), None)
    return isinstance(kind, str) and kind.upper().split("(")[0].strip() in _WHOLE_TYPES


def _declared_whole(column: exp.Column, select: exp.Select, types: dict | None) -> bool:
    """Whether the column is one of a declared table of this select and is declared a whole number."""

    sources = _sources(select)
    if column.table:
        tables = [source for source in sources if _lower(source.alias_or_name) == _lower(column.table)]
    else:
        tables = sources if len(sources) == 1 else []
    return len(tables) == 1 and _typed_whole(tables[0], column.name, types)


def _whole_column(column: exp.Column, top: exp.Select, types: dict | None) -> bool:
    """Whether a column somewhere in ``top`` is a column of a declared table with a whole-number type: the nearest
    select around it that has a source of its qualifier decides (a bare name is taken only from a select that reads
    one declared table holding it)."""

    if column.args.get("db") or column.args.get("catalog"):
        return False
    scope = column.find_ancestor(exp.Select)
    while scope is not None:
        sources = _sources(scope)
        if column.table:
            named = [source for source in sources if _lower(source.alias_or_name) == _lower(column.table)]
            if named:
                return len(named) == 1 and _typed_whole(named[0], column.name, types)
        elif sources:
            return len(sources) == 1 and _typed_whole(sources[0], column.name, types)
        if scope is top:
            return False
        scope = scope.find_ancestor(exp.Select)
    return False


def _whole_value(node: exp.Expression, top: exp.Select, types: dict | None) -> bool:
    """Whether ``node`` is plainly a whole number: written in digits, a column declared one, a sum, difference or
    product of such, or a scalar subquery that selects one. (Comparing a whole-number column with a text, a date or a
    decimal goes by conversion rules that are not the same for a column and for a number written out.)"""

    node = _unwrapped(node)
    if isinstance(node, exp.Neg):
        return _whole_value(node.this, top, types)
    if isinstance(node, exp.Literal):
        return _whole_literal(node)
    if isinstance(node, exp.Column):
        return _whole_column(node, top, types)
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul)):
        return _whole_value(node.left, top, types) and _whole_value(node.right, top, types)
    if isinstance(node, exp.Subquery) and isinstance(node.this, exp.Select) and len(node.this.expressions) == 1:
        item = node.this.expressions[0]
        return _whole_value(item.this if isinstance(item, exp.Alias) else item, top, types)
    return False


def _same_column(node: exp.Expression, pin: exp.Column, select: exp.Select) -> bool:
    """Whether ``node``, somewhere in the select's ``WHERE``, is the pinned column of this select."""

    if not isinstance(node, exp.Column) or _lower(node.name) != _lower(pin.name) or _lower(node.table) != _lower(pin.table):
        return False
    if node.args.get("db") or node.args.get("catalog"):
        return False
    scope = node.find_ancestor(exp.Select)
    while scope is not None and scope is not select:
        if not pin.table:
            return False  # a bare name inside a subquery is the subquery's own column first
        names = _source_names(scope)
        if names is None or _lower(pin.table) in names:
            return False
        scope = scope.find_ancestor(exp.Select)
    return scope is select


def pinned_columns(select: exp.Select, types: dict | None) -> exp.Select | None:
    """The select with a column its ``WHERE`` pins to a whole number written as that number where the other terms
    compare it (module doc), or None."""

    where = select.args.get("where")
    if where is None or not types or select.find(exp.With) is not None:
        return None
    node = select
    while node is not None:  # a WITH name could hide a declared table
        if node.args.get("with_") or node.args.get("with"):
            return None
        node = node.parent
    new = select.copy()
    where = new.args["where"]
    terms = []
    node = where.this
    stack = [node]
    while stack:  # the terms joined by AND, each in place (parentheses kept)
        part = stack.pop()
        inside = _unwrapped(part)
        if isinstance(inside, exp.And):
            stack.extend([inside.right, inside.left])
        else:
            terms.append(inside)
    wrote = False
    for term in terms:
        if not isinstance(term, exp.EQ):
            continue
        left, right = _unwrapped(term.left), _unwrapped(term.right)
        column, literal = (left, right) if isinstance(left, exp.Column) else (right, left)
        if not isinstance(column, exp.Column) or not _whole_literal(literal) or column.args.get("db") or column.args.get("catalog"):
            continue
        if not _declared_whole(column, new, types):
            continue
        for other in terms:
            if other is term:
                continue
            for use in list(other.find_all(exp.Column)):
                if not isinstance(use.parent, _COMPARISONS) or not _same_column(use, column, new):
                    continue
                against = use.parent.right if use.parent.left is use else use.parent.left
                if _whole_value(against, new, types):
                    use.replace(literal.copy())
                    wrote = True
    return new if wrote else None
