"""A self join whose two rows may change places: its condition read both ways round.

    -- "accounts with two sessions from different addresses that overlap", two ways
    SELECT DISTINCT a.account FROM sessions AS a JOIN sessions AS b
      ON a.account = b.account AND a.ip <> b.ip AND b.login BETWEEN a.login AND a.logout
    SELECT DISTINCT a.account FROM sessions AS a JOIN sessions AS b
      ON a.account = b.account AND a.ip <> b.ip AND a.login <= b.logout AND b.login <= a.logout

The first asks for a pair of sessions in which the second starts inside the first; the second for a pair that
overlaps. A pair that overlaps is, one way round or the other, a pair in which the second starts inside the first,
so the same accounts come out. No map from the rows of one query to the rows of the other shows that: which row
plays ``a`` depends on the data. This rule gives the join a form in which it no longer matters.

**The reading.** ``SELECT DISTINCT a.k, .. FROM t AS a JOIN t AS b ON C`` returns the values ``a.k`` of the pairs
``(a, b)`` of rows of ``t`` on which ``C`` is TRUE. When ``C`` holds ``a.k = b.k`` for every column ``k`` the select
lists, the pair ``(b, a)`` would put the same values out, because on a pair where ``C`` is TRUE the two rows carry
the same ``k``. So the select returns the same rows when its condition is ``C`` read either way round:

    C(a, b) OR C(b, a)

One direction is plain (a pair that passes ``C`` passes the new condition). For the other, take a pair that passes
``C(b, a)``: the pair ``(b, a)`` passes ``C``, so the original select returns ``b.k``, and ``b.k`` is ``a.k`` there.
``DISTINCT`` makes the number of pairs irrelevant. The terms of ``C`` that read the same both ways round (``S``:
``a.k = b.k``, ``a.ip <> b.ip``) are kept as they are and the rest (``R``) is written ``R OR R'`` with the two
names exchanged, which is the same condition in three-valued logic (AND distributes over OR). When ``R`` already
reads the same both ways round as a whole (``a.login <= b.logout AND b.login <= a.logout``), nothing is written.

**"The same value"** has to mean the same value and not two values an engine calls equal: a text compares equal to
the same text in other letter case under MySQL's default collation, and ``DISTINCT`` then keeps one of the two. So a
listed column must be declared a whole number.

**Left alone.** A select without ``DISTINCT``, or with ``GROUP BY``, ``HAVING``, a window, an aggregate, ``ORDER BY``,
``LIMIT`` or ``OFFSET``; sources other than exactly two occurrences of one declared table under two names, joined
by an inner or cross join (an outer join, ``USING``, a derived table, a name a ``WITH`` clause in or around the
select could have taken); a select item that is not a column of one of the two rows that the condition equates
with the other row's and that is declared a whole number; a condition with a subquery, a column without one of the
two names in front, or a call that gives another value each time it is read.

A rule of the twentieth hardest-cases batch. ``tests/test_mirrored_self_joins.py`` runs the reading on DuckDB over
made-up tables with NULLs and checks every refusal.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .filter_terms import _joined, _lower, _same_each_time, _sources, _terms, _typed_whole, _unwrapped

NAME = "mirrored_self_joins"

_COMMUTE = (exp.EQ, exp.NEQ, exp.NullSafeEQ, exp.NullSafeNEQ, exp.Add, exp.Mul)
_FLIPPED = {exp.GT: exp.LT, exp.GTE: exp.LTE}
_EITHER_ORDER = (exp.Greatest, exp.Least)
_REFUSED = (exp.Subquery, exp.Select, exp.Exists, exp.AggFunc, exp.Window, exp.Star, exp.Placeholder, exp.Parameter)


def _key(node: exp.Expression):
    """A value that is the same for two conditions that differ only in the order of the operands of ``AND``, ``OR``,
    ``=``, ``<>``, ``+``, ``*``, ``GREATEST`` and ``LEAST``, in ``x > y`` written ``y < x``, in parentheses and in the
    letter case of the names in a column. Each of those keeps the value of the expression."""

    node = _unwrapped(node)
    if isinstance(node, (exp.And, exp.Or)):
        return (type(node).__name__, tuple(sorted(repr(_key(term)) for term in _terms(node, type(node)))))
    if isinstance(node, _COMMUTE):
        return (type(node).__name__, tuple(sorted((repr(_key(node.left)), repr(_key(node.right))))))
    if isinstance(node, (exp.LT, exp.LTE, exp.GT, exp.GTE)):
        left, right = (node.right, node.left) if type(node) in _FLIPPED else (node.left, node.right)
        return (_FLIPPED.get(type(node), type(node)).__name__, repr(_key(left)), repr(_key(right)))
    if isinstance(node, _EITHER_ORDER):
        return (type(node).__name__, tuple(sorted(repr(_key(part)) for part in [node.this, *node.expressions])))
    if isinstance(node, exp.Column):
        return ("column", _lower(node.table), _lower(node.name))
    parts = []
    for name, value in sorted(node.args.items()):
        if isinstance(value, exp.Expression):
            parts.append((name, _key(value)))
        elif isinstance(value, list):
            parts.append((name, tuple(_key(item) if isinstance(item, exp.Expression) else repr(item) for item in value)))
        elif value is not None and value is not False:
            parts.append((name, repr(value)))
    return (type(node).__name__, tuple(parts))


def _exchanged(node: exp.Expression, one: exp.Identifier, two: exp.Identifier) -> exp.Expression:
    """A copy of ``node`` with the two names exchanged in front of its columns (each as it is written in ``FROM``)."""

    new = node.copy()
    for column in ([new] if isinstance(new, exp.Column) else []) + list(new.find_all(exp.Column)):
        column.set("table", (two if _lower(column.table) == _lower(one.name) else one).copy())
    return new


def _name(source: exp.Table) -> exp.Identifier | None:
    alias = source.args.get("alias")
    name = alias.this if alias is not None else source.this
    return name if isinstance(name, exp.Identifier) else None


def _under_with(select: exp.Select) -> bool:
    if select.find(exp.With) is not None:
        return True
    node: exp.Expression | None = select
    while node is not None:
        if node.args.get("with_") or node.args.get("with"):
            return True
        node = node.parent
    return False


def read_mirrored_self_joins(tree: exp.Expression, schema: dict | None, types: dict | None) -> tuple[exp.Expression, bool]:
    """``tree`` with the condition of every such self join read both ways round (module doc), and whether anything
    was written. The tree given is not changed. A pre-pass: the rules that follow take a self join apart (a row that
    is only tested becomes a derived table of the columns the test reads), and the two rows no longer look alike."""

    if not schema or not types or tree.find(exp.Join) is None or tree.find(exp.Distinct) is None or tree.find(exp.With) is not None:
        return tree, False
    new = tree.copy()
    wrote = False
    for _ in range(32):  # one select at a time: a rewritten select is a new node
        for select in new.find_all(exp.Select):
            read = mirrored_self_joins(select, schema, types)
            if read is None:
                continue
            wrote = True
            if select is new:
                new = read
            else:
                select.replace(read)
            break
        else:
            break
    return (new, True) if wrote else (tree, False)


def mirrored_self_joins(select: exp.Select, schema: dict | None, types: dict | None) -> exp.Select | None:
    """The select with the condition of its self join read both ways round (module doc), or None."""

    distinct = select.args.get("distinct")
    if distinct is None or distinct.args.get("on") is not None:
        return None
    if any(select.args.get(key) for key in ("group", "having", "qualify", "windows", "order", "limit", "offset", "laterals", "pivots", "sample", "into", "locks")):
        return None
    joins = select.args.get("joins") or []
    sources = _sources(select)
    if len(joins) != 1 or len(sources) != 2 or not all(isinstance(source, exp.Table) for source in sources):
        return None
    join = joins[0]
    if join.args.get("using") or join.args.get("side") or join.args.get("method") or (join.args.get("kind") or "").upper() not in ("", "INNER", "CROSS"):
        return None
    one, two = sources
    if any(source.args.get("db") or source.args.get("catalog") or source.args.get("pivots") or source.args.get("sample") for source in sources):
        return None
    if _lower(one.name) != _lower(two.name) or not any(_lower(name) == _lower(one.name) for name in (schema or {})):
        return None
    names = (_name(one), _name(two))
    if names[0] is None or names[1] is None:
        return None
    first, second = _lower(names[0].name), _lower(names[1].name)
    if not first or not second or first == second or _under_with(select):
        return None
    terms: list[exp.Expression] = []
    for holder in (join.args.get("on"), select.args.get("where")):
        if holder is not None:
            terms.extend(_terms(holder.this if isinstance(holder, exp.Where) else holder, exp.And))
    if not terms:
        return None
    for term in terms:
        for part in term.walk():
            if isinstance(part, _REFUSED):
                return None
            if isinstance(part, exp.Column) and (_lower(part.table) not in (first, second) or part.args.get("db") or part.args.get("catalog")):
                return None
        if not _same_each_time(term):
            return None
    equated = set()
    for term in terms:
        if isinstance(term, exp.EQ):
            left, right = _unwrapped(term.left), _unwrapped(term.right)
            if isinstance(left, exp.Column) and isinstance(right, exp.Column) and _lower(left.name) == _lower(right.name) and {_lower(left.table), _lower(right.table)} == {first, second}:
                equated.add(_lower(left.name))
    for item in select.expressions:
        value = _unwrapped(item.this if isinstance(item, exp.Alias) else item)
        if not isinstance(value, exp.Column) or _lower(value.table) not in (first, second) or value.args.get("db") or value.args.get("catalog"):
            return None
        if _lower(value.name) not in equated or not _typed_whole(one, value.name, types):
            return None
    same, rest = [], []
    for term in terms:
        (same if _key(_exchanged(term, *names)) == _key(term) else rest).append(term)
    if not rest:
        return None
    whole = _joined([term.copy() for term in rest], exp.And)
    mirrored = _exchanged(whole, *names)
    if _key(mirrored) == _key(whole):
        return None  # the rest reads the same both ways round as a whole
    either = exp.Or(this=exp.Paren(this=whole), expression=exp.Paren(this=mirrored))
    new = select.copy()
    new_join = new.args["joins"][0]
    new.set("where", None)
    if same:
        new_join.set("on", _joined([term.copy() for term in same], exp.And))
        if (new_join.args.get("kind") or "").upper() == "CROSS":
            new_join.set("kind", None)
        new.set("where", exp.Where(this=either))
    else:
        new_join.set("on", either)
        if (new_join.args.get("kind") or "").upper() == "CROSS":
            new_join.set("kind", None)
    return new
