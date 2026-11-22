"""Counts over a grouped table joined back to at most one of its table's own rows.

"Per first login date, how many players logged in first that day, and how many of them came back the day after" is
written with the next day's row looked up by an outer join, or with every row of the player joined and the next
day's counted:

    SELECT p.first, COUNT(p.player) AS players, COUNT(a.player) AS back
    FROM (SELECT player, MIN(day) AS first FROM logins GROUP BY player) AS p
    LEFT JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day
    GROUP BY p.first

    SELECT p.first, COUNT(DISTINCT p.player) AS players, COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) AS back
    FROM (SELECT player, MIN(day) AS first FROM logins GROUP BY player) AS p
    JOIN logins AS a ON p.player = a.player
    GROUP BY p.first

The first is read as the second, which the prover's aggregate rules read.

**Why the two are equal.** ``p`` (the *grouped table*) is the table grouped by one column ``k`` that is declared NOT
NULL, with nothing filtered: one row per value of ``k``, and every row of the table belongs to exactly one of them.
So the inner join on ``k`` alone (the *whole join*) has every row of the table once, beside the row of its group:
no row of ``p`` is lost, and a group of ``p`` rows is never empty there. The outer join has more in its ``ON``
(the *test*). When the test, with the equality on ``k``, gives the value of every column of a declared NOT NULL key
of the table (the *lookup finds at most one row*), each row of ``p`` appears in the outer join exactly once: with
the one row the test accepts, or padded with NULLs. Over the rows of a group (or over all rows, when there is no
``GROUP BY``):

- ``COUNT(*)``, and ``COUNT`` of a column of ``p`` that is never NULL, is the number of ``p`` rows: in the whole join
  that is ``COUNT(DISTINCT p.k)``, since ``k`` tells the rows of ``p`` apart and is never NULL. ``COUNT(DISTINCT p.k)``
  itself is the same number in both.
- ``COUNT`` of a column of the looked-up row that is declared NOT NULL is the number of ``p`` rows that found a row,
  which is the number of rows of the whole join that pass the test: ``COUNT(CASE WHEN test THEN 1 END)``. So is
  ``COUNT(DISTINCT a.k)``: the found rows have the ``k`` of their ``p`` row, one each.
- ``SUM(CASE WHEN a.c IS NULL THEN 0 ELSE 1 END)`` (``c`` declared NOT NULL; also written with ``IS NOT NULL`` or
  ``IF``) adds one for each ``p`` row that found a row: ``SUM(CASE WHEN test THEN 1 ELSE 0 END)`` over the whole join.
  Both are NULL exactly when there is no row at all, which only a select without ``GROUP BY`` over an empty table
  has. ``COUNT`` of such a flag that is 1 or NULL is the count above.

The groups are the same, since both joins hold every row of ``p`` and the ``GROUP BY`` reads columns of ``p`` only.

An inner join with the test (in ``ON`` or ``WHERE``) keeps only the ``p`` rows that found a row, so a group can be
lost: that form is read only without ``GROUP BY``, where ``COUNT(*)``, ``COUNT`` of a column that is never NULL and
``COUNT(DISTINCT k)`` of either side are all the number of rows of the whole join that pass the test.

Over the whole join without ``GROUP BY``, the number of ``p`` rows is also the number of distinct values of ``k`` in
the table, so ``(SELECT COUNT(DISTINCT k) FROM t)`` beside the aggregates is read as ``COUNT(DISTINCT p.k)``.

**When a lookup finds at most one row.** Each column ``c`` of the key other than ``k`` has a term of the test that
is an equality between ``c`` and a column of ``p`` holding ``MIN`` or ``MAX`` of a column declared with the type of
``c`` (or ``k`` itself, under that condition), a whole number possibly added to or taken from either: ``a.c = p.x``,
``a.c = p.x + 1``, ``a.c - 1 = p.x``, ``a.c - p.x = 1``. With a shift, ``c`` must be declared a whole number, or
``DATE`` in MySQL, where a date in arithmetic is the whole number ``YYYYMMDD``: two different values of ``c`` are two
different numbers, so at most one passes. (Whether that number is the next day is not this module's question: the
test is copied as it is written.)

**The found rows listed first.** The looked-up side may be the rows that pass, written as their own query:
``p LEFT JOIN (SELECT f.k FROM <grouped table> AS f JOIN t AS a ON f.k = a.k AND test) AS r ON p.k = r.k``. Both
grouped tables hold one row per value of ``k`` of the same table, so the row ``r`` starts from is the row of ``p``
with that ``k``; with a test that finds at most one row, ``r`` holds a ``k`` at most once, and a row of ``p`` finds
its ``k`` in ``r`` exactly when the test passes for one of its rows. So this is the outer join above with ``r.k``
standing for ``a.k``. The grouped table inside is the one written (the one outside may show fewer of its columns,
each with the same definition); ``r`` shows ``k`` only, and the join to it is the equality on ``k`` alone.
``waits_for_lookup`` tells such an ``r`` apart, so that in the last step of a proof no other rule writes it
differently before the join has been read.

**What is read** (anything else is left alone). The outer join where it stands: a select with a select list, an
optional ``GROUP BY`` and nothing else, over the grouped table ``LEFT JOIN`` the table (or the table ``RIGHT JOIN``
the grouped table; the join may stand in parentheses). A ``GROUP BY`` entry is a column of ``p``, or a name of the
select list that is no column of either source and names such a column (MySQL looks among the columns first). The
form the normalizer hands over: the same select over one derived table that selects plain columns from that join.
The inner form: a select without ``GROUP BY`` over the grouped table and the table, joined with ``ON`` or a comma,
with an optional ``WHERE``. The grouped table selects ``k`` and aggregates of the table's columns, groups by ``k``
and has nothing else. Every column names its source or belongs to one source only; no quoted names, nested queries
(but the total above), windows or volatile calls. Every aggregate is one of those above (``COUNT`` of a literal is
``COUNT(*)``; in MySQL, where a test is the number 1 or 0, ``SUM(a.c IS NOT NULL)`` is the sum of the flag), and
every column outside an aggregate is a grouped column of ``p``.

A rule of the seventeenth hardest-cases batch. ``tests/test_self_lookup_counts.py`` replays the reading on DuckDB and
shows each condition is needed.
"""

from __future__ import annotations

import re

from sqlglot import exp

from . import smt_values
from .ast_utils import FROM_KEY, visible_ctes
from .keyed_join_groups import _and_all
from .unmatched_outer_join import _conjuncts, _unwrap

NAME = "self_lookup_counts"

_AGGREGATES = (exp.Min, exp.Max, exp.Count, exp.Sum, exp.Avg)
_REFUSED = (exp.Query, exp.Subquery, exp.Window, exp.Star, exp.Filter, exp.WithinGroup, exp.Grouping, exp.IgnoreNulls, exp.RespectNulls, exp.AggFunc)
_SPJ_ARGS = frozenset({"expressions", "from_", "from", "joins"})
_GROUPED_ARGS = frozenset({"expressions", "from_", "from", "group"})


def _args(node: exp.Expression) -> set[str]:
    return {key for key, value in node.args.items() if value not in (None, [], False)}


def _identifier(node: exp.Expression | None) -> str | None:
    if isinstance(node, exp.Identifier) and not node.args.get("quoted"):
        return node.name.lower()
    return None


def _alias_of(source: exp.Expression) -> str | None:
    alias = source.args.get("alias")
    if alias is None:
        return _identifier(source.this) if isinstance(source, exp.Table) else None
    if alias.args.get("columns"):
        return None
    return _identifier(alias.this)


def _whole(node: exp.Expression | None) -> int | None:
    node = _unwrap(node) if node is not None else None
    sign = 1
    if isinstance(node, exp.Neg):
        sign, node = -1, _unwrap(node.this)
    if isinstance(node, exp.Literal) and not node.is_string and smt_values.is_integer_text(node.this):
        return sign * int(node.this)
    return None


def _kind(declared: str | None, dialect: str) -> str | None:
    if not declared:
        return None
    if smt_values.type_class(str(declared)) == smt_values.INT64:
        return "int"
    if dialect == "mysql" and re.sub(r"[(<].*", "", str(declared)).strip().upper() == "DATE":
        return "date"
    return None


class _Lookup:
    """The grouped table and the table it is joined to: who owns a column, and what the join says."""

    def __init__(self) -> None:
        self.grouped: exp.Subquery = None  # type: ignore[assignment]
        self.table: exp.Table = None  # type: ignore[assignment]
        self.p = self.a = ""  # the names the two sources go by
        self.k = ""  # the grouped column, as the table names it
        self.pk = ""  # the output of the grouped table that holds it
        self.outputs: dict[str, tuple[str, str | None]] = {}  # output of the grouped table -> ("key" | "min" | "agg", column)
        self.columns: set[str] = set()
        self.types: dict[str, str] = {}
        self.never_null: set[str] = set()
        self.keys: list[tuple[str, ...]] = []
        self.left = False
        self.test: list[exp.Expression] = []  # the terms of the join beyond the equality on k, columns qualified
        self.r = ""  # the name of a derived table of found rows that stands where the table would (module doc)
        self.via: set[str] = set()  # its outputs: each is k of the found row
        self.found: exp.Expression | None = None  # that derived table
        self.single = False  # the test was already seen to find at most one row
        self.mysql = False

    def owner(self, node: exp.Expression) -> tuple[str, str] | None:
        """``("p", output)`` or ``("a", column)`` for a plain column of one of the two sources."""

        if type(node) is not exp.Column or _identifier(node.this) is None or node.args.get("db") or node.args.get("catalog"):
            return None
        name = node.name.lower()
        theirs = self.via if self.r else self.columns  # what the looked-up side shows
        if node.args.get("table") is not None:
            table = _identifier(node.args["table"])
            if table == self.p and name in self.outputs:
                return "p", name
            if table == (self.r or self.a) and name in theirs:
                return "a", self.k if self.r else name
            return None
        in_p, in_a = name in self.outputs, name in theirs
        if in_p == in_a:
            return None
        return ("p", name) if in_p else ("a", self.k if self.r else name)

    def column(self, owner: tuple[str, str]) -> exp.Column:
        return exp.column(owner[1], table=self.p if owner[0] == "p" else self.a)

    def qualified(self, node: exp.Expression) -> exp.Expression | None:
        """A copy of ``node`` with every column naming its source, or None when one cannot be placed or a nested
        query, window or aggregate is inside."""

        if any(isinstance(inner, _REFUSED) for inner in node.walk()):
            return None
        node = node.copy()
        for column in list(node.find_all(exp.Column)):
            owner = self.owner(column)
            if owner is None:
                return None
            new = self.column(owner)
            if column is node:
                return new
            column.replace(new)
        return node

    def never_null_output(self, name: str) -> bool:
        kind, column = self.outputs[name]
        return kind == "key" or kind == "count" or (kind == "min" and column in self.never_null)

    # --- at most one row -----------------------------------------------------------------------------------------

    def _p_value(self, node: exp.Expression) -> tuple[str, bool] | None:
        """``(declared type, shifted)`` when ``node`` is a column of ``p`` holding k or a MIN or MAX of a column, a whole
        number possibly added to or taken from it."""

        node, shifted = _unwrap(node), False
        if isinstance(node, (exp.Add, exp.Sub)):
            left, right = _unwrap(node.this), _unwrap(node.expression)
            if _whole(right) is not None:
                node, shifted = left, True
            elif isinstance(node, exp.Add) and _whole(left) is not None:
                node, shifted = right, True
            else:
                return None
        owner = self.owner(node)
        if owner is None or owner[0] != "p":
            return None
        kind, column = self.outputs[owner[1]]
        if kind not in ("key", "min") or column is None or not self.types.get(column):
            return None
        return self.types[column].strip().upper(), shifted

    def _a_value(self, node: exp.Expression) -> tuple[str, bool] | None:
        """``(column, shifted)`` when ``node`` is a column of the table, a whole number possibly added or taken."""

        node, shifted = _unwrap(node), False
        if isinstance(node, (exp.Add, exp.Sub)):
            left, right = _unwrap(node.this), _unwrap(node.expression)
            if _whole(right) is not None:
                node, shifted = left, True
            elif isinstance(node, exp.Add) and _whole(left) is not None:
                node, shifted = right, True
            else:
                return None
        owner = self.owner(node)
        if owner is None or owner[0] != "a":
            return None
        return owner[1], shifted

    def _given(self, term: exp.Expression, dialect: str) -> str | None:
        """The column of the table whose value ``term`` gives from the row of ``p``, or None."""

        term = _unwrap(term)
        if type(term) is not exp.EQ:
            return None
        for one, other in ((term.this, term.expression), (term.expression, term.this)):
            found, value = self._a_value(one), self._p_value(other)
            shifted = False
            if found is None or value is None:
                # a.c - p.x = n and p.x - a.c = n
                difference, number = _unwrap(one), _whole(other)
                if type(difference) is not exp.Sub or number is None:
                    continue
                ends = [(self._a_value(difference.this), self._p_value(difference.expression)), (self._a_value(difference.expression), self._p_value(difference.this))]
                ends = [(a, p) for a, p in ends if a is not None and p is not None and not a[1] and not p[1]]
                if not ends:
                    continue
                found, value, shifted = ends[0][0], ends[0][1], True
            column, declared = found[0], self.types.get(found[0])
            if not declared or declared.strip().upper() != value[0]:
                continue
            if (shifted or found[1] or value[1]) and _kind(declared, dialect) is None:
                continue
            return column
        return None

    def at_most_one(self, dialect: str) -> bool:
        if self.single:
            return True
        given = {self.k} | {column for column in (self._given(term, dialect) for term in self.test) if column is not None}
        return any(set(key) <= given for key in self.keys)


def _grouped(source: exp.Expression, lookup: _Lookup, carried: frozenset = frozenset()) -> bool:
    """Read ``source`` as the grouped table: ``(SELECT k, MIN(c) AS x, .. FROM t GROUP BY k) AS p``. ``carried`` names
    the arguments the first source of a parenthesised join holds for the join."""

    if type(source) is not exp.Subquery or type(source.this) is not exp.Select or _args(source) - {"this", "alias"} - carried:
        return False
    alias, body = _alias_of(source), source.this
    if alias is None or _args(body) - _GROUPED_ARGS or not body.expressions:
        return False
    table = body.args[FROM_KEY].this if body.args.get(FROM_KEY) is not None else None
    if type(table) is not exp.Table or _args(table) - {"this", "alias"} or _identifier(table.this) is None:
        return False
    inner = _alias_of(table)
    group = body.args.get("group")
    if inner is None or group is None or _args(group) - {"expressions"} or len(group.expressions) != 1:
        return False

    def own(node: exp.Expression) -> str | None:
        node = _unwrap(node)
        if type(node) is not exp.Column or _identifier(node.this) is None or node.args.get("db") or node.args.get("catalog"):
            return None
        if node.args.get("table") is not None and _identifier(node.args["table"]) != inner:
            return None
        return node.name.lower()

    k = own(group.expressions[0])
    if k is None:
        return False
    outputs: dict[str, tuple[str, str | None]] = {}
    for item in body.expressions:
        value = _unwrap(item.unalias())
        name = _identifier(item.args.get("alias")) if isinstance(item, exp.Alias) else (own(value) if isinstance(value, exp.Column) else None)
        if name is None or name in outputs:
            return False
        if isinstance(value, exp.Column):
            if own(value) != k:
                return False
            outputs[name] = ("key", k)
            continue
        if type(value) not in _AGGREGATES or any(isinstance(n, _REFUSED) for n in value.walk() if n is not value and not isinstance(n, exp.Star)):
            return False
        if any(own(column) is None for column in value.find_all(exp.Column)):
            return False
        argument = _unwrap(value.this) if value.this is not None else None
        plain = type(value) in (exp.Min, exp.Max) and not (_args(value) - {"this"}) and isinstance(argument, exp.Column)
        if type(value) is exp.Count:
            outputs[name] = ("count", None)
        else:
            if any(isinstance(n, exp.Star) for n in value.walk()):
                return False
            outputs[name] = ("min", own(argument)) if plain else ("agg", None)
    keyed = [name for name, (kind, _) in outputs.items() if kind == "key"]
    if not keyed:
        return False
    lookup.grouped, lookup.p, lookup.k, lookup.pk, lookup.outputs = source, alias, k, keyed[0], outputs
    return True


_JOINS = frozenset({"joins"})


def _sources(select: exp.Select) -> tuple[exp.Expression, exp.Join, frozenset] | None:
    """The first source of ``select``, its one join, and the arguments the first source holds for the join: the join
    hangs on the select, or on the first source when the ``FROM`` is one parenthesised join."""

    from_ = select.args.get(FROM_KEY)
    if from_ is None or _args(from_) - {"this"}:
        return None
    first, joins, carried = from_.this, select.args.get("joins") or [], frozenset()
    if not joins and type(first) is exp.Subquery and not (_args(first) - {"this"}) and isinstance(first.this, (exp.Subquery, exp.Table)):
        first, carried = first.this, _JOINS  # FROM (p LEFT JOIN a ON ..)
        joins = first.args.get("joins") or []
    if len(joins) != 1 or _args(joins[0]) - {"this", "on", "side", "kind"}:
        return None
    return first, joins[0], carried


def _base(table: exp.Expression, lookup: _Lookup, schema: dict, keys: dict, not_null: dict | None, types: dict | None, carried: frozenset) -> bool:
    """Read ``table`` as the table the grouped table of ``lookup`` groups, and note what is declared about it."""

    if type(table) is not exp.Table or _args(table) - {"this", "alias"} - carried or _identifier(table.this) is None:
        return False
    name = table.name.lower()
    source = lookup.grouped.this.args[FROM_KEY].this
    lowered = {str(t).lower(): columns for t, columns in schema.items()}
    if source.name.lower() != name or name not in lowered:
        return False
    if name in visible_ctes(table) or name in visible_ctes(source):
        return False  # a WITH name hides the table: its rows and keys are not the declared ones
    lookup.table, lookup.a = table, _alias_of(table) or ""
    if not lookup.a or lookup.a == lookup.p:
        return False
    lookup.columns = {str(c).lower() for c in lowered[name]}
    lookup.types = {str(c).lower(): str(v) for c, v in ({str(t).lower(): v for t, v in (types or {}).items()}.get(name) or {}).items() if v}
    lookup.never_null = {str(c).lower() for c in ({str(t).lower(): v for t, v in (not_null or {}).items()}.get(name) or ())}
    lookup.keys = [key for key in (tuple(str(c).lower() for c in key) for key in ({str(t).lower(): v for t, v in keys.items()}.get(name) or [])) if key and set(key) <= lookup.never_null and set(key) <= lookup.columns]
    return lookup.k in lookup.never_null and lookup.k in lookup.columns


def _definitions(grouped: exp.Subquery) -> dict[str, str]:
    """What each output of a grouped table is, written without the name of its table."""

    table = grouped.this.args[FROM_KEY].this
    out: dict[str, str] = {}
    for item in grouped.this.expressions:
        value = _unwrap(item.unalias()).copy()
        for column in [value] if isinstance(value, exp.Column) else list(value.find_all(exp.Column)):
            column.set("table", None)
        name = _identifier(item.args.get("alias")) if isinstance(item, exp.Alias) else (value.name.lower() if isinstance(value, exp.Column) else None)
        out[name or ""] = value.sql().lower()
    return out


def _found_rows(source: exp.Expression, lookup: _Lookup, schema: dict, keys: dict, not_null: dict | None, types: dict | None, dialect: str) -> bool:
    """Read ``source``, the looked-up side of an outer join, as the found rows listed first:
    ``(SELECT p.k FROM <the grouped table> AS p JOIN t AS a ON p.k = a.k AND test) AS r`` (module doc)."""

    if type(source) is not exp.Subquery or type(source.this) is not exp.Select or _args(source) - {"this", "alias"}:
        return False
    alias, body = _alias_of(source), source.this
    if alias is None or alias == lookup.p or _args(body) - _SPJ_ARGS - {"where"} or not body.expressions:
        return False
    inner = _read_join(body, schema, keys, not_null, types, dialect)
    if inner is None or inner.left or inner.r or not inner.test or not inner.at_most_one(dialect):
        return False
    # Both grouped tables hold one row per value of k of the same table (``_base`` saw that it is the declared
    # table both times), so the row the found rows start from is the row the outer join starts from. The one inside
    # is written with the name of the one outside, which may show fewer columns of that row.
    if inner.table.name.lower() != lookup.grouped.this.args[FROM_KEY].this.name.lower() or inner.k != lookup.k or inner.pk != lookup.pk:
        return False
    if inner.a in (lookup.p, alias) or lookup.grouped.this.args[FROM_KEY].this.name.lower() in visible_ctes(lookup.grouped.this.args[FROM_KEY].this):
        return False
    theirs = _definitions(inner.grouped)
    if any(theirs.get(name) != text for name, text in _definitions(lookup.grouped).items()):
        return False
    shown: set[str] = set()
    for item in body.expressions:
        owner = inner.owner(_unwrap(item.unalias()))
        output = _identifier(item.args.get("alias")) if isinstance(item, exp.Alias) else (owner[1] if owner else None)
        if owner not in (("p", inner.pk), ("a", inner.k)) or output is None or output in shown:
            return False
        shown.add(output)
    lookup.table, lookup.a, lookup.r, lookup.via, lookup.single, lookup.found = inner.table, inner.a, alias, shown, True, source
    lookup.columns, lookup.types, lookup.never_null, lookup.keys = inner.columns, inner.types, inner.never_null, inner.keys
    grouped = inner.grouped.copy()
    grouped.set("alias", exp.TableAlias(this=exp.to_identifier(lookup.p)))
    lookup.grouped, lookup.outputs = grouped, inner.outputs
    for term in inner.test:
        term = term.copy()
        for column in term.find_all(exp.Column):
            if _identifier(column.args.get("table")) == inner.p:
                column.set("table", exp.to_identifier(lookup.p))
        lookup.test.append(term)
    return True


def _read_join(select: exp.Select, schema: dict, keys: dict, not_null: dict | None, types: dict | None, dialect: str) -> _Lookup | None:
    """The two sources of ``select`` as a grouped table and its table, with the terms of their join; ``WHERE`` terms
    count as terms of an inner join. The looked-up side of an outer join may be the found rows listed first."""

    sources = _sources(select)
    if sources is None:
        return None
    first, join, carried = sources
    side, kind = (join.args.get("side") or "").upper(), (join.args.get("kind") or "").upper()
    second = join.this
    lookup = _Lookup()
    lookup.mysql = dialect == "mysql"
    none = frozenset()
    if side == "LEFT" and kind in ("", "OUTER"):
        lookup.left = True
        if not _grouped(first, lookup, carried):
            return None
        table, held = second, none
    elif side == "RIGHT" and kind in ("", "OUTER"):
        lookup.left = True
        if not _grouped(second, lookup):
            return None
        table, held = first, carried
    elif side == "" and kind in ("", "INNER", "CROSS"):
        if _grouped(first, lookup, carried):
            table, held = second, none
        elif _grouped(second, lookup):
            table, held = first, carried
        else:
            return None
    else:
        return None
    if not _base(table, lookup, schema, keys, not_null, types, held):
        if not lookup.left or held or not _found_rows(table, lookup, schema, keys, not_null, types, dialect):
            return None
    terms = _conjuncts(join.args.get("on"))
    if not lookup.left:
        terms += _conjuncts(select.args["where"].this) if select.args.get("where") is not None else []
    elif select.args.get("where") is not None:
        return None
    linked = False
    for term in terms:
        term = _unwrap(term)
        if type(term) is exp.EQ and not linked:
            ends = {lookup.owner(_unwrap(term.this)), lookup.owner(_unwrap(term.expression))}
            if ends == {("p", lookup.pk), ("a", lookup.k)}:
                linked = True
                continue
        if lookup.r:
            return None  # the found rows are joined on k and nothing else
        written = lookup.qualified(term)
        if written is None:
            return None
        lookup.test.append(written)
    return lookup if linked else None


def _flag(node: exp.Expression, lookup: _Lookup, resolve) -> tuple[exp.Expression | None, exp.Expression | None] | None:
    """``(found, missing)``: what ``node`` is for a row of ``p`` that found a row and for one that did not, when
    ``node`` tests a NOT NULL column of the looked-up row for NULL: ``CASE WHEN a.c IS NULL THEN x ELSE y END``."""

    node = _unwrap(node)
    if lookup.mysql and (type(node) is exp.Is or (type(node) is exp.Not and type(_unwrap(node.this)) is exp.Is)):
        # in MySQL a test is the number 1 or 0: a.c IS NOT NULL is 1 for the row that found one
        test = _unwrap(node.this) if type(node) is exp.Not else node
        owner = resolve(_unwrap(test.this)) if type(_unwrap(test.expression)) is exp.Null else None
        if owner is None or owner[0] != "a" or owner[1] not in lookup.never_null:
            return None
        one, zero = exp.Literal.number(1), exp.Literal.number(0)
        return (one, zero) if type(node) is exp.Not else (zero, one)
    if type(node) is exp.Case:
        if node.args.get("this") is not None or len(node.args.get("ifs") or []) != 1:
            return None
        test, yes, no = node.args["ifs"][0].this, node.args["ifs"][0].args.get("true"), node.args.get("default")
    elif type(node) is exp.If:
        test, yes, no = node.this, node.args.get("true"), node.args.get("false")
    else:
        return None
    test, negated = _unwrap(test), False
    if type(test) is exp.Not:
        test, negated = _unwrap(test.this), True
    if type(test) is not exp.Is or type(_unwrap(test.expression)) is not exp.Null:
        return None
    owner = resolve(_unwrap(test.this))
    if owner is None or owner[0] != "a" or owner[1] not in lookup.never_null:
        return None
    for value in (yes, no):
        if value is not None and not isinstance(_unwrap(value), (exp.Literal, exp.Null)):
            return None
    return (yes, no) if negated else (no, yes)  # IS NULL is true for the row that found nothing


def _one(node: exp.Expression | None) -> bool:
    node = _unwrap(node) if node is not None else None
    return isinstance(node, exp.Literal) and not node.is_string and node.this == "1"


def _zero(node: exp.Expression | None) -> bool:
    node = _unwrap(node) if node is not None else None
    return isinstance(node, exp.Literal) and not node.is_string and node.this == "0"


def _nothing(node: exp.Expression | None) -> bool:
    return node is None or isinstance(_unwrap(node), exp.Null)


def _own_aggregates(select: exp.Select) -> list[exp.Expression]:
    """The aggregate calls of the select list of ``select`` itself, not those of a nested query."""

    return [node for item in select.expressions for node in item.walk() if isinstance(node, exp.AggFunc) and node.find_ancestor(exp.Select) is select]


def _total(node: exp.Expression, lookup: _Lookup) -> bool:
    """Whether ``node`` is ``(SELECT COUNT(DISTINCT k) FROM t)`` over the table of the lookup."""

    node = _unwrap(node)
    if type(node) is not exp.Subquery or _args(node) - {"this"} or type(node.this) is not exp.Select:
        return False
    body = node.this
    if _args(body) - {"expressions", "from_", "from"} or len(body.expressions) != 1:
        return False
    table = body.args[FROM_KEY].this if body.args.get(FROM_KEY) is not None else None
    if type(table) is not exp.Table or _args(table) - {"this", "alias"} or table.name.lower() != lookup.table.name.lower():
        return False
    if table.name.lower() in visible_ctes(table):
        return False
    call = _unwrap(body.expressions[0].unalias())
    if type(call) is not exp.Count or _args(call) - {"this", "big_int"} or type(_unwrap(call.this)) is not exp.Distinct:
        return False
    distinct = _unwrap(call.this)
    if distinct.args.get("on") or len(distinct.expressions) != 1:
        return False
    column = _unwrap(distinct.expressions[0])
    if type(column) is not exp.Column or _identifier(column.this) is None or column.args.get("db") or column.args.get("catalog"):
        return False
    if column.args.get("table") is not None and _identifier(column.args["table"]) != _alias_of(table):
        return False
    return column.name.lower() == lookup.k


class _Unread(Exception):
    pass


def _rewrite(expression: exp.Expression, lookup: _Lookup, resolve, outer: bool, grouped: set[tuple[str, str]]) -> exp.Expression:
    """``expression`` of the select list over the whole join: its aggregates as the module doc reads them, its other
    columns (they must be ``grouped`` ones) as columns of ``p``. ``outer`` says the select reads the outer join (each
    ``p`` row once); otherwise the inner join with the test (only the ``p`` rows that found a row). The total of the
    module doc is kept as it is. Raises ``_Unread``."""

    key = exp.column(lookup.pk, table=lookup.p)
    test = _and_all([term.copy() for term in lookup.test])

    def rows() -> exp.Expression:  # the number of p rows
        return exp.Count(this=exp.Distinct(expressions=[key.copy()]))

    def found() -> exp.Expression:  # the number of p rows that found a row
        return exp.Count(this=exp.Case(ifs=[exp.If(this=test.copy(), true=exp.Literal.number(1))]))

    def aggregate(call: exp.Expression) -> exp.Expression:
        if _args(call) - {"this", "big_int"}:
            raise _Unread
        inner, distinct = _unwrap(call.this), False
        if type(inner) is exp.Distinct:
            if inner.args.get("on") or len(inner.expressions) != 1:
                raise _Unread
            inner, distinct = _unwrap(inner.expressions[0]), True
        if type(call) is exp.Count:
            if (isinstance(inner, exp.Star) or type(inner) is exp.Literal) and not distinct:
                return rows() if outer else found()  # COUNT(1) counts the rows, as COUNT(*) does
            owner = resolve(inner)
            if owner is not None:
                side, name = owner
                if side == "p" and distinct and name == lookup.pk:
                    return rows() if outer else found()
                if side == "p" and not distinct and lookup.never_null_output(name):
                    return rows() if outer else found()
                if side == "a" and distinct and name == lookup.k:
                    return found()
                if side == "a" and not distinct and name in lookup.never_null:
                    return found()
                raise _Unread
            flag = None if distinct or not outer else _flag(inner, lookup, resolve)
            if flag is not None and not _nothing(flag[0]) and _nothing(flag[1]):
                return found()  # a value for the row that found one, NULL for the one that did not
            raise _Unread
        if type(call) is exp.Sum and not distinct and outer:
            flag = _flag(inner, lookup, resolve)
            if flag is not None and _one(flag[0]) and _zero(flag[1]):
                return exp.Sum(this=exp.Case(ifs=[exp.If(this=test.copy(), true=exp.Literal.number(1))], default=exp.Literal.number(0)))
        raise _Unread

    def walk(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.AggFunc):
            return aggregate(node)
        if _total(node, lookup):
            return node.copy()
        if isinstance(node, (exp.Query, exp.Subquery, exp.Window, exp.Star, exp.Filter, exp.WithinGroup, exp.Grouping)):
            raise _Unread
        if isinstance(node, exp.Column):
            owner = resolve(node)
            if owner is None or owner not in grouped:
                raise _Unread
            return lookup.column(owner)
        node = node.copy()
        for name, value in list(node.args.items()):
            if isinstance(value, exp.Expression):
                node.set(name, walk(value))
            elif isinstance(value, list):
                node.set(name, [walk(item) if isinstance(item, exp.Expression) else item for item in value])
        return node

    return walk(expression)


def _whole_join(lookup: _Lookup, items: list[exp.Expression], group: list[exp.Expression]) -> exp.Select:
    link = exp.EQ(this=exp.column(lookup.pk, table=lookup.p), expression=exp.column(lookup.k, table=lookup.a))
    select = exp.Select(expressions=items)
    grouped, table = lookup.grouped.copy(), lookup.table.copy()
    grouped.set("joins", None)
    table.set("joins", None)
    select.set(FROM_KEY, exp.From(this=grouped))
    select.set("joins", [exp.Join(this=table, on=link)])
    if group:
        select.set("group", exp.Group(expressions=group))
    return select


def _named(item: exp.Expression, new: exp.Expression) -> exp.Expression:
    if isinstance(item, exp.Alias):
        return exp.Alias(this=new, alias=item.args["alias"].copy())
    if not isinstance(item, exp.Column) or (isinstance(new, exp.Column) and new.name.lower() == item.name.lower()):
        return new  # an item without a name stays without one
    return exp.alias_(new, item.name)


def _outer_form(select: exp.Select, schema: dict, keys: dict, not_null: dict | None, types: dict | None, dialect: str) -> exp.Select | None:
    if _args(select) - {"expressions", "from_", "from", "group"} or select.args.get(FROM_KEY) is None:
        return None
    derived = select.args[FROM_KEY].this
    if type(derived) is not exp.Subquery or type(derived.this) is not exp.Select or _args(derived) - {"this", "alias"}:
        return None
    name, body = _alias_of(derived), derived.this
    if name is None or _args(body) - _SPJ_ARGS:
        return None
    lookup = _read_join(body, schema, keys, not_null, types, dialect)
    if lookup is None or not lookup.left or not lookup.test or not lookup.at_most_one(dialect):
        return None
    listed: dict[str, tuple[str, str]] = {}
    for item in body.expressions:
        owner = lookup.owner(_unwrap(item.unalias()))
        output = _identifier(item.args.get("alias")) if isinstance(item, exp.Alias) else (owner[1] if owner else None)
        if owner is None or output is None or output in listed:
            return None
        listed[output] = owner

    def resolve(node: exp.Expression) -> tuple[str, str] | None:
        if type(node) is not exp.Column or _identifier(node.this) is None or node.args.get("db") or node.args.get("catalog"):
            return None
        if node.args.get("table") is not None and _identifier(node.args["table"]) != name:
            return None
        return listed.get(node.name.lower())

    group = select.args.get("group")
    grouped: list[tuple[str, str]] = []
    if group is not None:
        if _args(group) - {"expressions"}:
            return None
        for column in group.expressions:
            owner = resolve(_unwrap(column))
            if owner is None or owner[0] != "p":
                return None
            grouped.append(owner)
    elif not _own_aggregates(select):
        return None  # not a select of aggregates: it has one row per row of the join
    try:
        items = [_named(item, _rewrite(item.unalias(), lookup, resolve, True, set(grouped))) for item in select.expressions]
    except _Unread:
        return None
    return _whole_join(lookup, items, [lookup.column(owner) for owner in grouped])


def _direct_form(select: exp.Select, schema: dict, keys: dict, not_null: dict | None, types: dict | None, dialect: str) -> exp.Select | None:
    """The outer join read where it stands: ``SELECT .. FROM p LEFT JOIN a ON .. [GROUP BY ..]`` with nothing else."""

    if _args(select) - {"expressions", "from_", "from", "joins", "group"}:
        return None
    lookup = _read_join(select, schema, keys, not_null, types, dialect)
    if lookup is None or not lookup.left or not lookup.test or not lookup.at_most_one(dialect):
        return None
    group = select.args.get("group")
    grouped: list[tuple[str, str]] = []
    if group is not None:
        if _args(group) - {"expressions"}:
            return None
        theirs = lookup.via if lookup.r else lookup.columns
        for column in group.expressions:
            column = _unwrap(column)
            owner = lookup.owner(column)
            if owner is None and type(column) is exp.Column and column.args.get("table") is None and _identifier(column.this) is not None:
                # a name of the select list, when it is no column of either source: the item it names
                name = column.name.lower()
                named = [item for item in select.expressions if isinstance(item, exp.Alias) and _identifier(item.args.get("alias")) == name]
                if name not in lookup.outputs and name not in theirs and len(named) == 1:
                    owner = lookup.owner(_unwrap(named[0].this))
            if owner is None or owner[0] != "p":
                return None
            grouped.append(owner)
    elif not _own_aggregates(select):
        return None
    try:
        items = [_named(item, _rewrite(item.unalias(), lookup, lookup.owner, True, set(grouped))) for item in select.expressions]
    except _Unread:
        return None
    return _whole_join(lookup, items, [lookup.column(owner) for owner in grouped])


def _inner_form(select: exp.Select, schema: dict, keys: dict, not_null: dict | None, types: dict | None, dialect: str) -> exp.Select | None:
    if _args(select) - {"expressions", "from_", "from", "joins", "where"}:
        return None
    lookup = _read_join(select, schema, keys, not_null, types, dialect)
    if lookup is None or lookup.left or not _own_aggregates(select):
        return None
    if lookup.test:
        # the join keeps the p rows that found a row: its counts are counts of the rows of the whole join that pass
        if not lookup.at_most_one(dialect):
            return None
        try:
            items = [_named(item, _rewrite(item.unalias(), lookup, lookup.owner, False, set())) for item in select.expressions]
        except _Unread:
            return None
        return _whole_join(lookup, items, [])
    # the whole join itself: the number of distinct k in the table is its number of p rows
    items, wrote = [], False
    for item in select.expressions:
        item = item.copy()
        for node in list(item.walk()):
            if _total(node, lookup):
                if node.find_ancestor(exp.AggFunc) is not None:
                    return None
                node.replace(exp.Count(this=exp.Distinct(expressions=[exp.column(lookup.pk, table=lookup.p)])))
                wrote = True
        for node in item.walk():
            if isinstance(node, (exp.Query, exp.Subquery, exp.Window, exp.Star)) and not (isinstance(node, exp.Star) and isinstance(node.parent, exp.Count)):
                return None
            if isinstance(node, exp.Column) and (node.find_ancestor(exp.AggFunc) is None or lookup.owner(node) is None):
                return None
        items.append(item)
    return _whole_join(lookup, items, []) if wrote else None


def waits_for_lookup(select: exp.Select, schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None, dialect: str) -> bool:
    """Whether ``select`` is the found rows of an outer join this module reads (module doc): another rule that would
    write them differently leaves them alone until the join has been read. ``select`` is not changed."""

    source = select.parent
    if not schema or not keys or type(source) is not exp.Subquery:
        return False
    outer = source.find_ancestor(exp.Select)
    if outer is None or any(isinstance(node, exp.Identifier) and node.args.get("quoted") for node in outer.walk()):
        return False
    lookup = _read_join(outer, schema, keys, not_null, types, dialect)
    return lookup is not None and lookup.found is source


def read_self_lookup_counts(select: exp.Select, schema: dict | None, keys: dict | None, not_null: dict | None, types: dict | None, dialect: str) -> exp.Select | None:
    """``select`` written over the whole join (module doc), or None when it is not one of the forms read. ``select``
    is not changed."""

    if not schema or not keys or type(select) is not exp.Select:
        return None
    from .counted_membership import _unstable

    if _unstable(select):
        return None
    if not (select.args.get("joins") or (select.args.get(FROM_KEY) is not None and type(select.args[FROM_KEY].this) is exp.Subquery)):
        return None
    if any(isinstance(node, exp.Identifier) and node.args.get("quoted") for node in select.walk()):
        return None
    for form in (_outer_form, _direct_form, _inner_form):
        new = form(select, schema, keys, not_null, types, dialect)
        if new is not None:
            return new
    return None
