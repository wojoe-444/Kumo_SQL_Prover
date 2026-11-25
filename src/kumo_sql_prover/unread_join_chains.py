"""An outer join nothing reads is dropped from a longer chain of joins, or beside windows over groups.

``X LEFT JOIN c ON cond`` returns every row of ``X`` at least once: once per matching row of ``c``, and once with
NULLs when none matches. When nothing outside ``cond`` reads ``c``, the join only repeats rows of ``X``. A select that
cannot see repeats returns the same rows without it:

    SELECT p.id, p.title, u.name FROM posts AS p JOIN users AS u ON p.owner = u.id LEFT JOIN votes AS v ON v.post = p.id
    GROUP BY p.id, p.title, u.name

is the same select without ``LEFT JOIN votes``. ``dedup_join_rules.drop_unread_outer_join`` reads a select whose only
join is the unread one. This module (hardest-cases batch 28) reads the two shapes it leaves alone:

* **a chain.** The unread join is one of several. The joins after it do not read ``c`` either (checked), so each of
  them sees the rows of ``X`` it saw, some of them more than once, and returns the rows it returned, some of them
  more than once.
* **windows over groups.** A grouped select may have window functions: they are evaluated over its groups, after
  ``GROUP BY`` and ``HAVING``, and the groups and their aggregates are the ones the select has without the join. (A
  ``DISTINCT`` select with a window is left alone: its windows read the repeated rows.)

**When repeats cannot matter.** The select has ``GROUP BY`` (no ``ROLLUP``, ``CUBE`` or grouping sets) or a plain
``DISTINCT``, and every aggregate it evaluates over its rows is ``MIN``, ``MAX`` or a ``COUNT`` / ``SUM`` / ``AVG`` of
``DISTINCT`` values. The function a window applies (``SUM(..) OVER``) is not such an aggregate: it reads groups.

**What is refused.** The select is left alone unless all of this holds:

* the join is a ``LEFT [OUTER] JOIN`` with an ``ON``, not ``NATURAL``, not ``USING``, not ``LATERAL``, of a table or a
  derived table with a name no other source of the select has;
* no column anywhere else in the select names the joined source. A bare name counts as a column of the joined
  source when that source has a column of the name, and when its columns are not known (a table that is not in the
  supplied schema or that a ``WITH`` binds, a derived table with a ``*``) every bare name does. Columns inside
  another derived table of the ``FROM`` clause cannot see the joined source and are not looked at;
* no bare ``*`` item and no ``c.*``, no ``LATERAL``, no ``DISTINCT ON``, no function the prover does not know (it
  could be an aggregate that counts repeats);
* every join written after it is an inner, cross, comma or left join with no ``USING`` and not ``NATURAL`` (those
  name columns of every source before them, the dropped one included).

The ``ON`` test may be anything: whatever it says of a pair of rows, a row of ``X`` with no match is still returned
once. Like every normalization rule this keeps the rows of a database on which nothing fails; a test that can raise
an error (a division, a cast) no longer can once the join is gone, which ``normalized_disclosure`` accounts for in
what a proof says about errors.

One join is dropped per call; the caller applies the rule until nothing changes.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY, binding_cte, extended_grouping

NAME = "unread_join_chains"

_BLIND = (exp.Min, exp.Max)
_DISTINCT_BLIND = (exp.Count, exp.Sum, exp.Avg)
_LATER_KINDS = frozenset({"", "INNER", "CROSS", "OUTER"})


def _own(select: exp.Select, kinds) -> list[exp.Expression]:
    return [node for node in select.find_all(kinds) if node.find_ancestor(exp.Select) is select]


def _window_head(call: exp.Expression) -> bool:
    """Whether ``call`` is the function a window applies (``SUM(x) OVER (..)``), which reads groups and not rows."""

    holder = call
    while isinstance(holder.parent, (exp.IgnoreNulls, exp.RespectNulls)) and holder.arg_key == "this":
        holder = holder.parent
    return isinstance(holder.parent, exp.Window) and holder.arg_key == "this"


def _blind(select: exp.Select) -> bool:
    """Whether the select returns the same rows when rows of its FROM clause are repeated."""

    distinct, group = select.args.get("distinct"), select.args.get("group")
    if distinct is not None and distinct.args.get("on"):
        return False
    if group is not None and (extended_grouping(group) or not group.expressions):
        return False  # ROLLUP, CUBE, grouping sets, GROUP BY ALL
    if group is None and (distinct is None or _own(select, exp.Window)):
        return False  # no GROUP BY: only DISTINCT hides repeats, and a window of such a select reads them
    for call in _own(select, exp.AggFunc):
        if _window_head(call) or isinstance(call, _BLIND):
            continue
        if isinstance(call, _DISTINCT_BLIND) and isinstance(call.this, exp.Distinct) and not call.this.args.get("on"):
            continue
        return False
    return not any(isinstance(node, exp.Anonymous) for node in select.walk())


def _name(source: exp.Expression) -> str | None:
    if isinstance(source, exp.Table):
        return (source.alias or source.name or "").lower() or None
    if type(source) is exp.Subquery:
        return (source.alias or "").lower() or None
    return None


def _under(node: exp.Expression, roots: list[exp.Expression], stop: exp.Expression) -> bool:
    while node is not None and node is not stop:
        if any(node is root for root in roots):
            return True
        node = node.parent
    return False


def _columns(source: exp.Expression, schema: dict[str, list[str]] | None) -> frozenset[str] | None:
    """The column names of a source of the FROM clause, lower case, or None when they are not known."""

    alias = source.args.get("alias")
    if alias is not None and alias.args.get("columns"):
        return None
    if isinstance(source, exp.Table):
        if source.args.get("db") or source.args.get("catalog") or binding_cte(source) is not None:
            return None
        found = [columns for table, columns in (schema or {}).items() if table.lower() == source.name.lower()]
        return frozenset(column.lower() for column in found[0]) if len(found) == 1 and found[0] else None
    query = source.this
    if not isinstance(query, exp.Select):
        return None
    names = [item.alias_or_name.lower() for item in query.expressions if not isinstance(item, exp.Star) and not (isinstance(item, exp.Column) and isinstance(item.this, exp.Star))]
    return frozenset(names) if len(names) == len(query.expressions) and all(names) else None


def drop_unread_chained_join(select: exp.Select, schema: dict[str, list[str]] | None = None) -> exp.Select | None:
    """``select`` without one unread ``LEFT JOIN`` (see the module docstring), or None."""

    joins = list(select.args.get("joins") or [])
    source = select.args.get(FROM_KEY)
    if not joins or source is None or select.args.get("laterals") or select.args.get("pivots"):
        return None
    if len(joins) < 2 and not _own(select, exp.Window):
        return None  # the one-join select without windows is read by dedup_join_rules.drop_unread_outer_join
    if not _blind(select):
        return None
    if any(not isinstance(node.parent, (exp.Count, exp.Column)) for node in _own(select, exp.Star)):
        return None  # a bare * lists the columns of every source
    sources = [source.this] + [join.this for join in joins]
    names = [_name(item) for item in sources]
    if None in names or len(set(names)) != len(names):
        return None
    derived = [item for item in sources if type(item) is exp.Subquery]
    for index, join in enumerate(joins):
        far, name, on = join.this, names[index + 1], join.args.get("on")
        if (join.side or "").upper() != "LEFT" or (join.kind or "").upper() not in ("", "OUTER") or join.args.get("method"):
            continue
        if on is None or join.args.get("using") or not isinstance(far, (exp.Table, exp.Subquery)):
            continue
        later = joins[index + 1 :]
        if any(
            (other.side or "").upper() not in ("", "LEFT") or (other.kind or "").upper() not in _LATER_KINDS or other.args.get("method") or other.args.get("using")
            for other in later
        ):
            continue
        columns, read = _columns(far, schema), False
        for column in select.find_all(exp.Column):
            if _under(column, derived, select) or _under(column, [on], select):
                continue  # inside a derived table of the FROM clause, which cannot see the joined source, or in its ON
            if column.args.get("db"):
                return None
            if column.table.lower() == name if column.table else (columns is None or column.name.lower() in columns):
                read = True
                break
        if read:
            continue
        copy = select.copy()
        kept = list(copy.args["joins"])
        del kept[index]
        copy.set("joins", kept or None)
        return copy
    return None
