"""A ``DISTINCT`` over the aggregates of each row's whole partition is the grouped aggregate.

    SELECT DISTINCT p.team, ROUND(AVG(e.years) OVER (PARTITION BY p.team), 2) AS mean
    FROM projects AS p JOIN staff AS e ON p.person = e.person

puts the team's mean beside every row of the team and then drops the repeats, which is

    SELECT p.team, ROUND(AVG(e.years), 2) AS mean
    FROM projects AS p JOIN staff AS e ON p.person = e.person GROUP BY p.team

The prover keeps a windowed select whole and knows a grouped one well, so the two spellings of one answer never met.
The same holds one level up, when the windows are computed in a derived table or a ``WITH`` query and the
``DISTINCT`` is taken by the select that reads it:

    SELECT DISTINCT mail FROM (SELECT mail, COUNT(*) OVER (PARTITION BY mail) AS n FROM people) AS t WHERE n > 1

reads ``SELECT mail, COUNT(*) AS n FROM people GROUP BY mail`` in the place of the derived table.

**Why it is exact.**

* A window with a ``PARTITION BY``, no ``ORDER BY`` and no frame aggregates every row of the row's partition: the
  rows the select reads (after its joins and its ``WHERE``) that agree on the partition expressions, NULL with NULL.
  ``GROUP BY`` on the same expressions makes the same sets of rows, and ``MIN``, ``MAX``, ``SUM``, ``AVG`` and
  ``COUNT`` return the same value and type for a set of rows as a window and as a grouped aggregate.
* Every output is a partition expression itself, a constant, or an expression of the windows and constants alone.
  So all rows of one partition give the same output row, ``DISTINCT`` keeps one of them, and the grouped select
  returns exactly that one. Rows of different partitions that give equal output rows are merged by the
  ``DISTINCT``, which therefore stays unless every partition expression is itself an output (the groups are then
  told apart by their outputs).
* A partition with no row does not exist in either spelling, so there is no empty group to get wrong: an empty input
  gives no row both ways (a ``PARTITION BY`` is required, and each of its expressions must read a column, because
  ``GROUP BY 1`` would name an output position).
* One level up: the rows of a partition are the same row of the derived table, repeated. The reading select may
  test them with a condition on the window columns, which is the same for every copy, and takes ``DISTINCT`` of
  what it shows, so one copy per partition gives the same answer.

**What is refused.** Nothing is rewritten unless all of this holds.

* The select with the windows: no ``GROUP BY``, ``HAVING``, ``QUALIFY``, ``LIMIT``, ``OFFSET``, named-window clause
  or aggregate outside a window; windows only in the select list; every window a plain ``MIN``, ``MAX``, ``SUM``,
  ``AVG`` or ``COUNT`` (no ``DISTINCT`` argument, ``FILTER``, ``IGNORE NULLS``, ``ORDER BY``, frame or window name)
  over the same set of partition expressions; no ``*``, subquery, parameter or call that can differ from one
  evaluation to the next.
* An output that reads a column outside a window must be a partition expression and nothing more. ``UPPER(team)``
  beside ``PARTITION BY team`` is refused: two spellings of a team that an engine's collation calls equal share a
  partition and would still show two rows. A bare and a qualified spelling of one column (``team``, ``p.team``)
  count as the same expression when the select has no ``USING`` or ``NATURAL`` join; a valid query cannot mean two
  columns by them.
* A partition expression may not be a bare name that an output alias of the select redefines.
* Direct form: the select has ``DISTINCT`` (not ``DISTINCT ON``). Its ``ORDER BY`` may name outputs and partition
  expressions only.
* One level up: the reading select has ``DISTINCT``, that one source and no join, grouping, aggregate, window,
  subquery or ``LIMIT``; it shows a partition column only as a whole output and reads only window columns
  elsewhere. A ``WITH`` query is rewritten only when it is read once in the whole statement and has no column list.

A pre-pass of the last proof step (``shape_fallback.TWENTY_FOURTH_BATCH``): the rules that follow isolate a
windowed select in a derived table and keep it whole.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .filter_terms import _same_each_time

_AGGREGATES = (exp.Min, exp.Max, exp.Sum, exp.Avg, exp.Count)
_REFUSED_CLAUSES = (
    "group", "having", "qualify", "windows", "limit", "offset", "laterals", "pivots", "sample", "into", "locks",
    "connect", "kind", "match", "distribute", "sort", "cluster", "prewhere", "exclude", "operation_modifiers",
)
_NOT_A_VALUE = (exp.Query, exp.Exists, exp.Star, exp.Placeholder, exp.Parameter)


def _unwrapped(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _text(node: exp.Expression) -> str:
    """The SQL of ``node`` with its unquoted names in lower case (a string literal keeps its letters)."""

    copy = node.copy()
    for name in copy.find_all(exp.Identifier):
        if not name.args.get("quoted"):
            name.set("this", name.name.lower())
    return copy.sql()


def _plain_column(node: exp.Expression) -> bool:
    return isinstance(node, exp.Column) and not isinstance(node.this, exp.Star) and not node.args.get("db") and not node.args.get("catalog")


def _is_key(node: exp.Expression, keys: list[exp.Expression], loose: bool) -> bool:
    """Whether ``node`` is one of the partition expressions ``keys`` (module doc: bare and qualified spellings)."""

    node = _unwrapped(node)
    text = _text(node)
    for key in keys:
        if _text(key) == text:
            return True
        if loose and _plain_column(node) and _plain_column(key) and node.name.lower() == key.name.lower() and not (node.table and key.table):
            if bool(node.this.args.get("quoted")) == bool(key.this.args.get("quoted")):
                return True
    return False


def _own_windows(select: exp.Select, under: list[exp.Expression] | None = None) -> list[exp.Window]:
    roots = [select] if under is None else under
    return [window for root in roots for window in root.find_all(exp.Window) if window.find_ancestor(exp.Select) is select]


def _window_keys(window: exp.Window) -> list[exp.Expression] | None:
    """The partition expressions of a window the rule reads, None for any other window."""

    call = window.this
    if not isinstance(call, _AGGREGATES) or isinstance(call.this, exp.Distinct):
        return None
    if window.args.get("order") or window.args.get("spec") or window.args.get("alias") or window.args.get("first"):
        return None
    if str(window.args.get("over") or "OVER").upper() != "OVER":
        return None
    keys = [_unwrapped(key) for key in window.args.get("partition_by") or []]
    if not keys:
        return None
    for key in keys:
        if key.find(exp.Column) is None or key.find(exp.Window, exp.AggFunc, *_NOT_A_VALUE) is not None or not _same_each_time(key):
            return None
    for part in call.walk():
        if part is not call and isinstance(part, (exp.Window, exp.AggFunc, exp.Query, exp.Exists, exp.Placeholder, exp.Parameter)):
            return None
    if not _same_each_time(call):
        return None
    return keys


def _reads_no_row(node: exp.Expression) -> bool:
    """Whether ``node`` reads the row it is computed for only through windows."""

    if isinstance(node, exp.Window):
        return True
    if isinstance(node, (exp.Column, exp.AggFunc, *_NOT_A_VALUE)):
        return False
    return all(_reads_no_row(child) for child in node.iter_expressions())


def _kinds(select: exp.Select) -> tuple[list[exp.Expression], list[str]] | None:
    """The partition expressions of a select the rule reads and, per output, ``key``, ``window`` or ``constant``."""

    if any(select.args.get(clause) for clause in _REFUSED_CLAUSES) or select.args.get(FROM_KEY) is None:
        return None
    windows = _own_windows(select)
    if not windows or len(_own_windows(select, list(select.expressions))) != len(windows):
        return None
    keys: list[exp.Expression] | None = None
    for window in windows:
        found = _window_keys(window)
        if found is None:
            return None
        if keys is None:
            keys = found
        elif {_text(key) for key in keys} != {_text(key) for key in found}:
            return None
    assert keys is not None
    joins = select.args.get("joins") or []
    loose = not any(join.args.get("using") or (join.method or "").upper() == "NATURAL" for join in joins)
    aliases = {
        item.alias.lower() for item in select.expressions
        if isinstance(item, exp.Alias) and not (isinstance(item.this, exp.Column) and item.this.name.lower() == item.alias.lower())
    }
    if any(not column.table and column.name.lower() in aliases for key in keys for column in key.find_all(exp.Column)):
        return None
    kinds = []
    for item in select.expressions:
        value = _unwrapped(item.this if isinstance(item, exp.Alias) else item)
        if isinstance(value, exp.Star) or isinstance(value, exp.Column) and isinstance(value.this, exp.Star):
            return None
        if _is_key(value, keys, loose):
            kinds.append("key")
        elif _reads_no_row(value) and _same_each_time(value):
            kinds.append("window" if value.find(exp.Window) is not None else "constant")
        else:
            return None
    unique: list[exp.Expression] = []
    for key in keys:
        if _text(key) not in {_text(other) for other in unique}:
            unique.append(key)
    return unique, kinds


def _grouped(select: exp.Select, keys: list[exp.Expression], kinds: list[str]) -> exp.Select:
    """A copy of ``select`` with each window read as the aggregate of its group."""

    loose = not any(join.args.get("using") or (join.method or "").upper() == "NATURAL" for join in select.args.get("joins") or [])
    shown = [_unwrapped(item.this if isinstance(item, exp.Alias) else item) for item, kind in zip(select.expressions, kinds) if kind == "key"]
    new = select.copy()
    for window in _own_windows(new):
        window.replace(window.this.copy())
    new.set("group", exp.Group(expressions=[key.copy() for key in keys]))
    if all(_is_key(key, shown, loose) for key in keys):
        new.set("distinct", None)  # one row per group, and the groups differ in what they show
    return new


def _ordered_by_outputs(select: exp.Select, keys: list[exp.Expression]) -> bool:
    order = select.args.get("order")
    if order is None:
        return True
    names = {item.alias_or_name.lower() for item in select.expressions if item.alias_or_name}
    loose = not any(join.args.get("using") or (join.method or "").upper() == "NATURAL" for join in select.args.get("joins") or [])
    for ordered in order.expressions:
        value = _unwrapped(ordered.this)
        if isinstance(value, exp.Literal) and not value.is_string:
            continue  # an output position
        if _is_key(value, keys, loose):
            continue
        if isinstance(value, exp.Column) and not value.table and value.name.lower() in names:
            continue
        return False
    return True


def distinct_window_groups(select: exp.Select) -> exp.Select | None:
    """The direct form (module doc): ``select`` as a grouped select, or None."""

    distinct = select.args.get("distinct")
    if distinct is None or distinct.args.get("on") is not None:
        return None
    read = _kinds(select)
    if read is None:
        return None
    keys, kinds = read
    if not _ordered_by_outputs(select, keys):
        return None
    return _grouped(select, keys, kinds)


def _definition(reader: exp.Select, root: exp.Expression) -> tuple[exp.Select, str] | None:
    """The select behind the one source of ``reader`` (a derived table, or a ``WITH`` query read once), and the name
    the reader knows it by."""

    source = reader.args[FROM_KEY].this
    if isinstance(source, exp.Subquery):
        alias = source.args.get("alias")
        if (alias is not None and alias.columns) or source.args.get("pivots") or source.args.get("sample"):
            return None
        inner = source.this
        return (inner, source.alias.lower()) if isinstance(inner, exp.Select) else None
    if not isinstance(source, exp.Table) or source.args.get("db") or source.args.get("catalog") or source.args.get("pivots") or source.args.get("sample"):
        return None
    alias = source.args.get("alias")
    if alias is not None and alias.columns:
        return None
    name = source.name.lower()
    if sum(1 for table in root.find_all(exp.Table) if table.name.lower() == name and not table.args.get("db")) != 1:
        return None
    node: exp.Expression | None = reader
    while node is not None:
        clause = node.args.get("with_") or node.args.get("with")
        if isinstance(clause, exp.With):
            for cte in clause.expressions:
                if cte.alias_or_name.lower() == name:
                    listed = cte.args.get("alias")
                    if clause.args.get("recursive") or (listed is not None and listed.columns) or not isinstance(cte.this, exp.Select):
                        return None
                    if any(ancestor is cte for ancestor in _ancestors(reader)):
                        return None
                    return cte.this, (source.alias or source.name).lower()
        node = node.parent
    return None


def _ancestors(node: exp.Expression):
    node = node.parent
    while node is not None:
        yield node
        node = node.parent


def windowed_source_groups(reader: exp.Select, root: exp.Expression) -> tuple[exp.Select, exp.Select] | None:
    """One level up (module doc): the select behind ``reader``'s source and what to put in its place, or None."""

    distinct = reader.args.get("distinct")
    if distinct is None or distinct.args.get("on") is not None:
        return None
    if any(reader.args.get(clause) for clause in _REFUSED_CLAUSES) or reader.args.get("joins") or reader.args.get(FROM_KEY) is None:
        return None
    found = _definition(reader, root)
    if found is None:
        return None
    inner, known_as = found
    if any(inner.args.get(clause) for clause in ("with_", "with")):
        return None
    read = _kinds(inner)
    if read is None:
        return None
    keys, kinds = read
    outputs: dict[str, str] = {}
    for item, kind in zip(inner.expressions, kinds):
        name = item.alias_or_name.lower() if (isinstance(item, exp.Alias) or isinstance(item, exp.Column)) else ""
        if not name or name in outputs:
            return None
        outputs[name] = kind
    whole = set()
    for item in reader.expressions:
        value = _unwrapped(item.this if isinstance(item, exp.Alias) else item)
        if isinstance(value, exp.Star):
            if value.args.get("except_") or value.args.get("except") or value.args.get("replace") or value.args.get("rename"):
                return None
            continue
        whole.add(id(value))
    order = reader.args.get("order")
    for ordered in (order.expressions if order is not None else []):
        whole.add(id(_unwrapped(ordered.this)))
    parts = list(reader.expressions) + [clause for clause in (reader.args.get("where"), order) if clause is not None]
    for part in parts:
        if not _same_each_time(part):
            return None
        for node in part.walk():
            if isinstance(node, (exp.Window, exp.AggFunc, exp.Query, exp.Exists, exp.Placeholder, exp.Parameter)):
                return None
            if not isinstance(node, exp.Column):
                continue
            if isinstance(node.this, exp.Star) or node.args.get("db") or (node.table and node.table.lower() != known_as):
                return None
            kind = outputs.get(node.name.lower())
            if kind is None or (kind == "key" and id(node) not in whole):
                return None
    new = _grouped(inner, keys, kinds)
    new.set("order", None)  # the order of a derived table's rows is not part of its value
    return inner, new


def read_distinct_window_groups(tree: exp.Expression) -> tuple[exp.Expression, bool]:
    """``tree`` with every such ``DISTINCT`` over partition aggregates read as a grouped select (module doc), and
    whether anything was written. The tree given is not changed."""

    if tree.find(exp.Window) is None or tree.find(exp.Distinct) is None:
        return tree, False
    new = tree.copy()
    wrote = False
    for _ in range(32):  # one select at a time: a rewritten select is a new node
        for select in new.find_all(exp.Select):
            read = distinct_window_groups(select)
            if read is not None:
                target = select
            else:
                above = windowed_source_groups(select, new)
                if above is None:
                    continue
                target, read = above
            wrote = True
            if target is new:
                new = read
            else:
                target.replace(read)
            break
        else:
            break
    return (new, True) if wrote else (tree, False)
