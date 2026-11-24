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
* One level up: the reading select shows partition columns and reads window columns, and those are the same for
  every row of a partition. It may test them with a condition on the window columns and takes ``DISTINCT`` of what
  it shows, so one row per partition gives the same answer. What the derived table shows besides (a ``*``, a
  column that differs from row to row) is left out, which the reader cannot notice: it reads none of it.
* A row's number. ``ROW_NUMBER() OVER (PARTITION BY k ..)`` numbers the n rows of a partition 1 to n, whatever its
  ``ORDER BY``. The reader keeps a partition when some row of it passes its ``WHERE``. For one test ``rn > c``,
  ``rn >= c`` or ``rn = c`` on the number (``c`` a whole number, at least 1, or 0 for ``>``), joined to the rest by
  ``AND`` and ``OR`` only, some number in 1 .. n passes exactly when ``n > c``, ``n >= c``, ``n >= c``: the rest of
  the condition is the same for every row, and "some row" passes through ``AND`` with such a term and through
  ``OR``. So the number is read as ``COUNT(*)`` and ``rn = c`` as ``rn >= c``. Under ``NOT`` or with a second test
  on the number that does not hold, and nothing is rewritten.

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
  elsewhere (a row number in its one test, never as an output). A partition column may come through the derived
  table's ``*`` only when that table has a single source. A ``WITH`` query is rewritten only when it is read once
  in the whole statement and has no column list.

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
_BOUNDS = (exp.GT, exp.GTE, exp.EQ, exp.LT, exp.LTE)
_MIRRORED = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.EQ: exp.EQ, exp.LT: exp.GT, exp.LTE: exp.GTE}


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


def _loose(select: exp.Select) -> bool:
    """Whether a bare and a qualified spelling of a column name mean one column in ``select`` (module doc)."""

    return not any(join.args.get("using") or (join.method or "").upper() == "NATURAL" for join in select.args.get("joins") or [])


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


def _own_windows(select: exp.Select, under: exp.Expression | None = None) -> list[exp.Window]:
    return [window for window in (select if under is None else under).find_all(exp.Window) if window.find_ancestor(exp.Select) is select]


def _steady(node: exp.Expression) -> bool:
    """Whether ``node`` is an expression of the row alone that gives the same value each time it is computed."""

    return node.find(exp.Window, exp.AggFunc, *_NOT_A_VALUE) is None and _same_each_time(node)


def _window(window: exp.Window) -> tuple[str, list[exp.Expression]] | None:
    """``("aggregate", keys)`` for an aggregate of the row's whole partition, ``("number", keys)`` for the row's
    number within its partition, None for a window the rule does not read. ``keys`` are the partition expressions."""

    if window.args.get("spec") or window.args.get("alias") or window.args.get("first"):
        return None
    if str(window.args.get("over") or "OVER").upper() != "OVER":
        return None
    keys = [_unwrapped(key) for key in window.args.get("partition_by") or []]
    if not keys or any(key.find(exp.Column) is None or not _steady(key) for key in keys):
        return None
    call, order = window.this, window.args.get("order")
    if isinstance(call, exp.RowNumber):
        # the rows of a partition of n rows are numbered 1 to n, in whatever order
        if order is not None and not all(_steady(ordered.this) for ordered in order.expressions):
            return None
        return "number", keys
    if order is not None or not isinstance(call, _AGGREGATES) or isinstance(call.this, exp.Distinct):
        return None
    if any(part is not call and isinstance(part, (exp.Window, exp.AggFunc, exp.Query, exp.Exists, exp.Placeholder, exp.Parameter)) for part in call.walk()):
        return None
    if not _same_each_time(call):
        return None
    return "aggregate", keys


def _reads_no_row(node: exp.Expression) -> bool:
    """Whether ``node`` reads the row it is computed for only through windows."""

    if isinstance(node, exp.Window):
        return True
    if isinstance(node, (exp.Column, exp.AggFunc, *_NOT_A_VALUE)):
        return False
    return all(_reads_no_row(child) for child in node.iter_expressions())


def _value(item: exp.Expression) -> exp.Expression:
    return _unwrapped(item.this if isinstance(item, exp.Alias) else item)


def _is_star(value: exp.Expression) -> bool:
    return isinstance(value, exp.Star) or isinstance(value, exp.Column) and isinstance(value.this, exp.Star)


def _shape(select: exp.Select, item: exp.Expression) -> tuple[str, list[exp.Expression] | None]:
    """What an output of ``select`` is made of: ``star``; ``window`` (aggregates of the partition and constants) or
    ``number`` (the row's number in its partition), each with its partition expressions; ``constant``; ``column``
    (an expression of the row, which may be a partition expression); ``row`` (anything else)."""

    value = _value(item)
    if _is_star(value):
        return "star", None
    windows = _own_windows(select, value)
    if not windows:
        if _reads_no_row(value):
            return ("constant", None) if _same_each_time(value) else ("row", None)
        return ("column", None) if _steady(value) else ("row", None)
    read = [_window(window) for window in windows]
    if any(found is None for found in read):
        return "row", None
    keys = read[0][1]
    if any({_text(key) for key in found[1]} != {_text(key) for key in keys} for found in read):
        return "row", None
    if isinstance(value, exp.Window) and read[0][0] == "number":
        return "number", keys
    if any(found[0] != "aggregate" for found in read) or not _reads_no_row(value) or not _same_each_time(value):
        return "row", None
    return "window", keys


def _plain_select(select: exp.Select) -> bool:
    if any(select.args.get(clause) for clause in _REFUSED_CLAUSES) or select.args.get(FROM_KEY) is None:
        return False
    windows = _own_windows(select)
    return bool(windows) and sum(len(_own_windows(select, item)) for item in select.expressions) == len(windows)


def _common_keys(select: exp.Select, shapes: list[tuple[str, list[exp.Expression] | None]]) -> list[exp.Expression] | None:
    """The partition expressions the given outputs share, each once; None when they differ or cannot be grouped by."""

    keys: list[exp.Expression] | None = None
    for kind, found in shapes:
        if kind not in ("window", "number"):
            continue
        if keys is None:
            keys = found
        elif {_text(key) for key in keys} != {_text(key) for key in found or []}:
            return None
    if keys is None:
        return None
    aliases = {
        item.alias.lower() for item in select.expressions
        if isinstance(item, exp.Alias) and not (isinstance(item.this, exp.Column) and item.this.name.lower() == item.alias.lower())
    }
    if any(not column.table and column.name.lower() in aliases for key in keys for column in key.find_all(exp.Column)):
        return None
    unique: list[exp.Expression] = []
    for key in keys:
        if _text(key) not in {_text(other) for other in unique}:
            unique.append(key)
    return unique


def _ungrouped_window(window: exp.Window) -> exp.Expression:
    return exp.Count(this=exp.Star()) if isinstance(window.this, exp.RowNumber) else window.this.copy()


def _ordered_by_outputs(select: exp.Select, keys: list[exp.Expression]) -> bool:
    order = select.args.get("order")
    if order is None:
        return True
    names = {item.alias_or_name.lower() for item in select.expressions if item.alias_or_name}
    loose = _loose(select)
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
    if distinct is None or distinct.args.get("on") is not None or not _plain_select(select):
        return None
    shapes = [_shape(select, item) for item in select.expressions]
    if any(kind not in ("window", "constant", "column") for kind, _ in shapes):
        return None
    keys = _common_keys(select, shapes)
    if keys is None:
        return None
    loose = _loose(select)
    shown = [_value(item) for item, (kind, _) in zip(select.expressions, shapes) if kind == "column"]
    if not all(_is_key(value, keys, loose) for value in shown) or not _ordered_by_outputs(select, keys):
        return None
    new = select.copy()
    for window in _own_windows(new):
        window.replace(_ungrouped_window(window))
    new.set("group", exp.Group(expressions=[key.copy() for key in keys]))
    if all(_is_key(key, shown, loose) for key in keys):
        new.set("distinct", None)  # one row per group, and the groups differ in what they show
    return new


def _ancestors(node: exp.Expression):
    node = node.parent
    while node is not None:
        yield node
        node = node.parent


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


def _whole_number(node: exp.Expression) -> int | None:
    node = _unwrapped(node)
    return int(node.this) if isinstance(node, exp.Literal) and not node.is_string and node.this.isdigit() else None


def _count_test(test: exp.Expression, column: exp.Column) -> exp.Expression | None:
    """``test`` on a row number of the partition, as the test on the partition's row count that says some row of
    the partition passes it: ``n > c``, ``n >= c`` and ``n = c`` for a whole number ``c`` of at least 1 (and at
    least 0 for ``>``) hold for some ``n`` in 1 .. count exactly when ``count > c``, ``count >= c``, ``count >= c``."""

    if type(test) not in _BOUNDS:
        return None
    left, right = _unwrapped(test.this), _unwrapped(test.expression)
    kind = type(test)
    if right is column:
        left, right, kind = right, left, _MIRRORED[kind]
    if left is not column:
        return None
    bound = _whole_number(right)
    if bound is None or kind in (exp.LT, exp.LTE) or (bound < 1 and kind is not exp.GT):
        return None
    return (exp.GT if kind is exp.GT else exp.GTE)(this=column.copy(), expression=exp.Literal.number(bound))


def windowed_source_groups(reader: exp.Select, root: exp.Expression) -> tuple[exp.Select, exp.Select] | None:
    """One level up (module doc): the select behind ``reader``'s source and what to put in its place, or None.

    A test of the reader on a row number is rewritten in place (in ``reader``, which must be the caller's own copy)
    when something is returned, and only then."""

    distinct = reader.args.get("distinct")
    if distinct is None or distinct.args.get("on") is not None:
        return None
    if any(reader.args.get(clause) for clause in _REFUSED_CLAUSES) or reader.args.get("joins") or reader.args.get(FROM_KEY) is None:
        return None
    found = _definition(reader, root)
    if found is None:
        return None
    inner, known_as = found
    if any(inner.args.get(clause) for clause in ("with_", "with")) or not _plain_select(inner):
        return None
    if inner.args.get("distinct") is not None and inner.args["distinct"].args.get("on") is not None:
        return None
    shapes = [_shape(inner, item) for item in inner.expressions]
    outputs: dict[str, int] = {}
    for position, (item, (kind, _)) in enumerate(zip(inner.expressions, shapes)):
        if kind == "star":
            if not isinstance(_value(item), exp.Star) or item.args.get("except_") or item.args.get("except") or item.args.get("replace") or item.args.get("rename"):
                return None
            continue
        name = item.alias_or_name.lower() if isinstance(item, (exp.Alias, exp.Column)) else ""
        if name in outputs:
            return None
        if name:
            outputs[name] = position
    starred = any(kind == "star" for kind, _ in shapes)
    # what the reader reads, and where
    mine = {item.alias.lower() for item in reader.expressions if isinstance(item, exp.Alias)}
    whole, everything = set(), False
    for item in reader.expressions:
        value = _value(item)
        if _is_star(value):
            if not isinstance(value, exp.Star) or value.args.get("except_") or value.args.get("except") or value.args.get("replace") or value.args.get("rename"):
                return None
            everything = True
            continue
        whole.add(id(value))
    where, order = reader.args.get("where"), reader.args.get("order")
    reads: list[tuple[exp.Column, str]] = []
    for part, place in [(item, "item") for item in reader.expressions] + [(where, "where")] + [(ordered, "order") for ordered in (order.expressions if order is not None else [])]:
        if part is None:
            continue
        if not _same_each_time(part):
            return None
        for node in part.walk():
            if isinstance(node, (exp.Window, exp.AggFunc, exp.Query, exp.Exists, exp.Placeholder, exp.Parameter)):
                return None
            if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star):
                continue
            if node.args.get("db") or (node.table and node.table.lower() != known_as):
                return None
            if place == "order":
                if node is not _unwrapped(part.this):
                    return None
                if not node.table and node.name.lower() in mine:
                    continue  # the reader's own output
            reads.append((node, place))
    read_positions = {outputs[node.name.lower()] for node, _ in reads if node.name.lower() in outputs}
    if everything:
        if starred:
            return None
        read_positions = set(range(len(shapes)))
    if any(shapes[position][0] == "row" for position in read_positions):
        return None
    keys = _common_keys(inner, [shapes[position] for position in sorted(read_positions)])
    if keys is None:
        return None
    loose = _loose(inner)
    if any(shapes[position][0] == "column" and not _is_key(_value(inner.expressions[position]), keys, loose) for position in read_positions):
        return None
    numbers = {position for position in read_positions if shapes[position][0] == "number"}
    through_star: list[exp.Expression] = []
    number_reads = []
    for node, place in reads:
        name = node.name.lower()
        if name not in outputs:
            # a column the star shows: read only when it is a partition column, and as a whole output
            key = next((key for key in keys if _plain_column(key) and key.name.lower() == name), None)
            # (of the one source: two joined sources could both have a column of that name)
            if not starred or key is None or inner.args.get("joins") or (place == "item" and id(node) not in whole) or place == "where":
                return None
            if _text(key) not in {_text(other) for other in through_star}:
                through_star.append(key)
            continue
        kind = shapes[outputs[name]][0]
        if kind == "column" and place == "item" and id(node) not in whole:
            return None
        if kind == "column" and place == "where":
            return None
        if kind == "number":
            number_reads.append((node, place))
    test, counted = None, None
    if numbers:
        if everything or len(number_reads) != 1 or number_reads[0][1] != "where":
            return None
        column = number_reads[0][0]
        test = column.parent
        while isinstance(test, exp.Paren):
            test = test.parent
        above = test.parent
        while above is not None and not isinstance(above, exp.Where):
            if not isinstance(above, (exp.And, exp.Or, exp.Paren)):
                return None  # under a NOT or a CASE "some row passes" is not what is asked
            above = above.parent
        counted = _count_test(test, column)
        if counted is None:
            return None
    new = inner.copy()
    items = []
    for position, (item, (kind, _)) in enumerate(zip(new.expressions, shapes)):
        if kind == "star":
            items.extend(exp.column(key.name, quoted=bool(key.this.args.get("quoted"))) if not key.table else key.copy() for key in through_star)
        elif position in read_positions:
            for window in _own_windows(new, item):
                if window is item:
                    return None  # a window without a name cannot be read
                window.replace(_ungrouped_window(window))
            items.append(item)
    if not items:
        return None
    new.set("expressions", items)
    new.set("distinct", None)  # one row for each partition: the reader takes DISTINCT of what it shows
    new.set("order", None)  # the order of a derived table's rows is not part of its value
    new.set("group", exp.Group(expressions=[key.copy() for key in keys]))
    if test is not None:
        test.replace(counted)  # in the reader, which the caller owns
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
                swaps = [(select, read)]
            else:
                above = windowed_source_groups(select, new)
                if above is None:
                    continue
                swaps = [above]
            wrote = True
            for old, fresh in swaps:
                if old is new:
                    new = fresh
                else:
                    old.replace(fresh)
            break
        else:
            break
    return (new, True) if wrote else (tree, False)
