"""What a key says about the rows that hold their group's ``MIN`` or ``MAX``.

    SELECT a.player_id, a.device_id FROM activity AS a
    WHERE a.event_date = (SELECT MIN(f.event_date) FROM activity AS f WHERE f.player_id = a.player_id)

keeps, for each player, the rows of the first day. When ``(player_id, event_date)`` is a declared key there is
exactly one such row per player. Three spellings then say the same thing, and the prover did not connect them:

* **``DISTINCT`` that removes nothing.** ``SELECT DISTINCT a.player_id, a.device_id ..`` lists the partition
  columns, the kept rows differ on them, so no two output rows are equal and ``DISTINCT`` is dropped.
* **``GROUP BY`` of one row per group.** ``.. GROUP BY a.player_id`` with no aggregate anywhere: each group is one
  row, and the columns selected are that row's, so the grouping is dropped.
* **The key and the value alone.** ``SELECT a.player_id, a.event_date ..`` reads nothing but the partition
  columns and the compared column, which on the kept row is the group's minimum, so it is
  ``SELECT a.player_id, MIN(a.event_date) FROM activity AS a GROUP BY a.player_id``. This one also holds without
  the key when the select is ``DISTINCT`` (all the rows kept for a group agree on those columns).

**Why one row.** The comparison is TRUE only when the row's partition columns are not NULL (``f.k = a.k``) and
its value equals a ``MIN`` or ``MAX`` that is not NULL. Two kept rows with the same partition columns have the
same value, hence agree on every column of a key that lies within the partition columns and the value, hence
are the same row. A filter inside the subquery only changes which value is the extreme; other ``AND`` parts of
the outer ``WHERE`` only remove rows.

**The grouped form in detail.** The rows of ``GROUP BY k`` are the groups of the rows the subquery reads (its
own filter becomes the ``WHERE``). A group whose key is NULL, or whose values are all NULL, has a row there but
no kept row here, so ``k IS NOT NULL`` is added to the ``WHERE`` and ``MIN(v) IS NOT NULL`` as ``HAVING``; where
the columns are declared NOT NULL the normalizer folds those tests away. It is only written when the comparison
is the whole ``WHERE``: another filter would have to hold for the row that has the extreme, which a grouped
query cannot say.

**What is read.** A select over one base table with supplied columns and no join, whose ``WHERE`` has, as a
top-level ``AND`` part, ``a.v = (SELECT MIN(f.v) | MAX(f.v) FROM same_table AS f WHERE f.k = a.k [AND ..])``:
the subquery reads the same table under another name, compares each of its partition columns with the same
column of ``a`` (qualified), and may filter its own rows. No window, no nested query besides that one, no star.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key, _unwrap
from .ast_utils import FROM_KEY, conjuncts, extended_grouping, plain_distinct, select_sources

_SCALAR_ARGS = frozenset({"expressions", "from", "from_", "where", "comments", "kind", "hint"})


def read_extreme_rows(
    tree: exp.Expression, schema: dict[str, list[str]] | None, keys: dict[str, list[tuple[str, ...]]] | None
) -> exp.Expression:
    """Rewrite, in place, each select described in the module docstring."""

    if not schema or not any(True for _ in tree.find_all(exp.Subquery)):
        return tree
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    declared = {key.lower(): [tuple(c.lower() for c in k) for k in ks] for key, ks in (keys or {}).items()}
    for select in list(tree.find_all(exp.Select))[::-1]:
        _rewrite(select, known, declared)
    return tree


class _Extreme:
    def __init__(self, part, partition, value, extreme, filters, inner_alias):
        self.part, self.partition, self.value, self.extreme, self.filters, self.inner_alias = part, partition, value, extreme, filters, inner_alias


def _column(node: exp.Expression, owner: str, columns: list[str], bare: bool) -> str | None:
    """The column of the table known as ``owner`` that ``node`` names (a bare name counts when ``bare``)."""

    node = _unwrap(node)
    if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star) or node.args.get("db") or node.args.get("catalog"):
        return None
    if node.name.lower() not in columns:
        return None
    if node.table:
        return node.name.lower() if node.table.lower() == owner else None
    return node.name.lower() if bare else None


def _extreme(part: exp.Expression, table: exp.Table, own: str, columns: list[str]) -> _Extreme | None:
    part = _unwrap(part)
    if not isinstance(part, exp.EQ):
        return None
    for mine, theirs in ((part.this, part.expression), (part.expression, part.this)):
        value = _column(mine, own, columns, bare=True)
        theirs = _unwrap(theirs) if not isinstance(theirs, exp.Subquery) else theirs
        if value is None or not isinstance(theirs, exp.Subquery) or any(v for k, v in theirs.args.items() if k not in ("this", "comments")):
            continue
        scalar = theirs.this
        if not isinstance(scalar, exp.Select) or any(v for k, v in scalar.args.items() if k not in _SCALAR_ARGS):
            continue
        sources = select_sources(scalar)
        if len(sources) != 1 or len(scalar.expressions) != 1:
            continue
        again = sources[0]
        if _base_table(again, {_table_key(table): columns}) is None or _table_key(again) != _table_key(table):
            continue
        inner = again.alias_or_name.lower()
        if not inner or inner == own:
            continue
        call = _unwrap(scalar.expressions[0].this if isinstance(scalar.expressions[0], exp.Alias) else scalar.expressions[0])
        if type(call) not in (exp.Min, exp.Max) or any(v for k, v in call.args.items() if k != "this"):
            continue
        if _column(call.this, inner, columns, bare=True) != value:
            continue
        if any(isinstance(n, (exp.Query, exp.Subquery, exp.Window, exp.Exists, exp.Lambda, exp.Star)) for n in scalar.walk() if n is not scalar):
            continue
        partition: list[str] = []
        filters: list[exp.Expression] = []
        where = scalar.args.get("where")
        good = True
        for test in conjuncts(where.this) if where is not None else []:
            test = _unwrap(test)
            pair = None
            if isinstance(test, exp.EQ):
                for x, y in ((test.this, test.expression), (test.expression, test.this)):
                    k = _column(x, inner, columns, bare=True)
                    if k is not None and _column(y, own, columns, bare=False) == k:
                        pair = k
            if pair is not None:
                partition.append(pair)
                continue
            # a filter of the subquery's own rows: every column in it is the inner table's
            for column in test.find_all(exp.Column):
                if _column(column, inner, columns, bare=True) is None:
                    good = False
            filters.append(test)
        if not good:
            continue
        return _Extreme(part, list(dict.fromkeys(partition)), value, type(call), filters, inner)
    return None


def _rewrite(select: exp.Select, known: dict[str, list[str]], declared: dict[str, list[tuple[str, ...]]]) -> None:
    sources = select_sources(select)
    if len(sources) != 1 or select.args.get("joins"):
        return
    table = sources[0]
    columns = _base_table(table, known)
    if columns is None or not table.alias_or_name:
        return
    own = table.alias_or_name.lower()
    where = select.args.get("where")
    if where is None or any(select.args.get(k) for k in ("limit", "offset", "qualify", "having", "windows", "laterals")):
        return
    parts = conjuncts(where.this)
    found = [(part, _extreme(part, table, own, columns)) for part in parts]
    extremes = [e for _, e in found if e is not None]
    if len(extremes) != 1:
        return
    extreme = extremes[0]
    chosen = next(part for part, e in found if e is extreme)
    subquery = next(n for n in _unwrap(chosen).find_all(exp.Subquery))
    # everything else in the select reads only this table, in this scope
    for walked in select.walk():
        if walked is select or walked is subquery or _inside(walked, subquery):
            continue
        if isinstance(walked, (exp.Query, exp.Subquery, exp.Window, exp.Exists, exp.Lambda)):
            return
        if isinstance(walked, exp.Star) and not isinstance(walked.parent, exp.Count):
            return
        if isinstance(walked, exp.Column) and _column(walked, own, columns, bare=True) is None:
            return
    within = set(extreme.partition) | {extreme.value}
    keyed = any(k and set(k) <= within for k in declared.get(_table_key(table), []))
    aggregates = [n for n in select.find_all(exp.AggFunc) if not _inside(n, subquery)]
    group = select.args.get("group")
    if group is not None and keyed and not aggregates:
        # one row per group: the grouping is the partition columns
        if extended_grouping(group) or any(v for k, v in group.args.items() if k != "expressions"):
            return
        grouped = [_column(key, own, columns, bare=True) for key in group.expressions]
        if None in grouped or set(grouped) != set(extreme.partition):
            return
        select.set("group", None)
        group = None
    if group is not None or aggregates:
        return
    outputs = [_column(item.this if isinstance(item, exp.Alias) else item, own, columns, bare=True) for item in select.expressions]
    lists_partition = set(extreme.partition) <= {o for o in outputs if o is not None}
    if plain_distinct(select) and select.args.get("distinct") is not None and keyed and lists_partition:
        select.set("distinct", None)
    distinct = select.args.get("distinct")
    if distinct is not None and not plain_distinct(select):
        return
    # the key and the value alone, as the grouped query
    if not extreme.partition or len(parts) != 1 or not (keyed or distinct is not None):
        return
    read = {_column(c, own, columns, bare=True) for c in select.find_all(exp.Column) if not _inside(c, subquery) and not _inside(c, where)}
    if not read <= within:
        return
    own_name = table.alias_or_name

    def aggregate() -> exp.Expression:
        return extreme.extreme(this=exp.column(extreme.value, table=own_name))

    def lifted(node: exp.Expression) -> exp.Expression:
        node = node.copy()
        if _column(node, own, columns, bare=True) == extreme.value:
            return aggregate()
        for column in list(node.find_all(exp.Column)):
            if _column(column, own, columns, bare=True) == extreme.value:
                column.replace(aggregate())
        return node

    items = []
    for item in select.expressions:
        if isinstance(item, exp.Alias):
            items.append(exp.alias_(lifted(item.this), item.alias))
        elif _column(item, own, columns, bare=True) == extreme.value:
            items.append(exp.alias_(aggregate(), item.name))
        else:
            items.append(lifted(item))
    order = select.args.get("order")
    if order is not None:
        names = {item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias)}
        if any(isinstance(c, exp.Column) and not c.table and c.name.lower() in names for c in order.find_all(exp.Column)):
            return  # ORDER BY may mean the select alias or the column
        select.set("order", exp.Order(expressions=[exp.Ordered(this=lifted(o.this), desc=o.args.get("desc"), nulls_first=o.args.get("nulls_first")) for o in order.expressions]))

    def owned(piece: exp.Expression) -> exp.Expression:
        piece = piece.copy()
        for column in piece.find_all(exp.Column):
            column.set("table", exp.to_identifier(own_name))
        return piece

    tests = [owned(test) for test in extreme.filters]
    tests = [exp.Paren(this=test) if isinstance(test, exp.Or) else test for test in tests]
    tests += [exp.Not(this=exp.Is(this=exp.column(k, table=own_name), expression=exp.Null())) for k in extreme.partition]
    select.set("expressions", items)
    select.set("where", exp.Where(this=_all(tests)))
    select.set("group", exp.Group(expressions=[exp.column(k, table=own_name) for k in extreme.partition]))
    select.set("having", exp.Having(this=exp.Not(this=exp.Is(this=aggregate(), expression=exp.Null()))))
    if distinct is not None and lists_partition:
        select.set("distinct", None)


def _inside(node: exp.Expression, ancestor: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


def _all(tests: list[exp.Expression]) -> exp.Expression:
    condition = tests[0]
    for test in tests[1:]:
        condition = exp.And(this=condition, expression=test)
    return condition
