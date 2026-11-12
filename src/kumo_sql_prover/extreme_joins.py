"""A join with the table's own grouped ``MIN`` or ``MAX`` read as a comparison with the group's value.

    SELECT a.player_id, a.device_id
    FROM activity AS a
    JOIN (SELECT player_id, MIN(event_date) AS first_day FROM activity GROUP BY player_id) AS g
      ON a.player_id = g.player_id AND a.event_date = g.first_day

keeps each player's first-day rows, as does

    SELECT a.player_id, a.device_id FROM activity AS a
    WHERE a.event_date = (SELECT MIN(f.event_date) FROM activity AS f WHERE f.player_id = a.player_id)

The normalizer already brings the inner join and the subquery to one form. This rule writes the join as the
subquery *before* that, for two reasons: the rules that read what a key says about such rows
(``extreme_rows``) look for the subquery spelling only, and an outer join from the grouped side is not read at
all otherwise.

**Why the two are the same.** The grouped table ``g`` has one row per group, and the join names every group
key, so a row of ``a`` whose keys are not NULL meets at most one row of ``g``: the row of its own group, when
the group exists. On that row ``g.m`` is the group's aggregate, which is what the scalar subquery computes; with
no such group the join drops the row, and the scalar subquery aggregates nothing, is NULL, and the comparison
drops it too (``MIN`` and ``MAX`` are NULL over no rows). A NULL key meets no group and equals nothing. So the
join is a filter on ``a``, and the rows keep their multiplicity.

Every other read of ``g`` is replaced: ``g.k`` by ``a.k`` (equal on every kept row, and the same column of the
same table, so the same type) and ``g.m`` by ``a.v`` (equal by the comparison itself).

**What is read.**

* ``g``: a derived table over one base table with supplied columns, an optional ``WHERE``, ``GROUP BY`` of plain
  columns, every key output, and exactly one more output, ``MIN(v)`` or ``MAX(v)`` of a plain column, named.
  ``DISTINCT`` (its rows are already distinct) and ``ORDER BY`` without a row limit are ignored.
* ``a``: a source of the same select that is the same base table.
* The conditions ``a.k = g.k`` for every key and ``a.v = g.m``, each a top-level ``AND`` part of the ``WHERE`` or
  of an ``ON``, in a select whose joins are all inner or cross joins (then ``ON`` and ``WHERE`` parts are
  interchangeable), with no ``USING``, star or nested query that reads ``g``.
* **An outer join that cannot fail.** ``g LEFT JOIN a ON a.k = g.k AND a.v = g.m`` (or ``a RIGHT JOIN g``) with
  exactly those conditions, where the keys and ``v`` are declared NOT NULL: every row of ``g`` then has a match,
  namely a row of the group that holds its minimum (the group is not empty, its ``v`` values are not NULL, and
  ``a`` reads every row of the table), so the outer join is the inner join. ``a LEFT JOIN g`` with the
  comparison ``a.v = g.m`` in the ``WHERE`` is the inner join too: the comparison is not TRUE on a row ``g`` did
  not match.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _membership_blind, _named_column, _names, _table_key, _unwrap
from .ast_utils import FROM_KEY, conjuncts, extended_grouping, select_sources

_PREFIX = "kqj"
_OUTER = ("LEFT", "RIGHT")


def read_extreme_joins(tree: exp.Expression, schema: dict[str, list[str]] | None, not_null: dict[str, frozenset[str]] | None) -> exp.Expression:
    """Rewrite, in place, each join described in the module docstring."""

    if not schema or not any(True for _ in tree.find_all(exp.Group)):
        return tree
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    declared = {key.lower(): {c.lower() for c in columns} for key, columns in (not_null or {}).items()}
    taken = _names(tree)
    for select in list(tree.find_all(exp.Select))[::-1]:
        for _ in range(4):  # one grouped table at a time
            if not _rewrite(select, known, declared, taken):
                break
    return tree


def _fresh(taken: set[str]) -> str:
    number = 0
    while f"{_PREFIX}{number}" in taken:
        number += 1
    taken.add(f"{_PREFIX}{number}")
    return f"{_PREFIX}{number}"


class _Grouped:
    """A derived table ``(SELECT k.., MIN(v) AS m FROM t [WHERE ..] GROUP BY k..) AS g``."""

    def __init__(self, source: exp.Subquery, table: exp.Table, keys: dict[str, str], name: str, value: str, extreme: type):
        self.source, self.table, self.keys, self.name, self.value, self.extreme = source, table, keys, name, value, extreme
        self.alias = source.alias.lower()


def _grouped(source: exp.Expression, known: dict[str, list[str]]) -> _Grouped | None:
    if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select) or not source.alias:
        return None
    alias = source.args.get("alias")
    if alias.args.get("columns") or any(value for key, value in source.args.items() if key not in ("this", "alias", "comments")):
        return None
    inner = source.this
    if not _membership_blind(inner) or inner.args.get("group") is None:
        return None
    sources = select_sources(inner)
    if len(sources) != 1:
        return None
    table = sources[0]
    columns = _base_table(table, known)
    if columns is None or not table.alias_or_name:
        return None
    own = table.alias_or_name.lower()
    order = inner.args.get("order")
    for walked in inner.walk():
        if order is not None and (walked is order or _inside(walked, order)):
            continue  # ORDER BY without a limit: not read
        if walked is not inner and isinstance(walked, (exp.Query, exp.Subquery, exp.Window, exp.Exists, exp.Lambda)):
            return None
        if isinstance(walked, exp.Column):
            if isinstance(walked.this, exp.Star) or walked.args.get("db") or walked.args.get("catalog"):
                return None
            if walked.name.lower() not in columns or (walked.table and walked.table.lower() != own):
                return None
        elif isinstance(walked, exp.Star) and not isinstance(walked.parent, exp.Count):
            return None
    group = inner.args["group"]
    if extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions"):
        return None
    grouped = [_named_column(key) for key in group.expressions]
    if not grouped or None in grouped or len(set(grouped)) != len(grouped):
        return None
    keys: dict[str, str] = {}  # output name -> key column
    name = value = extreme = None
    for item in inner.expressions:
        output = _unwrap(item.this if isinstance(item, exp.Alias) else item)
        label = item.alias_or_name.lower()
        if not label or label in keys or label == name:
            return None
        if isinstance(item, exp.Alias) and label in grouped and _named_column(output) != label:
            return None  # an alias that takes a grouped column's name
        if isinstance(output, exp.Column) and output.name.lower() in grouped:
            keys[label] = output.name.lower()
        elif type(output) in (exp.Min, exp.Max) and not any(v for k, v in output.args.items() if k != "this") and name is None:
            column = _named_column(output.this)
            if column is None or not isinstance(item, exp.Alias):
                return None
            name, value, extreme = label, column, type(output)
        else:
            return None
    if name is None or set(keys.values()) != set(grouped) or len(keys) != len(grouped):
        return None
    return _Grouped(source, table, keys, name, value, extreme)


def _inside(node: exp.Expression, ancestor: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


def _rewrite(select: exp.Select, known: dict[str, list[str]], declared: dict[str, set[str]], taken: set[str]) -> bool:
    sources = select_sources(select)
    if len(sources) < 2:
        return False
    names = [s.alias_or_name.lower() for s in sources]
    if "" in names or len(set(names)) != len(names):
        return False
    for source in sources:
        grouped = _grouped(source, known)
        if grouped is None:
            continue
        for partner in sources:
            if partner is source or _base_table(partner, known) is None or _table_key(partner) != _table_key(grouped.table):
                continue
            if _apply(select, sources, grouped, partner, known, declared, taken):
                return True
    return False


def _apply(select: exp.Select, sources: list, grouped: _Grouped, partner: exp.Table, known, declared, taken: set[str]) -> bool:
    joins = select.args.get("joins") or []
    table = _table_key(partner)
    columns = known[table]
    a, g = partner.alias_or_name.lower(), grouped.alias
    outputs = set(grouped.keys) | {grouped.name}
    # which columns each other source could answer for, when a bare name is read
    other_columns: set[str] = set()
    for source in sources:
        if source is grouped.source or source is partner:
            continue
        cols = _base_table(source, known)
        if cols is None:
            other_columns = None
            break
        other_columns |= set(cols)

    def side(node: exp.Expression) -> tuple[str, str] | None:
        """``("g", output)`` or ``("a", column)`` for a plain column of the grouped table or its partner."""

        node = _unwrap(node)
        if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star) or node.args.get("db"):
            return None
        name = node.name.lower()
        if node.table:
            if node.table.lower() == g and name in outputs:
                return "g", name
            if node.table.lower() == a and name in columns:
                return "a", name
            return None
        if other_columns is None or name in other_columns:
            return None
        if name in outputs and name not in columns:
            return "g", name
        if name in columns and name not in outputs:
            return "a", name
        return None

    outer = [j for j in joins if j.side or (j.kind and j.kind.upper() not in ("INNER", "CROSS"))]
    if any(j.args.get("using") or j.method or (j.kind and j.kind.upper() not in ("INNER", "CROSS", "OUTER")) for j in joins):
        return False
    where = select.args.get("where")
    where_parts = conjuncts(where.this) if where is not None else []
    on_parts = [part for j in joins if j.args.get("on") is not None for part in conjuncts(j.args["on"])]

    def classify(part: exp.Expression):
        part = _unwrap(part)
        if not isinstance(part, exp.EQ):
            return None
        left, right = side(part.this), side(part.expression)
        if left is None or right is None or {left[0], right[0]} != {"a", "g"}:
            return None
        ours, theirs = (left, right) if left[0] == "g" else (right, left)
        if ours[1] in grouped.keys and theirs[1] == grouped.keys[ours[1]]:
            return "key", ours[1]
        if ours[1] == grouped.name and theirs[1] == grouped.value:
            return "value", ours[1]
        return None

    if outer:
        # exactly two sources, one outer join between them
        if len(sources) != 2 or len(joins) != 1 or (joins[0].side or "").upper() not in _OUTER:
            return False
        join = joins[0]
        first = select.args[FROM_KEY].this
        preserved = first if join.side.upper() == "LEFT" else join.this
        on_kinds = [classify(part) for part in on_parts]
        if preserved is grouped.source:
            # every row of g has a match: only the keys and the value are compared, and none of them is NULL
            facts = declared.get(_table_key(grouped.table), set())
            if None in on_kinds or not all(k in facts for k in grouped.keys.values()) or grouped.value not in facts:
                return False
            if ("value", grouped.name) not in on_kinds:
                return False
        else:
            # a is preserved: the comparison with g.m must be in the WHERE, where an unmatched row fails it
            if not any(classify(part) == ("value", grouped.name) for part in where_parts):
                return False
    pool = on_parts + where_parts
    kinds = [classify(part) for part in pool]
    found_keys = {kind[1] for kind in kinds if kind is not None and kind[0] == "key"}
    if found_keys != set(grouped.keys) or ("value", grouped.name) not in kinds:
        return False
    # every other read of g must be a plain column read in this select's own scope; no star
    if any(isinstance(n, exp.Star) and not isinstance(n.parent, exp.Count) for e in select.expressions for n in e.walk()):
        return False
    if any(isinstance(e, exp.Column) and isinstance(e.this, exp.Star) for e in select.expressions):
        return False
    reads = []
    matched = [part for part, kind in zip(pool, kinds) if kind is not None]
    for column in select.find_all(exp.Column):
        if any(_inside(column, s) for s in sources if isinstance(s, exp.Subquery)):
            continue
        scope = column.find_ancestor(exp.Select)
        own_scope = scope is select
        which = side(column)
        if not own_scope:
            # a nested query: it must not read g at all
            if (column.table and column.table.lower() == g) or (not column.table and column.name.lower() in outputs):
                return False
            continue
        if which is not None and which[0] == "g":
            if not any(column is _unwrap(part.this) or column is _unwrap(part.expression) for part in matched):
                reads.append((column, which[1]))
        elif which is None and ((column.table and column.table.lower() == g) or (not column.table and column.name.lower() in outputs)):
            return False  # a read of g that cannot be placed
    # --- rewrite ---
    fresh = _fresh(taken)
    a_name = partner.alias_or_name
    inner = grouped.source.this

    def moved(piece: exp.Expression) -> exp.Expression:
        piece = piece.copy()
        for column in piece.find_all(exp.Column):
            column.set("table", exp.to_identifier(fresh))
        return piece

    inner_where = inner.args.get("where")
    tests = [moved(part) for part in conjuncts(inner_where.this)] if inner_where is not None else []
    tests = [exp.Paren(this=test) if isinstance(test, exp.Or) else test for test in tests]
    tests += [exp.EQ(this=exp.column(key, table=fresh), expression=exp.column(key, table=a_name)) for key in dict.fromkeys(grouped.keys.values())]
    again = grouped.table.copy()
    again.set("alias", exp.TableAlias(this=exp.to_identifier(fresh)))
    scalar = exp.Select(expressions=[grouped.extreme(this=exp.column(grouped.value, table=fresh))])
    scalar.set(FROM_KEY, exp.From(this=again))
    scalar.set("where", exp.Where(this=_all(tests)))
    comparison = exp.EQ(this=exp.column(grouped.value, table=a_name), expression=exp.Subquery(this=scalar))
    for column, output in reads:
        target = grouped.keys.get(output, grouped.value)
        replacement = exp.column(target, table=a_name)
        parent = column.parent
        if isinstance(parent, exp.Select) and column in parent.expressions and target != column.name.lower():
            replacement = exp.alias_(replacement, column.name)
        column.replace(replacement)
    kept = []
    done = False
    for part, kind in zip(pool, kinds):
        if kind is None:
            kept.append(part.copy())
        elif kind[0] == "value" and not done:
            kept.append(comparison)
            done = True
    kept = [exp.Paren(this=part) if isinstance(part, exp.Or) else part for part in kept]
    remaining = [s.copy() for s in sources if s is not grouped.source]
    select.set(FROM_KEY, exp.From(this=remaining[0]))
    select.set("joins", [exp.Join(this=s) for s in remaining[1:]] or None)
    select.set("where", exp.Where(this=_all(kept)))
    return True


def _all(tests: list[exp.Expression]) -> exp.Expression:
    condition = tests[0]
    for test in tests[1:]:
        condition = exp.And(this=condition, expression=test)
    return condition
