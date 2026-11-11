"""``RANK() OVER (PARTITION BY k ORDER BY v) = 1`` read as "the rows that hold their partition's smallest ``v``".

    SELECT player_id, device_id
    FROM (SELECT player_id, device_id, RANK() OVER (PARTITION BY player_id ORDER BY event_date) AS rk FROM activity) AS d
    WHERE rk = 1

keeps each player's first-day rows. The prover keeps a window whole and compares it by its text, so this never
met the same rows written with an aggregate. The rule writes the filter into the derived table as a comparison
the prover already reads, and drops the window:

    SELECT player_id, device_id
    FROM (SELECT player_id, device_id FROM activity
          WHERE activity.event_date = (SELECT MIN(f.event_date) FROM activity AS f WHERE f.player_id = activity.player_id)) AS d

**The windows read**, each as the only window of a derived table over one base table (``WHERE`` allowed), with
the test a top-level ``AND`` part of the ``WHERE`` of the select that reads the derived table:

* ``RANK()`` or ``DENSE_RANK()`` ``OVER (PARTITION BY k.. ORDER BY v [ASC|DESC])`` tested ``= 1`` (also
  ``<= 1`` and ``< 2``: a rank is a whole number from 1 up). Rank 1 is held by the rows that sort first, all of
  them: the rows whose ``v`` is the partition's ``MIN`` (``MAX`` for ``DESC``).
* ``ROW_NUMBER()`` the same way, but only when a declared key of the table lies within the partition columns
  and ``v``: then no two rows of a partition share ``v``, row 1 is the only row that sorts first, and the
  number is ``RANK()``. Without such a key the engine picks one of the tied rows as it likes and nothing is read.
* ``MIN(v) OVER (PARTITION BY k..)`` tested ``v = that`` (``MAX`` alike), also with ``ORDER BY v`` for ``MIN``
  and ``ORDER BY v DESC`` for ``MAX`` under the default frame: the running minimum up to a row and its peers is
  the partition's minimum for every row whose ``v`` is not NULL, and a row whose ``v`` is NULL fails the test
  either way.
* ``FIRST_VALUE(v) OVER (PARTITION BY k.. ORDER BY v [DESC])`` tested ``v = that``.

**Conditions.**

* ``v`` is a column declared NOT NULL for the ranking functions and ``FIRST_VALUE``: where NULLs sort differs by
  engine and by direction, and a NULL that sorts first would hold rank 1 while ``v = MIN(v)`` is not TRUE for it.
  ``MIN`` and ``MAX`` skip NULLs themselves, so they need nothing of ``v``.
* Every partition column is declared NOT NULL: a window puts the rows whose ``k`` is NULL in one partition, but
  ``f.k = t.k`` is never TRUE for them.
* The window's value is read nowhere but in the test (not selected, not ordered by, not in another test), the
  reading select has no star, and the derived table is not on the NULL-extended side of an outer join (every
  join of the reading select is an inner or cross join), since the filter moves below the join.
* The derived table has no other window (it would see fewer rows after the filter moves in), no grouping,
  ``DISTINCT``, ``LIMIT`` or nested query, and one base-table source whose columns the caller supplied.

The subquery added reads the same table under a fresh alias with the derived table's own ``WHERE``, because a
window is computed over the rows that pass it.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _names, _unwrap
from .ast_utils import FROM_KEY, conjuncts, select_sources

_INNER_ARGS = frozenset({"expressions", "from", "from_", "where", "comments", "kind", "hint"})
_RANKS = tuple(getattr(exp, name) for name in ("Rank", "DenseRank", "RowNumber") if hasattr(exp, name))
_PREFIX = "kqr"


def read_rank_filters(
    tree: exp.Expression,
    schema: dict[str, list[str]] | None,
    not_null: dict[str, frozenset[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
) -> exp.Expression:
    """Rewrite, in place, each window test described in the module docstring."""

    if not schema or not any(True for _ in tree.find_all(exp.Window)):
        return tree
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    facts = _Facts(
        {key.lower(): {c.lower() for c in columns} for key, columns in (not_null or {}).items()},
        {key.lower(): [tuple(c.lower() for c in k) for k in ks] for key, ks in (keys or {}).items()},
    )
    taken = _names(tree)
    for select in list(tree.find_all(exp.Select))[::-1]:
        for source in select_sources(select):
            if isinstance(source, exp.Subquery) and isinstance(source.this, exp.Select):
                _rewrite(select, source, known, facts, taken)
    return tree


class _Facts:
    def __init__(self, not_null: dict[str, set[str]], keys: dict[str, list[tuple[str, ...]]]):
        self.not_null, self.keys = not_null, keys


def _table_key(table: exp.Table) -> str:
    return ".".join(p.name for p in (table.args.get("catalog"), table.args.get("db"), table.this) if p is not None).lower()


def _own_column(node: exp.Expression, own: str, columns: list[str]) -> str | None:
    """The table column a plain column reference names, else None."""

    node = _unwrap(node)
    if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star) or node.args.get("db") or node.args.get("catalog"):
        return None
    if node.table and node.table.lower() != own:
        return None
    return node.name.lower() if node.name.lower() in columns else None


def _rewrite(select: exp.Select, source: exp.Subquery, known: dict[str, list[str]], facts: _Facts, taken: set[str]) -> None:
    inner = source.this
    alias = source.alias.lower() if source.alias else ""
    source_alias = source.args.get("alias")
    if not alias or (source_alias is not None and source_alias.args.get("columns")):
        return
    if any(value for key, value in source.args.items() if key not in ("this", "alias", "comments")):
        return
    if any(value for key, value in inner.args.items() if key not in _INNER_ARGS):
        return
    sources = select_sources(inner)
    if len(sources) != 1:
        return
    table = sources[0]
    columns = _base_table(table, known)
    if columns is None or not table.alias_or_name:
        return
    own = table.alias_or_name.lower()
    windows = [w for w in inner.find_all(exp.Window)]
    if len(windows) != 1 or len(inner.expressions) < 2:
        return
    window = windows[0]
    item = window.parent
    if not isinstance(item, exp.Alias) or item.parent is not inner or item.args.get("this") is not window:
        return
    name = item.alias.lower()
    if not name or name in columns:
        return
    for walked in inner.walk():
        if walked is not inner and isinstance(walked, (exp.Query, exp.Subquery, exp.Exists, exp.Lambda, exp.AggFunc)) and not _inside(walked, window):
            return
        if isinstance(walked, exp.Column) and not isinstance(walked.this, exp.Star):
            if walked.args.get("db") or walked.args.get("catalog") or walked.name.lower() not in columns:
                return
            if walked.table and walked.table.lower() != own:
                return
        if isinstance(walked, exp.Column) and isinstance(walked.this, exp.Star) and walked.table and walked.table.lower() != own:
            return
    outputs = _outputs(inner, item, own, columns)
    if outputs is None or name in outputs:
        return
    reading = _reading(window, own, columns, facts, _table_key(table))
    if reading is None:
        return
    kind, partition, value, extreme = reading
    # the reading select: inner or cross joins only, no star, the window's name read once, in a WHERE conjunct
    for join in select.args.get("joins") or []:
        if join.side or (join.kind and join.kind.upper() not in ("INNER", "CROSS")) or join.args.get("using") or join.method:
            return
    if any(isinstance(n, exp.Star) for e in select.expressions for n in e.walk() if not isinstance(n.parent, exp.Count)):
        return
    others = [s for s in select_sources(select) if s is not source]
    if any(s.alias_or_name.lower() == alias for s in others):
        return
    where = select.args.get("where")
    if where is None:
        return
    parts = conjuncts(where.this)
    chosen = None
    for part in parts:
        if _is_test(part, kind, name, alias, bool(others), value, outputs):
            chosen = part
            break
    if chosen is None:
        return
    for column in select.find_all(exp.Column):
        if _inside(column, source) or _inside(column, chosen) or column is chosen:
            continue
        if column.name.lower() == name and (not column.table or column.table.lower() == alias):
            return
    fresh = _fresh(taken)
    own_name = table.alias_or_name

    def moved(piece: exp.Expression) -> exp.Expression:
        piece = piece.copy()
        for column in piece.find_all(exp.Column):
            column.set("table", exp.to_identifier(fresh))
        return piece

    inner_where = inner.args.get("where")
    tests = [moved(part) for part in conjuncts(inner_where.this)] if inner_where is not None else []
    tests = [exp.Paren(this=test) if isinstance(test, exp.Or) else test for test in tests]
    tests += [exp.EQ(this=exp.column(k, table=fresh), expression=exp.column(k, table=own_name)) for k in partition]
    again = table.copy()
    again.set("alias", exp.TableAlias(this=exp.to_identifier(fresh)))
    scalar = exp.Select(expressions=[extreme(this=exp.column(value, table=fresh))])
    scalar.set(FROM_KEY, exp.From(this=again))
    if tests:
        scalar.set("where", exp.Where(this=_all(tests)))
    comparison = exp.EQ(this=exp.column(value, table=own_name), expression=exp.Subquery(this=scalar))
    kept = [part.copy() for part in conjuncts(inner_where.this)] if inner_where is not None else []
    kept = [exp.Paren(this=part) if isinstance(part, exp.Or) else part for part in kept]
    inner.set("where", exp.Where(this=_all(kept + [comparison])))
    inner.set("expressions", [e for e in inner.expressions if e is not item])
    rest = [part.copy() for part in parts if part is not chosen]
    rest = [exp.Paren(this=part) if isinstance(part, exp.Or) else part for part in rest]
    select.set("where", exp.Where(this=_all(rest)) if rest else None)


def _all(tests: list[exp.Expression]) -> exp.Expression:
    condition = tests[0]
    for test in tests[1:]:
        condition = exp.And(this=condition, expression=test)
    return condition


def _inside(node: exp.Expression, ancestor: exp.Expression) -> bool:
    parent = node.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


def _fresh(taken: set[str]) -> str:
    number = 0
    while f"{_PREFIX}{number}" in taken:
        number += 1
    taken.add(f"{_PREFIX}{number}")
    return f"{_PREFIX}{number}"


def _outputs(inner: exp.Select, item: exp.Expression, own: str, columns: list[str]) -> dict[str, str | None] | None:
    """Output name -> the table column it passes through (None for a computed output); None when names repeat."""

    outputs: dict[str, str | None] = {}
    for expression in inner.expressions:
        if expression is item:
            continue
        if isinstance(expression, exp.Star) or (isinstance(expression, exp.Column) and isinstance(expression.this, exp.Star)):
            if isinstance(expression, exp.Star) and any(value for key, value in expression.args.items()):
                return None
            named = {c: c for c in columns}
        else:
            name = expression.alias_or_name.lower()
            if not name:
                return None
            named = {name: _own_column(expression.this if isinstance(expression, exp.Alias) else expression, own, columns)}
        for name, column in named.items():
            if name in outputs:
                return None
            outputs[name] = column
    return outputs


def _reading(window: exp.Window, own: str, columns: list[str], facts: _Facts, table: str):
    """``(kind, partition columns, value column, MIN or MAX)`` for a window the module reads, else None."""

    if any(value for key, value in window.args.items() if key not in ("this", "partition_by", "order", "over")):
        return None
    if window.args.get("over") and str(window.args.get("over")).upper() != "OVER":
        return None
    not_null = facts.not_null.get(table, set())
    partition = []
    for key in window.args.get("partition_by") or []:
        column = _own_column(key, own, columns)
        if column is None or column not in not_null:
            return None
        partition.append(column)
    partition = list(dict.fromkeys(partition))
    order = window.args.get("order")
    ordered = list(order.expressions) if order is not None else []
    if len(ordered) > 1:
        return None
    key = descending = None
    if ordered:
        if not isinstance(ordered[0], exp.Ordered) or ordered[0].args.get("with_fill"):
            return None
        key = _own_column(ordered[0].this, own, columns)
        descending = bool(ordered[0].args.get("desc"))
        if key is None:
            return None
    call = window.this
    if isinstance(call, _RANKS):
        if key is None or key not in not_null or call.args.get("expressions") or (call.args.get("this") is not None and not isinstance(call, exp.RowNumber)):
            return None
        if isinstance(call, exp.RowNumber):
            if call.args.get("this") is not None:
                return None
            within = set(partition) | {key}
            if not any(k and set(k) <= within for k in facts.keys.get(table, [])):
                return None
        return "rank", partition, key, exp.Max if descending else exp.Min
    if type(call) in (exp.Min, exp.Max) and not any(value for name, value in call.args.items() if name != "this"):
        value = _own_column(call.this, own, columns)
        if value is None:
            return None
        if key is not None and (key != value or descending != (type(call) is exp.Max)):
            return None
        return "value", partition, value, type(call)
    if type(call) is exp.FirstValue and not any(value for name, value in call.args.items() if name != "this"):
        value = _own_column(call.this, own, columns)
        if value is None or key != value or value not in not_null:
            return None
        return "value", partition, value, exp.Max if descending else exp.Min
    return None


def _is_test(part: exp.Expression, kind: str, name: str, alias: str, qualified_only: bool, value: str, outputs: dict[str, str | None]) -> bool:
    """Whether a ``WHERE`` conjunct is the test that keeps the window's first rows."""

    part = _unwrap(part)

    def reads(node: exp.Expression, wanted: str) -> bool:
        node = _unwrap(node)
        if not isinstance(node, exp.Column) or node.name.lower() != wanted or node.args.get("db"):
            return False
        if node.table:
            return node.table.lower() == alias
        return not qualified_only

    def number(node: exp.Expression) -> int | None:
        node = _unwrap(node)
        return int(node.this) if isinstance(node, exp.Literal) and not node.is_string and node.this.isdigit() else None

    if kind == "rank":
        if isinstance(part, exp.EQ):
            return (reads(part.this, name) and number(part.expression) == 1) or (reads(part.expression, name) and number(part.this) == 1)
        if isinstance(part, exp.LTE):
            return reads(part.this, name) and number(part.expression) == 1
        if isinstance(part, exp.LT):
            return reads(part.this, name) and number(part.expression) == 2
        if isinstance(part, exp.GTE):
            return reads(part.expression, name) and number(part.this) == 1
        if isinstance(part, exp.GT):
            return reads(part.expression, name) and number(part.this) == 2
        return False
    if not isinstance(part, exp.EQ):
        return False
    passes = [n for n, column in outputs.items() if column == value]
    for a, b in ((part.this, part.expression), (part.expression, part.this)):
        if reads(a, name) and any(reads(b, n) for n in passes):
            return True
    return False
