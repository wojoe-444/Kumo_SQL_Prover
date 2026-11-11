"""A smaller encoding for a join whose right side is matched on a unique key.

The bounded checker gives every pair of a left row and a right row its own joined row, so three tables of three
rows joined in a chain are 27 symbolic rows, and every aggregate, grouping and bag comparison above them grows with
that number. Most joins people write follow a key: ``orders JOIN customers ON orders.customer_id = customers.id``
gives each order at most one customer. For such a join one joined row per *left* row is enough:

* it is present when the left row is and some right row matches;
* its right-hand values are those of the matching right row (an if-then-else over the right rows, which is a
  single choice because two present right rows cannot both match).

The joined rows keep the order of the left rows, which is the order the row-per-pair encoding gave the same rows,
so nothing that depends on row position (a tie under ``LIMIT``) reads another order. A ``LEFT JOIN`` row with no
match takes NULLs in the same single row.

"Unique" is tracked as positions on a relation (:attr:`Rel.keys`): a set of columns whose combinations, when no
column in it is NULL, are distinct among the rows present. It starts from the keys the schema declares (which the
symbolic database asserts), and survives a filter, a rename, a projection that keeps the columns as they are, and
``DISTINCT`` (all columns); a ``GROUP BY`` whose keys are all in the output makes them one. Everything is decided
from the *identity* of the symbolic values, never from names, so an alias that shadows a column cannot fool it.

A join is read this way only when its condition is a conjunction that holds, for every column of one such key,
an ``=`` between that column and an expression over the left side alone (columns, literals, ``+ - *`` and
casts), compared without converting the key column to another type (INT64 against a decimal is fine: that
conversion is one-to-one). ``<=>`` does not qualify (NULL keys match each other), nor does a condition under ``OR``.
Any other join is encoded as before.

:func:`inner_chain` applies the same reading to joins written with commas and a ``WHERE``: each ``WHERE`` conjunct
that only reads tables joined so far is used as that join's condition, which for inner joins selects the same
rows in the same order.
"""

from __future__ import annotations

from sqlglot import exp

_PLAIN = (exp.Column, exp.Literal, exp.Paren, exp.Add, exp.Sub, exp.Mul, exp.Neg, exp.Cast, exp.DataType, exp.Identifier, exp.TableAlias)
_NUMERIC = {"int", "real"}


def conjuncts(node: exp.Expression) -> list[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return conjuncts(node.this) + conjuncts(node.expression)
    return [node]


def table_keys(table) -> tuple[tuple[int, ...], ...]:
    """The declared keys of a schema table as column positions."""

    index = {c.name.lower(): i for i, c in enumerate(table.columns)}
    out = []
    for key in table.keys:
        if key and all(c.lower() in index for c in key):
            out.append(tuple(sorted(index[c.lower()] for c in key)))
    return tuple(dict.fromkeys(out))


def carried(source, out) -> tuple[tuple[int, ...], ...]:
    """The keys of ``source`` that ``out`` still shows: same rows in the same order, the key columns unchanged."""

    if not source.keys or len(source.rows) != len(out.rows) or not source.rows:
        return ()
    where: dict[int, int] = {}
    for position in {p for key in source.keys for p in key}:
        for target in range(len(out.cols)):
            if all(o.vals[target] is s.vals[position] for s, o in zip(source.rows, out.rows)):
                where[position] = target
                break
    return tuple(tuple(sorted(where[p] for p in key)) for key in source.keys if all(p in where for p in key))


def group_keys(out, keys: list[list]) -> tuple[tuple[int, ...], ...]:
    """The output positions holding every GROUP BY key (``keys[i]`` are the key values of output row ``i``)."""

    if not out.rows or not keys or not keys[0] or len(keys) != len(out.rows):
        return ()
    positions = []
    for k in range(len(keys[0])):
        found = next((t for t in range(len(out.cols)) if all(row.vals[t] is keys[i][k] for i, row in enumerate(out.rows))), None)
        if found is None:
            return ()
        positions.append(found)
    return (tuple(sorted(set(positions))),)


def _resolve(column: exp.Column, cols, hidden) -> list[int]:
    """The positions of the joined row that ``column`` can name (the rule of ``Scope.lookup``, by position)."""

    if isinstance(column.this, exp.Star):
        return [-1, -1]
    table, name = column.table.lower() or None, column.name.lower()
    return [i for i, (q, n) in enumerate(cols) if n == name and (q == table if table is not None else i not in hidden)]


def _reads_only(node: exp.Expression, cols, hidden, allowed: range) -> bool:
    """Whether ``node`` is plain arithmetic whose every column lies in ``allowed`` or outside this join altogether."""

    for part in node.walk():
        if not isinstance(part, _PLAIN):
            return False
        if isinstance(part, exp.Column):
            found = _resolve(part, cols, hidden)
            if len(found) > 1 or (found and found[0] not in allowed):
                return False
    return True


def _key_column(node: exp.Expression, cols, hidden, keyed: range) -> int | None:
    """The position, within the keyed side, of the column ``node`` is; ``None`` when it is anything else."""

    while isinstance(node, exp.Paren):
        node = node.this
    if not isinstance(node, exp.Column):
        return None
    found = _resolve(node, cols, hidden)
    return found[0] - keyed.start if len(found) == 1 and found[0] in keyed else None


def position_free(tree: exp.Expression) -> bool:
    """Whether no result of ``tree`` depends on the order of rows inside it (no LIMIT, OFFSET or window function)."""

    return tree.find(exp.Limit, exp.Offset, exp.Fetch, exp.Window) is None


def bound_key(compiler, keys, parts: list[exp.Expression], cols, hidden, keyed: range, scope, keyed_vals) -> tuple[int, ...] | None:
    """One of ``keys`` that the conjuncts ``parts`` bind, column by column, to expressions over the other side.

    ``cols`` are the columns of a joined row and ``keyed`` the positions of the side whose key is asked about. Which
    side a column belongs to is read from its position, never from the symbolic value: in a self-join both sides
    hold the same cells. ``scope`` and ``keyed_vals`` (one joined pair of rows) only give the types compared.
    """

    from .bounded_equivalence import Unsupported

    other_side = range(0, keyed.start) if keyed.start else range(keyed.stop, len(cols))
    bound: set[int] = set()
    for part in parts:
        if type(part) is not exp.EQ:
            continue
        for key_side, other in ((part.this, part.expression), (part.expression, part.this)):
            position = _key_column(key_side, cols, hidden, keyed)
            if position is None or not _reads_only(other, cols, hidden, other_side):
                continue
            try:
                value = compiler.expr(other, scope)
            except Unsupported:
                continue
            kind = keyed_vals[position].kind
            if value.kind == kind or {value.kind, kind} <= _NUMERIC:
                bound.add(position)
                break
    return next((key for key in keys if set(key) <= bound), None)


def keyed_side(compiler, left, right, parts, scope_for, *, try_right: bool = True, try_left: bool = False) -> str | None:
    """``"right"`` when each left row matches at most one right row, ``"left"`` for the reverse, else ``None``."""

    if not compiler.keyed_joins or not parts or not left.rows or not right.rows:
        return None
    a, b = left.rows[0], right.rows[0]
    cols = left.cols + right.cols
    width = len(left.cols)
    hidden = set(left.hidden) | {width + i for i in right.hidden}
    if try_right and right.keys and bound_key(compiler, right.keys, parts, cols, hidden, range(width, len(cols)), scope_for(a, b), b.vals) is not None:
        return "right"
    if try_left and left.keys and bound_key(compiler, left.keys, parts, cols, hidden, range(0, width), scope_for(a, b), a.vals) is not None:
        return "left"
    return None


def compact(compiler, left, right, conditions, outer_join: bool, keyed: str = "right"):
    """One joined row per row of the side that is not keyed; ``conditions[i][j]`` says left ``i`` matches right ``j``.

    With ``keyed == "right"`` each left row matches at most one right row, so the joined row of left row ``i`` takes
    the values of that right row (NULLs when ``outer_join`` and there is none). ``"left"`` is the mirror image.
    """

    from .bounded_equivalence import Rel, Row, null_value, z3

    many, one = (left, right) if keyed == "right" else (right, left)
    rows = []
    for m, row in enumerate(many.rows):
        matches = [z3.And(other.present, conditions[m][o] if keyed == "right" else conditions[o][m]) for o, other in enumerate(one.rows)]
        picked = []
        for position, kind in enumerate(one.kinds):
            branches = [(match, other.vals[position]) for match, other in zip(matches, one.rows)]
            if outer_join:
                picked.append(compiler._choose(branches, null_value(kind)))
            else:
                picked.append(compiler._choose(branches[:-1], one.rows[-1].vals[position]))
        present = row.present if outer_join else z3.And(row.present, z3.Or(*matches))
        rows.append(Row(present, row.vals + picked if keyed == "right" else picked + row.vals))
    width = len(left.cols)
    kinds = [v.kind for v in rows[0].vals]
    hidden = set(left.hidden) | {width + i for i in right.hidden}
    first = list(left.first) + [width + i for i in right.first]
    keys = left.keys if keyed == "right" else tuple(tuple(width + p for p in key) for key in right.keys)
    return Rel(left.cols + right.cols, kinds, rows, frozenset(hidden), tuple(first), keys)


def _plain_inner(join: exp.Join) -> bool:
    return (
        not isinstance(join.this, exp.Lateral)
        and not (join.side or "")
        and (join.kind or "").upper() in ("", "INNER", "CROSS")
        and not join.args.get("method")
        and not join.args.get("using")
    )


def _owner(column: exp.Column, factors) -> int | None:
    """The one FROM item ``column`` names, else ``None`` (not found here, or found twice)."""

    if isinstance(column.this, exp.Star):
        return None
    table, name = column.table.lower() or None, column.name.lower()
    owners = [
        index
        for index, rel in enumerate(factors)
        for position, (qualifier, column_name) in enumerate(rel.cols)
        if column_name == name and (qualifier == table if table is not None else position not in rel.hidden)
    ]
    return owners[0] if len(owners) == 1 else None


def _step(part: exp.Expression, factors) -> int | None:
    """The first join after which every column of ``part`` is in scope; ``None`` when it cannot be placed."""

    if part.find(exp.Select, exp.Subquery, exp.Window, exp.AggFunc) is not None:
        return None
    step = 0
    for column in part.find_all(exp.Column):
        owner = _owner(column, factors)
        if owner is None:
            return None
        step = max(step, owner)
    return step


def _ordinary(compiler, factors, joins, outer):
    """The joins as the compiler encodes them without this module, over the FROM items already compiled."""

    rel = factors[0]
    for join, right in zip(joins, factors[1:]):
        rel = compiler.join(rel, join, outer, right=right)
    return rel


def _binds(part: exp.Expression, factors) -> list[tuple[int, int, frozenset]]:
    """``(factor, position, the factors the other side reads)`` for each column an ``=`` conjunct could bind."""

    if type(part) is not exp.EQ:
        return []
    out = []
    for key_side, other in ((part.this, part.expression), (part.expression, part.this)):
        while isinstance(key_side, exp.Paren):
            key_side = key_side.this
        if not isinstance(key_side, exp.Column):
            continue
        owner = _owner(key_side, factors)
        if owner is None or not all(isinstance(n, _PLAIN) for n in other.walk()):
            continue
        reads = [_owner(c, factors) for c in other.find_all(exp.Column)]
        if None in reads or owner in reads:
            continue
        table, name = key_side.table.lower() or None, key_side.name.lower()
        position = next(i for i, (q, n) in enumerate(factors[owner].cols) if n == name and (table is None or q == table))
        out.append((owner, position, frozenset(reads)))
    return out


def _made(rel) -> tuple[int, ...]:
    """The solver's own numbering of the first row's cells: the same for one base table wherever it is read."""

    return tuple(v.val.get_id() for v in rel.rows[0].vals) if rel.rows else ()


def _plan(factors, parts_by_step) -> list[int]:
    """An order to join the FROM items in that keeps the joined relation small: keyed joins first, from the best start."""

    binds = [b for parts in parts_by_step.values() for part in parts for b in _binds(part, factors)]
    linked = [frozenset(o for c in part.find_all(exp.Column) if (o := _owner(c, factors)) is not None) for parts in parts_by_step.values() for part in parts]

    def keyed(candidate: int, joined: set[int]) -> bool:
        have = {position for owner, position, reads in binds if owner == candidate and reads <= joined}
        return any(set(key) <= have for key in factors[candidate].keys)

    # ties go to the table whose cells were made first, not to the one written first: ``a JOIN b`` and ``b JOIN a``
    # over the same two tables are then joined in one order and give their rows in one order
    ranked = sorted(range(len(factors)), key=lambda f: (_made(factors[f]), f))
    best = None
    for start in ranked:
        order, joined, rows, total = [start], {start}, len(factors[start].rows), 0
        while len(order) < len(factors):
            rest = [f for f in ranked if f not in joined]
            chosen = next((f for f in rest if keyed(f, joined)), None)
            if chosen is None:
                chosen = next((f for f in rest if any(f in link and link & joined for link in linked)), rest[0])
                rows *= len(factors[chosen].rows)
            order.append(chosen)
            joined.add(chosen)
            total += rows
        if best is None or (rows, total) < best[0]:
            best = ((rows, total), order)
    return best[1]


def inner_chain(compiler, node: exp.Select, outer):
    """``(relation, WHERE conjuncts still to apply)`` for a FROM of plain inner joins, else ``None``.

    ``None`` leaves the select to the ordinary path. Every ``ON`` and ``WHERE`` conjunct that reads only the joined
    tables becomes part of the condition of the first join after which its columns are in scope, so a condition
    written under ``ON`` in one query and under ``WHERE`` in another is encoded the same way in both.
    """

    from .bounded_equivalence import MAX_ROWS, Rel, Row, Scope, Unsupported, truth, z3

    clause = node.args.get("from_") or node.args.get("from")
    joins = node.args.get("joins") or []
    where = node.args.get("where")
    if not compiler.keyed_joins or clause is None or not joins or not all(_plain_inner(j) for j in joins):
        return None
    filters = conjuncts(where.this) if where is not None else []
    factors = [compiler.table_factor(clause.this, outer)] + [compiler.table_factor(j.this, outer) for j in joins]
    if any(rel.hidden for rel in factors):
        return _ordinary(compiler, factors, joins, outer), filters
    placed: dict[int, list[exp.Expression]] = {}
    remaining = []
    for index, join in enumerate(joins, start=1):
        for part in conjuncts(join.args["on"]) if join.args.get("on") is not None else []:
            step = _step(part, factors)
            if step is None or step > index:
                return _ordinary(compiler, factors, joins, outer), filters
            placed.setdefault(index, []).append(part)
    for part in filters:
        step = _step(part, factors)
        if step is None:
            remaining.append(part)
        else:
            placed.setdefault(step, []).append(part)

    def holds(parts, scope):
        # in one order whatever order (and whichever of ON and WHERE) the conjuncts were written in: two queries
        # that join the same way then build the very same terms, and the solver has nothing to work out
        terms = {t.get_id(): t for t in (truth(compiler.expr(part, scope)) for part in parts)}
        return z3.And(*[terms[k] for k in sorted(terms)]) if terms else z3.BoolVal(True)

    # every conjunct is applied as soon as the tables it reads are joined; without a LIMIT or a window function
    # the order of the joined rows is not observable, so the joins may run in the order that keeps them fewest
    order = _plan(factors, placed) if compiler.reorder else list(range(len(factors)))
    reads = {id(part): frozenset(_owner(c, factors) for c in part.find_all(exp.Column)) for parts in placed.values() for part in parts}
    pending = [part for step in sorted(placed) for part in placed[step]]

    def ready(joined: set[int]) -> list[exp.Expression]:
        now = [part for part in pending if reads[id(part)] <= joined]
        for part in now:
            pending.remove(part)
        return now

    rel = factors[order[0]]
    joined = {order[0]}
    parts = ready(joined)
    if parts:
        rows = [Row(z3.And(r.present, holds(parts, Scope(compiler._columns(rel, r), outer))), r.vals) for r in rel.rows]
        rel = Rel(rel.cols, rel.kinds, rows, keys=rel.keys)
    spans = {order[0]: (0, len(rel.cols))}
    for index in order[1:]:
        right = factors[index]
        joined.add(index)
        parts = ready(joined)
        cols = rel.cols + right.cols

        def scope_for(a, b, cols=cols):
            return Scope([(q, n, v) for (q, n), v in zip(cols, a.vals + b.vals)], outer)

        side = keyed_side(compiler, rel, right, parts, scope_for, try_left=compiler.reorder)
        if side is None and len(rel.rows) * len(right.rows) > MAX_ROWS:
            raise Unsupported("join too large for the bound")
        conditions = [[holds(parts, scope_for(a, b)) for b in right.rows] for a in rel.rows]
        spans[index] = (len(rel.cols), len(right.cols))
        if side is not None:
            rel = compact(compiler, rel, right, conditions, False, side)
        else:
            rows = [Row(z3.And(a.present, b.present, conditions[i][j]), a.vals + b.vals) for i, a in enumerate(rel.rows) for j, b in enumerate(right.rows)]
            rel = Rel(cols, rel.kinds + right.kinds, rows)
    if order != sorted(order):  # back to the column order of the FROM clause
        positions = [spans[f][0] + k for f in range(len(factors)) for k in range(spans[f][1])]
        where = {old: new for new, old in enumerate(positions)}
        rel = Rel(
            [rel.cols[p] for p in positions], [rel.kinds[p] for p in positions],
            [Row(r.present, [r.vals[p] for p in positions]) for r in rel.rows],
            keys=tuple(tuple(sorted(where[p] for p in key)) for key in rel.keys),
        )
    return rel, remaining
