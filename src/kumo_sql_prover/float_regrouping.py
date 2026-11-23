"""Refuse a proof that adds FLOAT64 values in a different grouping than the query as written.

A FLOAT64 sum is not associative, so ``SUM(f)`` and ``SUM(s)`` over ``(SELECT SUM(f) AS s FROM t GROUP BY x)``
return different values on rows ``(1, 1e16), (1, 1), (2, 1), (2, 1)``: the left adds 1e16 and the three ones in
some order, the right adds the two partial sums. ``float_sum_order`` covers the *order* of one fixed set of
additions; a normalization rule that folds a sum of partial sums into one sum (or splits one) changes the
*grouping*, which that assumption does not cover. The algebraic prover compares the normalized queries, so
without this check it proved ``float-sum-pre-aggregated`` of the numeric-traps eval (labelled not equivalent).

The test is on the number of ``SUM`` and ``AVG`` calls: a rule that regroups additions adds or removes a call
(the partial sum), and the rules that only reorder or refilter keep the count. Applied to each side, as written
against as normalized, when some call of the written pair can add FLOAT64 values:

* with declared column types, a call counts as exact only when ``sum_argument_types.is_exact`` places its
  argument: INT64 values, and also fixed-point (``NUMERIC``, ``DECIMAL``) values, whose additions are
  exact in any grouping (a product of them only while its decimal places stay within nine); if any other call
  exists and a side's number of calls changed, the proof is declined;
* without declared types nothing is known about the columns, so the proof is kept and carries the assumption
  :data:`UNTYPED_ASSUMPTION` (the old behaviour of every untyped corpus; a regrouped integer sum is exact).
"""

from __future__ import annotations

from typing import NamedTuple

from sqlglot import exp

from . import sum_argument_types
from .window_calls import is_window_function

_DEPTH = 12

UNTYPED_ASSUMPTION = (
    "sums of partial sums are read as one sum (exact for integers; a FLOAT64 sum is not associative, and no column "
    "types were given to rule it out)"
)
DECLINED = "the rewrite regroups the additions of a SUM or AVG that may add FLOAT64 values (a FLOAT64 sum is not associative)"


def _calls(tree: exp.Expression) -> list[exp.Expression]:
    return list(tree.find_all(exp.Sum, exp.Avg))


def _lowered(types) -> dict:
    return {t.lower(): {c.lower(): v for c, v in columns.items()} for t, columns in (types or {}).items()}


def _unwrap(query: exp.Expression) -> exp.Expression:
    while isinstance(query, (exp.Subquery, exp.Paren)):
        query = query.this
    return query


def _branches(query: exp.Expression) -> list[exp.Select] | None:
    """The selects of ``query`` (the branches of a set operation, left to right); ``None`` for any other shape."""

    query = _unwrap(query)
    if isinstance(query, exp.SetOperation):
        if query.args.get("by_name"):
            return None  # columns matched by name: positions say nothing
        left, right = _branches(query.this), _branches(query.expression)
        return None if left is None or right is None else left + right
    return [query] if isinstance(query, exp.Select) else None


def _literal(node: exp.Expression) -> bool:
    """An integer literal (also negated) or NULL: adding it rounds nothing."""

    while isinstance(node, (exp.Paren, exp.Neg)):
        node = node.this
    return isinstance(node, exp.Null) or (
        isinstance(node, exp.Literal) and not node.is_string and node.this.lstrip("-").isdigit()
    )


def _values_column(source: exp.Expression, name: str) -> bool | None:
    """For ``(VALUES (..), ..) AS t (a, b)``: whether column ``name`` holds only integer literals or NULL; ``None``
    when ``source`` is not such a table or has no column ``name``."""

    values = source if isinstance(source, exp.Values) else source.this
    alias = source.args.get("alias")
    if not isinstance(values, exp.Values) or alias is None or not alias.args.get("columns"):
        return None
    names = [c.name.lower() for c in alias.args["columns"]]
    if name not in names:
        return None
    position = names.index(name)
    rows = values.expressions
    return all(isinstance(r, exp.Tuple) and len(r.expressions) == len(names) and _literal(r.expressions[position]) for r in rows)


class _Source(NamedTuple):
    """One source of a select. ``kind`` is ``"table"`` (a base table: ``key`` is its name), ``"query"`` (a derived
    table or ``WITH`` name: ``query`` is what it holds), ``"values"`` or ``"opaque"`` (anything else: a lateral
    table, a table function, a recursive ``WITH`` name). ``names`` is the column list its alias gives, if any."""

    alias: str
    kind: str
    holder: exp.Expression
    query: exp.Expression | None = None
    names: tuple[str, ...] = ()
    key: str = ""


def _cte(name: str, node: exp.Expression) -> exp.CTE | str | None:
    """The nearest ``WITH`` definition called ``name`` around ``node``; ``"opaque"`` inside a recursive ``WITH``."""

    ancestor, inside = node, None
    while ancestor is not None:
        if isinstance(ancestor, exp.CTE):
            inside = ancestor  # a definition sees only the definitions before it
        with_ = ancestor.args.get("with_") or ancestor.args.get("with")
        if with_ is not None:
            visible = with_.expressions
            if any(cte is inside for cte in visible):
                visible = visible[: next(i for i, cte in enumerate(visible) if cte is inside)]
            for cte in with_.expressions if with_.args.get("recursive") else visible:
                if cte.alias.lower() == name:
                    return "opaque" if with_.args.get("recursive") else cte
        ancestor = ancestor.parent
    return None


def _column_list(alias: exp.Expression | None) -> tuple[str, ...]:
    return tuple(identifier.name.lower() for identifier in (alias.columns if isinstance(alias, exp.TableAlias) else []))


def _sources(select: exp.Select) -> list[_Source]:
    """Every source of ``select``: its ``FROM`` item and the right side of each join."""

    found: list[_Source] = []
    from_ = select.args.get("from_") or select.args.get("from")
    holders = ([from_.this] if from_ is not None else []) + [j.this for j in select.args.get("joins") or []]
    for position, holder in enumerate(holders):
        alias = (holder.alias_or_name or "").lower() or f"#{position}"
        names = _column_list(holder.args.get("alias"))
        if isinstance(holder, exp.Values) or (isinstance(holder, exp.Subquery) and isinstance(holder.this, exp.Values)):
            found.append(_Source(alias, "values", holder, names=names))
        elif isinstance(holder, exp.Subquery) and not holder.args.get("pivots") and not holder.args.get("sample"):
            found.append(_Source(alias, "query", holder, holder.this, names))
        elif isinstance(holder, exp.Table) and isinstance(holder.this, exp.Identifier) and not holder.args.get("pivots"):
            cte = None if holder.args.get("db") else _cte(holder.name.lower(), select)
            if cte is None:
                found.append(_Source(alias, "table", holder, names=names, key=".".join(part.name for part in holder.parts).lower()))
            elif cte == "opaque" or (names and _column_list(cte.args.get("alias"))):
                found.append(_Source(alias, "opaque", holder))
            else:
                found.append(_Source(alias, "query", holder, cte.this, names or _column_list(cte.args.get("alias"))))
        else:
            found.append(_Source(alias, "opaque", holder))
    return found


def _label(item: exp.Expression) -> str:
    """The name an output item is read by; empty for an item the engine has to name."""

    return (item.alias_or_name or "").lower()


def _is_star(item: exp.Expression) -> bool:
    return isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star))


def _unaliased(item: exp.Expression) -> exp.Expression:
    return item.this if isinstance(item, exp.Alias) else item


def _literal_valued(select: exp.Select, node: exp.Expression, depth: int = _DEPTH) -> bool:
    """Whether ``node`` is an integer literal or NULL in every row: the literal itself, or a column of a derived table
    (also ``VALUES``) that outputs only such values. A set operation gives its columns by position, so the item at
    the position of the name is read in every branch; a star, or a column list on the derived table, is not followed."""

    if _literal(node):
        return True
    if not isinstance(node, exp.Column) or depth <= 0:
        return False
    qualifier, name = node.table.lower(), node.name.lower()
    found = []
    for source in _sources(select):
        if qualifier and source.alias != qualifier:
            continue
        if source.kind == "opaque":
            return False
        if source.kind == "table":
            continue
        if source.kind == "values":
            values = _values_column(source.holder, name)
            if values is not None:
                found.append(values)
            continue
        branches = _branches(source.query)
        if branches is None or source.names or any(_is_star(item) for branch in branches for item in branch.expressions):
            return False
        positions = [index for index, item in enumerate(branches[0].expressions) if _label(item) == name]
        if not positions:
            continue
        if len(positions) > 1 or any(positions[0] >= len(branch.expressions) for branch in branches):
            return False
        found.extend(_literal_valued(branch, _unaliased(branch.expressions[positions[0]]), depth - 1) for branch in branches)
    return bool(found) and all(found)


class _Reader(NamedTuple):
    types: dict
    order: dict | None


def _exact(select: exp.Select, call: exp.Expression, reader: _Reader) -> bool:
    """A SUM or AVG that adds only integers, only fixed-point values or only NULLs (nothing to round)."""

    return sum_argument_types.is_exact(call, reader.types, reader.order) or (isinstance(call.this, exp.Expression) and _literal_valued(select, call.this))


def _may_round(call: exp.Expression, reader: _Reader) -> bool:
    """A SUM or AVG that may add FLOAT64 values (a window's own call is not one)."""

    select = call.find_ancestor(exp.Select)
    return select is not None and not is_window_function(call) and not _exact(select, call, reader)


def _item_sums(select: exp.Select, item: exp.Expression, reader: _Reader, depth: int) -> bool:
    """Whether output ``item`` of ``select`` is, holds or reads a SUM or AVG that may add FLOAT64 values. What cannot
    be followed (too deep, an unreadable source) counts as such a sum."""

    if any(_may_round(call, reader) for call in item.find_all(exp.Sum, exp.Avg)):
        return True
    if depth <= 0:
        return True
    for column in item.find_all(exp.Column):
        owner = column.find_ancestor(exp.Select) or select
        if not isinstance(column.this, exp.Star) and _reads_sum(owner, column, reader, depth - 1):
            return True
    return False


def _output_sums(source: _Source, name: str, reader: _Reader, depth: int) -> bool | None:
    """Whether output column ``name`` of a derived table or ``WITH`` name is such a sum; ``None`` when it has no
    column of that name. The outputs of a set operation, and of a source read with a column list, are matched by
    position; an item without a name stands for any name no other item has; a star is followed into its sources."""

    branches = _branches(source.query)
    if branches is None:
        return True
    first = branches[0]
    if name in source.names:
        positions = [source.names.index(name)]
    else:
        hidden = len(source.names)  # the first outputs are renamed by the column list
        positions = [index for index, item in enumerate(first.expressions) if index >= hidden and _label(item) == name]
        if not positions:
            positions = [index for index, item in enumerate(first.expressions) if index >= hidden and not _label(item) and not _is_star(item)]
    stars = [item for branch in branches for item in branch.expressions if _is_star(item)]
    if stars and (len(branches) > 1 or source.names):
        return True  # positions are not known past a star
    answer = None
    for position in positions:
        for branch in branches:
            if position >= len(branch.expressions) or _item_sums(branch, _unaliased(branch.expressions[position]), reader, depth):
                return True
        answer = False
    for star in stars:
        inner = star.this if isinstance(star, exp.Column) else star
        if any(inner.args.values()):
            return True  # EXCEPT, REPLACE or RENAME
        qualifier = star.table if isinstance(star, exp.Column) else None
        if _reads_sum(first, exp.column(name, table=qualifier or None), reader, depth - 1, outward=False):
            return True
        answer = False  # a star may pass the column through
    return answer


def _reads_sum(select: exp.Select, column: exp.Column, reader: _Reader, depth: int, outward: bool = True) -> bool:
    """Whether ``column``, read in ``select``, is an output of a derived table or ``WITH`` name that is such a sum.
    A name no source of ``select`` can own is looked up in the selects around it."""

    if depth <= 0:
        return True
    name, qualifier = column.name.lower(), column.table.lower()
    sources = [source for source in _sources(select) if not qualifier or source.alias == qualifier]
    owned = False
    for source in sources:
        if source.kind == "opaque":
            return True
        if source.kind == "values":
            if source.holder.find(exp.Sum, exp.Avg) is not None:
                return True
            owned = True
        elif source.kind == "table":
            declared = reader.types.get(source.key)
            owned = owned or declared is None or bool(source.names) or name in declared
        else:
            answer = _output_sums(source, name, reader, depth)
            if answer:
                return True
            owned = owned or answer is not None
    if owned or not outward:
        return False
    outer = select.find_ancestor(exp.Select)
    return outer is not None and _reads_sum(outer, column, reader, depth - 1)


def _site(call: exp.Expression, reader: _Reader) -> bool:
    """Whether ``call`` adds the output of another SUM or AVG that may add FLOAT64 values: a column that is such a
    sum (through any number of derived tables), or such a call inside its own argument."""

    select = call.find_ancestor(exp.Select)
    if select is None:
        return False
    if any(inner is not call and _may_round(inner, reader) for inner in call.find_all(exp.Sum, exp.Avg)):
        return True
    for column in call.find_all(exp.Column):
        if not isinstance(column.this, exp.Star) and _reads_sum(column.find_ancestor(exp.Select) or select, column, reader, _DEPTH):
            return True
    return False


def sum_of_sums(tree: exp.Expression, types, order: dict | None = None) -> bool:
    """Whether some SUM or AVG of ``tree`` adds the output of another SUM or AVG that may add FLOAT64 values."""

    reader = _Reader(types or {}, order)
    return any(_site(call, reader) for call in tree.find_all(exp.Sum, exp.Avg))


class _Uses:
    """How many times each part of a query is read: a ``WITH`` definition counts once for every place that reads
    it, so a call inside it counts as often as the normalized query, which holds a copy per place, shows it."""

    def __init__(self, tree: exp.Expression):
        self.readers: dict[int, list[exp.Table]] = {}
        self.memo: dict[int, int] = {}
        for table in tree.find_all(exp.Table):
            cte = None if table.args.get("db") or not isinstance(table.this, exp.Identifier) else _cte(table.name.lower(), table)
            if isinstance(cte, exp.CTE):
                self.readers.setdefault(id(cte), []).append(table)

    def of(self, node: exp.Expression) -> int:
        """The number of copies of ``node`` in the query with every ``WITH`` name written out."""

        cte = node.find_ancestor(exp.CTE)
        if cte is None:
            return 1
        with_ = cte.parent
        if isinstance(with_, exp.With) and with_.args.get("recursive"):
            return self.of(with_)  # not followed: counted where it stands
        key = id(cte)
        if key not in self.memo:
            self.memo[key] = 1  # a definition never reads itself; this only guards the walk
            self.memo[key] = sum(self.of(reader) for reader in self.readers.get(key, []))
        return self.memo[key]


def _profile(tree: exp.Expression, reader: _Reader) -> tuple[int, int]:
    """The number of SUM and AVG calls of ``tree`` and how many of them add the output of another such call that
    may add FLOAT64 values, each counted once per copy (:class:`_Uses`)."""

    uses = _Uses(tree)
    calls = sites = 0
    for call in _calls(tree):
        copies = uses.of(call)
        calls += copies
        if copies and _site(call, reader):
            sites += copies
    return calls, sites


def regrouped(written: list[exp.Expression], normalized: list[exp.Expression], types, order: dict | None = None) -> str | None:
    """``"declined"``, ``"untyped"`` or ``None`` for the two sides as written and as normalized (module doc).

    ``written`` and ``normalized`` hold one tree per side, in the same order. ``order`` gives each table's column
    names in declared order; a table read with a column list (``t AS u (b, a)``) is typed only with it."""

    declared = _lowered(types)
    reader = _Reader(declared, order)
    before = [_profile(tree, reader) for tree in written]
    if not any(sites for _, sites in before):
        return None  # no sum of partial sums as written: nothing to fold
    after = [_profile(tree, reader) for tree in normalized]
    if all(a[0] == b[0] and a[1] >= b[1] for b, a in zip(before, after)):
        return None  # every call is still there, and so is every sum of partial sums
    return "declined" if declared else "untyped"
