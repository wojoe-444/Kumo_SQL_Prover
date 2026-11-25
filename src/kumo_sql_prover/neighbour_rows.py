"""A row's neighbour over a column of consecutive whole numbers, read as a keyed lookup.

"Which free seats have a free seat next to them" is written with the neighbour looked up by its number, or with
``LAG`` and ``LEAD`` over the seats in order:

    SELECT s.seat FROM seats AS s
    WHERE s.free = 1 AND ((SELECT n.free FROM seats AS n WHERE n.seat = s.seat - 1) = 1
                       OR (SELECT n.free FROM seats AS n WHERE n.seat = s.seat + 1) = 1)

    SELECT d.seat FROM (SELECT seat, free, LAG(free) OVER (ORDER BY seat) AS before,
                               LEAD(free) OVER (ORDER BY seat) AS after FROM seats) AS d
    WHERE d.free = 1 AND (d.before = 1 OR d.after = 1)

The second is read as the first, which the prover's subquery rules read.

**Why the two are equal.** A column is declared *consecutive* (``TableConstraints.consecutive``) when its values
are whole numbers, never NULL, no two the same, and every value but the smallest has the value one below it in the
table: the values are ``m, m + 1, .., m + r - 1`` for the ``r`` rows. Ordered by that column ``k`` with no
``PARTITION BY``, over every row of the table, the row ``n`` places before the row with ``k = v`` is the row with
``k = v - n``, and there is one exactly when ``n`` rows come before. So ``LAG(x, n) OVER (ORDER BY k)`` is the ``x`` of
the row whose ``k`` is ``n`` less, and NULL when there is no such row, which is what the scalar subquery
``(SELECT b.x FROM t AS b WHERE b.k = a.k - n)`` gives: the key makes it at most one row, and no row is NULL.
``LEAD`` is the row whose ``k`` is ``n`` more, and ``ORDER BY k DESC`` swaps the two.

**Folding the derived table.** The derived table reads one table with no ``WHERE``, join, grouping or ``LIMIT``
(an ``ORDER BY`` without ``LIMIT`` changes no rows, and ``DISTINCT`` removes none when the consecutive column, which
never repeats, is one of the outputs), so it has exactly one row per row of the table, and each of its columns is an expression of that row
(the lookups read the table, which the row count does not depend on). The select that reads it and nothing else
is therefore the same select over the table itself with each column replaced by its expression.

**A default.** ``LEAD(x, 1, 0)`` gives 0 where no row follows, and the ``x`` of the next row where one does, NULL
included. When ``x`` is a column declared NOT NULL (or the consecutive column itself) the lookup is NULL exactly
when no row is there, so the neighbour is ``COALESCE((SELECT b.x ..), 0)`` (hardest-cases batch 31). For a column
that may be NULL the two differ on a neighbour whose value is NULL, and the window is left alone. The column must
also be declared with a whole-number type (or be a true/false column in MySQL, a small whole number there): DuckDB
gives the default the type of the value, so beside a BOOLEAN ``LEAD(b, 1, -1)`` is TRUE where the ``COALESCE`` is
-1, and it refuses the ``COALESCE`` of a text and a number.

**Left alone.** A window with ``PARTITION BY``, a frame, a name, ``IGNORE NULLS``, or a default value that is not a
whole number written out (DuckDB converts a default to the value's type, which ``COALESCE`` does not) or that comes
with a value which may be NULL; an offset that is not a positive whole number written out; an ``ORDER BY`` that is not exactly one column
declared consecutive; any other window, an aggregate or a subquery in the derived table; a derived table with a
``WHERE``, a join, ``GROUP BY``, ``HAVING``, ``LIMIT`` or a ``DISTINCT`` without the consecutive column among its
outputs, or one that is not read from a declared table; a query that still has a ``WITH`` clause (a name there could hide the declared table); an enclosing select with another source, ``GROUP BY``, ``HAVING``, ``LIMIT``, a window, a star or a
subquery of its own; a column of the enclosing select that is not a column of the derived table (it would name
the enclosing query's column, and the table's own column of that name would capture it).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY

NAME = "neighbour_rows"

_PASSES = 8


def _lower(name: str | None) -> str:
    return (name or "").lower()


def _from(select: exp.Select) -> exp.Expression | None:
    from_ = select.args.get(FROM_KEY)
    return from_.this if from_ is not None else None


def _whole(node: exp.Expression | None) -> int | None:
    """The value of a positive whole-number literal written without quotes."""

    if isinstance(node, exp.Literal) and not node.is_string and node.name.isdigit():
        return int(node.name)
    return None


class _Neighbour:
    """One ``LAG`` or ``LEAD`` the rule reads: the value read, the key, and how far before (negative) or after."""

    def __init__(self, value: exp.Expression, key: str, shift: int, default: exp.Expression | None = None):
        self.value, self.key, self.shift, self.default = value, key, shift, default


def _neighbour(window: exp.Window, table: exp.Table, columns: frozenset, consecutive: frozenset, never_null: frozenset = frozenset()) -> _Neighbour | None:
    call = window.this
    if not isinstance(call, (exp.Lag, exp.Lead)):
        return None
    if set(k for k, v in window.args.items() if v) - {"this", "order", "over"} or window.args.get("over") not in (None, "OVER"):
        return None  # PARTITION BY, a frame, a window name, OVER a named window
    order = window.args.get("order")
    if not isinstance(order, exp.Order) or len(order.expressions) != 1:
        return None
    ordered = order.expressions[0]
    key = ordered.this if isinstance(ordered, exp.Ordered) else ordered
    if not isinstance(key, exp.Column) or not _own(key, table, columns) or _lower(key.name) not in consecutive:
        return None
    if set(k for k, v in call.args.items() if v) - {"this", "offset", "default"}:
        return None
    default = call.args.get("default")
    if isinstance(default, exp.Null):
        default = None
    if default is not None and _whole(default.this if isinstance(default, exp.Neg) else default) is None and not (isinstance(default, exp.Literal) and default.name == "0"):
        # only a whole number written out: DuckDB converts the default to the value's type (LEAD(n, 1, 1.5) is 2 for
        # a whole-number n), and COALESCE does not
        return None
    offset = call.args.get("offset")
    places = 1 if offset is None else _whole(offset)
    if not places:
        return None
    value = call.this
    if value is None or not _plain(value, table, columns):
        return None
    if default is not None and not (isinstance(value, exp.Column) and _lower(value.name) in never_null | consecutive):
        # the default stands for a missing row only: a neighbour whose value is NULL gives NULL, which a lookup
        # cannot tell from no neighbour unless the value is a column that is never NULL
        return None
    back = isinstance(call, exp.Lag) != bool(isinstance(ordered, exp.Ordered) and ordered.args.get("desc"))
    return _Neighbour(value, _lower(key.name), -places if back else places, default)


def _own(column: exp.Column, table: exp.Table, columns: frozenset) -> bool:
    """Whether a column inside the derived table is a declared column of the table (bare, or named with the table's
    name or alias). A name the table does not have would be a column of an enclosing query."""

    if column.args.get("db") or column.args.get("catalog") or isinstance(column.this, exp.Star):
        return False
    return _lower(column.name) in columns and (not column.table or _lower(column.table) == _lower(table.alias_or_name))


def _plain(node: exp.Expression, table: exp.Table, columns: frozenset) -> bool:
    """An expression of one row of the table: no aggregate, window, subquery or star, and only the table's columns."""

    for part in node.walk():
        if isinstance(part, (exp.AggFunc, exp.Window, exp.Subquery, exp.Select, exp.Star, exp.Placeholder, exp.Parameter)):
            return False
        if isinstance(part, exp.Column) and not _own(part, table, columns):
            return False
    return True


def _outputs(inner: exp.Select, table: exp.Table, columns: list[str], consecutive: frozenset, never_null: frozenset = frozenset()) -> dict[str, exp.Expression] | None:
    """The derived table's columns by name, each an expression of the table's row with the windows still in it;
    None unless every window is a neighbour the rule reads and there is at least one."""

    outputs: dict[str, exp.Expression] = {}
    found, declared = False, frozenset(columns)
    for item in inner.expressions:
        if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            if isinstance(item, exp.Column) and (item.args.get("db") or item.args.get("catalog") or _lower(item.table) != _lower(table.alias_or_name)):
                return None
            if item.args.get("except_") or item.args.get("replace") or item.args.get("except"):
                return None
            named = [(name, exp.column(name)) for name in columns]
        else:
            value = item.this if isinstance(item, exp.Alias) else item
            name = _lower(item.alias_or_name)
            if not name or not isinstance(item, (exp.Alias, exp.Column)):
                return None
            named = [(name, value)]
        for name, value in named:
            if name in outputs:
                return None
            windows = list(value.find_all(exp.Window))
            for window in windows:
                if _neighbour(window, table, declared, consecutive, never_null) is None:
                    return None
                found = True
            rest = value.copy()
            for window in list(rest.find_all(exp.Window)):
                if window is rest:
                    rest = exp.Null()
                else:
                    window.replace(exp.Null())
            if not _plain(rest, table, declared):
                return None
            outputs[name] = value
    return outputs if found else None


def _names(tree: exp.Expression) -> set[str]:
    return {_lower(node.name) for node in tree.find_all(exp.Identifier)} | {_lower(node.alias_or_name) for node in tree.find_all(exp.Table, exp.Subquery)}


def _requalified(node: exp.Expression, alias: str) -> exp.Expression:
    node = node.copy()
    for column in [node] if isinstance(node, exp.Column) else list(node.find_all(exp.Column)):
        column.set("table", exp.to_identifier(alias))
    return node


def _lookup(neighbour: _Neighbour, table: exp.Table, alias: str, fresh: str) -> exp.Expression:
    """``(SELECT b.x FROM t AS b WHERE b.k = a.k - n)`` for a row before, ``b.k = a.k + n`` for a row after: the
    key alone on one side, which is the spelling the rules for keyed lookups read. With a default it is
    ``COALESCE((SELECT ..), default)``: the value is never NULL, so the lookup is NULL exactly when no row is there."""

    mine, theirs = exp.column(neighbour.key, table=alias), exp.column(neighbour.key, table=fresh)
    places = exp.Literal.number(abs(neighbour.shift))
    test = exp.EQ(this=theirs, expression=(exp.Sub if neighbour.shift < 0 else exp.Add)(this=mine, expression=places))
    source = exp.Table(this=table.this.copy(), db=table.args.get("db") and table.args["db"].copy(), catalog=table.args.get("catalog") and table.args["catalog"].copy())
    source.set("alias", exp.TableAlias(this=exp.to_identifier(fresh)))
    lookup = exp.Subquery(this=exp.Select(expressions=[_requalified(neighbour.value, fresh)]).from_(source).where(test))
    if neighbour.default is None:
        return lookup
    return exp.Coalesce(this=lookup, expressions=[neighbour.default.copy()])


_ATOMS = (exp.Column, exp.Literal, exp.Subquery, exp.Paren, exp.Func, exp.Case, exp.Null, exp.Boolean)


def _whole_numbers_never_null(declared: str, not_null: dict | None, types: dict | None, dialect: str | None) -> frozenset:
    """The columns of the declared table that are NOT NULL and hold whole numbers: a declared whole-number type, or a
    true/false column in MySQL, where it is a small whole number. Only beside such a value is a whole-number default
    the same value in the window and in ``COALESCE``: DuckDB turns the default of a BOOLEAN value into TRUE or FALSE
    (``LEAD(b, 1, -1)`` is TRUE where ``COALESCE(b, -1)`` is -1) and refuses ``COALESCE`` of a text and a number."""

    from .single_row_aggregates import column_kind

    def of(mapping: dict | None) -> object:
        return next((v for k, v in (mapping or {}).items() if _lower(k) == declared), None)

    kinds = {_lower(c): str(v) for c, v in (of(types) or {}).items() if v}
    flags = ("BOOL", "BOOLEAN") if dialect == "mysql" else ()
    return frozenset(
        _lower(c) for c in of(not_null) or () if column_kind(kinds.get(_lower(c))) == "int" or kinds.get(_lower(c), "").strip().upper() in flags
    )


def _read(
    select: exp.Select, schema: dict, consecutive: dict, taken: set[str], not_null: dict | None = None, types: dict | None = None, dialect: str | None = None
) -> bool:
    """Fold the one derived table of ``select`` when it is a table with neighbours read over a consecutive column."""

    source = _from(select)
    if not isinstance(source, exp.Subquery) or not source.alias or not isinstance(source.this, exp.Select):
        return False
    if set(k for k, v in source.args.items() if v) - {"this", "alias"} or source.args["alias"].args.get("columns"):
        return False
    inner = source.this
    extras = set(k for k, v in inner.args.items() if v) - {"expressions", FROM_KEY, "kind"}
    if extras - {"order", "distinct"}:  # an ORDER BY with no LIMIT does not change the rows of a derived table
        return False
    if "distinct" in extras and inner.args["distinct"].args.get("on"):
        return False
    table = _from(inner)
    if not isinstance(table, exp.Table) or set(k for k, v in table.args.items() if v) - {"this", "db", "catalog", "alias"}:
        return False
    if isinstance(table.args.get("alias"), exp.TableAlias) and table.args["alias"].args.get("columns"):
        return False
    declared = _lower(table.name)
    columns = [_lower(c) for c in (schema or {}).get(declared) or next((v for k, v in (schema or {}).items() if _lower(k) == declared), [])]
    keys = frozenset(_lower(c) for c in (consecutive or {}).get(declared) or next((v for k, v in (consecutive or {}).items() if _lower(k) == declared), ()))
    if not columns or not keys or not keys <= set(columns):
        return False
    never_null = _whole_numbers_never_null(declared, not_null, types, dialect)
    outputs = _outputs(inner, table, columns, keys, never_null)
    if outputs is None:
        return False
    if "distinct" in extras and not any(isinstance(value, exp.Column) and _lower(value.name) in keys for value in outputs.values()):
        return False  # DISTINCT removes nothing only when a column that never repeats is among the outputs
    # the enclosing select: this one source, and nothing that reads rows in another way
    if set(k for k, v in select.args.items() if v) - {"expressions", FROM_KEY, "where", "distinct", "order", "kind"}:
        return False
    alias = source.alias
    outside = [node for node in select.walk() if not _inside(node, source)]
    if any(isinstance(node, (exp.Subquery, exp.Select, exp.Window)) and node is not select for node in outside):
        return False
    if any(isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count) for node in outside):
        return False
    # an output name that renames something: ORDER BY reads it before a column of the source
    aliases = {_lower(item.alias) for item in select.expressions if isinstance(item, exp.Alias) and not (isinstance(item.this, exp.Column) and _lower(item.this.name) == _lower(item.alias))}
    uses = []
    for column in outside:
        if not isinstance(column, exp.Column):
            continue
        if column.args.get("db") or column.args.get("catalog"):
            return False
        name = _lower(column.name)
        if column.table:
            if _lower(column.table) != _lower(alias):
                continue  # a column of an enclosing query, named with that query's source
            if name not in outputs:
                return False
        elif name not in outputs:
            if column.find_ancestor(exp.Order) is not None and name in aliases:
                continue  # ORDER BY an output name of this select
            return False
        elif column.find_ancestor(exp.Order) is not None and name in aliases:
            return False  # an output name that is also a column of the derived table: which one ORDER BY means is left alone
        uses.append(column)
    fresh = _fresh(taken)  # one name for every lookup of this select: each is a subquery of its own
    for column in uses:
        value = outputs[_lower(column.name)]
        new = _requalified(value, alias)
        windows = [new] if isinstance(new, exp.Window) else list(new.find_all(exp.Window))
        for window in windows:
            neighbour = _neighbour(_original(window, table), table, frozenset(columns), keys, never_null)
            lookup = _lookup(neighbour, table, alias, fresh)
            if window is new:
                new = lookup
            else:
                window.replace(lookup)
        item = any(column is expression for expression in select.expressions)
        if item and not (isinstance(new, exp.Column) and _lower(new.name) == _lower(column.name)):
            new = exp.Alias(this=new, alias=exp.to_identifier(column.name))
        elif not isinstance(new, _ATOMS) and not isinstance(column.parent, (exp.Alias, exp.Paren)):
            new = exp.Paren(this=new)
        column.replace(new)
    renamed = exp.Table(this=table.this.copy(), db=table.args.get("db") and table.args["db"].copy(), catalog=table.args.get("catalog") and table.args["catalog"].copy())
    renamed.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
    source.replace(renamed)
    return True


def _original(window: exp.Window, table: exp.Table) -> exp.Window:
    """The window with its columns named as the derived table named them (bare): ``_neighbour`` reads that form."""

    window = window.copy()
    for column in window.find_all(exp.Column):
        column.set("table", None)
    return window


def _inside(node: exp.Expression, root: exp.Expression) -> bool:
    while node is not None:
        if node is root:
            return True
        node = node.parent
    return False


def _fresh(taken: set[str]) -> str:
    number = 0
    while f"kqn{number}" in taken:
        number += 1
    taken.add(f"kqn{number}")
    return f"kqn{number}"


def read_neighbour_rows(
    tree: exp.Expression, schema: dict | None, consecutive: dict | None, not_null: dict | None = None, types: dict | None = None, dialect: str | None = None
) -> tuple[exp.Expression, bool]:
    """``tree`` with every derived table of neighbours over a consecutive column folded into the select that reads it
    (module doc), and whether anything was written. The tree given is not changed."""

    if not schema or not consecutive or not any(v for v in consecutive.values()):
        return tree, False
    if not any(isinstance(window.this, (exp.Lag, exp.Lead)) for window in tree.find_all(exp.Window)):
        return tree, False
    if any(True for _ in tree.find_all(exp.With)):
        return tree, False  # a name of a WITH clause could hide the declared table
    new = tree.copy()
    taken = _names(new)
    wrote = False
    for _ in range(_PASSES):
        for select in list(new.find_all(exp.Select)):
            if _read(select, schema, consecutive, taken, not_null, types, dialect):
                wrote = True
                break
        else:
            break
    return (new, True) if wrote else (tree, False)
