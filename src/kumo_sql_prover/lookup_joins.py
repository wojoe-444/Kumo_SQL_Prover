"""A left join that looks up one row by its key, when only the filter reads that row: read as scalar lookups.

"Which free seats have a free seat next to them" with the neighbours joined on:

    SELECT a.seat FROM seats AS a
    LEFT JOIN seats AS p ON p.seat = a.seat - 1
    LEFT JOIN seats AS n ON n.seat - a.seat = 1
    WHERE a.free = 1 AND (p.free = 1 OR n.free = 1)

is read as

    SELECT a.seat FROM seats AS a
    WHERE a.free = 1 AND ((SELECT l.free FROM seats AS l WHERE l.seat = a.seat - 1) = 1
                       OR (SELECT l.free FROM seats AS l WHERE l.seat = a.seat + 1) = 1)

which the prover's subquery rules read (one alias for every lookup, so that two of them can be told to read the same
table).

**Why the two are equal.** ``seat`` is a declared key of the joined table and never NULL, so for each row built from
the sources before the join at most one row of the joined table has ``seat`` equal to a given number. A left join
that finds at most one row holds every row of its left side exactly once: beside the row it found, or padded with
NULLs. So removing the join changes no row and no row count, and each column of the joined table is a function of
the left row: the value in the one row found, or NULL. That is what the scalar subquery gives (at most one row by
the key, NULL for none; when the number looked for is NULL neither finds a row).

**Solving the condition.** The ``ON`` is one equality between sums and differences of whole numbers in which the key
column of the joined table appears once: ``p.seat + 1 = a.seat``, ``a.seat - p.seat = 1``, ``a.seat = (p.seat - 1)``.
Every column in it is a column of a declared table with a whole-number type, and so is every number, so the
equality is one of whole numbers and can be solved for the key (``p.seat = a.seat - 1``) without a conversion or a
rounding getting in the way. (A text key compared with a number matches ``'1'``, ``'01'`` and ``'1.0'`` alike in
MySQL, and a float plus one need not be another float; those are left alone.)

**Left alone.** A join that is not a ``LEFT`` join of a declared table by a single-column key declared NOT NULL; an
``ON`` with more than the one equality, or with a column of a source that comes after the join; a column of the
joined table read anywhere but in the select's own ``WHERE`` (the select list, ``GROUP BY``, ``ORDER BY``, another
join's ``ON``, a subquery); a select with a ``RIGHT``, ``FULL``, ``NATURAL`` or ``USING`` join, a lateral source, a star
in its select list, a column without a qualifier, or a ``WITH`` clause in or around it (a name there could hide the
declared table).
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .filter_terms import _lower, _typed_whole, _unwrapped, _whole_column, _whole_literal

NAME = "lookup_joins"


def _linear(node: exp.Expression, sign: int, key: tuple[str, str], out: dict) -> bool:
    """Add ``sign * node`` to ``out`` (the key's coefficient, the other columns with their signs, the constant);
    False when the node is not a sum or difference of columns and whole numbers written out."""

    node = _unwrapped(node)
    if isinstance(node, exp.Add):
        return _linear(node.left, sign, key, out) and _linear(node.right, sign, key, out)
    if isinstance(node, exp.Sub):
        return _linear(node.left, sign, key, out) and _linear(node.right, -sign, key, out)
    if isinstance(node, exp.Neg):
        return _linear(node.this, -sign, key, out)
    if isinstance(node, exp.Literal):
        if not _whole_literal(node):
            return False
        out["constant"] += sign * int(node.name)
        return True
    if isinstance(node, exp.Column):
        if (_lower(node.table), _lower(node.name)) == key:
            out["key"] += sign
        else:
            out["columns"].append((sign, node))
        return True
    return False


def _solved(on: exp.Expression, key: tuple[str, str]) -> tuple[list, int] | None:
    """``on`` solved for the key column: the signed columns and the constant whose sum the key equals, or None."""

    on = _unwrapped(on)
    if not isinstance(on, exp.EQ):
        return None
    out = {"key": 0, "columns": [], "constant": 0}
    # left - right = 0
    if not (_linear(on.left, 1, key, out) and _linear(on.right, -1, key, out)):
        return None
    if out["key"] not in (1, -1):
        return None
    flip = -out["key"]  # key * k + rest = 0, so k = -rest / key
    return [(sign * flip, column) for sign, column in out["columns"]], out["constant"] * flip


def _written(columns: list, constant: int) -> exp.Expression | None:
    """The sum as an expression, a column first; None when no column is added (a key equal to a number alone, or to
    minus a column, is not a lookup this rule reads)."""

    columns = sorted(columns, key=lambda item: -item[0])
    if not columns or columns[0][0] != 1:
        return None
    node: exp.Expression = columns[0][1].copy()
    for sign, column in columns[1:]:
        node = (exp.Add if sign == 1 else exp.Sub)(this=node, expression=column.copy())
    if constant:
        node = (exp.Add if constant > 0 else exp.Sub)(this=node, expression=exp.Literal.number(abs(constant)))
    return node


def _single_key(table: exp.Table, keys: dict | None, not_null: dict | None) -> str | None:
    """The column of a declared single-column key of the table that is declared NOT NULL, or None."""

    declared = next((value for name, value in (keys or {}).items() if _lower(name) == _lower(table.name)), None) or []
    never_null = {_lower(c) for c in next((value for name, value in (not_null or {}).items() if _lower(name) == _lower(table.name)), None) or ()}
    for key in declared:
        if len(key) == 1 and _lower(key[0]) in never_null:
            return _lower(key[0])
    return None


def read_lookup_joins(tree: exp.Expression, keys: dict | None, not_null: dict | None, types: dict | None) -> tuple[exp.Expression, bool]:
    """``tree`` with every such left join of every select written as scalar lookups (module doc), and whether
    anything was written. The tree given is not changed."""

    if not keys or not types or tree.find(exp.Join) is None or tree.find(exp.With) is not None:
        return tree, False
    new = tree.copy()
    wrote = False
    for _ in range(32):  # one select at a time: a rewritten select is a new node, and so are the selects inside it
        for select in new.find_all(exp.Select):
            read = lookup_joins(select, keys, not_null, types)
            if read is None:
                continue
            wrote = True
            if select is new:
                new = read
            else:
                select.replace(read)
            break
        else:
            break
    return (new, True) if wrote else (tree, False)


def lookup_joins(select: exp.Select, keys: dict | None, not_null: dict | None, types: dict | None) -> exp.Select | None:
    """The select with each left join that looks up one row by its key, for the filter alone, written as scalar
    lookups (module doc), or None."""

    joins = select.args.get("joins") or []
    where = select.args.get("where")
    from_ = select.args.get(FROM_KEY)
    if not joins or where is None or from_ is None or not types or not keys:
        return None
    if select.find(exp.With) is not None or select.args.get("laterals") or select.args.get("pivots"):
        return None
    node = select
    while node is not None:  # a WITH name could hide a declared table
        if node.args.get("with_") or node.args.get("with"):
            return None
        node = node.parent
    sources = [from_.this] + [join.this for join in joins]
    if any(not isinstance(source, (exp.Table, exp.Subquery)) or not source.alias_or_name for source in sources):
        return None
    names = [_lower(source.alias_or_name) for source in sources]
    if len(set(names)) != len(names):
        return None
    for join in joins:
        if join.args.get("using") or join.args.get("method") or _lower(join.args.get("side")) in ("right", "full") or _lower(join.args.get("kind")) not in ("", "inner", "outer", "cross"):
            return None
    if any(isinstance(item, exp.Star) or item.find(exp.Star) is not None and item.find(exp.Count) is None for item in select.expressions):
        return None
    columns = list(select.find_all(exp.Column))
    outputs = {_lower(item.alias_or_name) for item in select.expressions if item.alias_or_name}
    order = select.args.get("order")
    for column in columns:
        if column.args.get("db") or column.args.get("catalog"):
            return None
        if not column.table:
            # a bare name could be the joined table's; in the select's own ORDER BY a name of the select list is
            # that output (engines look among the output names first)
            if not (order is not None and column.find_ancestor(exp.Select) is select and column.find_ancestor(exp.Order) is order and _lower(column.name) in outputs):
                return None

    found: list[tuple[int, str, exp.Expression, exp.Expression]] = []  # position of the join, its name, its key column as written, what the key equals
    for position, join in enumerate(joins):
        table = join.this
        on = join.args.get("on")
        if _lower(join.args.get("side")) != "left" or not isinstance(table, exp.Table) or on is None or table.args.get("db") or table.args.get("catalog"):
            continue
        name = names[position + 1]
        key = _single_key(table, keys, not_null)
        if key is None or not _typed_whole(table, key, types):
            continue
        solved = _solved(on, (name, key))
        if solved is None:
            continue
        others, constant = solved
        before = set(names[: position + 1])
        if any(_lower(column.table) not in before or not _whole_column(column, select, types) for _, column in others):
            continue
        value = _written(others, constant)
        if value is None:
            continue
        uses = [column for column in columns if _lower(column.table) == name]
        inside_on = {id(column) for column in on.find_all(exp.Column)}
        if any(id(column) not in inside_on and not (column.find_ancestor(exp.Select) is select and column.find_ancestor(exp.Where) is where) for column in uses):
            continue  # read somewhere other than the select's own WHERE
        if not any(id(column) not in inside_on for column in uses):
            continue  # never read: other rules drop such a join
        written = next(column for column in on.find_all(exp.Column) if (_lower(column.table), _lower(column.name)) == (name, key))
        found.append((position, name, written.this, value))
    if not found:
        return None

    new = select.copy()
    taken = {_lower(identifier.name) for identifier in new.find_all(exp.Identifier)}
    number = 0
    while f"kql{number}" in taken:
        number += 1
    alias = f"kql{number}"
    new_joins = new.args["joins"]
    new_where = new.args["where"]
    for position, name, key, value in found:
        table = new_joins[position].this
        for column in list(new_where.find_all(exp.Column)):
            if _lower(column.table) != name:
                continue
            source = exp.Table(this=table.this.copy(), alias=exp.TableAlias(this=exp.to_identifier(alias)))
            lookup = (
                exp.Select()
                .select(exp.column(column.this.copy(), table=alias))
                .from_(source)
                .where(exp.EQ(this=exp.column(key.copy(), table=alias), expression=value.copy()))
            )
            column.replace(exp.Subquery(this=lookup))
    kept = [join for position, join in enumerate(new_joins) if position not in {item[0] for item in found}]
    new.set("joins", kept or None)
    return new
