"""Rows without a match, counted with a flag over a left join: read as a count over the rows the join pads.

    -- "per customer, the visits without a purchase", two ways
    SELECT v.customer, SUM(CASE WHEN p.id IS NULL THEN 1 ELSE 0 END) AS n
    FROM visits AS v LEFT JOIN purchases AS p ON p.visit = v.visit GROUP BY v.customer HAVING n > 0

    SELECT v.customer, COUNT(*) AS n
    FROM visits AS v LEFT JOIN purchases AS p ON p.visit = v.visit WHERE p.id IS NULL GROUP BY v.customer

The second is the anti join the prover reads (``WHERE p.id IS NULL`` over a left join); the first is an aggregate
over an outer join, which it does not. This pre-pass writes the first as the second.

**Why they return the same rows.** A left join holds each row of its left side that finds no match exactly once,
padded with NULLs, and each matched pair once. Take a column ``p.c`` that is never NULL on a matched pair: one the
table declares NOT NULL, or one the ``ON`` compares with ``=`` at its top level (the comparison is TRUE there, so
the column is not NULL). Then ``p.c IS NULL`` is TRUE exactly on the padded rows, and

    SUM(CASE WHEN p.c IS NULL THEN 1 ELSE 0 END)

over a group is the number of padded rows in it. ``HAVING n > 0`` keeps the groups that hold a padded row. The
second query groups the padded rows alone, by the same expressions: it has a group exactly for those keys, and
``COUNT(*)`` there is that number. Every other select item has to be an expression of the ``GROUP BY``, which has
one value per group in both; and it has to be a column declared a whole number, because a group of texts an engine
calls equal (``'a'`` and ``'A'`` under MySQL's default collation) shows one of them, and which one may change with
the rows in the group.

A count that is tested outside reads the same way: ``SELECT .. FROM (the grouped select, without HAVING) AS d WHERE
n > 0`` with the derived table alone in ``FROM``. The test goes and the derived table becomes the count.

**The flags read.** ``SUM(CASE WHEN p.c IS NULL THEN 1 ELSE 0 END)``, the same with ``IF``;
``COUNT(CASE WHEN p.c IS NULL THEN 1 END)`` (any value written out, no ``ELSE`` or ``ELSE NULL``), the same with
``IF``; ``all - matched`` where ``all`` is ``COUNT(*)``, ``COUNT(1)`` or ``COUNT`` of a left-side column declared
NOT NULL, and ``matched`` is ``COUNT(p.c)`` or ``SUM(CASE WHEN p.c IS NOT NULL THEN 1 ELSE 0 END)``; and where a
test is the number 1 or 0 (MySQL, SQLite) ``SUM(p.c IS NULL)`` and ``all - SUM(p.c IS NOT NULL)``. The
``HAVING`` is one test that the flag is at least 1: ``n > 0``, ``n <> 0``, ``n >= 1``, by the flag's name in the
select list or with the flag written again.

**Left alone.** A select without ``GROUP BY`` (over no rows the first returns nothing and the second a 0) or
without that ``HAVING``; sources other than one declared table ``LEFT JOIN`` one declared table with ``ON``; a
select with ``DISTINCT``, a window, ``ORDER BY``, ``LIMIT``, a subquery or a ``WITH`` clause in or around it; a
select item that is neither a flag nor a whole-number column of the ``GROUP BY`` (by position, by its name when no
source has a column of that name, or written the same); a name the select list gives twice; any other aggregate; a column whose table the names and the
declared columns do not settle; a flag over a column that can be NULL on a matched pair.

A rule of the twentieth hardest-cases batch. ``tests/test_unmatched_row_counts.py`` runs every reading on SQLite
over made-up tables with NULLs, checks every refusal, and holds tables on which refused shapes return other rows.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .filter_terms import _joined, _lower, _terms, _typed_whole, _unwrapped

NAME = "unmatched_row_counts"

_TRUTH_IS_NUMBER = ("mysql", "sqlite")
_REFUSED_CLAUSES = ("distinct", "qualify", "windows", "order", "limit", "offset", "laterals", "pivots", "sample", "into", "locks")


def read_unmatched_row_counts(tree: exp.Expression, schema: dict | None, not_null: dict | None, types: dict | None, dialect: str | None) -> tuple[exp.Expression, bool]:
    """``tree`` with every such flag count of every select written as a count over the padded rows (module doc), and
    whether anything was written. The tree given is not changed."""

    if not schema or not types or tree.find(exp.Join) is None or tree.find(exp.With) is not None:
        return tree, False
    new = tree.copy()
    wrote = False
    for _ in range(32):  # one select at a time: a rewritten select is a new node
        for select in new.find_all(exp.Select):
            read = unmatched_row_counts(select, schema, not_null, types, dialect)
            if read is None:
                read = _tested_outside(select, schema, not_null, types, dialect)
                if read is None:
                    continue
                select = select.parent_select
            wrote = True
            if select is new:
                new = read
            else:
                select.replace(read)
            break
        else:
            break
    return (new, True) if wrote else (tree, False)


def _tested_outside(select: exp.Select, schema: dict | None, not_null: dict | None, types: dict | None, dialect: str | None) -> exp.Select | None:
    """For a grouped select that is alone in the ``FROM`` of a select whose ``WHERE`` tests one of its counts: that
    outer select with the test gone and the count read (module doc), or None."""

    holder = select.parent
    if select.args.get("having") is not None or not isinstance(holder, exp.Subquery) or not isinstance(holder.parent, exp.From):
        return None
    outer = holder.parent.parent
    if not isinstance(outer, exp.Select) or outer.args.get("joins") or outer.args.get("laterals") or outer.args.get("where") is None:
        return None
    alias = holder.args.get("alias")
    if alias is not None and alias.args.get("columns"):
        return None
    for term in _terms(outer.args["where"].this, exp.And):
        columns = list(term.find_all(exp.Column))
        if len(columns) != 1 or columns[0].args.get("db") or (columns[0].table and _lower(columns[0].table) != _lower(holder.alias)):
            continue
        having = term.copy()
        for column in having.find_all(exp.Column):
            column.set("table", None)
        inner = select.copy()
        inner.set("having", exp.Having(this=having))
        read = unmatched_row_counts(inner, schema, not_null, types, dialect, named=True)
        if read is None:
            continue
        new = outer.copy()
        new.args[FROM_KEY].this.set("this", read)
        rest = [other.copy() for other in _terms(outer.args["where"].this, exp.And) if other is not term]
        new.set("where", exp.Where(this=_joined(rest, exp.And)) if rest else None)
        return new
    return None


def _declared(table: exp.Table, facts: dict | None) -> set[str] | None:
    found = next((value for name, value in (facts or {}).items() if _lower(name) == _lower(table.name)), None)
    return None if found is None else {_lower(column) for column in found}


def _number(node: exp.Expression, value: str) -> bool:
    node = _unwrapped(node)
    return isinstance(node, exp.Literal) and not node.is_string and node.name == value


class _Reader:
    def __init__(self, left: exp.Table, right: exp.Table, schema: dict, not_null: dict | None, on: exp.Expression, lax: bool):
        self.left, self.right = _lower(left.alias_or_name), _lower(right.alias_or_name)
        self.columns = {self.left: _declared(left, schema) or set(), self.right: _declared(right, schema) or set()}
        self.left_never_null = _declared(left, not_null) or set()
        self.never_null = set(_declared(right, not_null) or set())  # of the right table, on a matched pair
        for term in _terms(on, exp.And):
            if isinstance(term, exp.EQ):
                for side in (_unwrapped(term.left), _unwrapped(term.right)):
                    if isinstance(side, exp.Column) and self.owner(side) == self.right:
                        self.never_null.add(_lower(side.name))
        self.lax = lax
        self.tested: str | None = None  # the column of the first flag read: NULL exactly on the padded rows

    def owner(self, column: exp.Column) -> str | None:
        """The source a column belongs to by its qualifier, or by the one declared table that has it; or None."""

        if column.args.get("db") or column.args.get("catalog"):
            return None
        name = _lower(column.name)
        if column.table:
            table = _lower(column.table)
            return table if table in self.columns and name in self.columns[table] else None
        owners = [table for table in (self.left, self.right) if name in self.columns[table]]
        return owners[0] if len(owners) == 1 else None

    def _padded(self, node: exp.Expression, negated: bool = False) -> bool:
        """Whether ``node`` is ``c IS NULL`` (or with ``negated``, ``c IS NOT NULL``) for a column of the right
        table that is never NULL on a matched pair."""

        node = _unwrapped(node)
        if negated:
            if not isinstance(node, exp.Not):
                return False
            node = _unwrapped(node.this)
        if not isinstance(node, exp.Is) or not isinstance(node.expression, exp.Null):
            return False
        column = _unwrapped(node.this)
        if not isinstance(column, exp.Column) or self.owner(column) != self.right or _lower(column.name) not in self.never_null:
            return False
        if self.tested is None:
            self.tested = _lower(column.name)
        return True

    def _choice(self, node: exp.Expression) -> tuple[exp.Expression, exp.Expression, exp.Expression | None] | None:
        """The test, the value when it is TRUE and the value otherwise of a two-way ``CASE`` or ``IF``."""

        node = _unwrapped(node)
        if isinstance(node, exp.Case) and node.this is None and len(node.args.get("ifs") or []) == 1:
            branch = node.args["ifs"][0]
            return branch.this, branch.args["true"], node.args.get("default")
        if isinstance(node, exp.If) and not isinstance(node.parent, exp.Case):
            return node.this, node.args["true"], node.args.get("false")
        return None

    def _ones(self, node: exp.Expression, negated: bool) -> bool:
        """Whether ``SUM(node)`` adds 1 for each padded row (with ``negated``: for each matched pair) and 0 for the rest."""

        choice = self._choice(node)
        if choice is not None:
            test, yes, no = choice
            return no is not None and _number(yes, "1") and _number(no, "0") and self._padded(test, negated)
        return self.lax and self._padded(node, negated)

    def _all(self, node: exp.Expression) -> bool:
        """Whether ``node`` counts every row of the join."""

        node = _unwrapped(node)
        if not isinstance(node, exp.Count) or isinstance(node.this, exp.Distinct) or node.expressions:
            return False
        counted = _unwrapped(node.this)
        if isinstance(counted, exp.Star) or (isinstance(counted, exp.Literal) and not isinstance(counted, exp.Null)):
            return True
        return isinstance(counted, exp.Column) and self.owner(counted) == self.left and _lower(counted.name) in self.left_never_null

    def _matched(self, node: exp.Expression) -> bool:
        """Whether ``node`` counts the matched pairs."""

        node = _unwrapped(node)
        if isinstance(node, exp.Count) and not isinstance(node.this, exp.Distinct) and not node.expressions:
            counted = _unwrapped(node.this)
            if not isinstance(counted, exp.Column) or self.owner(counted) != self.right or _lower(counted.name) not in self.never_null:
                return False
            if self.tested is None:
                self.tested = _lower(counted.name)
            return True
        return isinstance(node, exp.Sum) and not isinstance(node.this, exp.Distinct) and self._ones(node.this, negated=True)

    def flag(self, node: exp.Expression) -> bool:
        """Whether ``node`` is the number of padded rows of its group (module doc)."""

        node = _unwrapped(node)
        if isinstance(node, exp.Sum) and not isinstance(node.this, exp.Distinct):
            return self._ones(node.this, negated=False)
        if isinstance(node, exp.Count) and not isinstance(node.this, exp.Distinct) and not node.expressions:
            choice = self._choice(node.this)
            if choice is None:
                return False
            test, yes, no = choice
            yes = _unwrapped(yes)
            if not isinstance(yes, exp.Literal) or not (no is None or isinstance(_unwrapped(no), exp.Null)):
                return False
            return self._padded(test)
        if isinstance(node, exp.Sub):
            return self._all(node.left) and self._matched(node.right)
        return False

    def key(self, node: exp.Expression) -> str | None:
        """The text of an expression with each column under its source's name, or None when a column's source is not
        settled."""

        new = _unwrapped(node).copy()
        for column in ([new] if isinstance(new, exp.Column) else []) + list(new.find_all(exp.Column)):
            owner = self.owner(column)
            if owner is None:
                return None
            column.set("table", exp.to_identifier(owner))
            column.this.set("this", _lower(column.name))
            column.this.set("quoted", False)
        return new.sql().lower()


def unmatched_row_counts(
    select: exp.Select, schema: dict | None, not_null: dict | None, types: dict | None, dialect: str | None, named: bool = False
) -> exp.Select | None:
    """The select with its flag counts over a left join written as counts over the padded rows (module doc), or None.
    With ``named`` the ``HAVING`` has to test a flag by its name in the select list."""

    joins = select.args.get("joins") or []
    from_ = select.args.get(FROM_KEY)
    group, having = select.args.get("group"), select.args.get("having")
    if len(joins) != 1 or from_ is None or group is None or having is None or not schema:
        return None
    if any(select.args.get(key) for key in _REFUSED_CLAUSES) or not group.expressions or any(value for key, value in group.args.items() if key != "expressions"):
        return None
    join = joins[0]
    left, right = from_.this, join.this
    if not isinstance(left, exp.Table) or not isinstance(right, exp.Table) or join.args.get("on") is None:
        return None
    if join.args.get("using") or join.args.get("method") or _lower(join.args.get("side")) != "left" or _lower(join.args.get("kind")) not in ("", "outer"):
        return None
    for table in (left, right):
        if table.args.get("db") or table.args.get("catalog") or table.args.get("pivots") or table.args.get("sample") or _declared(table, schema) is None:
            return None
    if not left.alias_or_name or not right.alias_or_name or _lower(left.alias_or_name) == _lower(right.alias_or_name):
        return None
    if any(part is not select for part in select.find_all(exp.Select)) or _has_star_or_subquery(select):
        return None
    node: exp.Expression | None = select
    while node is not None:  # a WITH name could hide a declared table
        if node.args.get("with_") or node.args.get("with"):
            return None
        node = node.parent
    reader = _Reader(left, right, schema, not_null, join.args["on"], _lower(dialect) in _TRUTH_IS_NUMBER)
    flags: dict[int, str] = {}  # position of a flag item -> its name
    names: dict[str, int] = {}
    for position, item in enumerate(select.expressions):
        value = item.this if isinstance(item, exp.Alias) else item
        name = _lower(item.alias_or_name)
        if name in names:
            return None  # a name given twice: which item a name means is not read here
        if name:
            names[name] = position
        if reader.flag(value):
            flags[position] = _lower(item.alias) if isinstance(item, exp.Alias) else ""
        elif value.find(exp.AggFunc) is not None:
            return None
    if not flags or reader.tested is None:
        return None
    owned = reader.columns[reader.left] | reader.columns[reader.right]
    grouped_positions: set[int] = set()
    grouped_keys: set[str] = set()
    for entry in group.expressions:
        entry = _unwrapped(entry)
        if isinstance(entry, exp.Literal):
            if entry.is_string or not entry.name.isdigit() or not 1 <= int(entry.name) <= len(select.expressions):
                return None
            grouped_positions.add(int(entry.name) - 1)
            continue
        if isinstance(entry, exp.Column) and not entry.table and _lower(entry.name) not in owned:
            if _lower(entry.name) not in names or not isinstance(select.expressions[names[_lower(entry.name)]], exp.Alias):
                return None
            grouped_positions.add(names[_lower(entry.name)])  # a name of the select list no source has
            continue
        key = reader.key(entry)
        if key is None or entry.find(exp.AggFunc) is not None:
            return None
        grouped_keys.add(key)
    if grouped_positions & set(flags):
        return None
    for position, item in enumerate(select.expressions):
        if position in flags:
            continue
        value = _unwrapped(item.this if isinstance(item, exp.Alias) else item)
        if not isinstance(value, exp.Column):
            return None
        owner = reader.owner(value)
        if owner is None or not _typed_whole(left if owner == reader.left else right, value.name, types):
            return None
        if position not in grouped_positions and reader.key(value) not in grouped_keys:
            return None
    # the HAVING: the flag is at least 1
    test = _unwrapped(having.this)
    if isinstance(test, (exp.GT, exp.NEQ)) and _number(test.right, "0"):
        counted = test.left
    elif isinstance(test, exp.GTE) and _number(test.right, "1"):
        counted = test.left
    elif isinstance(test, (exp.LT, exp.NEQ)) and _number(test.left, "0"):
        counted = test.right
    elif isinstance(test, exp.LTE) and _number(test.left, "1"):
        counted = test.right
    else:
        return None
    counted = _unwrapped(counted)
    by_name = isinstance(counted, exp.Column) and not counted.table and _lower(counted.name) in set(flags.values()) - {""}
    if named:  # the name of a column of the derived table: every flag keeps its name there
        if not by_name or "" in flags.values():
            return None
    elif not (by_name and _lower(counted.name) not in owned) and not reader.flag(counted):
        return None
    where = select.args.get("where")
    for holder in (where, join.args["on"]):
        if holder is not None and any(reader.owner(column) is None for column in holder.find_all(exp.Column)):
            return None
    new = select.copy()
    for position in flags:
        item = new.expressions[position]
        if isinstance(item, exp.Alias):
            item.set("this", exp.Count(this=exp.Star()))
        else:
            item.replace(exp.Count(this=exp.Star()))
    new.set("having", None)
    padded = exp.Is(this=exp.Column(this=exp.to_identifier(reader.tested), table=exp.to_identifier(right.alias_or_name)), expression=exp.Null())
    terms = [term.copy() for term in _terms(where.this, exp.And)] if where is not None else []
    new.set("where", exp.Where(this=_joined(terms + [padded], exp.And)))
    return new


def _has_star_or_subquery(select: exp.Select) -> bool:
    for part in select.walk():
        if isinstance(part, (exp.Subquery, exp.Window)):
            return True
        if isinstance(part, exp.Star) and not isinstance(part.parent, exp.Count):
            return True
    return False
