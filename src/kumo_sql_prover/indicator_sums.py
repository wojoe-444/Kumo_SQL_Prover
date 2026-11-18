"""Sums of indicators: several sums added up are one sum, and a filter the indicators repeat is a ``HAVING``.

    SELECT p.id, p.name,
           SUM(CASE WHEN p.id = g.first THEN 1 ELSE 0 END) + SUM(CASE WHEN p.id = g.second THEN 1 ELSE 0 END) AS wins
    FROM players AS p JOIN games AS g ON p.id = g.first OR p.id = g.second
    GROUP BY p.id

and

    SELECT p.id, p.name, SUM(CASE WHEN p.id = g.first THEN 1 ELSE 0 END + CASE WHEN p.id = g.second THEN 1 ELSE 0 END) AS wins
    FROM players AS p CROSS JOIN games AS g
    GROUP BY p.id
    HAVING SUM(CASE WHEN p.id = g.first THEN 1 ELSE 0 END + CASE WHEN p.id = g.second THEN 1 ELSE 0 END) > 0

return the same rows. The prover compares grouped selects aggregate by aggregate over the same input rows, so
it relates neither the sum of two sums to one sum nor a filtered input to an unfiltered one. Two rewrites bring
the first spelling to the second.

**1. A sum of indicator sums is one sum** (:func:`merge_indicator_sums`). ``SUM(a) + SUM(b)``, outside any
aggregate, becomes ``SUM(a + b)`` when ``a`` and ``b`` are *integer values*: built from integer literals alone
through ``CASE .. ELSE ..`` (or ``IF(c, x, y)``, which the merged sum writes as that ``CASE``), ``+``, ``-`` and
``*``, with any conditions. Such a value is an
integer and never NULL on any row, so neither sum skips a row and the two sums range over the same rows:
adding them row by row or sum by sum is the same integer. Over no rows (a global aggregate of an empty input,
the total row of a ``ROLLUP``) both sides are NULL. In a select with a plain ``GROUP BY``, where every group has
a row, ``COUNT(CASE WHEN c THEN k END)`` (no ``ELSE`` or ``ELSE NULL``, ``k`` a literal that is not NULL, or
``IF(c, k, NULL)``) is read as ``SUM(CASE WHEN c THEN 1 ELSE 0 END)`` when it is added to such sums: it counts
the rows on which ``c`` is TRUE. A lone ``COUNT`` is left as it is.

**2. A filter the indicators repeat is a ``HAVING``** (:func:`disjunction_to_having`). Let a select with a
plain ``GROUP BY`` over base tables, joined by inner or cross joins only, have a top-level ``AND`` part
``F = d_1 OR .. OR d_m`` in its ``WHERE`` or in an ``ON`` (``x IN (a, b)`` is read as ``x = a OR x = b``), and
let every aggregate of the select be a ``SUM`` of an integer value that is ``0`` on every row where no ``d_k``
is TRUE. Then ``F`` is removed and ``HAVING SUM(CASE WHEN d_1 THEN 1 ELSE 0 END + ..) > 0`` is added.

*Why.* With inner and cross joins an ``ON`` part is a filter like a ``WHERE`` part. Removing ``F`` adds to each
group the rows on which ``F`` is not TRUE, and adds the groups made of such rows only. On an added row every
``SUM`` argument is 0, so the sums of a group that was there do not change; they are never NULL before or
after, since their arguments never are and a group has a row. The added ``HAVING`` sum counts, per group, the
pairs (row, ``k``) with ``d_k`` TRUE: it is positive exactly when some row of the group passes ``F`` (``F`` is
TRUE exactly when some ``d_k`` is), which is exactly when the group existed before. That the argument is 0
where no ``d_k`` is TRUE is checked by evaluation: an arm of a ``CASE`` whose condition is one of the ``d_k``
(or an ``AND`` with one, or an ``OR`` of them) is not taken, any other arm must give the value the rest gives,
and the arithmetic over the literals must come to 0.

The select has more rows per group afterwards, so everything outside the aggregates must be fixed by the group
key: a key expression, or a column of a table whose declared key (all NOT NULL) is among the key columns.

**3. A count that is not zero is positive** (:func:`positive_tests`). A top-level ``AND`` part ``s <> 0`` or
``s >= 1`` of ``HAVING`` is written ``s > 0`` when ``s`` is a ``SUM`` of an integer value that is never negative
(built from the literals through ``CASE .. ELSE ..``, ``IF``, ``+`` and ``*`` only, no ``-``): ``s`` is then a
whole number that is not negative, or NULL over no rows, and the three tests are TRUE on the same values (none
is TRUE on NULL). ``s`` may be read by the name of the output that is such a sum, when every source is a base
table, none has a column of that name, and the select is the whole statement or a derived table (nested in a
condition, the name could be a column of the query around it). The prover then sees one spelling of "at least
one". A ``COUNT`` is left to the rules that already read its tests.

**What is refused.** Rule 3: any other comparison or number, a sum that could be negative or is not built from
literals. Rule 1: ``DISTINCT`` aggregates, ``FILTER``, window aggregates, a ``SUM`` of a column or of
anything that may be NULL (``SUM(a = b)`` is NULL for a group where every comparison is NULL, and the row
arithmetic ``(a = b) + (c = d)`` is NULL where one comparison is), subtraction of sums. Rule 2: outer joins,
``USING``, derived tables, nested queries, windows, grouping sets, ``LIMIT``, any aggregate other than such a
``SUM`` (a ``COUNT(*)`` would count the added rows), an argument that is not provably 0 off the filter, a
column no single source owns, a loosely grouped column.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key
from .ast_utils import FROM_KEY, conjuncts, extended_grouping, select_sources
from .union_all_counts import _SELECT_ARGS, _all, _fixed, _integer_value, _owner, _plain_joins, _row_value, _stands_alone, _with_positive


def indicator_sum_rules(
    select: exp.Expression,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression | None:
    """One step of one of the rewrites of the module docstring, or ``None``."""

    if not isinstance(select, exp.Select):
        return None
    return merge_indicator_sums(select) or disjunction_to_having(select, schema, keys, not_null) or positive_tests(select, schema)


def _own(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select


def _plain_sum(node: exp.Expression) -> bool:
    return (
        type(node) is exp.Sum
        and not isinstance(node.parent, (exp.Filter, exp.Window))
        and not any(value for key, value in node.args.items() if key != "this")
        and node.this is not None
        and _row_value(node.this)
        and _integer_value(node.this)
    )


def _counted_test(node: exp.Expression) -> exp.Expression | None:
    """``CASE WHEN c THEN 1 ELSE 0 END`` for ``COUNT(CASE WHEN c THEN k END)`` or ``COUNT(IF(c, k, NULL))``."""

    if type(node) is not exp.Count or isinstance(node.parent, (exp.Filter, exp.Window)):
        return None
    if any(value for key, value in node.args.items() if key not in ("this", "big_int")):
        return None
    value = node.this
    while isinstance(value, exp.Paren):
        value = value.this
    if value is None or not _row_value(value):
        return None

    def missing(part: exp.Expression | None) -> bool:
        return part is None or isinstance(part, exp.Null)

    def present(part: exp.Expression | None) -> bool:
        return isinstance(part, exp.Literal)

    if isinstance(value, exp.Case) and value.args.get("this") is None and len(value.args["ifs"]) == 1 and missing(value.args.get("default")):
        arm = value.args["ifs"][0]
        test, kept = arm.this, arm.args.get("true")
    elif isinstance(value, exp.If) and missing(value.args.get("false")):
        test, kept = value.this, value.args.get("true")
    else:
        return None
    if not present(kept):
        return None
    return exp.Case(ifs=[exp.If(this=test.copy(), true=exp.Literal.number(1))], default=exp.Literal.number(0))


def _terms(node: exp.Expression) -> list[exp.Expression]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Add):
        return _terms(node.this) + _terms(node.expression)
    return [node]


def merge_indicator_sums(select: exp.Select) -> exp.Expression | None:
    """Rule 1 of the module docstring, applied to every sum of sums of the select."""

    group = select.args.get("group")
    counts = group is not None and bool(group.expressions) and not extended_grouping(group)
    copy = select.copy()
    changed = False
    for node in list(copy.find_all(exp.Add)):
        if isinstance(node.parent, exp.Add) or (isinstance(node.parent, exp.Paren) and isinstance(node.parent.parent, exp.Add)):
            continue  # not the top of its sum
        if not _own(node, copy) or node.find_ancestor(exp.AggFunc, exp.Window) is not None:
            continue
        terms = _terms(node)
        merged = []
        for term in terms:
            if _plain_sum(term):
                merged.append(term.this.copy())
                continue
            test = _counted_test(term) if counts else None
            if test is None:
                break
            merged.append(test)
        if len(merged) != len(terms) or len(terms) < 2:
            continue
        total = None
        for value in merged:
            if not isinstance(value, (exp.Case, exp.If, exp.Literal, exp.Paren)):
                value = exp.Paren(this=value)
            total = value if total is None else exp.Add(this=total, expression=value)
        node.replace(exp.Sum(this=_cases(total)))
        changed = True
    return copy if changed else None


def _cases(node: exp.Expression) -> exp.Expression:
    """An integer value with each ``IF(c, x, y)`` written as ``CASE WHEN c THEN x ELSE y END``, the same value."""

    node = exp.Paren(this=node)
    for call in list(node.find_all(exp.If)):
        if not isinstance(call.parent, exp.Case) and call.args.get("false") is not None:
            call.replace(exp.Case(ifs=[exp.If(this=call.this, true=call.args["true"])], default=call.args["false"]))
    return node.this


def _disjuncts(node: exp.Expression) -> list[exp.Expression] | None:
    """The ``d_k`` of a filter ``d_1 OR .. OR d_m`` or ``x IN (a, ..)``; ``None`` for any other condition."""

    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Or):
        left, right = _disjuncts(node.this), _disjuncts(node.expression)
        return (left or [node.this.unnest()]) + (right or [node.expression.unnest()])
    if isinstance(node, exp.In) and node.expressions and not any(value for key, value in node.args.items() if key not in ("this", "expressions")):
        return [exp.EQ(this=node.this.copy(), expression=item.copy()) for item in node.expressions]
    return None


def _spellings(test: exp.Expression) -> set[str]:
    while isinstance(test, exp.Paren):
        test = test.this
    found = {test.sql()}
    if isinstance(test, exp.EQ):
        found.add(exp.EQ(this=test.expression.copy(), expression=test.this.copy()).sql())
    return found


def _not_taken(test: exp.Expression, off: set[str]) -> bool:
    """Whether ``test`` is certainly not TRUE on a row where none of the conditions spelled in ``off`` is."""

    while isinstance(test, exp.Paren):
        test = test.this
    if test.sql() in off:
        return True
    if isinstance(test, exp.And):
        return _not_taken(test.this, off) or _not_taken(test.expression, off)
    if isinstance(test, exp.Or):
        return _not_taken(test.this, off) and _not_taken(test.expression, off)
    return False


def _value_off(node: exp.Expression, off: set[str]) -> int | None:
    """The integer an integer value takes on every row where none of the conditions in ``off`` is TRUE, if one."""

    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Literal):
        return int(node.this) if not node.is_string and node.this.isdigit() else None
    if isinstance(node, exp.Neg):
        value = _value_off(node.this, off)
        return None if value is None else -value
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul)):
        left, right = _value_off(node.this, off), _value_off(node.expression, off)
        if left is None or right is None:
            return None
        return left + right if isinstance(node, exp.Add) else left - right if isinstance(node, exp.Sub) else left * right
    if isinstance(node, exp.Case) and node.args.get("this") is None and node.args.get("default") is not None:
        arms = [(arm.this, arm.args["true"]) for arm in node.args["ifs"]]
        rest = _value_off(node.args["default"], off)
    elif isinstance(node, exp.If) and node.args.get("false") is not None:
        arms = [(node.this, node.args["true"])]
        rest = _value_off(node.args["false"], off)
    else:
        return None
    for test, value in reversed(arms):
        if _not_taken(test, off):
            continue
        if rest is None or _value_off(value, off) != rest:
            return None
    return rest


def disjunction_to_having(
    select: exp.Select,
    schema: dict[str, list[str]] | None,
    keys: dict[str, list[tuple[str, ...]]] | None,
    not_null: dict[str, frozenset[str]] | None,
) -> exp.Expression | None:
    """Rule 2 of the module docstring."""

    group = select.args.get("group")
    if not schema or group is None or not group.expressions or extended_grouping(group):
        return None
    if any(value for key, value in select.args.items() if key not in _SELECT_ARGS) or not _plain_joins(select):
        return None
    distinct = select.args.get("distinct")
    if distinct is not None and distinct.args.get("on"):
        return None
    if any(node is not select for node in select.find_all(exp.Select, exp.Subquery, exp.Exists, exp.Window)):
        return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    columns: dict[str, list[str]] = {}
    spelling: dict[str, str] = {}
    base: dict[str, str] = {}
    sources = select_sources(select)
    for source in sources:
        listed = _base_table(source, known)
        alias = source.alias_or_name.lower() if isinstance(source, exp.Table) else ""
        if listed is None or not alias or alias in columns:
            return None
        columns[alias], spelling[alias], base[alias] = listed, source.alias_or_name, _table_key(source)
    # an output is read by its name only where no query around the select can own the name
    names = {item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias)} if _stands_alone(select) else set()

    def resolved(node: exp.Expression, allowed: set[str] = frozenset()) -> exp.Expression | None:
        node = exp.Paren(this=node.copy())
        for column in list(node.find_all(exp.Column)):
            name = column.name.lower()
            if not column.table and name in allowed:
                if any(name in listed for listed in columns.values()):
                    return None  # the output of that name or the source column: the dialects disagree
                continue
            owner = _owner(column, columns)
            if owner is None:
                return None
            column.replace(exp.column(name, spelling[owner]))
        return node.this

    pool: list[exp.Expression] = []
    for join in select.args.get("joins") or []:
        if join.args.get("on") is not None:
            pool += conjuncts(join.args["on"])
    if select.args.get("where") is not None:
        pool += conjuncts(select.args["where"].this)
    pool = [resolved(part) for part in pool]
    group_keys = [resolved(key) for key in group.expressions]
    items = [resolved(item.unalias()) for item in select.expressions]
    having = resolved(select.args["having"].this, names) if select.args.get("having") is not None else None
    ordered = [resolved(term.this, names) for term in select.args["order"].expressions] if select.args.get("order") is not None else []
    read = items + ordered + ([having] if select.args.get("having") is not None else [])
    if any(node is None for node in pool + group_keys + read) or any(key.find(exp.AggFunc) is not None for key in group_keys):
        return None
    aggregates = [call for node in read for call in node.find_all(exp.AggFunc)]
    if not aggregates or not all(_plain_sum(call) for call in aggregates):
        return None
    declared = {name.lower(): {c.lower() for c in listed} for name, listed in (not_null or {}).items()}
    declared_keys = {name.lower(): listed for name, listed in (keys or {}).items()}
    if not _fixed(read, group_keys, declared_keys, declared, base, names):
        return None
    for position, part in enumerate(pool):
        tests = _disjuncts(part)
        if tests is None or not _row_value(part):
            continue
        off = {spelled for test in tests for spelled in _spellings(test)}
        if any(_value_off(call.this, off) != 0 for call in aggregates):
            continue
        total = None
        for test in tests:
            term = exp.Case(ifs=[exp.If(this=test.copy(), true=exp.Literal.number(1))], default=exp.Literal.number(0))
            total = term if total is None else exp.Add(this=total, expression=term)
        test = exp.GT(this=exp.Sum(this=total), expression=exp.Literal.number(0))
        copy = select.copy()
        tables = [source.copy() for source in sources]
        copy.set("expressions", [exp.alias_(value, item.alias, copy=False) if isinstance(item, exp.Alias) else value for item, value in zip(select.expressions, items)])
        copy.set(FROM_KEY, exp.From(this=tables[0]))
        copy.set("joins", [exp.Join(this=table, kind="CROSS") for table in tables[1:]] or None)
        where = _all(pool[:position] + pool[position + 1 :])
        copy.set("where", exp.Where(this=where) if where is not None else None)
        copy.set("group", exp.Group(expressions=group_keys))
        copy.set("having", exp.Having(this=test if having is None else _with_positive(having, test, copy.expressions)))
        for term, value in zip(copy.args["order"].expressions if ordered else [], ordered):
            term.set("this", value)
        return copy
    return None


def _never_negative(node: exp.Expression) -> bool:
    """An integer value (see rule 1) written without ``-``: never NULL and never below 0."""

    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Literal):
        return not node.is_string and node.this.isdigit()
    if isinstance(node, (exp.Add, exp.Mul)):
        return _never_negative(node.this) and _never_negative(node.expression)
    if isinstance(node, exp.Case):
        default = node.args.get("default")
        return node.this is None and default is not None and _never_negative(default) and all(_never_negative(arm.args["true"]) for arm in node.args["ifs"])
    if isinstance(node, exp.If):
        return node.args.get("false") is not None and _never_negative(node.args["true"]) and _never_negative(node.args["false"])
    return False


def _counts(node: exp.Expression) -> bool:
    """A ``SUM`` of a value that is a whole number and never negative."""

    while isinstance(node, exp.Paren):
        node = node.this
    return _plain_sum(node) and _never_negative(node.this)


def positive_tests(select: exp.Select, schema: dict[str, list[str]] | None) -> exp.Expression | None:
    """Rule 3 of the module docstring."""

    having = select.args.get("having")
    if having is None:
        return None
    named: set[str] = set()
    if schema and _stands_alone(select):
        known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
        listed = [_base_table(source, known) for source in select_sources(select)]
        if listed and all(columns is not None for columns in listed):
            taken = {column for columns in listed for column in columns}
            names = [item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias)]
            named = {
                item.alias.lower()
                for item in select.expressions
                if isinstance(item, exp.Alias) and _counts(item.this) and item.alias.lower() not in taken and names.count(item.alias.lower()) == 1
            }
    copy = select.copy()
    changed = False
    for part in conjuncts(copy.args["having"].this):
        test = part.unnest()
        if not ((isinstance(test, exp.NEQ) and _zero(test.expression, "0")) or (isinstance(test, exp.GTE) and _zero(test.expression, "1"))):
            continue
        left = test.this.unnest()
        by_name = isinstance(left, exp.Column) and not left.table and left.name.lower() in named
        if not by_name and not _counts(left):
            continue
        test.replace(exp.GT(this=test.this.copy(), expression=exp.Literal.number(0)))
        changed = True
    return copy if changed else None


def _zero(node: exp.Expression, text: str) -> bool:
    node = node.unnest()
    return isinstance(node, exp.Literal) and not node.is_string and node.this == text
