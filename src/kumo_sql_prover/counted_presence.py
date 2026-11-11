"""A ``HAVING`` that counts the rows of a group meeting a condition asks whether such a row exists.

    SELECT s.buyer_id
    FROM sales AS s JOIN product AS p ON s.product_id = p.product_id
    GROUP BY s.buyer_id
    HAVING SUM(CASE WHEN p.name = 'S8' THEN 1 ELSE 0 END) > 0
       AND SUM(CASE WHEN p.name = 'iPhone' THEN 1 ELSE 0 END) = 0

lists the buyers with a row where the name is S8 and no row where it is iPhone. The same question is commonly
written with a filter and a membership test, and that is the form this rule gives it:

    SELECT DISTINCT s.buyer_id
    FROM sales AS s JOIN product AS p ON s.product_id = p.product_id
    WHERE p.name = 'S8'
      AND NOT EXISTS (SELECT 1 FROM sales AS s2 JOIN product AS p2 ON s2.product_id = p2.product_id
                      WHERE p2.name = 'iPhone' AND s2.buyer_id = s.buyer_id)

Why it holds. The rows the select reads (its sources, joins and ``WHERE``) are split into groups by the grouping
columns. ``SUM(CASE WHEN c THEN 1 ELSE 0 END)`` is the number of rows of the group on which ``c`` is true, never
NULL, so ``> 0`` keeps the groups that have such a row and ``= 0`` those that have none. The rewritten select
keeps a row when the first "has a row" condition is true on it and, for every other condition, a row of the same
group meeting it exists (or does not). The rows kept are those of the groups the ``HAVING`` keeps that meet the
first condition, each such group has at least one, and ``DISTINCT`` over the grouping columns lists each group once.
"A row of the same group" is spelled with ``=`` on the grouping columns, which is why each of them must be a column
declared NOT NULL of a source no outer join pads.

The rule fires on a select that

* has ``GROUP BY`` on plain columns and lists exactly those columns (in any order, renamed or not);
* has a ``HAVING`` that is a conjunction of comparisons of such a count with 0 or 1: ``> 0``, ``>= 1``, ``<> 0``
  ("has a row"), ``= 0``, ``< 1``, ``<= 0`` ("has none"). The count is ``SUM(CASE WHEN c THEN n ELSE 0 END)`` with
  a positive whole number ``n``, ``COUNT(CASE WHEN c THEN x END)`` with a literal ``x``, the same written with
  ``IF``, or ``COUNTIF(c)``. ``MAX(CASE WHEN c THEN 1 ELSE 0 END)`` compared with ``= 1`` or ``= 0`` reads the same way;
* reads base tables only, each under its own name, joined by inner, cross or left joins;
* has no window, subquery, other aggregate, ``ORDER BY``, ``LIMIT`` or volatile call.

Every grouping column must be declared NOT NULL, not floating-point, and belong to the first table or to one
joined by an inner or cross join (the right side of a left join can be padded with NULLs). Without NOT NULL
facts nothing fires.
"""

from __future__ import annotations

from sqlglot import exp

from . import smt_values

_BANNED = ("qualify", "windows", "with_", "with", "limit", "offset", "order", "laterals", "pivots", "into", "locks", "sample", "cluster", "distribute", "sort", "connect", "match", "kind", "hint")
_JOIN_EXTRAS = ("method", "using", "match_condition", "global_", "hint", "directed", "expressions", "pivots")


def _unparen(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _whole(node: exp.Expression | None) -> int | None:
    node = _unparen(node)
    if isinstance(node, exp.Literal) and not node.is_string and node.this.isdigit():
        return int(node.this)
    return None


def _branch(node: exp.Expression | None) -> tuple[exp.Expression, exp.Expression, exp.Expression | None] | None:
    """``(condition, value when true, value otherwise)`` of a one-branch ``CASE`` or an ``IF``."""

    node = _unparen(node)
    if isinstance(node, exp.Case) and node.args.get("this") is None and len(node.args.get("ifs") or []) == 1:
        only = node.args["ifs"][0]
        return only.this, only.args.get("true"), node.args.get("default")
    if isinstance(node, exp.If) and node.args.get("true") is not None:
        return node.this, node.args["true"], node.args.get("false")
    return None


def _counted(node: exp.Expression | None) -> tuple[exp.Expression, str] | None:
    """``(condition, "count" | "flag")`` for an aggregate that counts, or flags, the rows of a group meeting a condition."""

    node = _unparen(node)
    if isinstance(node, exp.CountIf) and not node.args.get("distinct") and not isinstance(node.this, exp.Distinct):
        return node.this, "count"
    if not isinstance(node, (exp.Sum, exp.Count, exp.Max)) or isinstance(node.this, exp.Distinct) or node.args.get("distinct") or node.args.get("expressions"):
        return None
    branch = _branch(node.this)
    if branch is None:
        return None
    condition, value, otherwise = branch
    otherwise = _unparen(otherwise)
    if isinstance(node, exp.Count):
        value = _unparen(value)
        if isinstance(value, exp.Literal) and (otherwise is None or isinstance(otherwise, exp.Null)):
            return condition, "count"
        return None
    if _whole(otherwise) != 0:
        return None
    if isinstance(node, exp.Sum):
        return (condition, "count") if (_whole(value) or 0) >= 1 else None
    return (condition, "flag") if _whole(value) == 1 else None


_FLIPPED = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE, exp.EQ: exp.EQ, exp.NEQ: exp.NEQ}
# comparison of a count (a whole number, 0 or more) with a number: is it "has a row" (True) or "has none" (False)
_COUNT = {(exp.GT, 0): True, (exp.GTE, 1): True, (exp.NEQ, 0): True, (exp.EQ, 0): False, (exp.LT, 1): False, (exp.LTE, 0): False}
# the same for a flag, which is 0 or 1
_FLAG = {**_COUNT, (exp.EQ, 1): True, (exp.NEQ, 1): False}


def _presence(condition: exp.Expression) -> tuple[exp.Expression, bool] | None:
    """``(c, wanted)`` when ``condition`` says the group has (``wanted``) or lacks a row on which ``c`` is true."""

    condition = _unparen(condition)
    if type(condition) not in _FLIPPED:
        return None
    for kind, aggregate, number in ((type(condition), condition.this, condition.expression), (_FLIPPED[type(condition)], condition.expression, condition.this)):
        found, value = _counted(aggregate), _whole(number)
        if found is None or value is None:
            continue
        wanted = (_COUNT if found[1] == "count" else _FLAG).get((kind, value))
        return None if wanted is None else (found[0], wanted)
    return None


def _sources(select: exp.Select) -> tuple[dict[str, exp.Expression], set[str]] | None:
    """``(name -> source, names no outer join pads)`` for a select over tables and derived tables joined by inner, cross or left joins."""

    from_ = select.args.get("from_") or select.args.get("from")
    if from_ is None:
        return None
    found: dict[str, exp.Expression] = {}
    real: set[str] = set()
    for source, join in [(from_.this, None), *((j.this, j) for j in select.args.get("joins") or [])]:
        padded = False
        if join is not None:
            side, kind = (join.side or "").upper(), (join.kind or "").upper()
            if side not in ("", "LEFT") or kind not in ("", "INNER", "CROSS", "OUTER") or (kind == "OUTER" and side != "LEFT") or any(join.args.get(k) for k in _JOIN_EXTRAS):
                return None
            padded = side == "LEFT"
        if isinstance(source, exp.Table):
            if not isinstance(source.this, exp.Identifier) or any(source.args.get(k) for k in ("db", "catalog", "pivots", "sample", "joins", "version", "when")):
                return None
        elif isinstance(source, exp.Subquery):
            alias = source.args.get("alias")
            if not isinstance(source.this, exp.Query) or alias is None or alias.args.get("columns") or any(source.args.get(k) for k in ("pivots", "sample", "joins")):
                return None
            if source.find(exp.Limit, exp.Offset, exp.Fetch, exp.TableSample) is not None:
                return None  # which rows it keeps is not fixed, and it would be read twice
        else:
            return None
        name = source.alias_or_name.lower()
        if not name or name in found:
            return None
        found[name] = source
        if not padded:
            real.add(name)
    return found, real


def _columns(source: exp.Expression, listed: dict[str, set[str]]) -> set[str] | None:
    """The column names ``source`` gives, or None when they are not known."""

    if isinstance(source, exp.Table):
        return listed.get(source.name.lower())
    names = [name.lower() for name in source.this.named_selects]
    return None if "*" in names or "" in names or len(set(names)) != len(names) else set(names)


class _Facts:
    def __init__(self, not_null, types, schema):
        self.declared = {t.lower(): {c.lower() for c in cs} for t, cs in not_null.items()}
        self.listed = {t.lower().split(".")[-1]: {str(c).lower() for c in cs} for t, cs in (schema or {}).items()}
        self.kinds = {t.lower(): {c.lower(): k for c, k in cs.items()} for t, cs in (types or {}).items()}

    def owner(self, column: exp.Expression, sources: dict[str, exp.Expression]) -> tuple[str, str] | None:
        """``(source name, column)`` of a plain column of one of ``sources``."""

        column = _unparen(column)
        if not isinstance(column, exp.Column) or not isinstance(column.this, exp.Identifier) or column.args.get("db"):
            return None
        name = column.name.lower()
        if column.table:
            return (column.table.lower(), name) if column.table.lower() in sources else None
        given = {alias: _columns(source, self.listed) for alias, source in sources.items()}
        if any(columns is None for columns in given.values()):
            return None
        holders = [alias for alias, columns in given.items() if name in columns]
        return (holders[0], name) if len(holders) == 1 else None

    def identifies(self, source: exp.Expression, name: str, depth: int = 0) -> bool:
        """Is column ``name`` of ``source`` never NULL and not floating-point, so that ``=`` on it says "the same value"?"""

        if isinstance(source, exp.Table):
            table = source.name.lower()
            return name in self.declared.get(table, ()) and smt_values.type_class(self.kinds.get(table, {}).get(name)) != smt_values.FLOAT64
        inner = source.this
        if depth > 3 or not isinstance(inner, exp.Select) or any(inner.args.get(k) for k in ("group", "having", *_BANNED)):
            return False
        distinct = inner.args.get("distinct")
        if distinct is not None and distinct.args.get("on"):
            return False
        sourced = _sources(inner)
        outputs = [item for item in inner.expressions if item.alias_or_name.lower() == name]
        if sourced is None or len(outputs) != 1:
            return False
        found = self.owner(outputs[0].unalias(), sourced[0])
        return found is not None and found[0] in sourced[1] and self.identifies(sourced[0][found[0]], found[1], depth + 1)


def read_counted_presence(select: exp.Select, not_null: dict[str, frozenset[str]] | None, types: dict[str, dict[str, str]] | None = None, schema: dict | None = None) -> exp.Expression | None:
    from .algebraic_equivalence import _conjuncts
    from .counted_membership import _unstable

    group, having = select.args.get("group"), select.args.get("having")
    if not not_null or group is None or having is None or any(select.args.get(k) for k in _BANNED):
        return None
    if not group.expressions or any(group.args.get(k) for k in ("rollup", "cube", "grouping_sets", "all", "totals")):
        return None
    distinct = select.args.get("distinct")
    if distinct is not None and distinct.args.get("on"):
        return None
    sourced = _sources(select)
    if sourced is None or _unstable(select):
        return None
    sources, real = sourced
    own = [select.args.get("where"), having, group, *select.expressions, *(j.args.get("on") for j in select.args.get("joins") or [])]
    for clause in own:  # the select's own clauses; a derived table is read as it is
        if clause is not None and (clause.find(exp.Window, exp.Subquery, exp.Exists, exp.Star, exp.Select) is not None or any(c.args.get("db") for c in clause.find_all(exp.Column))):
            return None
    facts = _Facts(not_null, types, schema)

    keys: list[tuple[str, str]] = []
    for item in group.expressions:
        key = facts.owner(item, sources)
        if key is None or key in keys or key[0] not in real or not facts.identifies(sources[key[0]], key[1]):
            return None
        keys.append(key)
    outputs = [facts.owner(item.unalias(), sources) for item in select.expressions]
    if None in outputs or sorted(outputs) != sorted(keys):
        return None
    if any(isinstance(item, exp.Alias) and item.alias.lower() in sources for item in select.expressions):
        return None
    wanted: list[tuple[exp.Expression, bool]] = []
    for condition in _conjuncts(having.this):
        found = _presence(condition)
        if found is None or found[0].find(exp.AggFunc) is not None:
            return None
        wanted.append((_unparen(found[0]), found[1]))
    for clause in own:
        if clause is not None and clause is not having and clause.find(exp.AggFunc) is not None:
            return None
    if any(f"kqc{i}_{alias}" in sources for i in range(len(wanted)) for alias in sources):
        return None

    result = select.copy()
    result.set("group", None)
    result.set("having", None)
    result.set("distinct", exp.Distinct())
    for item, key in zip(result.expressions, outputs):  # the row outside is named in full, so a probe can refer to it
        item.unalias().replace(exp.column(key[1], table=sources[key[0]].alias_or_name))
    first = next((i for i, (_, has) in enumerate(wanted) if has), None)
    conditions: list[exp.Expression] = []
    if first is not None:
        conditions.append(exp.paren(wanted[first][0].copy(), copy=False))
    for index, (condition, has) in enumerate(wanted):
        if index == first:
            continue
        probe = select.copy()
        for name in ("group", "having", "distinct"):
            probe.set(name, None)
        probe.set("expressions", [exp.Literal.number(1)])
        renamed = {alias: f"kqc{index}_{alias}" for alias in sources}
        holder = exp.Paren(this=condition.copy())
        for column in [*probe.find_all(exp.Column), *holder.find_all(exp.Column)]:
            inside = column.find_ancestor(exp.Select)
            if column.table and column.table.lower() in renamed and (inside is probe or inside is None):  # not a column of a derived table's own select
                column.set("table", exp.to_identifier(renamed[column.table.lower()]))
        for alias, source in _sources(probe)[0].items():
            source.set("alias", exp.TableAlias(this=exp.to_identifier(renamed[alias])))
        same_group = [exp.EQ(this=exp.column(name, table=renamed[alias]), expression=exp.column(name, table=sources[alias].alias_or_name)) for alias, name in keys]
        _filter(probe, [holder, *same_group])
        exists = exp.Exists(this=probe)
        conditions.append(exists if has else exp.Not(this=exists))
    _filter(result, conditions)
    return result


def _filter(select: exp.Select, conditions: list[exp.Expression]) -> None:
    """Add ``conditions`` to the ``WHERE`` of ``select`` as one flat conjunction."""

    from .algebraic_equivalence import _conjuncts

    where = select.args.get("where")
    parts = [*(_conjuncts(where.this) if where is not None else []), *conditions]
    if parts:
        select.set("where", exp.Where(this=exp.and_(*parts, copy=False)))
