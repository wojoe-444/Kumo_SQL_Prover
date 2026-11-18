"""The join of a table with its own grouped aggregate: kept as written, and an outer join that cannot fail made inner.

    SELECT s.id, CASE WHEN g.top > 100 THEN s.pay * 0.5 ELSE s.pay END
    FROM staff AS s JOIN (SELECT team, MAX(pay) AS top FROM staff GROUP BY team) AS g ON s.team = g.team

is the spelling of "the group's aggregate beside each row" that the SMT prover reasons about: it keeps the grouped
table of a base table as one relation and knows what it says about the rows of that table (a row's value is not
above its group's ``MAX``, a NULL ``MAX`` means the row's value is NULL, and so on). Two things moved a query
away from that spelling.

**1. Pulling the aggregate up** (``eager_aggregation.pull_up_aggregate``). With a key declared on ``staff`` the
normalizer rewrites the join as ``FROM staff AS s, staff AS k WHERE s.team = k.team GROUP BY s.id, ..`` with
``MAX(k.pay)`` in the select list. That is the same query, but in it the prover no longer sees that ``s`` is one
of the rows aggregated, and whether the rewrite fires depends on how the query spells its columns (it needs every
column qualified), so two spellings of one query ended in different forms. ``own_group_join`` recognizes the
join, and ``normalize`` does not pull the aggregate up for a select it recognizes. Not applying a rewrite cannot
make a normal form wrong; like every shape rule this is switched off on the prover's second attempt, where the
aggregate is pulled up as before.

*What is recognized:* a select with exactly two sources, a plain base table with supplied columns and a derived
table that is one ``GROUP BY`` of plain columns over the same base table, with no join, ``WHERE``, ``HAVING``,
``DISTINCT``, ``LIMIT`` or window of its own.

**2. An outer join.** ``staff AS s LEFT JOIN (SELECT team, .. FROM staff GROUP BY team) AS g ON s.team = g.team``
is the inner join when ``team`` is declared NOT NULL (``inner_own_group_join``):

* the grouped table has no ``WHERE`` or ``HAVING``, so it has one row for every value of ``team`` that occurs in
  ``staff``;
* ``s`` is a row of ``staff`` (it is a base-table source that no other outer join of the select can NULL-pad)
  and its ``team`` is not NULL, so the group of its own ``team`` exists and equals it on the key: every row has
  a match, and the outer join pads nothing. (It has exactly one match, the groups being distinct on the key,
  but that does not matter for the rewrite.)
* the ``ON`` condition holds nothing but the key equalities, one per grouping column, so nothing else can make
  the match fail.

*What is read:* a ``LEFT [OUTER] JOIN`` whose right side is such a grouped table (any output expressions; every
grouping column output as a plain column), or the mirrored ``g RIGHT JOIN s`` when these are the select's only
two sources. The ``ON`` condition is exactly ``s.k = g.k`` for every grouping column ``k``, each side named by
its qualifier, and every ``k`` is declared NOT NULL. ``s`` is a plain base table with supplied columns, the same
table the grouped one reads; the select has no ``RIGHT`` or ``FULL`` join (other than the mirrored one) and
``s`` is not itself the right side of a ``LEFT JOIN``. A grouped table with a ``WHERE`` (its groups need not
include the row's), a nullable key, an extra ``ON`` condition or a key that is not output is left alone.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_membership import _base_table, _table_key, _unwrap
from .ast_utils import conjuncts, extended_grouping, select_sources, visible_ctes

_GROUPED_ARGS = frozenset({"expressions", "from", "from_", "group", "comments", "kind", "hint"})


def _plain_table(source: exp.Expression, known: dict[str, list[str]]) -> list[str] | None:
    columns = _base_table(source, known)
    if columns is None or not source.alias_or_name:
        return None
    if not source.args.get("db") and source.name.lower() in visible_ctes(source):
        return None
    return columns


def _grouped(source: exp.Expression, known: dict[str, list[str]]) -> tuple[str, dict[str, str], list[str]] | None:
    """``(table, {output name: key column}, grouping columns)`` for a derived table that groups one base table."""

    if not isinstance(source, exp.Subquery) or not source.alias or not isinstance(source.this, exp.Select):
        return None
    alias = source.args.get("alias")
    if alias.args.get("columns") or any(value for key, value in source.args.items() if key not in ("this", "alias", "comments")):
        return None
    inner = source.this
    if any(value for key, value in inner.args.items() if key not in _GROUPED_ARGS):
        return None
    sources = select_sources(inner)
    if len(sources) != 1:
        return None
    table = sources[0]
    columns = _plain_table(table, known)
    group = inner.args.get("group")
    if columns is None or group is None or extended_grouping(group) or any(value for key, value in group.args.items() if key != "expressions"):
        return None
    own = table.alias_or_name.lower()

    def column(node: exp.Expression) -> str | None:
        node = _unwrap(node)
        if type(node) is not exp.Column or isinstance(node.this, exp.Star) or node.args.get("db") or node.args.get("catalog"):
            return None
        if node.table and node.table.lower() != own:
            return None
        return node.name.lower() if node.name.lower() in columns else None

    keys = [column(key) for key in group.expressions]
    if not keys or None in keys or len(set(keys)) != len(keys):
        return None
    if any(isinstance(node, (exp.Window, exp.Query, exp.Subquery)) for node in inner.walk() if node is not inner):
        return None
    outputs: dict[str, str] = {}
    for item in inner.expressions:
        if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            return None
        name = item.alias_or_name.lower()
        key = column(item.this if isinstance(item, exp.Alias) else item)
        if key in keys and name:
            if name in outputs:
                return None
            outputs[name] = key
        elif name in outputs or (name in keys and key != name):
            return None
    return _table_key(table), outputs, keys


def own_group_join(select: exp.Expression, schema: dict[str, list[str]] | None) -> bool:
    """Whether ``select`` joins a base table with a grouped table over the same base table and nothing else."""

    if not schema or not isinstance(select, exp.Select):
        return False
    sources = select_sources(select)
    if len(sources) != 2:
        return False
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    for table, derived in (sources, sources[::-1]):
        if _plain_table(table, known) is None:
            continue
        grouped = _grouped(derived, known)
        if grouped is not None and grouped[0] == _table_key(table):
            return True
    return False


def inner_own_group_join(select: exp.Select, schema: dict[str, list[str]] | None, not_null: dict[str, frozenset[str]] | None) -> exp.Expression | None:
    """``select`` (in place) with each outer join described in the module docstring made inner, else ``None``."""

    joins = select.args.get("joins") or []
    if not schema or not not_null or not any(join.side for join in joins):
        return None
    sources = select_sources(select)
    if len(sources) != len(joins) + 1:
        return None
    names = [source.alias_or_name.lower() for source in sources]
    if "" in names or len(set(names)) != len(names):
        return None
    sides = [(join.side or "").upper() for join in joins]
    if any(join.method or join.args.get("using") or (join.kind or "").upper() not in ("", "INNER", "CROSS", "OUTER") for join in joins):
        return None
    known = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    declared = {key.lower(): {c.lower() for c in columns} for key, columns in not_null.items()}
    changed = False
    for index, join in enumerate(joins):
        if sides[index] == "LEFT":
            if any(side in ("RIGHT", "FULL") for side in sides):
                continue
            derived, partners = join.this, [(i, s) for i, s in enumerate(sources) if i <= index and (i == 0 or sides[i - 1] != "LEFT")]
        elif sides[index] == "RIGHT" and len(joins) == 1:
            derived, partners = sources[0], [(1, sources[1])]
        else:
            continue
        grouped = _grouped(derived, known)
        on = join.args.get("on")
        if grouped is None or on is None:
            continue
        table, outputs, keys = grouped
        if not set(keys) <= declared.get(table, set()):
            continue
        g = derived.alias.lower()
        for _, partner in partners:
            if _plain_table(partner, known) is None or _table_key(partner) != table:
                continue
            if _only_key_equalities(on, partner.alias_or_name.lower(), g, outputs, keys):
                join.set("side", None)
                join.set("kind", None)
                sides[index] = ""
                changed = True
                break
    return select if changed else None


def _only_key_equalities(on: exp.Expression, s: str, g: str, outputs: dict[str, str], keys: list[str]) -> bool:
    """Whether ``on`` is exactly ``s.k = g.k`` for every grouping column."""

    found = set()
    for part in conjuncts(on):
        if not isinstance(part, exp.EQ):
            return False
        left, right = _unwrap(part.this), _unwrap(part.expression)
        if type(left) is not exp.Column or type(right) is not exp.Column:
            return False
        if left.table.lower() == g and right.table.lower() == s:
            left, right = right, left
        if left.table.lower() != s or right.table.lower() != g or left.args.get("db") or right.args.get("db"):
            return False
        key = outputs.get(right.name.lower())
        if key is None or key != left.name.lower():
            return False
        found.add(key)
    return found == set(keys)
