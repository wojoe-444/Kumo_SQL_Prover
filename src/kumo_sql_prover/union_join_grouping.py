"""A key-only ``GROUP BY`` over a join that reads a derived ``UNION`` is ``SELECT DISTINCT``.

    SELECT c.k, c.a
    FROM (SELECT x AS id FROM s1 UNION ALL SELECT y AS id FROM s2) AS t, c
    WHERE c.k = t.id
    GROUP BY c.k, c.a

has no aggregate and no ``HAVING`` and outputs exactly its grouping keys, so it returns one row per distinct
``(k, a)``: the rows of ``SELECT DISTINCT c.k, c.a FROM ..``, NULL keys in one group in both. TPC-DS 54 collects
its customers this way, and an optimizer writes the same thing as ``SELECT .. FROM s1, c .. UNION DISTINCT
SELECT .. FROM s2, c ..``. The prover distributes a ``DISTINCT`` select over a ``UNION ALL`` source and meets that
form; the ``GROUP BY`` spelling stopped it.

A select that outputs only some of its keys (``SELECT c.k .. GROUP BY c.k, c.a``, which is what is left of the
select above once the query around it reads only ``k``) still has one row per distinct ``(k, a)``. It becomes
``SELECT kqg.kq_g0 AS k FROM (SELECT DISTINCT c.k AS kq_g0, c.a AS kq_g1 FROM ..) AS kqg``, which keeps the
duplicate ``k`` values the dropped key leaves. This form is read only when every key is a plain column.

``algebraic_equivalence._group_by_to_distinct`` makes this step for a select that reads the derived union alone.
This rule makes it when the union is one of several sources joined by inner or cross joins. The identity holds
for any ``FROM`` clause; the rule is kept to this shape because only there does the ``DISTINCT`` spelling read
further, and a select it does not help keeps its ``GROUP BY``.

Declined: an aggregate or window of the select itself, ``HAVING``, ``DISTINCT``, ``QUALIFY``, ``ORDER BY``,
``LIMIT``, ``OFFSET``, ``ROLLUP`` / ``CUBE`` / ``GROUPING SETS``, a star, a key that is a number (a position), an
output that is not a key (compared by text), an outer join, and a select none of
whose sources is a derived set operation.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping
from .grouped_source_filters import _inner_sources


def group_over_union_join_to_distinct(select: exp.Select) -> exp.Expression | None:
    group = select.args.get("group")
    if group is None or not select.args.get("joins"):
        return None  # a select that reads the union alone is ``_group_by_to_distinct``
    if any(select.args.get(k) for k in ("having", "distinct", "qualify", "windows", "with_", "with", "limit", "offset", "order")):
        return None
    if extended_grouping(group):
        return None
    if any(node.find_ancestor(exp.Select) is select for node in select.find_all(exp.AggFunc)) or any(node.find_ancestor(exp.Select) is select for node in select.find_all(exp.Window)):
        return None
    if any(isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star)) for e in select.expressions):
        return None
    sources = _inner_sources(select)
    if sources is None or not any(isinstance(s, exp.Subquery) and isinstance(s.this, exp.Union) for s in sources.values()):
        return None
    keys = [key.sql() for key in group.expressions]
    values = [(item.this if isinstance(item, exp.Alias) else item).sql() for item in select.expressions]
    if not set(values) <= set(keys) or len(set(keys)) != len(keys) or any(isinstance(key, exp.Literal) and not key.is_string for key in group.expressions):
        return None
    copy = select.copy()
    copy.set("group", None)
    copy.set("distinct", exp.Distinct())
    if set(values) == set(keys):
        return copy
    # a key that is not output still splits the groups: the rows are those of the DISTINCT over every key, read
    # through a projection that keeps the duplicates the dropped keys leave
    if not all(isinstance(key, exp.Column) and isinstance(key.this, exp.Identifier) for key in group.expressions) or not all(item.alias_or_name for item in select.expressions):
        return None
    hidden = {text: f"kq_g{index}" for index, text in enumerate(keys)}
    copy.set("expressions", [exp.alias_(key.copy(), hidden[key.sql()], quoted=False) for key in group.expressions])
    outputs = [exp.alias_(exp.column(hidden[value], table="kqg"), item.alias_or_name, quoted=False) for item, value in zip(select.expressions, values)]
    return exp.select(*outputs).from_(exp.Subquery(this=copy, alias=exp.TableAlias(this=exp.to_identifier("kqg"))))
