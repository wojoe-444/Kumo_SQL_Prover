"""A select with no ``FROM`` whose outputs each read one aggregate subquery over the same rows, as one aggregate select.

    SELECT (SELECT MAX(n) FROM nums WHERE n < 10) AS top

is

    SELECT MAX(n) AS top FROM nums WHERE n < 10

The prover compares a select with no ``FROM`` subquery by subquery and a select of aggregates block by block, and had
no way from one to the other. ``aggregate_scalars`` goes the other way for an output that combines several
aggregates; this module reads only outputs that hold one aggregate each, the shape that module leaves alone, so the
two never undo each other.

**Why it holds.** A select with no ``FROM`` returns exactly one row. A select with aggregates, no ``GROUP BY`` and no
``HAVING`` returns exactly one row whatever its sources hold. Each subquery here is such a select with one output, so
as a value it is that output, computed over the rows of its sources that pass its ``WHERE``. All the subqueries of
the select have the same sources and the same ``WHERE``, so one select over those sources computes every one of the
aggregates over the same rows, and the expressions around them are the ones that were around the subqueries.
Nothing is volatile, so reading the sources once gives the values that reading them once per subquery gives.

**No quotient is moved.** MySQL keeps more digits of a quotient while it is used directly than where it is stored
(Kumo_SQL_Prover issue 45, and the note in ``aggregate_scalars``). So the output of each subquery must be built
from one ``COUNT``, ``SUM``, ``MIN`` or ``MAX`` with no division and no ``AVG`` anywhere in it: a value that is the
same wherever it is read.

**What is read.** Only the query itself, when it is a ``SELECT`` with no ``FROM``, ``WHERE``, ``GROUP BY``, ``HAVING``,
``DISTINCT``, window, ``ORDER BY``, ``LIMIT`` or set operation, in which

* at least one output holds a subquery, and no output holds more than one;
* each subquery stands in the place of a value, outside every aggregate, and is a ``SELECT`` with a ``FROM`` and one
  output and none of the clauses above (it may have a ``WHERE``);
* that output holds exactly one aggregate of the kind described, reads columns only inside it, and holds no
  subquery; every column of the subquery resolves inside it (``closed_scalar_filters.closed``);
* all the subqueries have the same sources, joins and ``WHERE`` (compared as trees, names without case);
* no unqualified column in that ``WHERE`` or in a join condition has the name of an output of the query (some
  engines would read it as the output once the two stand in one select).

A pre-pass of the thirteenth hardest-cases batch (``shape_fallback.THIRTEENTH_BATCH``). ``tests/test_lifted_aggregates.py``
replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .aggregate_scalars import _CLAUSES, _ancestors, _exact
from .ast_utils import FROM_KEY, _shape, is_aggregate
from .clause_alias_reads import _volatile
from .closed_scalar_filters import closed
from .keyed_join_groups import declared_columns

_FRAME = (FROM_KEY, "joins", "where")


def _own_subqueries(node: exp.Expression, select: exp.Select) -> list[exp.Subquery]:
    return [sub for sub in node.find_all(exp.Subquery) if sub.find_ancestor(exp.Select) is select]


def _value(sub: exp.Subquery, tree: exp.Select, declared: dict) -> exp.Expression | None:
    """The output of an aggregate subquery of ``tree`` that can be lifted (module doc), or None."""

    if any(v for key, v in sub.args.items() if key != "this") or isinstance(sub.parent, (exp.In, exp.Exists, exp.Any, exp.All, exp.From, exp.Join)):
        return None
    if any(is_aggregate(up) for up in _ancestors(sub, tree)):
        return None
    body = sub.this
    if type(body) is not exp.Select or body.args.get(FROM_KEY) is None or any(body.args.get(key) for key in _CLAUSES):
        return None
    if len(body.expressions) != 1 or _volatile(body) or any(True for _ in body.find_all(exp.Window)):
        return None
    item = body.expressions[0]
    output = item.unalias()
    if isinstance(output, exp.Star) or item.is_star or any(isinstance(node, (exp.Query, exp.Subquery, exp.Div, exp.IntDiv, exp.Avg)) for node in output.walk()):
        return None
    calls = [node for node in output.find_all(exp.AggFunc, exp.Anonymous) if is_aggregate(node)]
    if len(calls) != 1 or not _exact(calls[0]):
        return None
    for column in output.find_all(exp.Column):
        if not any(up is calls[0] for up in _ancestors(column, body)):
            return None
    if not closed(body, declared):
        return None
    return output


def lift_scalar_aggregates(tree: exp.Expression, schema: dict | None) -> tuple[exp.Expression, bool]:
    """``tree`` as one aggregate select when it is a select of aggregate subqueries over the same rows; whether it changed."""

    if type(tree) is not exp.Select or not schema or not tree.expressions:
        return tree, False
    if any(tree.args.get(key) for key in _CLAUSES + _FRAME) or _volatile(tree) or any(True for _ in tree.find_all(exp.Window)):
        return tree, False
    if any(isinstance(item, exp.Star) or item.is_star for item in tree.expressions):
        return tree, False
    declared = declared_columns(schema)
    subs = _own_subqueries(tree, tree)
    if not subs or any(len(_own_subqueries(item, tree)) > 1 for item in tree.expressions):
        return tree, False
    if any(is_aggregate(node) and node.find_ancestor(exp.Select) is tree for node in tree.find_all(exp.AggFunc, exp.Anonymous)):
        return tree, False
    if any(_value(sub, tree, declared) is None for sub in subs):
        return tree, False
    frames = {tuple(_shape(sub.this.args.get(key)) for key in _FRAME) for sub in subs}
    if len(frames) != 1:
        return tree, False
    names = {(item.alias_or_name or "").lower() for item in tree.expressions} - {""}
    for key in ("joins", "where"):
        clause = subs[0].this.args.get(key)
        for part in clause if isinstance(clause, list) else [clause]:
            if part is not None and any(not column.table and column.name.lower() in names for column in part.find_all(exp.Column)):
                return tree, False  # an output name of the query would stand next to a test that reads the same name
    result = tree.copy()
    frame = None
    for sub in _own_subqueries(result, result):
        body = sub.this
        frame = frame or body
        value = body.expressions[0].unalias().copy()
        sub.replace(value if isinstance(value, exp.Func) else exp.paren(value))
    for key in _FRAME:
        value = frame.args.get(key)
        if value is not None:
            result.set(key, [v.copy() for v in value] if isinstance(value, list) else value.copy())
    return result, True
