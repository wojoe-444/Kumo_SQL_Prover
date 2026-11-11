"""An anti-join written as ``LEFT JOIN .. WHERE joined.column IS NULL`` is read as ``NOT EXISTS``.

    SELECT a.g, COUNT(DISTINCT a.x) FROM a LEFT JOIN t ON a.k = t.k AND t.f = 1 WHERE t.k IS NULL GROUP BY a.g

is

    SELECT a.g, COUNT(DISTINCT a.x) FROM a WHERE NOT EXISTS (SELECT 1 FROM t WHERE a.k = t.k AND t.f = 1) GROUP BY a.g

Take a row ``r`` of everything joined before ``t``. The ``LEFT JOIN`` keeps it once per row of ``t`` that makes the
``ON`` condition TRUE, or once, padded with NULLs, when no row of ``t`` does. The ``WHERE`` conjunct
``t.c IS NULL`` is FALSE on every matched row when ``c`` cannot be NULL there: ``c`` is declared NOT NULL, or the
``ON`` condition has a top-level conjunct that compares the bare column ``t.c`` (``=``, ``<>``, ``<``, ``<=``, ``>``,
``>=``), which is not TRUE for a NULL. On the padded row it is TRUE. So the rows that pass are exactly the rows
``r`` for which no row of ``t`` makes the ``ON`` condition TRUE, each once: ``NOT EXISTS (SELECT 1 FROM t WHERE on)``
is TRUE for exactly those rows (it is never NULL), and keeps each once. An empty ``t`` pads every row and makes
``NOT EXISTS`` TRUE for every row; duplicates of ``r`` are kept as they are on both sides.

The prover reads the join form directly when nothing is above it, but an aggregate over an outer join is first
wrapped into a derived table (``algebraic_equivalence._wrap_outer_join_aggregate``), and the anti-join inside it
never meets a ``NOT IN`` or ``NOT EXISTS`` on the other side. This rule runs before the wrap.

The test need not be spelled ``IS NULL``. Calcite's rewrites join a derived table with a constant column and
test that (``LEFT JOIN (SELECT k, TRUE AS i FROM t WHERE ..) AS d ON a.k = d.k WHERE NOT d.i IS TRUE``, or
``NOT (d.i IS TRUE AND a.k IS NOT NULL)``). Any ``WHERE`` conjunct works that is TRUE on every padded row and FALSE
or NULL on every matched row. ``_truth`` decides that in three-valued logic from what is known about a row of
each kind: on a padded row every column of the joined source is NULL; on a matched row a constant output of a
derived source has its value, a column declared NOT NULL is not NULL, and neither is any column, of either side,
that a top-level conjunct of the ``ON`` condition compares. Anything it cannot decide is "unknown" and the join is
left alone. The probe reads the joined source as it is written, derived table included.

Conditions, all checked:

* the join is a plain ``LEFT [OUTER] JOIN`` with an ``ON`` condition (no ``USING``, ``NATURAL``, lateral or sampled
  source) of a base table whose columns the schema lists, or of a derived table whose select names every output
  once without a star; its name is used once in the select;
* no join of the select uses ``USING`` or ``NATURAL`` (they read columns by name), and every join after it is an
  inner, cross or ``LEFT`` join (a ``RIGHT`` or ``FULL`` join pads the left side with NULLs, and those rows pass
  ``IS NULL`` without having met ``t`` at all);
* the test is a top-level ``WHERE`` conjunct that reads the joined source and is TRUE when padded and FALSE or
  NULL when matched (above);
* the ``ON`` condition holds no subquery, aggregate or window, and every column in it is qualified with the
  joined source's name or with a source that comes before the join, so it means the same inside the subquery;
* nothing else in the select, at any depth, reads the joined source (``unmatched_outer_join._unread``, applied
  to the rewritten select): the padded columns are never seen.

Later inner and ``LEFT`` joins do not read ``t``, so each surviving row is repeated once per match either way.

When the test is not a conjunct of its own (Calcite's ``NOT IN`` expansion has ``g.c = 0 OR d.i IS NULL AND ..``)
the join is read only if it matches at most once (``_matches_once``: a derived table grouped by, or distinct on,
columns the ``ON`` condition equates with values from the other sources). Then every row comes out exactly once,
matched or padded, so the join neither repeats nor drops a row, and each ``WHERE`` expression that tells the two
apart is an existence test wherever it stands: ``NOT EXISTS (probe)`` if it is TRUE when padded and FALSE when
matched, ``EXISTS (probe)`` the other way round (``_existence_tests``). Every read of the joined source must be
inside such an expression of the ``WHERE`` clause; otherwise the join is left alone.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import extended_grouping, plain_distinct, select_sources
from .unmatched_outer_join import _columns_of, _conjuncts, _inside, _plain_column, _plain_table, _unread

_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
_JOIN_ARGS = frozenset({"this", "side", "kind", "on"})


def _and_all(parts: list[exp.Expression]) -> exp.Expression:
    parts = [exp.Paren(this=p) if isinstance(p, (exp.Or, exp.Xor)) else p for p in parts]  # they bind looser than AND
    result = parts[0]
    for part in parts[1:]:
        result = exp.And(this=result, expression=part)
    return result


def _output_facts(source: exp.Subquery) -> tuple[set[str], dict[str, str]] | None:
    """The output names of a derived table and, for its constant outputs, ``"T"``, ``"F"`` or ``"NN"`` (not NULL)."""

    alias = source.args.get("alias")
    if alias is None or alias.args.get("columns") or not source.alias:
        return None
    if any(value for key, value in source.args.items() if key not in ("this", "alias")):
        return None
    inner = source.this
    if type(inner) is not exp.Select:
        return None
    names: set[str] = set()
    constants: dict[str, str] = {}
    for item in inner.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        name = (item.alias_or_name or "").lower()
        if not name or name in names or isinstance(value, exp.Star) or (isinstance(value, exp.Column) and isinstance(value.this, exp.Star)):
            return None
        names.add(name)
        if type(value) is exp.Boolean:
            constants[name] = "T" if value.this else "F"
        elif type(value) is exp.Literal:
            constants[name] = "NN"
    return names, constants


def _compared(on: exp.Expression) -> set[tuple[str, str]]:
    """The qualified bare columns a top-level conjunct of ``on`` compares: none of them is NULL on a matched row."""

    found: set[tuple[str, str]] = set()
    for part in _conjuncts(on):
        if type(part) not in _COMPARISONS:
            continue
        for side in (part.this, part.expression):
            column = _plain_column(side)
            if column is not None and column.table:
                found.add((column.table.lower(), column.name.lower()))
    return found


_FLIP = {"T": "F", "F": "T"}


def _truth(node: exp.Expression, name: str, padded: bool, known: dict[str, str], compared: set[tuple[str, str]]) -> str:
    """What ``node`` is on a padded row of the join (``padded``) or on a matched one.

    ``"T"``, ``"F"``, ``"N"`` (NULL), ``"NN"`` (a value that is not NULL) or ``"U"`` (not known). On a padded row every
    column of the joined source ``name`` is NULL; on a matched row its constant outputs have their value, its
    columns in ``known`` are not NULL and neither is any column the ``ON`` condition compares.
    """

    while isinstance(node, exp.Paren):
        node = node.this
    if type(node) is exp.Boolean:
        return "T" if node.this else "F"
    if type(node) is exp.Null:
        return "N"
    if type(node) is exp.Literal:
        return "NN"
    if type(node) is exp.Column:
        column = _plain_column(node)
        if column is None or not column.table:
            return "U"
        if column.table.lower() == name:
            if padded:
                return "N"
            return known.get(column.name.lower()) or ("NN" if (name, column.name.lower()) in compared else "U")
        return "NN" if not padded and (column.table.lower(), column.name.lower()) in compared else "U"
    if type(node) is exp.Is:
        if any(value for key, value in node.args.items() if key not in ("this", "expression", "negate")):
            return "U"
        value = _truth(node.this, name, padded, known, compared)
        target = node.expression
        if type(target) is exp.Null:
            result = "T" if value == "N" else "F" if value in ("T", "F", "NN") else "U"
        elif type(target) is exp.Boolean:
            if value in ("T", "F"):
                result = "T" if (value == "T") == bool(target.this) else "F"
            else:
                result = "F" if value == "N" else "U"
        else:
            return "U"
        return _FLIP.get(result, result) if node.args.get("negate") else result  # IS NOT .. is IS with a negate flag
    if type(node) is exp.Not:
        value = _truth(node.this, name, padded, known, compared)
        return _FLIP.get(value, "N" if value == "N" else "U")
    if type(node) in (exp.And, exp.Or):
        absorbing, neutral = ("F", "T") if type(node) is exp.And else ("T", "F")
        values = [_truth(side, name, padded, known, compared) for side in (node.this, node.expression)]
        if absorbing in values:
            return absorbing
        if all(value == neutral for value in values):
            return neutral
        return "N" if all(value in (neutral, "N") for value in values) else "U"
    return "U"


def _later_joins_keep_rows(joins: list[exp.Join]) -> bool:
    for join in joins:
        if (join.args.get("side") or "").upper() not in ("", "LEFT"):
            return False
        if (join.args.get("kind") or "").upper() not in ("", "INNER", "CROSS", "OUTER"):
            return False
    return True


def read_anti_join(select: exp.Select, schema: dict | None, not_null: dict | None) -> exp.Expression | None:
    """``select`` with one ``LEFT JOIN t .. WHERE t.c IS NULL`` anti-join written as ``NOT EXISTS``, or ``None``."""

    joins = select.args.get("joins") or []
    where = select.args.get("where")
    if not joins or where is None or not schema or select.args.get("laterals"):
        return None
    if any(j.args.get("using") or j.args.get("method") for j in joins):
        return None
    declared = {str(t).lower(): {str(c).lower() for c in columns} for t, columns in schema.items()}
    never_null = {str(t).lower(): {str(c).lower() for c in columns} for t, columns in (not_null or {}).items()}
    names = [(s.alias_or_name or "").lower() for s in select_sources(select)]
    tests = _conjuncts(where.this)
    for index, join in enumerate(joins):
        joined = join.this
        if (join.args.get("side") or "").upper() != "LEFT" or (join.args.get("kind") or "").upper() not in ("", "OUTER"):
            continue
        on = join.args.get("on")
        if on is None or any(v for key, v in join.args.items() if key not in _JOIN_ARGS):
            continue
        if not _later_joins_keep_rows(joins[index + 1:]):
            continue
        name = (joined.alias_or_name or "").lower()
        if not name or names.count(name) != 1:
            continue
        if _plain_table(joined):
            columns = _columns_of(joined, declared)
            full = ".".join(part.name for part in joined.parts).lower()
            known = dict.fromkeys(never_null.get(full) or never_null.get(joined.name.lower()) or (), "NN")
        elif type(joined) is exp.Subquery:
            facts = _output_facts(joined)
            columns, known = facts if facts is not None else (None, {})
        else:
            continue
        if not columns:
            continue
        if any(isinstance(n, (exp.Query, exp.Subquery, exp.Exists, exp.AggFunc, exp.Window, exp.Star)) for n in on.walk()):
            continue
        before = set(names[: index + 1])
        if any(_plain_column(c) is None or c.table.lower() not in before | {name} for c in on.find_all(exp.Column)):
            continue
        compared = _compared(on)
        for position, test in enumerate(tests):
            if not any(c.table.lower() == name for c in test.find_all(exp.Column)):
                continue
            if _truth(test, name, True, known, compared) != "T" or _truth(test, name, False, known, compared) not in ("F", "N"):
                continue
            copy = select.copy()
            new_join = copy.args["joins"][index]
            probe = exp.Select(expressions=[exp.Literal.number(1)]).from_(new_join.this.copy())
            probe.set("where", exp.Where(this=new_join.args["on"].copy()))
            absent = exp.Not(this=exp.Exists(this=probe))
            parts = [t.copy() for t in _conjuncts(copy.args["where"].this)]
            parts[position] = absent
            copy.set("where", exp.Where(this=_and_all(parts)))
            new_join.pop()
            if not copy.args.get("joins"):
                copy.set("joins", None)
            if not _unread(copy, absent, name, columns, declared):
                continue
            return copy
        if type(joined) is exp.Subquery and _matches_once(joined, on, name):
            tested = _existence_tests(select, index, name, columns, known, compared)
            if tested is not None:
                return tested
    return None


def _matches_once(joined: exp.Subquery, on: exp.Expression, name: str) -> bool:
    """Whether at most one row of the derived table makes ``on`` TRUE for a row of the sources before it.

    Its select groups by plain columns (or is ``SELECT DISTINCT``), outputs only columns and constants, and outputs
    every grouping column: no two of its rows agree on all of those outputs. ``on`` has, for each of them, a
    top-level conjunct ``d.output = e`` with ``e`` free of the derived table, so a matching row is fixed by the
    values of the ``e`` (two rows that both matched would agree on every such output).
    """

    inner = joined.this
    if type(inner) is not exp.Select:
        return False
    items: dict[str, exp.Expression] = {}
    for item in inner.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        if type(value) not in (exp.Column, exp.Literal, exp.Boolean, exp.Null):
            return False
        items[(item.alias_or_name or "").lower()] = value
    group = inner.args.get("group")
    if group is not None:
        if extended_grouping(group) or not group.expressions:
            return False
        unique = []
        for key in group.expressions:
            if type(key) is not exp.Column:
                return False
            outputs = [n for n, value in items.items() if type(value) is exp.Column and value.sql() == key.sql()]
            if not outputs:
                return False
            unique.append(outputs)
    elif plain_distinct(inner):
        unique = [[n] for n, value in items.items() if type(value) is exp.Column]
        if not unique:
            return False  # a row of constants: any two rows of the source agree
    else:
        return False
    equated = set()
    for part in _conjuncts(on):
        if type(part) is not exp.EQ:
            continue
        for mine, other in ((part.this, part.expression), (part.expression, part.this)):
            column = _plain_column(mine)
            if column is None or column.table.lower() != name:
                continue
            if any(c.table.lower() in ("", name) for c in other.find_all(exp.Column)):
                continue
            equated.add(column.name.lower())
    return all(any(n in equated for n in outputs) for outputs in unique)


def _existence_tests(select: exp.Select, index: int, name: str, columns: set[str], known: dict[str, str], compared) -> exp.Expression | None:
    """``select`` without its join ``index``, every read of the joined source written as ``[NOT] EXISTS``, or ``None``.

    For a join that matches at most once: each row of the sources before it comes out exactly once, matched or
    padded. A ``WHERE`` expression that is TRUE on a padded row and FALSE on a matched one is ``NOT EXISTS (probe)``
    wherever it stands, under ``OR`` and ``NOT`` too, and one that is FALSE and TRUE is ``EXISTS (probe)``.
    """

    copy = select.copy()
    join = copy.args["joins"][index]
    where = copy.args["where"]
    for item in copy.expressions:
        value = item.unalias()
        if isinstance(value, exp.Star) or (isinstance(value, exp.Column) and isinstance(value.this, exp.Star)):
            return None
    targets: dict[int, tuple[exp.Expression, bool]] = {}
    for column in copy.find_all(exp.Column):
        if _inside(column, join):
            continue
        if not column.table:
            if column.name.lower() in columns:
                return None  # a bare name the joined source has
            continue
        if column.table.lower() != name:
            continue
        if _plain_column(column) is None or column.find_ancestor(exp.Select) is not copy or not _inside(column, where):
            return None
        node = column.parent
        while node is not where:
            padded, matched = _truth(node, name, True, known, compared), _truth(node, name, False, known, compared)
            if (padded, matched) in (("T", "F"), ("F", "T")):
                break
            node = node.parent
        if node is where:
            return None
        targets[id(node)] = (node, padded == "T")
    if not targets:
        return None
    probe = exp.Select(expressions=[exp.Literal.number(1)]).from_(join.this.copy())
    probe.set("where", exp.Where(this=join.args["on"].copy()))
    for node, absent in targets.values():
        found = exp.Exists(this=probe.copy())
        node.replace(exp.Not(this=found) if absent else found)
    join.pop()
    if not copy.args.get("joins"):
        copy.set("joins", None)
    return copy
