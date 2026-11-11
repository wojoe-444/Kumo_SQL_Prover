"""Two derived tables the prover shows equal are read as one text.

The prover keeps some derived tables whole and matches them by their text: a grouped select it cannot open (an
aggregate over an outer join, ``COUNT(DISTINCT ..)``), a select with ``DISTINCT`` or a ``LIMIT``. Two spellings of
one such relation then never meet when they are *parts* of the two queries, although the prover proves the same
two texts equal when they are the whole queries:

    SELECT ps.id, us.n FROM (SELECT p.id, p.day, COUNT(c.id) AS n FROM p LEFT JOIN c ON c.post = p.id GROUP BY p.id, p.day) AS ps JOIN ..
    SELECT t9.id, t12.n FROM (SELECT p1.id, p1.day, COUNT(c1.id) AS n FROM p AS p1 LEFT JOIN c AS c1 ON p1.id = c1.post GROUP BY p1.day, p1.id) AS t9 JOIN ..

``unify(left, right, prove=...)`` works on the two normalized queries after the prover has left them unproven
(and after ``window_inputs`` had its turn). For a derived table of each query that aggregates, removes duplicates
or cuts rows:

1. **An identity on each query.** The select items of the derived table are listed in the order in which the
   select that reads it uses them. A derived table in ``FROM`` is read by column name only (the conditions below
   exclude a star and a column list), so the order of its outputs carries no meaning.
2. **A proof.** ``prove(left_body, right_body)`` asks the prover itself whether the two bodies return the same bag
   of rows, column by column in that order.
3. **A replacement.** The right derived table then reads the left body, with each output given the name the right
   body gave to the output in the same position. Its rows are the proven-equal rows under the right query's own
   column names, so the right query is unchanged as a query, and the two queries now hold one text for the relation.

The prover is then run again on the two texts.

Conditions checked (the derived table is left alone otherwise):

* neither query has a ``WITH``; the derived table is an aliased ``FROM`` or join item without a column list,
  reached from the query through derived tables only (never inside a scalar, ``EXISTS`` or ``IN`` subquery, a
  ``LATERAL`` or a set operation), so no column in it can name a row of an enclosing query, and the body means the
  same wherever it is written;
* the select that reads it has no star and no ``USING`` or ``NATURAL`` join (they would read outputs by position or
  without naming them);
* the body is one ``SELECT`` with a ``GROUP BY``, ``HAVING``, aggregate, ``DISTINCT`` or ``LIMIT`` (a plain join or
  filter is flattened by the prover and needs no help), no window function of its own, no ``QUALIFY``, named
  ``WINDOW``, ``OFFSET`` or locking clause, and nothing that can differ between two evaluations (``RANDOM()``, an
  unknown function);
* its outputs have distinct names, none is a star, and no column of the body outside the select list is an
  unqualified name equal to the name of an output, unless that output is the same bare column (a ``GROUP BY`` or
  ``ORDER BY`` that reads an output by its alias would read another column after the renaming);
* every qualified column of the body names a source declared inside it;
* the body's ``GROUP BY`` and ``ORDER BY`` name no output by its position (``GROUP BY 1``): the outputs are listed in
  another order here, and the position would name another one;
* the two bodies have the same number of outputs and read the same tables the same number of times.

A sub-proof's assumptions are passed on by the caller. At most ``MAX_PROOFS`` sub-proofs are tried for a pair, on
bodies of at most ``MAX_CHARS`` characters, and (with ``window_inputs``) only within ``budget``.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from .ast_utils import faithful_sql, select_sources
from .window_inputs import _UNSTABLE, _reachable_through_derived_tables, _reader_rank

MAX_PROOFS = 4
MAX_CHARS = 4000
_BODY_ARGS = frozenset({"expressions", "from_", "from", "joins", "where", "group", "having", "distinct", "order", "limit", "kind"})


def _own(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select


def _kept_whole(select: exp.Select) -> bool:
    if select.args.get("group") or select.args.get("having") or select.args.get("distinct") or select.args.get("limit"):
        return True
    return any(_own(call, select) for call in select.find_all(exp.AggFunc))


def _candidate(select: exp.Select) -> bool:
    holder = select.parent
    if not isinstance(holder, exp.Subquery) or not holder.alias or holder.args["alias"].args.get("columns"):
        return False
    if not _reachable_through_derived_tables(select) or not _kept_whole(select):
        return False
    if any(value for key, value in select.args.items() if key not in _BODY_ARGS):
        return False
    if any(_own(w, select) for w in select.find_all(exp.Window)) or any(isinstance(n, _UNSTABLE) for n in select.walk()):
        return False
    reader = holder.find_ancestor(exp.Select)
    if reader is None or any(isinstance(s, exp.Star) and not isinstance(s.parent, exp.Count) and _own(s, reader) for s in reader.find_all(exp.Star)):
        return False
    if any(join.args.get("using") or join.args.get("method") for join in reader.args.get("joins") or []):
        return False
    if _positional_key(select):
        return False  # the items are reordered below, and a position would then name another one
    names = [(item.alias_or_name or "").lower() for item in select.expressions]
    if "" in names or len(set(names)) != len(names):
        return False
    if any(isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)) for item in select.expressions):
        return False
    declared = {n.alias_or_name.lower() for n in select.walk() if isinstance(n, (exp.Table, exp.Subquery)) and n.alias_or_name}
    in_list = {id(c) for item in select.expressions for c in item.find_all(exp.Column)}
    for column in select.find_all(exp.Column):
        if isinstance(column.this, exp.Star) or column.args.get("db") or (column.table and column.table.lower() not in declared):
            return False
        if not column.table and id(column) not in in_list and column.name.lower() in names and _own(column, select):
            item = select.expressions[names.index(column.name.lower())]
            if not (isinstance(item, exp.Column) and not item.table):
                return False  # GROUP BY / ORDER BY / HAVING may be reading an output by its alias
    return True


def _positional_key(select: exp.Select) -> bool:
    """Whether the select's own ``GROUP BY`` or ``ORDER BY`` names an output by its position (``GROUP BY 1``)."""

    group, order = select.args.get("group"), select.args.get("order")
    for node in group.walk() if group is not None else ():
        if isinstance(node, exp.Literal) and not node.is_string and isinstance(node.parent, (exp.Group, exp.Rollup, exp.Cube, exp.GroupingSets, exp.Tuple, exp.Paren)):
            return True
    for key in order.expressions if order is not None else ():
        key = key.this if isinstance(key, exp.Ordered) else key
        if isinstance(key, exp.Literal) and not key.is_string:
            return True
    return False


def _candidates(tree: exp.Expression) -> list[exp.Select]:
    return sorted((s for s in tree.find_all(exp.Select) if _candidate(s)), key=lambda s: -s.depth)  # innermost first


def _in_reader_order(select: exp.Select) -> exp.Select:
    """A copy of ``select`` with its items in the order its reader first uses them (unread ones last, as written)."""

    rank = _reader_rank(select)
    copy = select.copy()
    items = list(copy.expressions)
    order = sorted(range(len(items)), key=lambda i: (rank.get(items[i].alias_or_name.lower(), len(rank)), i))
    copy.set("expressions", [items[i] for i in order])
    return copy


def _renamed(body: exp.Select, names: list[str]) -> exp.Select:
    """A copy of ``body`` whose outputs carry ``names``, position by position."""

    copy = body.copy()
    items = []
    for item, name in zip(copy.expressions, names):
        value = item.this if isinstance(item, exp.Alias) else item
        items.append(exp.alias_(value, name, quoted=False))
    copy.set("expressions", items)
    return copy


def _signature(select: exp.Select) -> tuple:
    return len(select.expressions), tuple(sorted(t.name.lower() for t in select.find_all(exp.Table)))


def unify(left_sql: str, right_sql: str, *, dialect: str, prove) -> tuple[str, str] | None:
    """The two queries with provably equal derived tables written as one text, or None when nothing was unified.

    ``prove(a, b)`` says whether two queries provably return the same bag of rows, column by column.
    """

    try:
        left, right = sqlglot.parse_one(left_sql, read=dialect), sqlglot.parse_one(right_sql, read=dialect)
    except sqlglot.errors.SqlglotError:
        return None
    if any(tree.find(exp.With) is not None or tree.find(exp.CTE) is not None for tree in (left, right)):
        return None
    proofs = 0
    changed = False
    installed: dict[int, exp.Select] = {}  # left bodies already put in reader order, by the select that replaced them
    for right_select in _candidates(right):
        right_body = _in_reader_order(right_select)
        right_text = None
        for left_select in _candidates(left):
            if _signature(left_select) != _signature(right_select):
                continue
            left_body = installed.get(id(left_select)) or _in_reader_order(left_select)
            try:
                left_text = faithful_sql(left_body, dialect)
                right_text = right_text or faithful_sql(right_body, dialect)
            except Exception:  # noqa: BLE001 - a body that cannot be printed faithfully is not compared
                continue
            if max(len(left_text), len(right_text)) > MAX_CHARS:
                continue
            if left_text != right_text:
                if proofs >= MAX_PROOFS:
                    continue
                proofs += 1
                if not prove(left_text, right_text):
                    continue
            if id(left_select) not in installed:
                new_left = left_body.copy()
                left_select.replace(new_left)
                installed[id(new_left)] = new_left
            right_select.replace(_renamed(left_body, [item.alias_or_name for item in right_body.expressions]))
            changed = True
            break
    if not changed:
        return None
    try:
        return faithful_sql(left, dialect), faithful_sql(right, dialect)
    except Exception:  # noqa: BLE001
        return None
