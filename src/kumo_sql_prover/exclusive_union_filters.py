"""A ``UNION ALL`` of one query under filters that exclude each other is that query under their ``OR``.

    SELECT l.x FROM l JOIN p ON p.k = l.pk WHERE p.b = 'A' AND l.q < 5
    UNION ALL
    SELECT l.x FROM l JOIN p ON p.k = l.pk WHERE p.b = 'B' AND l.q > 9

is

    SELECT l.x FROM l JOIN p ON p.k = l.pk WHERE (p.b = 'A' AND l.q < 5) OR (p.b = 'B' AND l.q > 9)

(TPC-H Q19 and the rewrites that split its three-way ``OR`` into branches). The prover compares a ``UNION ALL``
branch by branch, so the one-block spelling never met the three-block one.

Why this is exact. Every branch returns, for each row of the same joined sources that passes its filter, the
same output row. A joined row passes at most one of the filters, because two branches each hold a conjunct
``c = literal`` on the same column with certainly different literals; so the union returns it once if it passes
some filter and never otherwise. ``A OR B`` is TRUE exactly when ``A`` is TRUE or ``B`` is TRUE, so the single
select keeps the same joined rows, once each.

Conditions, all checked:

* a ``UNION ALL`` (through nested ``UNION ALL`` nodes; no ``UNION``, ``INTERSECT``, ``EXCEPT``, ``BY NAME``, and no
  ``ORDER BY`` or ``LIMIT`` on the union);
* every branch is a select with only a select list, sources, joins and ``WHERE`` (no ``DISTINCT``, grouping,
  window, tail), no aggregate and no row-making function in its select list, and nothing volatile anywhere;
* the branches are the same text once their ``WHERE`` is removed, so each name binds the same way in each; or,
  when every join of every branch is a comma, ``CROSS JOIN`` or inner ``JOIN .. ON`` whose condition holds no
  nested query and names the source of each column, the same text once the join conditions are removed too. An
  inner join condition is a filter on the joined row, so each branch's filter is then its join conditions and
  its ``WHERE`` together (TPC-H Q7 split by nation pair, with each nation's name tested in its ``ON``); the single
  select lists the sources with every shared conjunct once and the ``OR`` of what is left of each branch;
* every two branches differ on a column: both have a top-level conjunct ``column = literal`` on the same column
  text, with literals that are certainly different (``disjoint_union._tag``: integers as numbers; strings with
  case, spaces and punctuation ignored, so no collation or padding calls them equal).
"""

from __future__ import annotations

from sqlglot import exp

from .disjoint_union import _one_row_per_input, _own_aggregate, _tag, _unwrap_branch
from .unmatched_outer_join import _conjuncts, _unwrap

_UNION_ARGS = frozenset({"this", "expression", "distinct"})
_BRANCH_ARGS = frozenset({"expressions", "from_", "from", "joins", "where"})


def _all_branches(node: exp.Expression) -> list[exp.Select] | None:
    """The selects under a ``UNION ALL`` tree; ``None`` when a node under it is another set operation or a ``UNION``."""

    node = _unwrap_branch(node)
    if type(node) is exp.Select:
        return [node]
    if type(node) is not exp.Union or node.args.get("distinct") is not False:
        return None
    if any(value for key, value in node.args.items() if key not in _UNION_ARGS):
        return None
    left, right = _all_branches(node.this), _all_branches(node.expression)
    return None if left is None or right is None else left + right


def _pins(where: exp.Expression) -> dict[str, tuple[str, object]]:
    """``{column text: literal tag}`` for the top-level conjuncts ``column = literal`` of a filter."""

    pins: dict[str, tuple[str, object]] = {}
    for part in _conjuncts(where):
        if type(part) is not exp.EQ:
            continue
        for column, literal in ((part.this, part.expression), (part.expression, part.this)):
            column, literal = _unwrap(column), _unwrap(literal)
            if type(column) is exp.Column and type(literal) is exp.Literal:
                tag = _tag(literal)
                text = column.sql().lower()
                if tag is not None and text not in pins:
                    pins[text] = tag
    return pins


def _exclusive(a: dict, b: dict) -> bool:
    return any(column in b and a[column][0] == b[column][0] and a[column][1] != b[column][1] for column in a)


def _frame(select: exp.Select) -> str:
    copy = select.copy()
    copy.set("where", None)
    return copy.sql()


def _inner_only(select: exp.Select) -> bool:
    """Whether every join is a comma, ``CROSS JOIN`` or inner ``JOIN .. ON`` whose condition names the source of
    each column it reads, so the condition means the same in ``WHERE``."""

    for join in select.args.get("joins") or []:
        if any(value for key, value in join.args.items() if key not in ("this", "kind", "on")):
            return False
        if (join.args.get("kind") or "").upper() not in ("", "INNER", "CROSS") or isinstance(join.this, exp.Lateral):
            return False
        on = join.args.get("on")
        if on is not None and any(isinstance(n, (exp.Query, exp.Subquery)) or (type(n) is exp.Column and not n.table) for n in on.walk()):
            return False
    return True


def _flat(select: exp.Select) -> tuple[exp.Select, list[exp.Expression]]:
    """``select`` with its join conditions and ``WHERE`` taken out, and their conjuncts in order."""

    copy = select.copy()
    parts: list[exp.Expression] = []
    for join in copy.args.get("joins") or []:
        parts.extend(_conjuncts(join.args.get("on")))
        join.set("on", None)
        join.set("kind", None)
    where = copy.args.get("where")
    parts.extend(_conjuncts(where.this if where is not None else None))
    copy.set("where", None)
    return copy, parts


def _and(parts: list[exp.Expression]) -> exp.Expression:
    test = None
    for part in parts:
        part = part.copy()
        test = part if test is None else exp.And(this=test, expression=part)
    return test


def _merged_flat(branches: list[exp.Select]) -> exp.Select | None:
    """One select for branches that are the same inner join under conditions that exclude each other, or ``None``."""

    if not all(_inner_only(b) for b in branches):
        return None
    flats = [_flat(b) for b in branches]
    if len({frame.sql() for frame, _ in flats}) != 1 or any(not parts for _, parts in flats):
        return None
    pins = [_pins(_and(parts)) for _, parts in flats]
    if not all(_exclusive(pins[i], pins[j]) for i in range(len(pins)) for j in range(i + 1, len(pins))):
        return None
    texts = [[part.sql() for part in parts] for _, parts in flats]
    shared = [part for part, text in zip(flats[0][1], texts[0]) if all(text in other for other in texts[1:])]
    shared_texts = {part.sql() for part in shared}
    rests = [[part for part in parts if part.sql() not in shared_texts] for _, parts in flats]
    if any(not rest for rest in rests):
        return None
    test = None
    for rest in rests:
        part = exp.Paren(this=_and(rest))
        test = part if test is None else exp.Or(this=test, expression=part)
    merged = flats[0][0]
    merged.set("where", exp.Where(this=_and([*shared, exp.Paren(this=test)])))
    return merged


def merge_exclusive_unions(tree: exp.Expression) -> exp.Expression:
    """``tree`` with every such ``UNION ALL`` written as one select (a pre-pass of ``normalize``)."""

    from .counted_membership import _unstable

    for union in list(tree.find_all(exp.Union)):
        if isinstance(union.parent, exp.Union) or (isinstance(union.parent, exp.Subquery) and not union.parent.alias and isinstance(union.parent.parent, exp.Union)):
            continue  # read from the top of its union tree
        root = union
        while root is not None and root is not tree:
            root = root.parent
        if root is None:
            continue  # replaced along with an enclosing union
        branches = _all_branches(union)
        if branches is None or len(branches) < 2:
            continue
        if any(type(b) is not exp.Select or any(value for key, value in b.args.items() if key not in _BRANCH_ARGS) for b in branches):
            continue
        if any(_own_aggregate(b) or not _one_row_per_input(b) or _unstable(b) for b in branches):
            continue
        if any(isinstance(n, exp.Window) for b in branches for n in b.walk()):
            continue
        merged = None
        if all(b.args.get("where") is not None for b in branches) and len({_frame(b) for b in branches}) == 1:
            pins = [_pins(b.args["where"].this) for b in branches]
            if all(_exclusive(pins[i], pins[j]) for i in range(len(pins)) for j in range(i + 1, len(pins))):
                merged = branches[0].copy()
                test = None
                for branch in branches:
                    part = exp.Paren(this=branch.args["where"].this.copy())
                    test = part if test is None else exp.Or(this=test, expression=part)
                merged.set("where", exp.Where(this=test))
        if merged is None:
            merged = _merged_flat(branches)
        if merged is None:
            continue
        if union is tree:
            tree = merged
        else:
            union.replace(merged)
    return tree
