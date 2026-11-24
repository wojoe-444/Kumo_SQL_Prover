"""A column list on a ``WITH`` name, written as aliases on the outputs it renames.

    WITH t (id) AS (SELECT gold FROM rounds UNION ALL SELECT silver FROM rounds) SELECT t.id FROM t

is

    WITH t AS (SELECT gold AS id FROM rounds UNION ALL SELECT silver FROM rounds) SELECT t.id FROM t

The prover reads a ``WITH`` query by its own output names and did not read the list, so it declined every query
that used a listed name ("t.id is not a column of t").

**Why it holds.** A column list on a ``WITH`` name renames the outputs of its query by position, and the list must
name every output. A set operation takes its output names from its first operand. So giving each output of the
first select the listed name as its alias, and dropping the list, leaves a ``WITH`` query with the same rows whose
outputs have the listed names. An alias changes no row and no value; what could change is a reference to an output
by its old name from inside the query, so the cases where one can exist are not read (below).

**What is read.** A ``WITH`` entry with a column list (not ``RECURSIVE``), whose query is a ``SELECT`` or a set
operation whose first operand, through any parentheses, is a ``SELECT``, when

* the list has exactly one name per output of that select, the names differ from one another, and no output is a
  star;
* neither that select nor any set operation it is the first operand of has an ``ORDER BY`` or a ``LIMIT`` (an
  ``ORDER BY`` names outputs, and the one after a set operation names those of the first operand);
* no unqualified column anywhere in that select has a name that an alias here gives, or that an alias written in
  the query loses (engines differ on where such a name means an output and where a column). A plain column that
  loses its output name is still found as the column it is;
* no listed name is the name another output of that select has (``t (a, b) AS (SELECT x.b, x.a FROM x)``).

**A list that is not read is refused.** The solver layer looked a ``WITH`` name up by the output names of its query
and did not look at the list. A listed name that the query does not give was then "not a column of t", and the
pair was declined. A listed name that the query gives to another output was read as that output:
``WITH t (a, b) AS (SELECT b, a FROM x) SELECT a FROM t`` was proved equal to ``SELECT a FROM x``, and the true
pair got a counterexample (Kumo_SQL_Prover issue 67). ``smt_equivalence`` now asks :func:`listed_body` for the query
of an entry that has a list: the query as it is when the list gives every output the name it already has, the
query with the listed names as aliases when the list is read, and otherwise the pair is declined.

Every dialect: this is how standard SQL names the columns of a ``WITH`` query. The list on a derived table
(``FROM (..) AS d (a, b)``) is read by ``ast_utils.expand_alias_columns``, which already looks through a set
operation.

A pre-pass of the thirteenth hardest-cases batch (``shape_fallback.THIRTEENTH_BATCH``); ``tests/test_listed_cte_names.py``
replays the rewrite on DuckDB.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import with_clause

def _first_select(query: exp.Expression) -> exp.Select | None:
    """The select whose outputs name the columns of ``query``, or None when an ``ORDER BY`` or ``LIMIT`` on the way reads them."""

    while True:
        if isinstance(query, exp.Subquery):
            if any(v for key, v in query.args.items() if key != "this"):
                return None
            query = query.this
        elif isinstance(query, exp.SetOperation):
            if query.args.get("order") or query.args.get("limit") or query.args.get("offset") or query.args.get("with_") or query.args.get("with"):
                return None
            query = query.this
        else:
            break
    if type(query) is not exp.Select or query.args.get("order") or query.args.get("limit") or query.args.get("offset"):
        return None
    return query


def _plan(cte: exp.CTE) -> tuple[exp.Select, list[exp.Identifier]] | None:
    """The select whose outputs take the listed names, and those names, or None when the entry is not read."""

    alias = cte.args.get("alias")
    listed = [column for column in (alias.args.get("columns") or [])] if isinstance(alias, exp.TableAlias) else []
    if not listed or not all(isinstance(column, exp.Identifier) and column.name for column in listed):
        return None
    names = [column.name if column.args.get("quoted") else column.name.lower() for column in listed]
    select = _first_select(cte.this)
    if select is None or len(names) != len(select.expressions) or len({name.lower() for name in names}) != len(names):
        return None
    own = [(item.alias_or_name or "").lower() for item in select.expressions]
    if any(name.lower() != old and name.lower() in own for name, old in zip(names, own)):
        # The list gives an output the name another output has (``t (a, b) AS (SELECT x.b, x.a ..)``). The checks
        # made on a query as it is written look a ``WITH`` column up by the names of its query: not read.
        return None
    touched = set()
    for item, name in zip(select.expressions, names):
        if isinstance(item, exp.Star) or item.is_star:
            return None
        if isinstance(item, exp.Alias) and not isinstance(item.args.get("alias"), exp.Identifier):
            return None
        old = (item.alias_or_name or "").lower()
        if old != name.lower():
            touched.add(name.lower())  # a name the alias gives
            if isinstance(item, exp.Alias):
                touched.add(old)  # a name it takes away; a plain column that loses its output name is still that column
    if touched and any(not column.table and column.name.lower() in touched for column in select.find_all(exp.Column)):
        return None
    return select, [column.copy() if column.args.get("quoted") else exp.to_identifier(name) for column, name in zip(listed, names)]


def _named_as_listed(cte: exp.CTE) -> bool:
    """Whether the column list of ``cte`` gives every output of its query the name it already has."""

    alias = cte.args.get("alias")
    listed = list(alias.args.get("columns") or []) if isinstance(alias, exp.TableAlias) else []
    query = cte.this
    while isinstance(query, (exp.Subquery, exp.SetOperation)):
        query = query.this  # a set operation takes its output names from its first operand
    if type(query) is not exp.Select or len(listed) != len(query.expressions):
        return False
    for item, column in zip(query.expressions, listed):
        if isinstance(item, exp.Star) or item.is_star or not isinstance(column, exp.Identifier):
            return False
        if not item.alias_or_name or item.alias_or_name.lower() != column.name.lower():
            return False
    return True


def renames_an_output(tree: exp.Expression) -> bool:
    """Whether some ``WITH`` column list in ``tree`` gives an output another name than the one its query gives it."""

    for cte in tree.find_all(exp.CTE):
        alias = cte.args.get("alias")
        if isinstance(alias, exp.TableAlias) and alias.args.get("columns") and not _named_as_listed(cte):
            return True
    return False


def listed_body(cte: exp.CTE) -> exp.Expression | None:
    """The query of a ``WITH`` entry that has a column list, with its outputs named as listed; None when the list is
    not read (module doc).

    For a reader that looks a ``WITH`` name up by the output names of its query: read without its list, an entry
    ``t (a, b) AS (SELECT b, a FROM x)`` gives ``t.a`` the values of ``x.a``, which is the other column.
    """

    if _named_as_listed(cte):
        return cte.this
    copy = cte.copy()
    plan = _plan(copy)
    if plan is None:
        return None
    select, names = plan
    select.set("expressions", [exp.Alias(this=item.unalias(), alias=name) for item, name in zip(list(select.expressions), names)])
    return copy.this


def name_listed_cte_outputs(tree: exp.Expression) -> tuple[exp.Expression, bool]:
    """``tree`` with each ``WITH`` column list written as aliases (module doc); whether anything changed."""

    if not any(isinstance(cte.args.get("alias"), exp.TableAlias) and cte.args["alias"].args.get("columns") for cte in tree.find_all(exp.CTE)):
        return tree, False
    result, changed = tree.copy(), False
    for cte in list(result.find_all(exp.CTE)):
        clause = cte.parent
        if not isinstance(clause, exp.With) or clause.args.get("recursive"):
            continue
        plan = _plan(cte)
        if plan is None:
            continue
        select, names = plan
        select.set("expressions", [exp.Alias(this=item.unalias(), alias=name) for item, name in zip(list(select.expressions), names)])
        cte.args["alias"].set("columns", None)
        changed = True
    return (result, True) if changed else (tree, False)
