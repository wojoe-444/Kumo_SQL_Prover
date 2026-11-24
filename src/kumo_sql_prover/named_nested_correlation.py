"""A nested test of the outer row, read when its query names its columns bare.

    SELECT p.id, p.name FROM product AS p
    WHERE p.id IN (SELECT pid FROM sales WHERE NOT EXISTS (SELECT 1 FROM sales AS t WHERE day > 9 AND pid = sales.pid))

``nested_correlation`` reads the inner ``NOT EXISTS`` as a test of ``p.id``, asked once beside the ``IN``:

    WHERE p.id IN (SELECT sales.pid FROM sales) AND NOT EXISTS (SELECT 1 FROM sales AS t WHERE t.day > 9 AND t.pid = p.id)

but only when every column in the subquery names its source, so that replacing a read cannot change the row it
means. A query written with bare names (``SELECT pid FROM sales WHERE ..``) was left as it was, and "has a sale and
none outside the window" written as a join to ``SELECT DISTINCT pid FROM sales WHERE pid NOT IN (..)`` stayed
unproved against the bounds of a group (hardest-cases batch 30, LeetCode 1084).

This rule gives the bare columns of the nested one-table queries their table first
(``keyed_semijoin.name_nested_sources``, after ``rename_nested_sources`` has given a nested source named like an
outer one a name of its own), and then asks ``nested_correlation``. Both steps are the ones other rules already
take: SQL looks a bare name up in the innermost query that has a source with a column of that name, a nested query
over one declared table whose column list has the name is that query, and only the select list and the ``WHERE``
are read (a ``GROUP BY``, ``HAVING`` or ``ORDER BY`` may read an output name first). A bare name the table does not
have stays bare, and ``nested_correlation`` then declines as before.

It also moves the test out when it is the subquery's whole ``WHERE``, which ``nested_correlation`` leaves in place:

    x IN (SELECT s.c FROM s WHERE b)   is   x IN (SELECT s.c FROM s) AND b      when b reads no row of s

As a top-level conjunct of a ``WHERE`` the ``IN`` is TRUE exactly when some row of ``s`` has ``s.c = x`` and ``b`` is
TRUE, and ``b`` is the same for every row: so exactly when ``b`` is TRUE and some row has ``s.c = x``. ``EXISTS``
the same way. (When neither is TRUE the row is dropped, whether the test is FALSE or NULL.)

Nothing is read when ``nested_correlation`` gives the same select by itself. It runs in the last step of batch 27
only, so a pair an earlier step decides is decided as it was.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import FROM_KEY
from .keyed_semijoin import name_nested_sources, rename_nested_sources
from .nested_correlation import flatten_nested_correlation

NAME = "named_nested_correlation"


def flatten_named_correlation(select: exp.Select, types: dict | None, schema: dict | None) -> exp.Expression | None:
    """``select`` with a nested test of its own row asked once outside, when naming the bare columns of its nested
    one-table queries lets ``nested_correlation`` read it; else ``None``. The input is not changed."""

    where = select.args.get("where")
    if not schema or where is None or where.find(exp.Subquery, exp.Exists) is None:
        return None
    from_ = select.args.get(FROM_KEY)
    if from_ is None:
        return None
    taken = {(source.alias_or_name or "").lower() for source in (from_.this, *(join.this for join in select.args.get("joins") or []))}
    named = name_nested_sources(rename_nested_sources(select, taken - {""}), schema)
    read = flatten_nested_correlation(named, types, whole=True)
    if read is None:
        return None
    plain = flatten_nested_correlation(select, types)
    if plain is not None and plain.sql() == read.sql():
        return None  # nested_correlation reads the select this way by itself
    return read
