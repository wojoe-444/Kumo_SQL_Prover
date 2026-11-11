"""A quoted name that needs no quotes is the unquoted name.

    SELECT "t0"."title" FROM "posts" AS "t0"    is    SELECT t0.title FROM posts AS t0

PostgreSQL folds an unquoted identifier to lower case and takes a quoted one as written, so a quoted identifier
made only of lower-case letters, digits and underscores (not starting with a digit) names exactly what the same
characters name without quotes: the same table, column, alias or output name. DuckDB compares identifiers without
regard to case, quoted or not, so the same holds there. Calcite quotes every name it prints, and the prover matches
some relations by their text (a select with a window function, a derived table kept whole), where ``"t"."id"`` and
``t.id`` were two different texts.

A name is unquoted only when (it keeps its quotes otherwise):

* the dialect is PostgreSQL or DuckDB;
* it matches ``[a-z_][a-z0-9_]*``: an upper-case letter, a space, a dot or any other character keeps the quotes,
  because unquoted the name would be folded or would not be one name;
* it is not a reserved key word of PostgreSQL or DuckDB (``"order"``, ``"select"``, ``"user"`` keep their quotes),
  and written without quotes as a table, a table alias, a column and an output alias, sqlglot's parser for the
  dialect reads it back as that same name.

Nothing else changes: the tree is the same query with a flag cleared on some identifiers.
"""

from __future__ import annotations

import functools
import re

import sqlglot
from sqlglot import exp

_DIALECTS = ("postgres", "duckdb")
_PLAIN = re.compile(r"[a-z_][a-z0-9_]*")
# PostgreSQL's reserved key words (appendix C of its manual: "reserved" and "reserved, can be function or type"),
# with the few more DuckDB reserves. sqlglot's parser is more lenient than either engine and reads some of these
# as names, so the list is checked before its answer is.
_RESERVED = frozenset(
    """all analyse analyze and any array as asc asymmetric authorization between binary both case cast check collate
    collation column concurrently constraint create cross current_catalog current_date current_role current_schema
    current_time current_timestamp current_user default deferrable desc distinct do else end except false fetch for
    foreign freeze from full grant group having ilike in initially inner intersect into is isnull join lateral leading
    left like limit localtime localtimestamp natural not notnull null offset on only or order outer overlaps placing
    primary references returning right select session_user similar some symmetric system_user table tablesample then
    to trailing true union unique user using variadic verbose when where window with
    anti asof describe pivot positional qualify semi show summarize unpivot""".split()
)


@functools.lru_cache(maxsize=8192)
def _reads_back(name: str, dialect: str) -> bool:
    """Whether ``name`` without quotes parses as the identifier ``name`` in every position a name is written."""

    if name in _RESERVED:
        return False
    try:
        bare = sqlglot.parse_one(f"SELECT {name}.{name} AS {name} FROM {name} AS {name} WHERE {name}.{name} = {name}", read=dialect)
        quoted = sqlglot.parse_one(f'SELECT "{name}"."{name}" AS "{name}" FROM "{name}" AS "{name}" WHERE "{name}"."{name}" = "{name}"', read=dialect)
    except sqlglot.errors.SqlglotError:
        return False
    for identifier in quoted.find_all(exp.Identifier):
        identifier.set("quoted", False)
    return bare == quoted


def unquote_plain_names(tree: exp.Expression, dialect: str) -> exp.Expression:
    """``tree`` with the quotes dropped from every name that means the same without them (in place)."""

    if dialect not in _DIALECTS:
        return tree
    for identifier in tree.find_all(exp.Identifier):
        name = identifier.this
        if identifier.args.get("quoted") and isinstance(name, str) and _PLAIN.fullmatch(name) and _reads_back(name, dialect):
            identifier.set("quoted", False)
    return tree
