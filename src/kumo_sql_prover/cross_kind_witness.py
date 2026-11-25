"""No counterexample from a model that sets a text against a number for one column across the two queries.

    SELECT id FROM t WHERE c = '1'        SELECT id FROM t WHERE c = 1

The provers hold a value as a number, a text or a truth value and call values of different kinds unequal. For a column
the schema gives no numeric or text type, the model of the first query makes ``c`` the text ``'1'`` to pass its test
and the model of the second makes it the number 1, and a database with the number 1 in ``c`` came back as a
counterexample: no row from the first query, one from the second (Kumo_SQL_Prover issue 60). No engine reads it so.
MySQL compares a number with a text as numbers, so the number 1 passes both tests; DuckDB and PostgreSQL convert the
text to the column's type. Where the pair does differ (a text column that holds ``'1.0'`` in MySQL), the database
shown is not the one that shows it.

Inside one query the same clash (``c = 'abc' AND c = 0``) is declined before any proof is tried
(``string_number_compare.problem``). Across the two queries nothing is wrong with a proof: each query compares the
column with one kind only, and no step of a proof sets the text of one query against the number of the other. Only
the counterexample reads the two kinds against each other, so only the counterexample is withheld: a ``NOT_EQUIVALENT``
answer for such a pair becomes ``NOT_PROVEN``.

**What counts.** A value compared with a string literal in one query and with a number literal in the other, in a
comparison, ``BETWEEN``, an ``IN`` list, a simple ``CASE`` or ``NULLIF``, where the value is

* a column that has no declared numeric or text type (none at all, a ``BOOL``, a date), or a value that hands such a
  column on (``MIN(c)``, ``COALESCE(c, d)``); columns are matched by name alone, whatever source they are read from,
  since the two queries name their sources independently;
* or another expression whose kind depends on such columns, matched by its text without source names.

A column declared numeric or text is not counted: the solver reads its comparison with the other kind as an unknown
predicate and takes no counterexample from that already.

**The cost.** SQLite compares a column of no declared type holding the integer 1 with the text ``'1'`` as unequal, so
for an untyped column there the withheld database was right. The schema does not say whether a type was left out or
never declared, and unknown is better than wrong.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from . import string_number_compare as snc

REASON = "a value is compared with a string in one query and with a number in the other (the engines convert one of them; the difference found is not a counterexample)"


def _bare(node: exp.Expression) -> str:
    """The text of ``node`` with every column written without its source."""

    copy = node.copy()
    for column in list(copy.find_all(exp.Column)):
        bare = exp.column(column.name)
        if column is copy:
            copy = bare
        else:
            column.replace(bare)
    return copy.sql().lower()


def literal_kinds(sql: str, dialect: str, types=None, schema=None) -> dict[str, set[str]]:
    """For each value of open kind in ``sql`` (see the module doc), the kinds of the literals it is compared with."""

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001 - the prover reports the parse error itself
        return {}
    kinds = snc._Kinds(tree, types, schema)
    found: dict[str, set[str]] = {}
    for left, right in snc._pairs(tree):
        for value, other in ((left, right), (right, left)):
            other = snc._unwrap(other)
            if isinstance(other, exp.Neg) and isinstance(other.this, exp.Literal) and not other.this.is_string:
                other = other.this
            if not isinstance(other, exp.Literal):
                continue
            carried = snc._carried(value)
            if isinstance(carried, exp.Literal) or kinds.fixed(carried) not in (None, "date"):
                continue
            found.setdefault(_bare(carried), set()).add("string" if other.is_string else "number")
    return found


def clash(left_sql: str, right_sql: str, dialect: str, types=None, schema=None) -> bool:
    """Whether some value of open kind meets a string literal in one query and a number literal in the other."""

    if "'" not in left_sql + right_sql and '"' not in left_sql + right_sql:
        return False
    left, right = (literal_kinds(sql, dialect, types, schema) for sql in (left_sql, right_sql))
    for name in left.keys() & right.keys():
        if (left[name] | right[name]) == {"string", "number"} and left[name] != right[name]:
            return True
    return False


def without_cross_kind_counterexample(result, left_sql: str, right_sql: str, kwargs: dict):
    """``result`` unless it is a counterexample for a pair with such a clash, which is ``NOT_PROVEN``."""

    from .smt_equivalence import SmtEquivalenceResult, SmtStatus

    if result.status is not SmtStatus.NOT_EQUIVALENT:
        return result
    dialect = kwargs.get("dialect", "bigquery") or "bigquery"
    if not clash(left_sql, right_sql, dialect, kwargs.get("types"), kwargs.get("schema")):
        return result
    return SmtEquivalenceResult(SmtStatus.NOT_PROVEN, REASON, assumptions=result.assumptions)
