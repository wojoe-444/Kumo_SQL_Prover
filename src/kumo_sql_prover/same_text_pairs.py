"""A query is equal to itself: a pair whose two queries are one text is proved, whatever the prover can read of it.

The provers show two queries equal by reading both. A query they cannot read (a correlated scalar subquery over a
windowed ``WITH`` table, say) left even this pair unproved:

    WITH ranked AS (SELECT .., ROW_NUMBER() OVER (..) AS rn FROM ..) SELECT .., (SELECT AVG(x) FROM ranked AS r WHERE r.rn = f.rn) FROM ranked AS f
    WITH ranked AS (SELECT .., ROW_NUMBER() OVER (..) AS rn FROM ..) SELECT .., (SELECT AVG(x) FROM ranked AS r WHERE r.rn = f.rn) FROM ranked AS f;

A rewriting tool that finds nothing to change returns its input, so benchmarks of rewrites hold such pairs.

``same_text(left, right, dialect)`` is true when the two texts give the same tokens in the pair's dialect: the same
kinds of token with the same spelling, letter for letter, and the same comments on the same tokens. Only white space
and semicolons at the end may differ. The comparison is of tokens and not of parsed trees, so nothing depends on how
a parser groups them; and it is of spelling and not of meaning, so two names that differ in case are two texts (a
table name's case matters on some servers). Comments are compared because one kind of them is not a comment
(MySQL runs ``/*!50000 .. */``). The text must also parse, as one statement.

``algebraic_equivalence`` asks this only for a pair every other step left unproved (hardest-cases batch 28), and
passes the proof through the checks every proof passes: a text with a call that can give another value the next time
it is evaluated (``RAND()``, ``nextval``) is not proved (``smt_equivalence.without_volatile_proof``), and one that
reads the clock is proved under a condition (``statement_values``). What the proof assumes is that one statement
gives one result on one database (``SAME_TEXT_ASSUMPTION``): rows tied in an ``ORDER BY`` that a ``LIMIT`` or a
window cuts, and a floating-point sum, are the server's choice, and the proof says they are the same choice twice.
That is what the provers already assume of two windows or two ``LIMIT`` subqueries with one text
(``WINDOW_SOURCE_ASSUMPTION``, ``LIMIT_SOURCE_ASSUMPTION``).
"""

from __future__ import annotations

import sqlglot
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType

NAME = "same_text_pairs"
SAME_TEXT_ASSUMPTION = "the two queries are one text, and one statement gives one result on one database (ties and the order of a floating-point sum resolve alike)"
REASON = "the two queries are the same text"


def _tokens(sql: str, dialect: str) -> list[tuple] | None:
    try:
        tokens = list(Dialect.get_or_raise(dialect).tokenize(sql))
    except Exception:  # noqa: BLE001 - a text that cannot be tokenized is not compared
        return None
    while tokens and tokens[-1].token_type is TokenType.SEMICOLON and not tokens[-1].comments:
        tokens.pop()
    if not tokens or any(token.token_type is TokenType.SEMICOLON for token in tokens):
        return None  # nothing, or more than one statement
    return [(token.token_type, token.text, tuple(token.comments or ())) for token in tokens]


def same_text(left_sql: str, right_sql: str, dialect: str | None) -> bool:
    """Whether the two texts are one query in ``dialect``: the same tokens and comments, and a text that parses."""

    dialect = dialect or "bigquery"
    left = _tokens(left_sql, dialect)
    if left is None or left != _tokens(right_sql, dialect):
        return False
    try:
        return len(sqlglot.parse(left_sql, read=dialect)) == 1 and isinstance(sqlglot.parse_one(left_sql, read=dialect), sqlglot.exp.Query)
    except Exception:  # noqa: BLE001 - a text that does not parse is not a query the prover speaks about
        return False
