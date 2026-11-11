"""Proofs that hold when both queries read the same statement time: ``NOW()``, ``CURRENT_DATE``, ``SESSION_USER()``.

A query that calls ``CURRENT_DATE`` is not a function of the tables alone: it also depends on when it runs. The
solver-based provers therefore never call two such queries equivalent (:mod:`kumo_sql_prover.volatile_calls`).
What can be proved is weaker and still useful: *run at the same statement time, the two queries return the same
rows on every database*. That is what a rewrite of a query with a date filter needs.

How it is done. Within one statement these calls return one value however often they are evaluated, so each is
a constant the prover knows nothing about. Before the provers see the pair, every such call is replaced by a
call to a function with no arguments that the provers have no definition for (``KQ_FIXED_1_CURRENT_DATE()``);
the same call gets the same name in both queries. The provers already read an unknown function as "equal
arguments, equal result", which for no arguments is exactly "one value". Nothing else about the value is
assumed: not its type, not that it is non-NULL, not that ``CURRENT_DATE`` is the date of ``CURRENT_TIMESTAMP``.
So a proof holds for every value the clock could have, and the result is reported as
``PROVEN_CONDITIONALLY`` with a :class:`~kumo_sql_prover.conditional_equivalence.Condition` naming the calls.
It is never ``PROVEN_EQUIVALENT`` and never counted as a proof.

What stays out:

* calls that may differ between two evaluations in one statement (``RAND()``, ``nextval``, ``clock_timestamp()``,
  MySQL's ``SYSDATE()``): still refused, by ``volatile_calls``;
* dialects whose engine does not promise one value for the statement, or where that was not checked. SQLite
  promises it only within one ``sqlite3_step`` (one result row; version 3.53.1 did keep the value across rows when
  tried), and SQL Server's ``GETDATE()`` was not checked, so both stay refused. ``DIALECTS`` lists the ones that
  are in;
* spellings whose meaning depends on the reader: a bare ``CURRENT_SCHEMA`` a dialect reads as a column,
  ``'now'::timestamp``, ``date('now')``, PostgreSQL's ``age(t)``;
* refutations. The solver would be free to pick the clock value, and a replay runs the two queries at two
  different times, so a difference found for such a pair is reported as ``NOT_PROVEN``, not ``NOT_EQUIVALENT``.
"""

from __future__ import annotations

import dataclasses
import re
from functools import lru_cache
from typing import Callable

import sqlglot
from sqlglot import exp

from . import volatile_calls

# Engines that document one value per statement (or per transaction) for these calls. BigQuery: "set at the start
# of the query statement". PostgreSQL and DuckDB: the start of the transaction (``statement_timestamp()``: of the
# statement). MySQL: "evaluated only once per query at the start of query execution". Spark and Databricks: "all
# calls within the same query return the same value". Trino and Presto: "as of the start of the query".
DIALECTS = frozenset({"bigquery", "postgres", "mysql", "duckdb", "spark", "databricks", "trino", "presto"})

SAME_TIME = "same_statement_time"
SAME_SESSION = "same_session"
KINDS = (SAME_TIME, SAME_SESSION)

PREFIX = "KQ_FIXED_"

_CLOCK_TYPES = {
    "CurrentDate": "CURRENT_DATE",
    "CurrentDatetime": "CURRENT_DATETIME",
    "CurrentTime": "CURRENT_TIME",
    "CurrentTimestamp": "CURRENT_TIMESTAMP",
    "Localtime": "LOCALTIME",
    "Localtimestamp": "LOCALTIMESTAMP",
    "UtcDate": "UTC_DATE",
    "UtcTime": "UTC_TIME",
    "UtcTimestamp": "UTC_TIMESTAMP",
}
_SESSION_TYPES = {
    "CurrentUser": "CURRENT_USER",
    "SessionUser": "SESSION_USER",
    "CurrentSchema": "CURRENT_SCHEMA",
    "CurrentDatabase": "CURRENT_DATABASE",
    "CurrentCatalog": "CURRENT_CATALOG",
    "CurrentVersion": "VERSION",
}
# Functions sqlglot has no class for in some dialect, by upper-case name. The value says which call it is: two
# spellings of one value share it (PostgreSQL, MySQL, DuckDB, Spark and Trino all define NOW() as CURRENT_TIMESTAMP).
_CLOCK_NAMES = {
    "NOW": "CURRENT_TIMESTAMP",
    "TRANSACTION_TIMESTAMP": "CURRENT_TIMESTAMP",
    "GET_CURRENT_TIMESTAMP": "CURRENT_TIMESTAMP",
    "CURRENT_TIMESTAMP": "CURRENT_TIMESTAMP",
    "CURRENT_DATE": "CURRENT_DATE",
    "CURRENT_TIME": "CURRENT_TIME",
    "CURRENT_DATETIME": "CURRENT_DATETIME",
    "STATEMENT_TIMESTAMP": "STATEMENT_TIMESTAMP",
    "CURDATE": "CURDATE",
    "CURTIME": "CURTIME",
    "TODAY": "TODAY",
    "GET_CURRENT_TIME": "GET_CURRENT_TIME",
    "LOCALTIME": "LOCALTIME",
    "LOCALTIMESTAMP": "LOCALTIMESTAMP",
    "UTC_DATE": "UTC_DATE",
    "UTC_TIME": "UTC_TIME",
    "UTC_TIMESTAMP": "UTC_TIMESTAMP",
    "UNIX_TIMESTAMP": "UNIX_TIMESTAMP",  # with no argument only: volatile_calls reads UNIX_TIMESTAMP(d) as a function of d
}
_SESSION_NAMES = {
    "USER": "USER",
    "CURRENT_USER": "CURRENT_USER",
    "SESSION_USER": "SESSION_USER",
    "SYSTEM_USER": "SYSTEM_USER",
    "CURRENT_SCHEMA": "CURRENT_SCHEMA",
    "CURRENT_DATABASE": "CURRENT_DATABASE",
    "CURRENT_CATALOG": "CURRENT_CATALOG",
    "DATABASE": "DATABASE",
    "CONNECTION_ID": "CONNECTION_ID",
    "VERSION": "VERSION",
}
_CONDITION_TEXT = {
    SAME_TIME: "both queries run at the same statement time, so each of these has one value in both: {calls}",
    SAME_SESSION: "both queries run in the same session, so each of these has one value in both: {calls}",
}
_NO_COUNTEREXAMPLE = (
    "the queries may differ, but a counterexample is not reported for queries that read the statement time or the session ({calls}): "
    "the difference may depend on when each query runs"
)


def fixed(node: exp.Expression, dialect: str | None) -> tuple[str, str] | None:
    """``(kind, call)`` when ``node`` is a call with one value for the whole statement, else ``None``.

    ``kind`` is ``SAME_TIME`` or ``SAME_SESSION``; ``call`` is a canonical text of the call
    (``CURRENT_TIMESTAMP(6)``), the same for two spellings of one value. Only calls ``volatile_calls`` reads as
    per statement qualify, written as a call or a keyword sqlglot knows, with literal arguments."""

    if dialect not in DIALECTS or volatile_calls.kind(node, dialect) != volatile_calls.PER_STATEMENT:
        return None
    type_name = type(node).__name__
    if isinstance(node, exp.Anonymous):
        if isinstance(node.parent, exp.Dot):
            return None  # owner.NOW(): somebody's own function
        name = str(node.this or "").upper()
        arguments = list(node.expressions)
        if name in _CLOCK_NAMES:
            found = (SAME_TIME, _CLOCK_NAMES[name])
        elif name in _SESSION_NAMES:
            found = (SAME_SESSION, _SESSION_NAMES[name])
        else:
            return None
    elif type_name in _CLOCK_TYPES or type_name in _SESSION_TYPES:
        found = (SAME_TIME, _CLOCK_TYPES[type_name]) if type_name in _CLOCK_TYPES else (SAME_SESSION, _SESSION_TYPES[type_name])
        arguments = []
        for key, value in node.args.items():
            if value is None or value is False or value == []:
                continue
            if key != "this" or not isinstance(value, exp.Expression):
                return None  # a flag that changes the meaning, such as Snowflake's SYSDATE read as CURRENT_TIMESTAMP
            arguments.append(value)
    else:
        return None
    if not all(isinstance(a, exp.Literal) for a in arguments):
        return None
    text = ", ".join(a.sql(dialect="bigquery") for a in arguments)
    return found[0], f"{found[1]}({text})" if text else found[1]


@dataclasses.dataclass(frozen=True)
class Reading:
    """A pair with its statement values replaced: the two texts to prove and what each name stands for."""

    left_sql: str
    right_sql: str
    names: tuple[tuple[str, str, str], ...]  # (function name, kind, call), ordered by name

    @property
    def calls(self) -> dict[str, tuple[str, ...]]:
        """The calls of each kind, sorted."""

        found: dict[str, list[str]] = {}
        for _, kind, call in self.names:
            found.setdefault(kind, []).append(call)
        return {kind: tuple(sorted(calls)) for kind, calls in found.items()}

    def conditions(self) -> tuple:
        from .conditional_equivalence import Condition

        return tuple(
            Condition(kind, "", calls, text=_CONDITION_TEXT[kind].format(calls=", ".join(calls)))
            for kind, calls in sorted(self.calls.items())
        )

    def restore(self, text: str) -> str:
        """``text`` with each stand-in name replaced by the call it stands for."""

        for name, _, call in sorted(self.names, key=lambda item: -len(item[0])):
            text = re.sub(re.escape(name) + r"(\(\))?", call, text, flags=re.IGNORECASE)
        return text


def _safe_name(index: int, call: str) -> str:
    name = f"{PREFIX}{index}_{re.sub(r'[^A-Z0-9]+', '_', call.upper()).strip('_')[:40]}"
    # The stand-in must not itself read as volatile (an argument such as a time zone may spell anything).
    return name if volatile_calls._name_kind(name, []) is None else f"{PREFIX}{index}"


_SPELLINGS = frozenset(_CLOCK_NAMES) | frozenset(_SESSION_NAMES) | frozenset(_CLOCK_TYPES.values()) | frozenset(_SESSION_TYPES.values())
# A text that mentions none of them is not parsed here at all: most pairs pay one regular expression.
_MENTIONED = re.compile(r"(?<![A-Za-z0-9_$])(?:%s)(?![A-Za-z0-9_$])" % "|".join(sorted(_SPELLINGS, key=len, reverse=True)), re.IGNORECASE)


def _spans(sql: str, dialect: str) -> list[tuple[int, int, tuple[str, str]]]:
    """``(start, end, (kind, call))`` of each statement value written in ``sql``, found in its tokens.

    The text is cut at token positions so that everything else in the query stays exactly as it was written.
    A word counts when it is not quoted, not part of a dotted name, stands alone or is followed by parentheses
    holding only literals, and reads as a statement value by itself."""

    from sqlglot.dialects.dialect import Dialect
    from sqlglot.tokens import TokenType

    tokens = Dialect.get_or_raise(dialect).tokenize(sql)
    found = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token.token_type in (TokenType.STRING, TokenType.IDENTIFIER) or token.text.upper() not in _SPELLINGS:
            continue
        if index >= 2 and tokens[index - 2].token_type is TokenType.DOT:
            continue
        last = index - 1
        if index < len(tokens) and tokens[index].token_type is TokenType.L_PAREN:
            last = index + 1
            while last < len(tokens) and tokens[last].token_type in (TokenType.STRING, TokenType.NUMBER, TokenType.COMMA):
                last += 1
            if last >= len(tokens) or tokens[last].token_type is not TokenType.R_PAREN:
                continue  # NOW(x), UNIX_TIMESTAMP(d): not a value of the statement alone
        if last + 1 < len(tokens) and tokens[last + 1].token_type is TokenType.DOT:
            continue
        text = sql[token.start : tokens[last].end + 1]
        try:
            alone = sqlglot.parse_one(f"SELECT {text}", read=dialect).expressions[0]
        except (sqlglot.errors.SqlglotError, IndexError):
            continue
        value = fixed(alone, dialect)
        if value is not None:
            found.append((token.start, tokens[last].end + 1, value))
            index = last + 1
    return found


@lru_cache(maxsize=256)
def reading(left_sql: str, right_sql: str, dialect: str = "bigquery") -> Reading | None:
    """The pair with its statement values replaced by stand-in functions, or ``None`` when there is nothing to
    replace or the pair cannot be read that way (another dialect, text sqlglot cannot read, more than one
    statement, a stand-in name already in the text, a call the token scan and the parser do not agree on).

    The replacement is made twice, in the text (``_spans``) and in the syntax tree, and the text is used only
    when it parses to the tree: every statement value the parser sees was replaced, and nothing else was."""

    from .ast_utils import _shape, canonical_negation, quiet_parser

    if dialect not in DIALECTS or PREFIX.lower() in left_sql.lower() or PREFIX.lower() in right_sql.lower():
        return None
    if not _MENTIONED.search(left_sql) and not _MENTIONED.search(right_sql):
        return None
    try:
        with quiet_parser():
            trees = []
            for sql in (left_sql, right_sql):
                statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
                if len(statements) != 1:
                    return None
                trees.append(statements[0])
            sites = [(node, fixed(node, dialect)) for tree in trees for node in list(tree.walk())]
            sites = [(node, found) for node, found in sites if found is not None]
            if not sites:
                return None
            spans = [_spans(sql, dialect) for sql in (left_sql, right_sql)]
            keys = sorted({found for _, found in sites} | {found for side in spans for _, _, found in side})
            names = {found: _safe_name(index, found[1]) for index, found in enumerate(keys, 1)}
            for node, found in sites:
                if node.parent is None:
                    return None
                node.replace(exp.Anonymous(this=names[found], expressions=[]))
            texts = []
            for sql, side, tree in zip((left_sql, right_sql), spans, trees):
                for start, end, found in sorted(side, reverse=True):
                    sql = f"{sql[:start]}{names[found]}(){sql[end:]}"
                back = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
                if len(back) != 1 or _shape(canonical_negation(back[0])) != _shape(canonical_negation(tree)):
                    return None
                texts.append(sql)
    except (sqlglot.errors.SqlglotError, RecursionError):
        return None
    used = {found for _, found in sites}
    return Reading(texts[0], texts[1], tuple(sorted((names[found], found[0], found[1]) for found in keys if found in used)))


def _without_stand_ins(assumptions, found: Reading, unlisted_format: str) -> tuple[str, ...]:
    """The assumptions with the stand-in names taken out of the list of functions the prover has no definition for."""

    prefix = unlisted_format.split("{")[0]
    stand_ins = {name.upper() for name, _, _ in found.names}
    kept = []
    for assumption in assumptions:
        if assumption.startswith(prefix):
            names = [n for n in assumption[len(prefix) :].split(", ") if n.upper() not in stand_ins]
            if not names:
                continue
            assumption = unlisted_format.format(names=", ".join(names))
        kept.append(found.restore(assumption))
    return tuple(dict.fromkeys(kept))


def proved(prove: Callable, left_sql: str, right_sql: str, kwargs: dict):
    """``prove(left_sql, right_sql, **kwargs)``, with the statement values of the pair read as constants.

    A pair that holds none is proved as it is. Otherwise the stand-in pair is proved, and a proof comes back as
    ``PROVEN_CONDITIONALLY`` with one condition per kind of call; a refutation comes back as ``NOT_PROVEN``."""

    from .smt_equivalence import UNLISTED_FUNCTION_ASSUMPTION, SmtEquivalenceResult, SmtStatus

    found = reading(left_sql, right_sql, kwargs.get("dialect", "bigquery") or "bigquery")
    if found is None:
        return prove(left_sql, right_sql, **kwargs)
    result = prove(found.left_sql, found.right_sql, **kwargs)
    assumptions = _without_stand_ins(result.assumptions, found, UNLISTED_FUNCTION_ASSUMPTION)
    calls = ", ".join(call for calls in found.calls.values() for call in calls)
    if result.status is SmtStatus.NOT_EQUIVALENT:
        return SmtEquivalenceResult(SmtStatus.NOT_PROVEN, _NO_COUNTEREXAMPLE.format(calls=calls), assumptions=assumptions)
    if result.status not in (SmtStatus.PROVEN_EQUIVALENT, SmtStatus.PROVEN_CONDITIONALLY):
        return dataclasses.replace(result, reason=found.restore(result.reason), assumptions=assumptions)
    conditions = tuple(result.conditions) + found.conditions()
    wording = " and ".join("at the same statement time" if kind == SAME_TIME else "in the same session" for kind in sorted(found.calls))
    return dataclasses.replace(
        result,
        status=SmtStatus.PROVEN_CONDITIONALLY,
        reason=f"{found.restore(result.reason)}; equivalent when both queries run {wording} ({calls})",
        assumptions=assumptions,
        conditions=conditions,
        counterexample=None,
    )
