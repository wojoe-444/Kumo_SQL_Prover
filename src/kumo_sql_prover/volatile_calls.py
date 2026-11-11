"""Which calls do not give one value per argument list: the single list every prover asks.

The provers model a function they have no rule for as *uninterpreted*: equal arguments give equal results, so
``F(x) = F(x)`` is ``F(x) IS NOT NULL``. That is wrong for a call that draws a new value each time
(``RAND()``, ``nextval('q')``, ``clock_timestamp()``, ``UUID_SHORT()``) and it is not a fact about the tables for a
call that reads the clock, the session or the server (``NOW()``, ``CONNECTION_ID()``, ``date('now')``). Until
2026-11-11 each prover kept its own short list, written for BigQuery, and a name outside it was taken as
deterministic: in PostgreSQL ``WHERE nextval('q') = nextval('q')`` returns no row and the provers called it
``WHERE nextval('q') IS NOT NULL``.

``reason(node, dialect)`` says why a node is refused, or ``None``. Two kinds are told apart, because the
structural prover accepts either one only when a rewrite leaves it identical and in place, while the solver-based
provers refuse both:

``PER_CALL``
    Two evaluations in one statement may differ, or the call has an effect (a sequence advances, a lock is taken).
``PER_STATEMENT``
    One value for the whole statement, taken from the clock, the session, the connection or the server.

A name counts in every dialect once it is volatile in one. That refuses a deterministic function that shares a
name with a volatile one somewhere else, which costs a proof and never a wrong one. A function sqlglot has no
class for and that is not listed here is still read as deterministic: it may be a built-in this list does not
know to be volatile, or a user-defined function that is. ``unlisted_calls`` names those, and the solver-based
provers record them as an assumption of the proof.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

import sqlglot
from sqlglot import exp

PER_CALL = "per call"
PER_STATEMENT = "per statement"

# --- sqlglot node types -------------------------------------------------------------------------------------------
_PER_CALL_TYPES = {
    "Rand",
    "Randn",
    "Randstr",
    "Uniform",
    "Normal",
    "Zipf",
    "Uuid",
    "Shuffle",
    "Seq1",
    "Seq2",
    "Seq4",
    "Seq8",
    "NextValueFor",
    "TableSample",
    # Snowflake's ENCRYPT draws a fresh initialization vector unless one is passed.
    "Encrypt",
    "EncryptRaw",
    # A model's answer is not a function of its arguments.
    "AIAgg",
    "AIClassify",
    "AIEmbed",
    "AIForecast",
    "AIGenerate",
    "AISimilarity",
    "AISummarizeAgg",
    "GenerateBool",
    "GenerateDouble",
    "GenerateEmbedding",
    "GenerateInt",
    "GenerateTable",
    "GenerateText",
    "MLForecast",
    "MLTranslate",
    "Predict",
}
_PER_STATEMENT_TYPES = {
    "SessionUser",
    "Localtime",
    "Localtimestamp",
    "Systimestamp",
    "UtcDate",
    "UtcTime",
    "UtcTimestamp",
}  # and every type whose name starts with "Current" (CurrentDate, CurrentUser, CurrentSchema, CurrentVersion, ...)

# --- function names sqlglot has no class for (upper case) -------------------------------------------------------------
_PER_CALL_NAMES = {
    # random values (names starting with RAND or holding RANDOM are matched below)
    "GEN_RANDOM_BYTES",
    "CRYPT_GEN_RANDOM",
    "UNIFORM",
    "NORMAL",
    "ZIPF",
    "SETSEED",
    "SHUFFLE",
    "ARRAYSHUFFLE",
    "ARRAY_SHUFFLE",
    "ARRAY_SAMPLE",
    "SAMPLE",
    # fresh identifiers (names holding UUID, ULID or GUID are matched below)
    "NEWID",
    "NEWSEQUENTIALID",
    "GENERATESNOWFLAKEID",
    "GENERATESERIALID",
    # encryption with a fresh salt, nonce or key
    "GEN_SALT",
    "PGP_SYM_ENCRYPT",
    "PGP_PUB_ENCRYPT",
    "AEAD.ENCRYPT",
    "KEYS.NEW_KEYSET",
    "KEYS.ROTATE_KEYSET",
    "KEYS.NEW_WRAPPED_KEYSET",
    "KEYS.ROTATE_WRAPPED_KEYSET",
    # sequences and counters
    "NEXTVAL",
    "CURRVAL",
    "LASTVAL",
    "SETVAL",
    "MONOTONICALLY_INCREASING_ID",
    "SPARK_PARTITION_ID",
    "INPUT_FILE_NAME",
    "INPUT_FILE_BLOCK_START",
    "INPUT_FILE_BLOCK_LENGTH",
    "ROWNUMBERINALLBLOCKS",
    "ROWNUMBERINBLOCK",
    "BLOCKNUMBER",
    "SEQ1",
    "SEQ2",
    "SEQ4",
    "SEQ8",
    # the clock, read at each call
    "CLOCK_TIMESTAMP",
    "TIMEOFDAY",
    "SYSDATE",
    "NOWINBLOCK",
    # effects and waits
    "SLEEP",
    "SLEEPEACHROW",
    "BENCHMARK",
    "GET_LOCK",
    "RELEASE_LOCK",
    "RELEASE_ALL_LOCKS",
    "IS_FREE_LOCK",
    "IS_USED_LOCK",
    "MASTER_POS_WAIT",
    "SOURCE_POS_WAIT",
    "WAIT_FOR_EXECUTED_GTID_SET",
    "LOAD_FILE",
    "LOAD_EXTENSION",
    "SET_CONFIG",
    "SETVARIABLE",
    # calls into arbitrary code
    "JAVA_METHOD",
    "REFLECT",
    "TRY_REFLECT",
    "SYSTEM$WAIT",
}
_PER_STATEMENT_NAMES = {
    # the clock, read once
    "NOW",
    "NOW64",
    "TODAY",
    "YESTERDAY",
    "CURDATE",
    "CURTIME",
    "LOCALTIME",
    "LOCALTIMESTAMP",
    "UTC_DATE",
    "UTC_TIME",
    "UTC_TIMESTAMP",
    "GETDATE",
    "GETUTCDATE",
    "SYSDATETIME",
    "SYSUTCDATETIME",
    "SYSDATETIMEOFFSET",
    "SYSTIMESTAMP",
    "STATEMENT_TIMESTAMP",
    "TRANSACTION_TIMESTAMP",
    "GET_CURRENT_TIMESTAMP",
    "GET_CURRENT_TIME",
    "UNIX_MICROS_NOW",
    "UPTIME",
    # the transaction, the connection and what it did last
    "TXID_CURRENT",
    "TXID_CURRENT_IF_ASSIGNED",
    "TXID_CURRENT_SNAPSHOT",
    "CONNECTION_ID",
    "LAST_INSERT_ID",
    "LAST_INSERT_ROWID",
    "CHANGES",
    "TOTAL_CHANGES",
    "FOUND_ROWS",
    "ROW_COUNT",
    "SCOPE_IDENTITY",
    "IDENT_CURRENT",
    "LAST_QUERY_ID",
    "LAST_TRANSACTION",
    "QUERYID",
    "QUERY_ID",
    "INITIALQUERYID",
    "ERROR_NUMBER",
    "ERROR_MESSAGE",
    "ERROR_LINE",
    "ERROR_SEVERITY",
    "ERROR_STATE",
    "ERROR_PROCEDURE",
    "XACT_STATE",
    # who and where
    "USER",
    "SESSION_USER",
    "SYSTEM_USER",
    "USER_NAME",
    "USER_ID",
    "SUSER_NAME",
    "SUSER_SNAME",
    "SUSER_ID",
    "SUSER_SID",
    "ORIGINAL_LOGIN",
    "HOST_NAME",
    "HOSTNAME",
    "HOST_ID",
    "APP_NAME",
    "DB_NAME",
    "DB_ID",
    "DATABASE",
    "SCHEMA",
    "SCHEMA_NAME",
    "VERSION",
    "SQLITE_VERSION",
    "SQLITE_SOURCE_ID",
    "OBJECT_ID",
    "SERVERTIMEZONE",
    "SESSIONTIMEZONE",
    "DBTIMEZONE",
    "GETVARIABLE",
    "GETSETTING",
    "GETMACRO",
    "USERENV",
    "SYS_CONTEXT",
    "SESSION_CONTEXT",
    "CONTEXT_INFO",
    "INET_CLIENT_ADDR",
    "INET_CLIENT_PORT",
    "INET_SERVER_ADDR",
    "INET_SERVER_PORT",
    "ICU_VERSION",
    "SESSION_ID",
    "IS_ROLE_IN_SESSION",
    "IS_MEMBER",
    "IS_SRVROLEMEMBER",
    "HAS_TABLE_PRIVILEGE",
    "HAS_SCHEMA_PRIVILEGE",
    "HAS_DATABASE_PRIVILEGE",
    "HAS_COLUMN_PRIVILEGE",
}
# Deterministic functions whose names hold UUID: conversions and name-based (hashed) identifiers.
_UUID_CONVERSIONS = {
    "UUID_TO_BIN",
    "BIN_TO_UUID",
    "IS_UUID",
    "UUID_NIL",
    "UUID_NS_DNS",
    "UUID_NS_URL",
    "UUID_NS_OID",
    "UUID_NS_X500",
    "UUID_GENERATE_V3",
    "UUID_GENERATE_V5",
    "UUID_EXTRACT_TIMESTAMP",
    "UUID_EXTRACT_VERSION",
    "UUIDSTRINGTONUM",
    "UUIDNUMTOSTRING",
    "TOUUID",
    "TOUUIDORNULL",
    "TOUUIDORZERO",
    "TOUUIDORDEFAULT",
    "TRY_TO_UUID",
    "TO_UUID",
    "PARSE_UUID",
}
# PostgreSQL system functions (pg_backend_pid, pg_sleep, pg_advisory_lock, pg_current_wal_lsn, ...) read or change
# state outside the tables; these few depend only on their argument.
_PG_PURE = {"PG_TYPEOF", "PG_COLUMN_SIZE", "PG_INPUT_IS_VALID", "PG_SIZE_PRETTY", "PG_SIZE_BYTES"}

# SQLite's date and time functions read the clock when no time value is given.
_SQLITE_TIME_FUNCTIONS = {"DATE", "TIME", "DATETIME", "JULIANDAY", "UNIXEPOCH"}
# Strings PostgreSQL and SQLite turn into the current date or time when read as one ...
_CLOCK_STRINGS = {"now", "today", "tomorrow", "yesterday"}
# ... and the nodes that read their argument as one, in any dialect.
_TIME_READERS = {
    "Cast",
    "TryCast",
    "Date",
    "Datetime",
    "Time",
    "Timestamp",
    "TsOrDsToDate",
    "TsOrDsToDatetime",
    "TsOrDsToTime",
    "TsOrDsToTimestamp",
    "TimeStrToTime",
    "TimeStrToDate",
    "TimeStrToUnix",
    "DateStrToDate",
    "StrToTime",
    "StrToDate",
    "StrToUnix",
}
_TIME_READER_NAMES = _SQLITE_TIME_FUNCTIONS | {"STRFTIME", "TIMEDIFF"}
# Dialects in which a string literal takes the type of what it is compared with, so ``ts > 'now'`` reads the clock.
_UNTYPED_LITERAL_DIALECTS = {"postgres", "redshift", "materialize", "risingwave"}

# Niladic functions written without parentheses. A dialect whose parser does not know one reads it as a column.
_KEYWORD_VALUES = {
    "CURRENT_ROLE",
    "CURRENT_SCHEMA",
    "CURRENT_CATALOG",
    "CURRENT_PATH",
    "CURRENT_USER",
    "CURRENT_DATE",
    "CURRENT_TIME",
    "CURRENT_TIMESTAMP",
    "LOCALTIME",
    "LOCALTIMESTAMP",
    "SESSION_USER",
    "SYSTEM_USER",
}
_DIALECT_KEYWORD_VALUES = {
    "oracle": {"USER", "UID", "SYSDATE", "SYSTIMESTAMP", "SESSIONTIMEZONE", "DBTIMEZONE"},
    "postgres": {"USER"},
    "redshift": {"USER", "SYSDATE"},
}
# Oracle numbers the rows of a result as they are produced, before ORDER BY.
_DIALECT_ROW_COUNTERS = {"oracle": {"ROWNUM"}}


def _name_kind(name: str, arguments: list) -> str | None:
    """The kind of a call to the function called ``name`` (upper case), which sqlglot has no class for."""

    if name in _PER_CALL_NAMES or name.startswith("RAND") or "RANDOM" in name:
        return PER_CALL
    if ("UUID" in name or "ULID" in name or "GUID" in name) and name not in _UUID_CONVERSIONS:
        return PER_CALL
    if name.startswith("PG_") and name not in _PG_PURE:
        return PER_CALL
    if name.startswith("CURRENT") or name in _PER_STATEMENT_NAMES:
        return PER_STATEMENT
    if name == "TIMEZONE" and not arguments:
        return PER_STATEMENT  # ClickHouse: the server's zone; PostgreSQL's timezone(zone, t) is a function of both
    if name == "UNIX_TIMESTAMP" and not arguments:
        return PER_STATEMENT  # MySQL: the current time; UNIX_TIMESTAMP(d) is a function of d
    if name == "AGE" and len(arguments) == 1:
        return PER_STATEMENT  # PostgreSQL: age(t) is measured from CURRENT_DATE
    if (name in _SQLITE_TIME_FUNCTIONS and not arguments) or (name == "STRFTIME" and len(arguments) < 2):
        return PER_STATEMENT
    return None


def _call_name(node: exp.Anonymous) -> str:
    """Upper-case name of the call, with the name before the dot when it has one (``DBMS_RANDOM.STRING``)."""

    owner = node.parent
    prefix = f"{owner.this.sql(dialect='bigquery')}." if isinstance(owner, exp.Dot) and owner.expression is node else ""
    return (prefix + str(node.this or "")).upper()


def _reads_time(node: exp.Expression | None) -> bool:
    if type(node).__name__ in _TIME_READERS:
        return True
    return isinstance(node, exp.Anonymous) and str(node.this or "").upper() in _TIME_READER_NAMES


def _clock_string(node: exp.Expression) -> bool:
    return isinstance(node, exp.Literal) and node.is_string and str(node.this).strip().lower() in _CLOCK_STRINGS


def kind(node: exp.Expression, dialect: str | None = None) -> str | None:
    """``PER_CALL``, ``PER_STATEMENT`` or ``None`` for this node alone (its children are not looked at)."""

    type_name = type(node).__name__
    if type_name in _PER_CALL_TYPES:
        return PER_CALL
    if type_name in _PER_STATEMENT_TYPES or type_name.startswith("Current"):
        return PER_STATEMENT
    if isinstance(node, exp.Anonymous):
        return _name_kind(_call_name(node), node.expressions)
    if isinstance(node, exp.Date) and node.this is None:
        return PER_STATEMENT  # SQLite: date() is today
    if _clock_string(node):
        if _reads_time(node.parent) or dialect in _UNTYPED_LITERAL_DIALECTS:
            return PER_STATEMENT  # date('now'), 'now'::timestamp, ts > 'now'
    if isinstance(node, exp.Column):
        name = node.name.upper()
        quoted = isinstance(node.this, exp.Identifier) and node.this.quoted
        if node.table and name in ("NEXTVAL", "CURRVAL"):
            return PER_CALL  # Oracle and Snowflake: sequence.NEXTVAL
        if not node.table and not quoted:
            if name in _DIALECT_ROW_COUNTERS.get(dialect or "", ()):
                return PER_CALL
            if name in _KEYWORD_VALUES or name in _DIALECT_KEYWORD_VALUES.get(dialect or "", ()):
                return PER_STATEMENT
    return None


def is_volatile(node: exp.Expression, dialect: str | None = None) -> bool:
    return kind(node, dialect) is not None


def sites(tree: exp.Expression, dialect: str | None = None) -> Iterator[tuple[exp.Expression, str]]:
    """Every volatile node under ``tree`` (itself included) with its kind, in walk order."""

    for node in tree.walk():
        found = kind(node, dialect)
        if found is not None:
            yield node, found


def first(tree: exp.Expression, dialect: str | None = None) -> exp.Expression | None:
    return next((node for node, _ in sites(tree, dialect)), None)


def contains(tree: exp.Expression, dialect: str | None = None) -> bool:
    return first(tree, dialect) is not None


def describe(node: exp.Expression) -> str:
    """A short text of the call for a refusal message."""

    if isinstance(node, exp.Literal):
        return (node.parent if _reads_time(node.parent) else node).sql(dialect="bigquery")
    return node.sql(dialect="bigquery")


def reason(tree: exp.Expression, dialect: str | None = None) -> str | None:
    """Why a statement holding this tree is not compared by the solver-based provers, or ``None``."""

    for node, found in sites(tree, dialect):
        if found == PER_CALL:
            return f"nondeterministic: {describe(node)} can give a different value at each call"
        return f"nondeterministic: {describe(node)} reads the clock or the session, not the tables"
    return None


def unlisted_calls(tree: exp.Expression) -> tuple[str, ...]:
    """Upper-case names of the functions under ``tree`` that sqlglot has no class for and this module does not list.

    A proof that reads one as a function of its arguments rests on that; the name may belong to a user-defined
    function or to a built-in that is volatile."""

    names = set()
    for node in tree.walk():
        if isinstance(node, exp.Anonymous) and kind(node) is None:
            names.add(_call_name(node))
    return tuple(sorted(names))


@lru_cache(maxsize=512)
def read(sql: str, dialect: str = "bigquery") -> tuple[str | None, tuple[str, ...]]:
    """``(reason, unlisted_calls)`` of a query as it is written: the refusal for its first volatile call, if any,
    and the names of the functions it calls that this module does not know.

    Text sqlglot cannot read gives ``(None, ())``: the prover that was asked reports the parse error itself."""

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except (sqlglot.errors.SqlglotError, RecursionError):
        return None, ()
    names: set[str] = set()
    for statement in statements:
        why = reason(statement, dialect)
        if why:
            return why, ()
        names.update(unlisted_calls(statement))
    return None, tuple(sorted(names))
