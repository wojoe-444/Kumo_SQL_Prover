"""Cross-dialect parse and proof coverage on DLBench's SQL translation pairs, with no language model.

DLBench (https://github.com/DLBenchll/DLBench, Apache-2.0, ASE 2025) pairs a query with its
translation into another database's dialect: BIRDTrans (3,206 pairs from BIRD's SQLite queries)
and BUTTERTrans (3,196 pairs from MySQL's and PostgreSQL's own test suites), translated into
MySQL, MariaDB, PostgreSQL, ClickHouse, MonetDB and DuckDB. Its labels (6,199 "exact" and 203
"approximate" equivalence) come from language models and human review, not proofs.

For each pair KumoSQL:

1. **parses** the source in its dialect and the translation in the target's (sqlglot has no MariaDB
   or MonetDB dialect: they are read as MySQL and PostgreSQL);
2. **proves** the translation equal to the source in the source's dialect: the translation is
   renamed back to the source's column and table names (BIRD's translations respell some, for
   example ``Type`` to ``_Type`` and ``Sales Orders`` to ``Sales_Orders``), written in the source
   dialect by sqlglot, and compared with ``prove_equivalent_algebraic`` (no keys, output names
   ignored). As in Spider, a source ending in ``ORDER BY`` is compared as a list. A MariaDB
   translation that is its MySQL source token for token is also decided, when one run of the
   statement says what another would (``same_statement``).

A proof is about sqlglot's reading of each dialect. Where the two databases mean different things
by the same text and sqlglot does not translate it, no proof is taken (``dialect gap``):

* ``LIKE`` ignores case in SQLite and MySQL but not in the others;
* MySQL and MariaDB compare strings without case (their default collation): a proof between a
  MySQL-family query and another dialect is taken only for queries with no string and no column;
* ``/`` on integers truncates in PostgreSQL and MonetDB but not in MySQL; dividing by zero gives NULL
  in SQLite and MySQL but infinity in DuckDB and ClickHouse (an error elsewhere, which proofs exclude);
* ``REAL`` (and ``FLOAT`` in most of the targets) is a 32-bit float, but sqlglot writes it as SQLite's
  64-bit ``REAL``;
* ``1.0`` is an exact decimal in PostgreSQL, MonetDB and MySQL (so ``1.0 / 3`` keeps 20 digits), a float
  in SQLite;
* ClickHouse returns 0 rather than NULL for ``SUM`` of no rows and fills the missing side of an outer
  join with default values, not NULL;
* SQLite compares a text column with a number after converting one side (see
  ``llm_sql_solver_bench.mixed_type_comparison``).

Every proof of a BIRDTrans pair is re-checked: the source and the translation run on 300 random
SQLite databases, and a translation into DuckDB also runs natively in DuckDB on the same data. A
difference makes the proof ``wrong``.

The repository pins a subset (``tests/fixtures/dlbench``): every "approximate" pair and one in ten of
the rest, by a hash of the pair's dataset, target and id. One pair in five of the subset is held out.

    python tools/dlbench_bench.py                                 # the pinned subset
    python tools/dlbench_bench.py --data DLBench                  # all 6,402 pairs from a checkout
    python tools/dlbench_bench.py --data DLBench --write-subset   # rebuild the pinned subset
    python tools/dlbench_bench.py --write-results
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import datetime
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.getLogger("sqlglot").setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "dlbench"
SOURCE_COMMIT = "a3525919033faac73e60a13a88c2d14ab6953f23"
DATASETS = ("BIRDTrans", "BUTTERTrans")
TARGETS = ("mysql", "mariadb", "postgresql", "clickhouse", "monetdb", "duckdb")
SQLGLOT = {"sqlite": "sqlite", "mysql": "mysql", "mariadb": "mysql", "postgresql": "postgres", "clickhouse": "clickhouse", "monetdb": "postgres", "duckdb": "duckdb"}
CASELESS_LIKE = {"sqlite", "mysql", "mariadb"}
CASELESS_STRINGS = {"mysql", "mariadb"}
INTEGER_DIVISION = {"sqlite", "postgresql", "monetdb"}
EXACT_DECIMALS = {"postgresql", "monetdb", "mysql", "mariadb"}  # 1.0 is an exact decimal, so 1.0 / 3 is not a float
DIVIDE_BY_ZERO = {"sqlite": "null", "mysql": "null", "mariadb": "null", "duckdb": "infinity", "clickhouse": "infinity"}  # the others raise an error
SAME_READING = {"mysql", "mariadb"}  # one sqlglot dialect reads both, and no gap rule tells them apart
# Calls whose answer is not a function of the tables: the clock, random values, sequences, the session, the server
VOLATILE_CALLS = {
    "benchmark", "changes", "clock_timestamp", "connection_id", "curdate", "current_database", "current_setting",
    "currentdatabase", "currentuser", "currval", "curtime", "database", "found_rows", "gen_random_uuid",
    "generateuuidv4", "generateuuidv7", "get_current_time", "get_current_timestamp", "get_lock", "getdate", "hostname",
    "is_free_lock", "is_used_lock", "last_insert_id", "last_insert_rowid", "lastval", "load_file", "nextval", "now",
    "now64", "pg_backend_pid", "pg_sleep", "rand", "rand32", "rand64", "randcanonical", "randconstant", "random",
    "random_bytes", "randomblob", "release_all_locks", "release_lock", "row_count", "schema", "servername", "setseed",
    "setval", "sleep", "sqlite_version", "statement_timestamp", "sysdate", "timeofday", "today", "total_changes",
    "transaction_timestamp", "txid_current", "uptime", "user", "uuid", "uuid_generate_v1", "uuid_generate_v4",
    "uuid_short", "uuidv4", "uuidv7", "version", "yesterday",
}
# The same without parentheses
VOLATILE_WORDS = {
    "current_catalog", "current_date", "current_role", "current_schema", "current_time", "current_timestamp",
    "current_user", "localtime", "localtimestamp", "session_user", "system_user", "utc_date", "utc_time", "utc_timestamp",
}
VOLATILE_NO_ARGUMENT = {"date", "datetime", "julianday", "time", "unix_timestamp", "unixepoch"}  # the clock when called with nothing
VOLATILE_TEXTS = {"now", "today", "tomorrow", "yesterday"}  # date('now'), 'today'::date
# A statement that uses one of these need not return the same thing twice either (the same-statement rule)
RERUN_DEPENDENT = VOLATILE_CALLS | VOLATILE_WORDS | {"any_value", "group_concat", "json_arrayagg", "json_objectagg", "over"}
ROW_STRICT = {"duckdb", "clickhouse", "monetdb"}  # any compared row value is a gap: DuckDB and ClickHouse compare structs, MonetDB is not replayed
DECIMAL_QUOTIENTS = {"mysql", "mariadb"}
# Functions that compare their string arguments (by the collation, where there is one)
STRING_COMPARERS = {"field", "find_in_set", "greatest", "instr", "interval", "least", "locate", "nullif", "position", "strcmp"}
ROW_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.NullSafeNEQ, exp.In, exp.Between)
# Unix time: MySQL 8 counts seconds past 2038 and MariaDB (before 11.5) does not; MySQL answers NULL before 1970 and
# ClickHouse answers 1970-01-01. Only a literal inside the range every engine keeps is let through.
EPOCH_CALLS = {"unix_timestamp", "from_unixtime", "fromunixtimestamp", "fromunixtimestamp64milli", "tounixtimestamp", "epoch", "epoch_ms", "unixepoch"}
EPOCH_YEARS = range(1971, 2038)
# Population in MySQL and MariaDB, sample in PostgreSQL, DuckDB and MonetDB; sqlglot reads them as one function
AMBIGUOUS_SPREAD = {"std", "stddev", "variance"}
MYSQL_FAMILY = {"mysql", "mariadb"}
# One name, different functions: what two engines of different families compute for the same call (each seen natively)
SET_RETURNING = {"generate_series", "unnest", "range", "generate_subscripts", "arrayjoin", "numbers", "json_each", "regexp_split_to_table"}
NAN_IN_CLICKHOUSE = {"acos", "asin", "ln", "log10", "log1p", "log2", "pow", "power", "sqrt"}  # NULL in MySQL outside the domain
SUBSTRINGS = {"mid", "substr", "substring"}
BITWISE = (exp.BitwiseAnd, exp.BitwiseOr, exp.BitwiseXor, exp.BitwiseNot, exp.BitwiseLeftShift, exp.BitwiseRightShift)
DATE_BUILDERS = {"from_days", "make_date", "make_timestamp", "makedate", "makedate32", "makedatetime", "makedatetime64"}
CONTRADICTIONS = FIXTURES / "native_contradictions.json"
PROVER_TIMEOUT_MS = 3000
CHECK_TRIALS = 300
UNBOUNDED = 1_000_000_000


@dataclass
class Pair:
    dataset: str
    sql_id: int
    database: str
    source_dbms: str
    target_dbms: str
    source_query: str
    target_query: str
    label_raw: str  # as published: exact_equivalence, appr_equivalence, approximate_equivalence or "Approximate equivalence"
    source_schema: list[str] = field(default_factory=list)  # source_related_schemas, as published
    renames: dict[str, str] = field(default_factory=dict)  # translation's column name -> source's, where they differ

    @property
    def id(self) -> str:
        return f"{self.dataset}/{self.target_dbms}/{self.sql_id}"

    @property
    def label(self) -> str:
        return "approximate" if "appr" in self.label_raw.lower() else "exact"

    @property
    def held_out(self) -> bool:
        return int(hashlib.sha1(f"dlbench-held-out\n{self.id}".encode()).hexdigest(), 16) % 5 == 0


def in_subset(dataset: str, target: str, sql_id: int, label_raw: str) -> bool:
    if "appr" in label_raw.lower():
        return True
    return int(hashlib.sha1(f"dlbench\n{dataset}\n{target}\n{sql_id}".encode()).hexdigest(), 16) % 10 == 0


# -- loading --------------------------------------------------------------------------

_TABLE = re.compile(r"Table:\s*`?([^`\n]+?)`?\s*\nColumns:\s*\n(.*)", re.S)
_COLUMN = re.compile(r"^\(\s*`?([^`,]+?)`?\s*,\s*((?:[^,()]|\([^()]*\))*)(?:,\s*([^()]*))?\)\s*$")


def described_tables(texts: list[str]) -> list[tuple[str, list[tuple[str, str, bool]]]]:
    """``[(table, [(column, type, primary key)])]`` from DLBench's ``Table: .. Columns: ..`` text."""

    tables = []
    for text in texts:
        match = _TABLE.match(text.strip())
        if not match:
            continue
        columns = []
        for line in match.group(2).splitlines():
            column = _COLUMN.match(line.strip())
            if column:
                columns.append((column.group(1).strip(), column.group(2).strip().lower(), "primary key" in (column.group(3) or "").lower()))
        tables.append((match.group(1).strip(), columns))
    return tables


def created_tables(texts: list[str], dialect: str) -> list[tuple[str, list[tuple[str, str, bool]]]]:
    """The same shape from CREATE TABLE statements (BUTTERTrans lists the statements a test runs first)."""

    tables = []
    for text in texts:
        try:
            statements = sqlglot.parse(text, read=SQLGLOT[dialect])
        except sqlglot.errors.SqlglotError:
            continue
        for statement in statements:
            if isinstance(statement, exp.Create) and isinstance(statement.this, exp.Schema):
                columns = [
                    (c.name, (c.args.get("kind").sql() if c.args.get("kind") else "").lower(), False)
                    for c in statement.this.expressions
                    if isinstance(c, exp.ColumnDef)
                ]
                tables.append((statement.this.this.name, columns))
    return tables


def schema_of(pair: Pair) -> dict[str, dict[str, str]]:
    """Lower-case table -> lower-case column -> declared type, for the tables the source describes."""

    tables = described_tables(pair.source_schema) or created_tables(pair.source_schema, pair.source_dbms)
    return {t.lower(): {c.lower(): k for c, k, _ in cols} for t, cols in tables}


def _renames(source: list[str], target: list[str]) -> dict[str, str]:
    """Column renames between the source's and the translation's schema text, matched by position."""

    renames = {}
    for (_, left), (_, right) in zip(described_tables(source), described_tables(target)):
        if len(left) != len(right):
            continue
        for (a, _, _), (b, _, _) in zip(left, right):
            if a.lower() != b.lower():
                renames[b] = a
    return renames


def read_checkout(data: Path, subset_only: bool = False) -> list[Pair]:
    pairs = []
    for dataset in DATASETS:
        for target in TARGETS:
            for row in json.loads((data / "datasets" / dataset / f"{target}.json").read_text(encoding="utf-8")):
                if subset_only and not in_subset(dataset, target, row["sql_id"], row["semantic_equivalent_type"]):
                    continue
                pairs.append(Pair(
                    dataset, row["sql_id"], row["database_name"], row["source_dbms"], row["target_dbms"],
                    row["source_query"], row["target_query"], row["semantic_equivalent_type"],
                    row.get("source_related_schemas") or [],
                    _renames(row.get("source_related_schemas") or [], row.get("target_related_schemas") or []),
                ))
    return pairs


def _schema_key(pair: Pair) -> str:
    return f"{pair.dataset}/{pair.database}/{pair.source_dbms}"


def _renames_key(pair: Pair) -> str:
    return f"{pair.dataset}/{pair.database}/{pair.target_dbms}"


def load_pairs(fixtures: Path = FIXTURES) -> list[Pair]:
    """The pinned subset; schemas and renames are stored once per database (``schemas.json``, ``renames.json``)."""

    schemas = json.loads((fixtures / "schemas.json").read_text(encoding="utf-8"))
    renames = json.loads((fixtures / "renames.json").read_text(encoding="utf-8"))
    pairs = []
    for line in (fixtures / "pairs.jsonl").read_text(encoding="utf-8").splitlines():
        pair = Pair(**json.loads(line))
        pair.source_schema = schemas.get(_schema_key(pair), [])
        pair.renames = renames.get(_renames_key(pair), {})
        pairs.append(pair)
    return pairs


def write_subset(data: Path, fixtures: Path = FIXTURES) -> int:
    pairs = read_checkout(data, subset_only=True)
    schemas: dict[str, dict[str, str]] = defaultdict(dict)  # key -> table -> its schema text
    renames: dict[str, dict[str, str]] = defaultdict(dict)
    with open(fixtures / "pairs.jsonl", "w", encoding="utf-8") as handle:
        for pair in pairs:
            for text in pair.source_schema:
                described = described_tables([text])
                schemas[_schema_key(pair)].setdefault(described[0][0] if described else text, text)
            renames[_renames_key(pair)].update(pair.renames)
            row = {k: v for k, v in pair.__dict__.items() if k not in ("source_schema", "renames")}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name, value in (("schemas.json", {k: list(v.values()) for k, v in schemas.items()}), ("renames.json", {k: v for k, v in renames.items() if v})):
        (fixtures / name).write_text(json.dumps(dict(sorted(value.items())), ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    return len(pairs)


# -- reading and translating ----------------------------------------------------------


def parse(sql: str, dbms: str):
    try:
        trees = [t for t in sqlglot.parse(sql, read=SQLGLOT[dbms]) if t is not None and not isinstance(t, exp.Semicolon)]
    except Exception:  # sqlglot raises more than ParseError on unusual input
        return None
    return trees[0] if len(trees) == 1 and not isinstance(trees[0], exp.Command) else None


def _spelling(name: str) -> str:
    """A name as a translated schema would spell it: lower-case, every other character an underscore."""

    return re.sub(r"[^a-z0-9]", "_", name.lower())


def _respelled(name: str, known: set[str]) -> str | None:
    """The one name in ``known`` that ``name`` (not itself in it) is a translated spelling of, if any.

    DLBench's translated schemas respell names the target would not take as written: a space or a
    hyphen becomes an underscore (``Date received`` to ``Date_received``, ``current-terms`` to
    ``current_terms``) and a reserved word gets an underscore in front (``date`` to ``_date``). The
    fixture's ``renames.json`` only lists the tables a pair's schema text describes in the same order
    on both sides, so a name it misses is matched this way, and only when one source name fits.
    """

    if name in known:
        return None
    spelled = name.lower()
    fits = {k for k in known if spelled in (_spelling(k), "_" + _spelling(k))}
    return fits.pop() if len(fits) == 1 else None


def _plain(tree, renames: dict[str, str], schema: dict[str, dict[str, str]] | None = None):
    """Translation's column and table names back to the source's; every name lower-case, quoted only when needed.

    ``renames`` is the fixture's list (its keys keep the quotes the translated schema was written
    with). With the source's ``schema``, a name the list misses is read back by its spelling
    (``_respelled``).
    """

    lowered = {k.strip('"`').lower(): v.strip('"`') for k, v in renames.items()}
    tables = set(schema or ())
    columns = {c for cols in (schema or {}).values() for c in cols}
    local = {n.alias_or_name.lower() for n in tree.find_all(exp.CTE, exp.TableAlias, exp.Subquery) if n.alias_or_name}
    aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}  # ORDER BY _date may name an output, not a column
    table_names = {}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name and name not in local and not table.args.get("db") and (found := _respelled(name, tables)):
            table_names[name] = found
    for identifier in tree.find_all(exp.Identifier):
        name = identifier.name
        parent = identifier.parent
        if isinstance(parent, exp.Column) and parent.this is identifier:
            if parent.args.get("table") is not None or name.lower() not in aliases:
                name = lowered.get(name.lower(), name)
                name = _respelled(name.lower(), columns) or name
        elif (isinstance(parent, exp.Table) and parent.this is identifier) or (isinstance(parent, exp.Column) and parent.args.get("table") is identifier):
            name = table_names.get(name.lower(), name)
        name = name.lower()
        identifier.set("this", name)
        identifier.set("quoted", not re.fullmatch(r"[a-z_][a-z0-9_]*", name))
    return tree


# strftime specifiers that mean the same in SQLite, DuckDB and C (which MonetDB's timestamp_to_str calls):
# four-digit year, two-digit month, day, hour (00-23), minute and second. Everything else is declined:
# %J and %f are SQLite's own, %s, %j, %w, %u, %e, %p and the rest differ or were not confirmed.
_SHARED_FORMAT = re.compile(r"(?:[^%]|%[YmdHMS])*")


def _format_of(arguments: list) -> tuple[object, object] | None:
    """``(value, format)`` when exactly one of two arguments is a string literal with a shared ``%`` specifier."""

    if len(arguments) != 2:
        return None
    formats = [a for a in arguments if isinstance(a, exp.Literal) and a.is_string and "%" in a.name]
    if len(formats) != 1 or not _SHARED_FORMAT.fullmatch(formats[0].name):
        return None
    value = next(a for a in arguments if a is not formats[0])
    if isinstance(value, exp.Literal) and value.is_string:
        return None  # two string literals: which one is the format is a guess
    return value, formats[0]


_WHOLE_TARGETS = {exp.DataType.Type.INT, exp.DataType.Type.BIGINT, exp.DataType.Type.SMALLINT, exp.DataType.Type.TINYINT}


def _year_padding_cannot_matter(node, form: str) -> bool:
    """Whether a year written without leading zeros would change nothing where ``node`` stands.

    MonetDB formats through C's ``strftime``, which writes the year 99 as ``99`` where SQLite writes
    ``0099`` (glibc; checked on this C library, MonetDB itself was not available). The two agree from
    the year 1000 on. Before it, the text differs, so the call is read back only where that cannot
    show: the format has no ``%Y``; or the format is ``%Y`` alone and the text is cast to a whole
    number at once (``0099`` and ``99`` are both 99); or it is tested for equality (``=``, ``<>``,
    ``IN``) against string literals that are themselves a possible output with a four-digit year not
    starting with 0, which neither spelling of an early year can equal. An ordering comparison is
    refused: ``'0099' < '1960'`` and ``'99' > '1960'``.
    """

    if "%Y" not in form:
        return True
    parent = node.parent
    while isinstance(parent, exp.Paren):
        node, parent = parent, parent.parent
    if isinstance(parent, exp.Cast):
        return form == "%Y" and parent.to.this in _WHOLE_TARGETS
    output = re.compile("".join(
        r"[1-9]\d{3}" if piece == "%Y" else r"\d{2}" if piece.startswith("%") else re.escape(piece) for piece in re.findall(r"%.|[^%]+", form)
    ))

    def possible(literal) -> bool:
        return isinstance(literal, exp.Literal) and literal.is_string and output.fullmatch(literal.name) is not None

    if isinstance(parent, (exp.EQ, exp.NEQ)):
        return possible(parent.expression if parent.this is node else parent.this)
    if isinstance(parent, exp.In) and parent.this is node and parent.args.get("query") is None and parent.args.get("field") is None:
        return bool(parent.expressions) and all(possible(e) for e in parent.expressions)
    return False


def _unshared_date_format(tree) -> bool:
    """Whether a date-formatting call in a DuckDB or MonetDB translation uses a specifier outside the shared ones."""

    for node in tree.find_all(exp.TimeToStr):
        literals = [a for a in (node.this, node.args.get("format")) if isinstance(a, exp.Literal) and a.is_string and "%" in a.name]
        if any(not _SHARED_FORMAT.fullmatch(a.name) for a in literals) or (not literals and node.args.get("format") is not None):
            return True
    for node in tree.find_all(exp.Anonymous):
        if node.name.lower() in ("timestamp_to_str", "date_to_str", "time_to_str", "strftime"):
            if any(isinstance(a, exp.Literal) and a.is_string and "%" in a.name and not _SHARED_FORMAT.fullmatch(a.name) for a in node.expressions):
                return True
    return False


def _date_formats_as_sqlite(tree, target_dbms: str):
    """Two date-formatting calls sqlglot leaves as they are, read as SQLite's ``STRFTIME(format, value)``.

    * DuckDB takes ``strftime(value, format)`` and also ``strftime(format, value)``; sqlglot reads the
      first argument as the value either way, so the second spelling comes out with its arguments
      swapped.
    * MonetDB's ``sys.timestamp_to_str(value, format)`` is not a function sqlglot knows. It is read
      back only where the unpadded year of C's ``strftime`` cannot matter (``_year_padding_cannot_matter``).

    Only for a format made of ``%Y %m %d %H %M %S`` and plain text, and only when exactly one argument
    is a string literal holding a ``%`` specifier, so the order is not a guess.
    """

    if target_dbms == "duckdb":
        for node in list(tree.find_all(exp.TimeToStr)):
            read = _format_of([node.this, node.args.get("format")]) if not node.args.get("culture") and not node.args.get("zone") else None
            if read is not None and read[1] is node.this:  # format first: sqlglot took it for the value
                value, form = read[0].copy(), read[1].copy()
                node.set("this", value)
                node.set("format", form)
    elif target_dbms == "monetdb":
        for node in list(tree.find_all(exp.Anonymous)):
            if node.name.lower() != "timestamp_to_str":
                continue
            read = _format_of(list(node.expressions))
            if read is None or read[1] is not node.expressions[1]:
                continue  # MonetDB takes the value first
            outer = node.parent if isinstance(node.parent, exp.Dot) and node.parent.expression is node and node.parent.name.lower() == "timestamp_to_str" and node.parent.this.name.lower() == "sys" else node
            if isinstance(node.parent, exp.Dot) and outer is node:
                continue  # qualified by something other than sys
            if not _year_padding_cannot_matter(outer, read[1].name):
                continue
            outer.replace(exp.TimeToStr(this=read[0].copy(), format=read[1].copy()))
    return tree


def as_source(pair: Pair) -> tuple[str | None, str | None]:
    """Both queries in the source's dialect, names normalised; ``None`` where a query does not parse."""

    dialect = SQLGLOT[pair.source_dbms]
    source, target = parse(pair.source_query, pair.source_dbms), parse(pair.target_query, pair.target_dbms)
    try:
        left = _plain(source, {}).sql(dialect=dialect) if source is not None else None
        if target is not None and pair.source_dbms == "sqlite":
            target = _date_formats_as_sqlite(target, pair.target_dbms)
        right = _plain(target, pair.renames, schema_of(pair)).sql(dialect=dialect) if target is not None else None
    except Exception:  # a construct sqlglot cannot write in the source dialect
        return None, None
    return left, right


# -- dialect gaps ---------------------------------------------------------------------


def dialect_gap(pair: Pair, left: str, right: str, schema: dict[str, dict[str, str]]) -> str | None:
    """Why a proof would not carry over between the two databases, or ``None``."""

    source, target = pair.source_dbms, pair.target_dbms
    trees = [parse(q, d) for q, d in ((pair.source_query, source), (pair.target_query, target))]
    if any(t is None for t in trees):
        return "unparsed"
    types = _Types(schema)
    if any(_row_compared(t, types, strict=bool({source, target} & ROW_STRICT)) for t in trees):
        return "row comparison"
    if source != target and (_epoch_outside(pair.source_query, source) or _epoch_outside(pair.target_query, target)):
        return "timestamp range"
    words = {text for query, dbms in ((pair.source_query, source), (pair.target_query, target)) for kind, text in _tokens(query, dbms) or () if "STRING" not in kind}
    if "clickhouse" in (source, target) and words & DATE_BUILDERS:
        return "ClickHouse date range"
    if "postgresql" in (source, target) and source != target and "to_timestamp" in words:
        return "PostgreSQL time zone"  # TO_TIMESTAMP gives a timestamp with a time zone, whose text ends in +00
    if (source in MYSQL_FAMILY) != (target in MYSQL_FAMILY) and words & AMBIGUOUS_SPREAD:
        return "population variance"
    if {source, target} == MYSQL_FAMILY and words & (AMBIGUOUS_SPREAD | {"stddev_pop", "stddev_samp", "var_pop", "var_samp"}):
        return "deviation precision"  # of whole numbers: a double in MySQL, a decimal with four more digits in MariaDB
    if _family(source) != _family(target) and (name := _engine_specific(pair, trees, words, types)) is not None:
        return name
    if (source in CASELESS_LIKE) != (target in CASELESS_LIKE) and any(t.find(exp.Like) for t in trees):
        return "LIKE and case"
    if (source in MYSQL_FAMILY) != (target in MYSQL_FAMILY) and any(
        not all(isinstance(a.this if isinstance(a, exp.Neg) else a, exp.Literal) for a in [n.this, *n.expressions])
        for t in trees for n in t.find_all(exp.Least, exp.Greatest)
    ):
        return "LEAST and GREATEST with NULL"  # NULL in MySQL when any argument is, the others skip a NULL argument
    if (source in CASELESS_STRINGS) != (target in CASELESS_STRINGS):
        if "monetdb" in (source, target):  # no MonetDB engine to replay on: the first, coarse rule stays
            if any(t.find(exp.Literal) and any(l.is_string for l in t.find_all(exp.Literal)) or t.find(exp.Column) for t in trees):
                return "MySQL string comparison"
        elif any(_string_site(t, types, literals_agree=False) for t in trees):
            return "MySQL string comparison"
    if {source, target} == {"mysql", "mariadb"} and any(_string_site(t, types, literals_agree=True) for t in trees):
        return "MySQL and MariaDB collations"
    if target in INTEGER_DIVISION and source not in INTEGER_DIVISION and trees[1].find(exp.Div):
        return "integer division"
    zero = {DIVIDE_BY_ZERO.get(source), DIVIDE_BY_ZERO.get(target)}
    if len(zero - {None}) == 2 and any(t.find(exp.Div) for t in trees):
        return "division by zero"
    if target != "sqlite" and any(d.this == exp.DataType.Type.FLOAT for d in trees[1].find_all(exp.DataType)):
        return "32-bit float"
    if target in EXACT_DECIMALS and source not in EXACT_DECIMALS and trees[1].find(exp.Div) and any(
        not l.is_string and "." in l.name for l in trees[1].find_all(exp.Literal)
    ):
        return "decimal arithmetic"
    if (source in DECIMAL_QUOTIENTS) != (target in DECIMAL_QUOTIENTS) and any(t.find(exp.Div, exp.Avg) for t in trees):
        return "decimal arithmetic"  # MySQL: 1 / 3 and AVG of whole numbers are decimals with four more digits, 0.3333
    if target == "clickhouse" and (
        any(not isinstance(a, exp.Count) for a in trees[1].find_all(exp.AggFunc))
        or any(j.side for j in trees[1].find_all(exp.Join))
    ):
        return "ClickHouse defaults"
    if source == "sqlite" and (_text_against_number(left) or _text_against_number(right)):
        return "SQLite text against a number"
    if source == "sqlite" and target in ("duckdb", "monetdb") and _unshared_date_format(trees[1]):
        return "date format"  # only %Y %m %d %H %M %S are known to mean the same as in SQLite
    if source == "sqlite":
        import llm_sql_solver_bench as solver

        tables = {t: {c: ("TEXT" if _texty(k) else "INTEGER") for c, k in cols.items()} for t, cols in schema.items()}
        if solver.mixed_type_comparison(left, tables) or solver.mixed_type_comparison(right, tables):
            return "SQLite type conversion"
    return None


def _family(dbms: str) -> str:
    return "mysql" if dbms in MYSQL_FAMILY else dbms


def _calls(tokens: list[tuple[str, str]], names: set[str]):
    """``(name, arguments)`` for every call of one of ``names``: the arguments as token lists, split at top-level commas."""

    for index, (kind, text) in enumerate(tokens):
        if "STRING" in kind or text not in names or tokens[index + 1:index + 2] != [("L_PAREN", "(")]:
            continue
        depth, arguments = 0, [[]]
        for token in tokens[index + 1:]:
            depth += (token[0] == "L_PAREN") - (token[0] == "R_PAREN")
            if depth == 0:
                break
            if depth == 1 and token[0] == "COMMA":
                arguments.append([])
            elif not (depth == 1 and token[0] == "L_PAREN"):
                arguments[-1].append(token)
        yield text, arguments


_MONTHS = re.compile(r"\s*[+-]?\s*\d+\s+(month|year)s?\s*", re.I)


def _whole(node) -> int | None:
    """The value of a whole-number literal (``2``, ``-2``), or ``None``."""

    literal = node.this if isinstance(node, exp.Neg) else node
    if not (isinstance(literal, exp.Literal) and not literal.is_string and literal.name.isdigit()):
        return None
    return -int(literal.name) if isinstance(node, exp.Neg) else int(literal.name)


def _substring_differs(node, families: set[str]) -> bool:
    """Whether the two engines can answer this ``SUBSTR`` differently.

    From running ``SUBSTR(s, i)`` and ``SUBSTR(s, i, n)`` on every engine for short strings and every
    small ``i`` and ``n``: all agree for a position of 1 or more and a length of 0 or more; SQLite and
    DuckDB agree everywhere; from a negative position ClickHouse agrees with them and MySQL and
    PostgreSQL do not; from position 0 PostgreSQL agrees with them and MySQL and ClickHouse do not; a
    negative length is each engine's own. A position or length that is not a literal can be any of these.
    """

    if families == {"sqlite", "duckdb"}:
        return False
    start = _whole(node.args["start"]) if node.args.get("start") is not None else None
    length = node.args.get("length")
    if start is None or (length is not None and (_whole(length) is None or _whole(length) < 0)):
        return True
    if start >= 1:
        return False
    return not families <= ({"sqlite", "duckdb", "clickhouse"} if start < 0 else {"sqlite", "duckdb", "postgresql"})


def _engine_specific(pair: Pair, trees: list, words: set[str], types: "_Types") -> str | None:
    """A call that the pair's two engines compute differently although sqlglot reads it as one function (named).

    Each rule is a difference seen by running one statement on the engines themselves
    (docs/evals/dlbench.md has a made-up statement for each):

    * ``ROUND``: a half goes away from zero in SQLite and DuckDB and to the even neighbour in MySQL,
      PostgreSQL and ClickHouse for a double; ``ROUND(2.675, 2)`` is 2.67 in SQLite and 2.68 elsewhere.
    * ``LOG(x)``: the natural logarithm in MySQL and ClickHouse, base 10 in PostgreSQL and DuckDB.
    * A logarithm, root or power outside its domain is NULL in MySQL and NaN in ClickHouse.
    * ``SUBSTR`` from position 0 or a negative position, or with a negative length (``_substring_differs``).
    * ``CAST(x AS CHAR)`` without a length is one character in PostgreSQL.
    * Bitwise operators work on unsigned 64-bit numbers in MySQL (``~3`` is 18446744073709551612).
    * ``DIV``: sqlglot does not write MySQL's integer division back as it read it.
    * ``HEX`` of a number: its digits in MySQL and DuckDB, padded to whole bytes in ClickHouse, the hex
      of its decimal text in SQLite. Of a string literal or a column every table declares as text, it
      is the same everywhere.
    * SQLite's ``date(x, '+1 month')`` runs past the end of a shorter month (January 31 gives March 2);
      the others stop at its last day.
    * ``generate_series`` and other calls that return rows in one engine and a list in another.
    """

    engines = {pair.source_dbms, pair.target_dbms}
    tokens = [token for query, dbms in ((pair.source_query, pair.source_dbms), (pair.target_query, pair.target_dbms)) for token in _tokens(query, dbms) or ()]
    if words & SET_RETURNING:
        return "set-returning function"
    if "round" in words:
        return "rounding"
    if "log" in words:
        return "logarithm base"
    hexes = [node for tree in trees for node in tree.find_all(exp.Hex, exp.LowerHex)]
    if len(hexes) != len(list(_calls(tokens, {"hex"}))) or any(
        not ((isinstance(node.this, exp.Literal) and node.this.is_string) or (isinstance(node.this, exp.Column) and node.this.name.lower() in types.text))
        for node in hexes
    ):
        return "HEX of a number"
    if pair.source_dbms == "sqlite" and any(
        _MONTHS.fullmatch(text) if "STRING" in kind else text == "interval" for kind, text in _tokens(pair.source_query, "sqlite") or ()
    ):
        return "SQLite month arithmetic"
    if "clickhouse" in engines:
        for _, arguments in _calls(tokens, NAN_IN_CLICKHOUSE):
            if not all(kind in ("NUMBER", "DASH") for argument in arguments for kind, _ in argument):
                return "ClickHouse NaN"
    families = {_family(pair.source_dbms), _family(pair.target_dbms)}
    substrings = [node for tree in trees for node in tree.find_all(exp.Substring)]
    if len(substrings) != len(list(_calls(tokens, SUBSTRINGS))) and families != {"sqlite", "duckdb"}:
        return "substring position"  # a spelling sqlglot did not read as SUBSTR
    if any(_substring_differs(node, families) for node in substrings):
        return "substring position"
    for tree in trees:
        for node in tree.walk():
            if isinstance(node, BITWISE):
                return "bitwise operators"
            if isinstance(node, exp.IntDiv):
                return "integer division"
            if isinstance(node, exp.Cast) and node.to.this in (exp.DataType.Type.CHAR, exp.DataType.Type.NCHAR) and not node.to.expressions:
                return "CHAR without a length"
    return None


_NUMERIC_TYPE = re.compile(r"\s*((tiny|small|medium|big)?int(eger)?[248]?|real|float[48]?|double|dec(imal)?|numeric|number|bit|bool(ean)?|year|serial)\b")
_TEXT_TYPE = re.compile(r"\s*((var)?char|(tiny|medium|long)?text|string|character)\b")
_BOOLEAN = (
    exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.NullSafeNEQ, exp.In, exp.Between, exp.Is, exp.And, exp.Or,
    exp.Not, exp.Exists, exp.Like, exp.ILike, exp.Boolean,
)
_ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.IntDiv, exp.Mod)
_NUMERIC_OF_NUMERIC = (exp.Sum, exp.Avg, exp.Min, exp.Max, exp.Abs, exp.Round, exp.Floor, exp.Ceil, exp.Coalesce, exp.Neg, exp.Paren, exp.Sign, exp.Sqrt)
_NUMERIC_TARGETS = {
    exp.DataType.Type.INT, exp.DataType.Type.BIGINT, exp.DataType.Type.SMALLINT, exp.DataType.Type.TINYINT, exp.DataType.Type.FLOAT,
    exp.DataType.Type.DOUBLE, exp.DataType.Type.DECIMAL, exp.DataType.Type.UINT, exp.DataType.Type.UBIGINT,
}


class _Types:
    """What the source's schema text says about a column name: a number in every table that has it, or not."""

    def __init__(self, schema: dict[str, dict[str, str]]):
        declared: dict[str, list[str]] = defaultdict(list)
        for columns in schema.values():
            for column, kind in columns.items():
                declared[column.lower()].append(kind)
        self.declared = set(declared)
        self.text = {c for c, kinds in declared.items() if all(_TEXT_TYPE.match(k.lower()) for k in kinds)}
        self.numeric = {c for c, kinds in declared.items() if all(_NUMERIC_TYPE.match(k.lower()) for k in kinds)}


def _is_number(node, types: _Types, outputs: dict[str, list], depth: int = 0) -> bool:
    """Whether ``node`` is provably a number (or NULL): never a string, whatever the tables hold."""

    if depth > 12:
        return False
    again = lambda n: _is_number(n, types, outputs, depth + 1)  # noqa: E731
    if isinstance(node, exp.Null) or isinstance(node, _BOOLEAN):
        return True
    if isinstance(node, exp.Literal):
        return not node.is_string
    if isinstance(node, exp.Column):
        name = node.name.lower()
        stands_for = [e for e in outputs.get(name, ()) if not (isinstance(e, exp.Column) and e.name.lower() == name)]
        if not all(again(e) for e in stands_for):  # an output name of the query itself: what it stands for decides too
            return False
        return name in types.numeric if name in types.declared or not stands_for else True
    if isinstance(node, exp.Count) or isinstance(node, (exp.Length, exp.Year, exp.Month, exp.Day)):
        return True
    if isinstance(node, _ARITHMETIC):
        return again(node.this) and again(node.expression)
    if isinstance(node, (*_NUMERIC_OF_NUMERIC, exp.Least, exp.Greatest)):
        return all(again(e) for e in [node.this, *node.expressions] if isinstance(e, exp.Expression))
    if isinstance(node, exp.Cast):
        return node.to.this in _NUMERIC_TARGETS and again(node.this)
    if isinstance(node, exp.Case):
        results = [i.args["true"] for i in node.args.get("ifs", [])] + ([node.args["default"]] if node.args.get("default") is not None else [])
        return bool(results) and all(again(r) for r in results)
    if isinstance(node, exp.If):
        return all(again(node.args[k]) for k in ("true", "false") if node.args.get(k) is not None)
    if isinstance(node, exp.Subquery) and isinstance(node.this, exp.Select) and len(node.this.expressions) == 1:
        return again(node.this.expressions[0].unalias())
    if isinstance(node, exp.Distinct):
        return all(again(e) for e in node.expressions)
    return False


def _plain_text(node) -> bool:
    """A string literal that every collation reads alike: ASCII, no space at either end."""

    return isinstance(node, exp.Literal) and node.is_string and node.name.isascii() and node.name == node.name.strip()


def _string_site(tree, types: _Types, literals_agree: bool) -> str | None:
    """Where ``tree`` may compare, group, order or convert a string (described), or ``None``.

    Between a database that compares text without case and one that compares it exactly
    (``literals_agree`` false), a proof carries over only when no string is ever compared: every
    operand of a comparison, IN, BETWEEN or CASE, every key of GROUP BY, ORDER BY, DISTINCT, a window
    or a UNION, every MIN and MAX, and every operand of arithmetic or a condition (a string would be
    converted, and the databases convert differently) is provably a number. LIKE and the functions
    that compare strings are always such a place. A column counts as a number only when every table of
    the source's schema that has a column of that name declares it one; anything unknown is a string.

    Between MySQL and MariaDB (``literals_agree``) only the collation differs (trailing spaces,
    some letters), so two plain ASCII literals may meet, conversions are alike, and only a string that
    is not such a literal makes a place unsafe.
    """

    outputs: dict[str, list] = defaultdict(list)
    for alias in tree.find_all(exp.Alias):
        if alias.alias:
            outputs[alias.alias.lower()].append(alias.this)

    def fine(node) -> bool:
        if _is_row(node):  # a row value of such fields; _row_compared has the rules about rows themselves
            return all(fine(e) for e in node.expressions)
        return _is_number(node, types, outputs) or (literals_agree and _plain_text(node))

    def keyed(select, key):
        """An ORDER BY or GROUP BY key that is a position or an output name stands for that output."""

        if isinstance(key, exp.Literal) and not key.is_string and key.name.isdigit() and isinstance(select, exp.Select):
            index = int(key.name) - 1
            return select.expressions[index].unalias() if 0 <= index < len(select.expressions) else key
        return key

    for node in tree.walk():
        if isinstance(node, (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.NullSafeNEQ)):
            sides = [node.this, node.expression]
        elif isinstance(node, exp.In):
            query = node.args.get("query")
            sides = [node.this, *node.expressions, *([query] if query is not None else [])]
            if node.args.get("field") is not None or node.args.get("unnest") is not None:
                return "IN"
        elif isinstance(node, exp.Between):
            sides = [node.this, node.args["low"], node.args["high"]]
        elif isinstance(node, (exp.Like, exp.ILike, exp.RegexpLike, exp.RegexpILike, exp.SimilarTo, exp.Glob)):
            if literals_agree and fine(node.this) and fine(node.expression):
                continue
            return "LIKE"
        elif isinstance(node, exp.Case) and node.args.get("this") is not None:
            sides = [node.this, *[i.this for i in node.args.get("ifs", [])]]
        elif isinstance(node, exp.Select) and node.args.get("distinct") is not None:
            sides = [e.unalias() for e in node.expressions]
        elif isinstance(node, exp.Group):
            sides = [keyed(node.parent, k) for k in node.expressions]
        elif isinstance(node, exp.Order):
            sides = [keyed(node.parent, o.this) for o in node.expressions]
        elif isinstance(node, (exp.Min, exp.Max)):
            sides = [node.this]
        elif isinstance(node, exp.Distinct):
            sides = list(node.expressions)
        elif isinstance(node, exp.SetOperation):
            if node.args.get("distinct") is False and isinstance(node, exp.Union):
                continue
            sides = [e.unalias() for side in (node.this, node.expression) for e in (side.selects if hasattr(side, "selects") else [])] or [node]
        elif isinstance(node, exp.Join) and (node.args.get("using") or node.args.get("method") == "NATURAL" or node.kind == "NATURAL"):
            sides = list(node.args.get("using") or [node])
            sides = [exp.column(u.name) if isinstance(u, exp.Identifier) else u for u in sides]
        elif isinstance(node, exp.Window):
            sides = list(node.args.get("partition_by") or [])
        elif isinstance(node, (exp.Nullif, exp.Greatest, exp.Least, exp.StrPosition)) or (
            isinstance(node, exp.Anonymous) and node.name.lower() in STRING_COMPARERS
        ):
            sides = [e for e in [node.args.get("this"), node.args.get("expression"), node.args.get("substr"), *node.expressions] if isinstance(e, exp.Expression)]
        elif not literals_agree and isinstance(node, _ARITHMETIC):
            sides = [node.this, node.expression]
        elif not literals_agree and isinstance(node, (exp.Sum, exp.Avg)):
            sides = [node.this]
        elif not literals_agree and isinstance(node, exp.Cast) and node.to.this in _NUMERIC_TARGETS:
            sides = [node.this]
        elif not literals_agree and isinstance(node, (exp.And, exp.Or, exp.Not, exp.Where, exp.Having)):
            sides = [e for e in (node.this, node.args.get("expression")) if isinstance(e, exp.Expression)]
        elif not literals_agree and isinstance(node, exp.If):
            sides = [node.this]
        else:
            continue
        for side in sides:
            if isinstance(side, exp.Star) or (isinstance(side, exp.Column) and isinstance(side.this, exp.Star)) or not fine(side):
                return f"{type(node).__name__.lower()} of {side.sql()[:40]}"
    return None


def _is_row(node) -> bool:
    return isinstance(node, (exp.Tuple, exp.Struct)) or (isinstance(node, exp.Anonymous) and node.name.upper() == "ROW")


def _row_compared(tree, types: _Types, strict: bool) -> bool:
    """Whether a row value is compared in a way the databases answer differently.

    ``ROW(1, NULL) IN (...)``, ``(a, b) = (1, 2)``: MariaDB compares the fields of an IN list of rows
    in one common type (``'b'`` and ``'a'`` are both 0 beside a number), and DuckDB compares a NULL
    inside a nested row as a value where MySQL answers NULL. So a compared row value is a gap when a
    field is not provably a number (or NULL, or another such row), and always (``strict``) where one of
    the databases has its own struct comparison or cannot be replayed.
    """

    row = _is_row

    def numbers(node) -> bool:
        return all(numbers(e) if row(e) else _is_number(e, types, {}) for e in node.expressions)

    for node in tree.walk():
        compared = row(node) and (isinstance(node.parent, ROW_COMPARISONS) or (isinstance(node, exp.Anonymous) and not row(node.parent)))
        if compared and (strict or not numbers(node)):
            return True
    return False


def _epoch_outside(sql: str, dbms: str) -> bool:
    """Whether ``sql`` converts to or from Unix time anything but a literal inside the range both engines keep."""

    tokens = _tokens(sql, dbms) or []
    for index, (_, text) in enumerate(tokens):
        if text == "epoch" and tokens[index + 1:index + 2] == [("FROM", "from")]:
            return True  # EXTRACT(EPOCH FROM ...)
        if text not in EPOCH_CALLS or tokens[index + 1:index + 2] != [("L_PAREN", "(")]:
            continue
        depth, inside = 0, []
        for kind, value in tokens[index + 1:]:
            depth += (kind == "L_PAREN") - (kind == "R_PAREN")
            if depth == 0:
                break
            inside.append((kind, value))
        first = inside[1:2]  # after the opening parenthesis
        if text in ("unix_timestamp", "tounixtimestamp", "epoch", "unixepoch"):
            year = re.match(r"\s*(\d{4})-\d{1,2}-\d{1,2}", first[0][1]) if len(inside) == 2 and "STRING" in first[0][0] else None
            if year is None or int(year.group(1)) not in EPOCH_YEARS:
                return True
        elif not (first and first[0][0] == "NUMBER" and first[0][1].isdigit() and 86400 <= int(first[0][1]) < 2**31 - 86400 and (len(inside) == 2 or inside[2][0] == "COMMA")):
            return True
    return False


@functools.lru_cache(maxsize=1)
def contradicted() -> dict[str, dict]:
    """Pairs whose proof a native replay contradicted (``native_contradictions.json``): never counted as proved."""

    if not CONTRADICTIONS.exists():
        return {}
    return json.loads(CONTRADICTIONS.read_text(encoding="utf-8"))


_TEXT_VALUED = (exp.TimeToStr, exp.Substring, exp.Trim, exp.Upper, exp.Lower, exp.DPipe, exp.Concat, exp.TsOrDsToDate, exp.Date, exp.Replace)
_TEXT_NAMES = {"strftime", "date", "datetime", "time", "substr", "substring", "replace", "ltrim", "rtrim", "printf", "format", "hex"}


def _text_against_number(sql: str) -> bool:
    """Whether a SQLite query compares the text a function returns with a bare number.

    SQLite does not convert here: ``STRFTIME('%Y', d) < 1997`` is false for every row, because any
    text sorts above any number, and ``STRFTIME('%Y', d) = 1997`` is false too. The other databases
    convert one side, so a translation that keeps the comparison means something else, and the
    provers do not model SQLite's rule. No proof is taken for such a query, whatever its label says.
    """

    tree = parse(sql, "sqlite")
    if tree is None:
        return False

    def text(node) -> bool:
        node = node.unnest()
        return isinstance(node, _TEXT_VALUED) or (isinstance(node, exp.Anonymous) and node.name.lower() in _TEXT_NAMES)

    def number(node) -> bool:
        node = node.unnest()
        return (isinstance(node, exp.Literal) and not node.is_string) or isinstance(node, (exp.Neg, exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Count, exp.Sum, exp.Avg))

    for node in tree.walk():
        if isinstance(node, (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)):
            sides = [(node.this, node.expression)]
        elif isinstance(node, exp.Between):
            sides = [(node.this, node.args["low"]), (node.this, node.args["high"])]
        elif isinstance(node, exp.In):
            sides = [(node.this, e) for e in node.expressions]
        else:
            continue
        if any((text(a) and number(b)) or (number(a) and text(b)) for a, b in sides):
            return True
    return False


def _texty(declared: str) -> bool:
    return any(w in declared.lower() for w in ("char", "text", "clob", "date", "time", "string", "blob")) or not declared


def _ordered(sql: str, dialect: str) -> bool:
    """Whether the query ends in ORDER BY; ``dialect`` is sqlglot's name (``postgres``), not the database's."""

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return False
    while isinstance(tree, exp.Subquery):
        tree = tree.this
    return tree is not None and tree.args.get("order") is not None


def _with_unbounded_limit(sql: str, dialect: str) -> str:
    tree = sqlglot.parse_one(sql, read=dialect)
    top = tree
    while isinstance(top, exp.Subquery):
        top = top.this
    if top.args.get("order") is not None and top.args.get("limit") is None and top.args.get("offset") is None:
        top.set("limit", exp.Limit(expression=exp.Literal.number(UNBOUNDED)))
    return tree.sql(dialect=dialect)


def prove(left: str, right: str, schema: dict[str, dict[str, str]], dialect: str) -> bool:
    from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
    from kumo_sql_prover.smt_equivalence import SmtStatus

    if _ordered(left, dialect):
        if not _ordered(right, dialect):
            return False  # a list against an unordered result
        left, right = _with_unbounded_limit(left, dialect), _with_unbounded_limit(right, dialect)
    try:
        result = prove_equivalent_algebraic(
            left, right, schema={t: list(c) for t, c in schema.items()} or None, compare_names=False,
            dialect=dialect, timeout_ms=PROVER_TIMEOUT_MS,
        )
    except Exception:  # a crash is a failure to prove, never a proof
        return False
    return result.status is SmtStatus.PROVEN_EQUIVALENT


# -- the same statement ---------------------------------------------------------------


def _tokens(sql: str, dbms: str) -> list[tuple[str, str]] | None:
    """The statement's tokens: comments and the closing semicolon dropped, everything but a literal lower-case."""

    try:
        tokens = sqlglot.tokenize(sql, read=SQLGLOT[dbms])
    except Exception:
        return None
    while tokens and tokens[-1].token_type.name == "SEMICOLON":
        tokens.pop()
    return [(t.token_type.name, t.text if "STRING" in t.token_type.name else t.text.lower()) for t in tokens]


def volatile(sql: str, dbms: str) -> str | None:
    """A call in ``sql`` whose answer is not a function of the tables (named), or ``None``.

    The provers read a function they do not know as a function of its arguments, which is wrong for
    the clock, random values, sequences and the session: ``nextval('q') = nextval('q')`` is never true.
    sqlglot keeps most of these as plain calls outside BigQuery, so the harness refuses them by name,
    from the tokens of each query in its own dialect.
    """

    tokens = _tokens(sql, dbms)
    for index, (kind, text) in enumerate(tokens or ()):
        called = tokens[index + 1:index + 2] == [("L_PAREN", "(")]
        if "STRING" in kind:
            if text.strip().lower() in VOLATILE_TEXTS:
                return f"'{text.strip().lower()}'"
        elif kind in ("PARAMETER", "SESSION_PARAMETER") and SQLGLOT[dbms] == "mysql":
            return "a variable"
        elif text in VOLATILE_WORDS or (called and text in VOLATILE_CALLS):
            return text
        elif called and text in VOLATILE_NO_ARGUMENT and tokens[index + 2:index + 3] == [("R_PAREN", ")")]:
            return f"{text}()"
    return None


def same_statement(pair: Pair) -> bool:
    """Whether the translation is its source, token for token, and one run of it says what another would.

    Only between MySQL and MariaDB, which this harness reads as one dialect: there the translation
    often changes nothing but letter case and spacing, and the provers have nothing to read (a column
    of a table whose definition was not published, an INSERT, a CREATE TABLE AS). The tokens are
    compared, not the trees, so nothing rests on how sqlglot parses the statement. Refused: anything
    that is not a query or a data change (EXPLAIN's output is the engine's own); a statement whose
    answer can change between two runs (the clock, random values, the session, variables, locks,
    ``GROUP_CONCAT`` and window functions, ``LIMIT`` without ``ORDER BY``); and a string literal that
    ends in a space, because MySQL 8's default collation compares trailing spaces and MariaDB's
    ignores them.
    """

    if pair.source_dbms not in SAME_READING or pair.target_dbms not in SAME_READING:
        return False
    left, right = _tokens(pair.source_query, pair.source_dbms), _tokens(pair.target_query, pair.target_dbms)
    if not left or left != right:
        return False
    tree = parse(pair.source_query, pair.source_dbms)
    if not isinstance(tree, (exp.Query, exp.Insert, exp.Create, exp.Update, exp.Delete)):
        return False
    return not _rerun_dependent(pair.source_query, pair.source_dbms)


def _rerun_dependent(sql: str, dbms: str) -> bool:
    """Whether two runs of the statement need not do the same thing (or it cannot be told)."""

    tokens, tree = _tokens(sql, dbms), parse(sql, dbms)
    if not tokens or tree is None:
        return True
    for index, (kind, text) in enumerate(tokens):
        if kind in ("PARAMETER", "SESSION_PARAMETER", "PLACEHOLDER") or text in RERUN_DEPENDENT:
            return True
        if text in VOLATILE_NO_ARGUMENT and tokens[index + 1:index + 3] == [("L_PAREN", "("), ("R_PAREN", ")")]:
            return True
        if "STRING" in kind and text != text.rstrip():
            return True
    from kumo_sql_prover.smt_equivalence import _NONDETERMINISTIC_NAMES, _NONDETERMINISTIC_TYPES

    for node in tree.walk():
        if type(node).__name__ in _NONDETERMINISTIC_TYPES or isinstance(node, (exp.Window, exp.Parameter, exp.SessionParameter, exp.Placeholder, exp.TableSample)):
            return True
        if isinstance(node, exp.Anonymous) and node.name.upper() in _NONDETERMINISTIC_NAMES:
            return True
        if (node.args.get("limit") or node.args.get("offset")) and not node.args.get("order"):
            return True
    return False


def _inner_query(tree):
    """The query an ``INSERT`` or a ``CREATE TABLE ... AS`` takes its rows from, or ``None`` for any other statement."""

    if isinstance(tree, exp.Insert) or (isinstance(tree, exp.Create) and str(tree.args.get("kind") or "").upper() == "TABLE"):
        inner = tree.args.get("expression")
        return inner if isinstance(inner, (exp.Query, exp.Values)) else None
    return None


def _rows_as_select(inner):
    """``VALUES (1, 'a'), (2, 'b')`` as ``SELECT 1, 'a' UNION ALL SELECT 2, 'b'``; a query as it is."""

    if not isinstance(inner, exp.Values):
        return inner
    selects = []
    for row in inner.expressions:
        if not isinstance(row, exp.Tuple) or not row.expressions:
            return None
        selects.append(exp.Select(expressions=[exp.alias_(value.unnest().copy(), f"c{i}") for i, value in enumerate(row.expressions)]))
    if not selects or len({len(select.expressions) for select in selects}) != 1:
        return None
    query = selects[0]
    for select in selects[1:]:
        query = exp.Union(this=query, expression=select, distinct=False)
    return query


def statement_proof(pair: Pair, left: str, right: str, schema: dict[str, dict[str, str]]) -> str | None:
    """How an ``INSERT`` or ``CREATE TABLE ... AS`` pair is decided although the provers read queries only, or ``None``.

    ``left`` and ``right`` are the two statements as sqlglot writes them in the source's dialect (the
    translation's names read back), so the dialect-gap rules have passed already.

    * ``"same statement, read back"``: the two are then the same statement token for token, the way
      ``INSERT INTO t1 (data) VALUES (repeat('a',1*1024))`` and its re-capitalised translation are.
    * ``"inner query proved"``: everything outside the inner query is the same text (the table, the
      column list, ``IGNORE``, every table property), and the provers prove the two inner queries
      equal, ``VALUES`` rows read as ``SELECT ... UNION ALL SELECT ...``.

    Refused, as by the same-statement rule: a statement whose second run need not do what the first
    did (the clock, random values, the session, variables, ``GROUP_CONCAT``, windows, ``LIMIT``
    without ``ORDER BY``) on either side, and any statement that is not an ``INSERT`` or a
    ``CREATE TABLE ... AS`` (a table definition means what its engine's types mean). The decision is
    about the rows the statement writes: what the table's column types then do to them (rounding,
    truncation, an error) is the engine's own and is not modelled.
    """

    dialect = SQLGLOT[pair.source_dbms]
    if "monetdb" in (pair.source_dbms, pair.target_dbms):
        return None  # read as PostgreSQL and never run: a statement is not decided on a name alone
    try:
        trees = [sqlglot.parse_one(left, read=dialect), sqlglot.parse_one(right, read=dialect)]
    except Exception:
        return None
    inner = [_inner_query(tree) for tree in trees]
    if inner[0] is None or inner[1] is None or type(trees[0]) is not type(trees[1]):
        return None
    if _rerun_dependent(pair.source_query, pair.source_dbms) or _rerun_dependent(pair.target_query, pair.target_dbms):
        return None
    if _tokens(left, pair.source_dbms) == _tokens(right, pair.source_dbms):
        return "same statement, read back"
    outside = []
    for tree in trees:
        rest = tree.copy()
        rest.set("expression", exp.Placeholder())
        outside.append(_tokens(rest.sql(dialect=dialect), pair.source_dbms))
    queries = [_rows_as_select(query) for query in inner]
    if outside[0] is None or outside[0] != outside[1] or queries[0] is None or queries[1] is None:
        return None
    if isinstance(trees[0], exp.Create):  # the inner query's output names are the new table's columns
        try:
            names = [[name.lower() for name in query.named_selects] for query in inner]
        except Exception:
            return None
        if names[0] != names[1] or any(e.find(exp.Star) for query in inner for e in query.selects):
            return None
    try:
        proved = prove(queries[0].sql(dialect=dialect), queries[1].sql(dialect=dialect), schema, dialect)
    except Exception:
        return None
    return "inner query proved" if proved else None


# -- checking proofs by running them --------------------------------------------------


def _random_rows(schema: dict[str, dict[str, str]], rng: random.Random, literals: list) -> dict[str, list[tuple]]:
    numbers = [0, 1, 2, 3, -1] + [v for v in literals if isinstance(v, (int, float))]
    whole = [v for v in numbers if isinstance(v, int)]  # an INTEGER column holds no 0.1: the translation's could not
    texts = ["a", "b", "A"] + [v for v in literals if isinstance(v, str)]
    data = {}
    for table, columns in schema.items():
        rows = []
        for _ in range(rng.choice([0, 1, 2, 3, 4, 6])):
            rows.append(tuple(
                None if rng.random() < 0.1 else rng.choice(texts if _texty(k) else numbers if _fractional(k) else whole)
                for k in columns.values()
            ))
        data[table] = rows
    return data


def _literals(*queries: str) -> list:
    values: list = []
    for sql in queries:
        tree = parse(sql, "sqlite")
        for literal in tree.find_all(exp.Literal) if tree is not None else ():
            if literal.is_string:
                values.append(literal.name)
            else:
                try:
                    number = float(literal.name)
                    values.extend([int(number) if number.is_integer() else number])
                except ValueError:
                    pass
    return list(dict.fromkeys(values))[:20]


def _bag(rows) -> Counter:
    def norm(v):
        if isinstance(v, float):
            return round(v, 6) + 0.0
        if isinstance(v, datetime.datetime):
            return v.isoformat(sep=" ")
        if isinstance(v, datetime.date):  # DuckDB's DATE against the ISO text SQLite's date() returns
            return v.isoformat()
        return v

    return Counter(tuple(norm(v) for v in row) for row in rows)


def check_proof(pair: Pair, left: str, right: str, schema: dict[str, dict[str, str]]) -> str | None:
    """A database on which a proved BIRDTrans pair differs (described), or ``None``.

    Both queries (the translation written as SQLite) run in SQLite; a DuckDB translation also runs as
    published in DuckDB. Results are compared as bags (lists when the source ends in ORDER BY and the
    ordering covers ties, which a second load in reverse row order checks).
    """

    import sqlite3

    if pair.source_dbms != "sqlite" or not schema:
        return None
    rng = random.Random(5)
    literals = _literals(left, right)
    ordered = _ordered(left, "sqlite")
    native = None
    if pair.target_dbms == "duckdb":
        try:
            import duckdb

            native = _plain(parse(pair.target_query, "duckdb"), pair.renames, schema).sql(dialect="duckdb")
        except Exception:
            native = None
    for _ in range(CHECK_TRIALS):
        data = _random_rows(schema, rng, literals)
        results = []
        for reverse in (False, True):
            connection = sqlite3.connect(":memory:")
            try:
                for table, columns in schema.items():
                    connection.execute(_create(table, columns, lambda k: k or "TEXT"))
                    rows = data[table][::-1] if reverse else data[table]
                    if rows:
                        connection.executemany(f'INSERT INTO "{table}" VALUES ({", ".join("?" * len(columns))})', rows)
                results.append(tuple(connection.execute(q).fetchall() for q in (left, right)))
            except sqlite3.Error:
                return None  # a query SQLite cannot run is not checked
            finally:
                connection.close()
        (a, b), (a2, b2) = results
        if ordered and (a != a2 or b != b2):
            continue  # ties: the order is not determined
        if (a != b) if ordered else (_bag(a) != _bag(b)):
            return f"SQLite: {data}"
        if native is not None:
            from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

            db = duckdb.connect()
            try:
                for table, columns in schema.items():
                    db.execute(_create(table, columns, _duck_type))
                    if data[table]:
                        insert_rows(db, f'"{table}"', data[table])
                theirs = db.execute(native).fetchall()
                if _bag(theirs) != _bag(a) and _bag(run_unoptimized(db, native)[0]) != _bag(a):
                    return f"DuckDB: {data}"
            except Exception:
                pass  # DuckDB cannot load these values or run the query: not a difference
            finally:
                db.close()
    return None


def native_replay(pair: Pair, schema: dict[str, dict[str, str]]) -> dict | None:
    """The pair run on its two databases (``dlbench_native``), or ``None`` when ``DLBENCH_NATIVE`` is not set."""

    import dlbench_native as native

    if native.configured() is None:
        return None
    source, target = parse(pair.source_query, pair.source_dbms), parse(pair.target_query, pair.target_dbms)
    try:
        source, target = _plain(source, {}), _plain(target, pair.renames, schema)
    except Exception:
        return {"route": f"{pair.source_dbms}>{pair.target_dbms}", "compared": 0, "difference": None, "skipped": "not written"}
    values: list = []
    for literal in [*source.find_all(exp.Literal), *target.find_all(exp.Literal)]:
        if literal.is_string:
            values.append(literal.name)
        else:
            try:
                number = float(literal.name)
                values.append(int(number) if number.is_integer() else number)
            except (ValueError, OverflowError):
                pass
    top = source
    while isinstance(top, exp.Subquery):
        top = top.this
    return native.replay(
        pair.source_dbms, pair.target_dbms, source, target, schema, SQLGLOT,
        ordered=top.args.get("order") is not None, literals=list(dict.fromkeys(values))[:20],
    )


def _create(table: str, columns: dict[str, str], typed) -> str:
    definitions = ", ".join(f'"{column}" {typed(declared)}' for column, declared in columns.items())
    return f'CREATE TABLE "{table}" ({definitions})'


def _fractional(declared: str) -> bool:
    return any(w in declared.lower() for w in ("real", "float", "double", "dec", "num"))


def _duck_type(declared: str) -> str:
    if _texty(declared):
        return "VARCHAR"
    return "DOUBLE" if _fractional(declared) else "BIGINT"


# -- deciding -------------------------------------------------------------------------


def decide(pair: Pair) -> dict:
    started = time.time()
    source, target = parse(pair.source_query, pair.source_dbms), parse(pair.target_query, pair.target_dbms)
    parsed = {"source": source is not None, "target": target is not None}
    outcome, detail, wrong = "unsupported", "", False
    if source is not None and target is not None:
        left, right = as_source(pair)
        schema = schema_of(pair)
        if left is None or right is None:
            detail = "not written in the source dialect"
        elif (gap := dialect_gap(pair, left, right, schema)) is not None:
            outcome, detail = "unknown", f"dialect gap: {gap}"
        elif (call := volatile(pair.source_query, pair.source_dbms) or volatile(pair.target_query, pair.target_dbms)) is not None:
            outcome, detail = "unknown", f"volatile: {call}"
        elif pair.id in contradicted():
            outcome, detail = "unknown", f"contradicted natively: {contradicted()[pair.id]['route']}"
        elif prove(left, right, schema, SQLGLOT[pair.source_dbms]):
            outcome = "proven"
            difference = check_proof(pair, left, right, schema)
            if difference is not None:
                wrong, detail = True, difference
        elif same_statement(pair):
            outcome, detail = "proven", "same statement"
        elif (how := statement_proof(pair, left, right, schema)) is not None:
            outcome, detail = "proven", how
        else:
            outcome = "unknown"
    else:
        detail = "parse: " + ", ".join(k for k, v in parsed.items() if not v)
    replayed = native_replay(pair, schema) if outcome == "proven" and not wrong else None
    if replayed is not None and replayed["difference"]:
        wrong, detail = True, replayed["difference"]
    return {
        **({"replay": {k: v for k, v in replayed.items() if k != "difference"}} if replayed is not None else {}),
        "id": pair.id, "dataset": pair.dataset, "target": pair.target_dbms, "source": pair.source_dbms,
        "label": pair.label, "parsed": all(parsed.values()), **{f"parsed_{k}": v for k, v in parsed.items()},
        "outcome": outcome, "detail": detail, "wrong": wrong, "held_out": pair.held_out,
        "seconds": round(time.time() - started, 2),
    }


def run(pairs: list[Pair], jobs: int = 1) -> list[dict]:
    if jobs > 1 and len(pairs) > 1:
        with ProcessPoolExecutor(jobs) as pool:
            return list(pool.map(decide, pairs, chunksize=4))
    return [decide(p) for p in pairs]


def table(results: list[dict]) -> list[str]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        groups[(r["dataset"], r["target"])].append(r)
    lines = [f"{'dataset':12} {'target':11} {'pairs':>5} {'parsed':>7} {'proven':>7} {'gap':>5} {'approx proven':>14} {'wrong':>6}"]
    for (dataset, target), rows in sorted(groups.items()):
        lines.append(
            f"{dataset:12} {target:11} {len(rows):5} {sum(r['parsed'] for r in rows):7} "
            f"{sum(r['outcome'] == 'proven' for r in rows):7} {sum(r['detail'].startswith('dialect gap') for r in rows):5} "
            f"{sum(r['outcome'] == 'proven' and r['label'] == 'approximate' for r in rows):14} {sum(r['wrong'] for r in rows):6}"
        )
    return lines


def results_rows(results: list[dict]) -> dict[str, dict]:
    from bench_common import today

    exact = [r for r in results if r["label"] == "exact"]
    approx = [r for r in results if r["label"] == "approximate"]
    held = [r for r in exact if r["held_out"]]
    counts = Counter(r["outcome"] for r in results)
    proven = sum(r["outcome"] == "proven" for r in exact)
    parsed = sum(r["parsed"] for r in results)
    gaps = sum(r["detail"].startswith("dialect gap") for r in results)
    same = sum(r["outcome"] == "proven" and r["detail"] == "same statement" for r in exact)
    read_back = sum(r["outcome"] == "proven" and r["detail"] == "same statement, read back" for r in exact)
    inner = sum(r["outcome"] == "proven" and r["detail"] == "inner query proved" for r in exact)
    volatile_calls = sum(r["detail"].startswith("volatile") for r in results)
    listed = sum(r["detail"].startswith("contradicted natively") for r in results)
    return {
        "dlbench": {
            "suite": "DLBench cross-dialect translations (pinned subset)",
            "order": 35,
            "size": len(results),
            "score": f"{proven}/{len(exact)} exact pairs proved, {sum(r['wrong'] for r in results)} wrong; {parsed}/{len(results)} parsed",
            "metric": "Translations from SQLite, MySQL and PostgreSQL into six databases' dialects, proved equal to the source in the source's dialect after sqlglot reads both; parsed means both sides parse in their own dialect.",
            "evidence": "proof",
            "correctness": f"Every BIRDTrans proof is re-run on 300 random SQLite databases (DuckDB translations also natively in DuckDB); {gaps} pairs where the two databases read the same text differently (LIKE case, string collations, integer and decimal division, Unix time ranges, row values, SQLite type conversion; the docs list every rule) are left unproved, and so are {volatile_calls} that call the clock, a random value, a sequence or the session{f' and {listed} whose proof an earlier native replay contradicted' if listed else ''}. {replay_sentence(results)} {sum(r['outcome'] == 'proven' for r in approx)}/{len(approx)} 'approximate' pairs are proved (listed in the docs). {same} of the exact proofs are a MariaDB translation that is its MySQL source token for token, where the provers have nothing to read (an INSERT, a table with no published definition); statements whose answer can change between two runs are refused. {read_back} more are an INSERT or CREATE TABLE AS that, read back into the source's dialect, is the source statement token for token, and {inner} an INSERT or CREATE TABLE AS whose inner queries the provers prove equal with everything else the same text; these decide the rows written, not what the table's column types do to them.",
            "coverage": {k: counts[k] for k in ("proven", "unknown", "unsupported") if counts[k]},
            "held_out": f"{sum(r['outcome'] == 'proven' for r in held)}/{len(held)} exact pairs proved",
            "docs": "docs/evals/dlbench.md",
            "command": "python tools/dlbench_bench.py --write-results",
            "date": today(),
            "caveats": f"Pinned subset of the 6,402 pairs (every approximate pair, one in ten of the rest); run --data with a DLBench checkout for all. Labels are LLM plus human review, not proofs. Proofs are about sqlglot's reading of each dialect; MariaDB is read as MySQL and MonetDB as PostgreSQL. The 32-bit float, division-by-zero, decimal and ClickHouse rules were added after a first run on the whole subset showed two proofs contradicted natively in DuckDB (tuned on test). The 2026-11-11 harness changes (translated column and table spellings read back, the same-statement rule, the volatile-call refusal, and the gap rules that the native replay of every proof led to) were developed on the development split only; held-out pairs were replayed for their verdicts, never read.",
        },
    }


def replay_sentence(results: list[dict]) -> str:
    """What the native replay of this run covered, for the results file."""

    import dlbench_native as native

    replays = [r["replay"] for r in results if r.get("replay")]
    if not replays:
        return f"No native replay in this run ({native.ENV} not set); the docs record the last one."
    routes = Counter(x["route"] for x in replays if x["compared"])
    engines = sorted({d for route in routes for d in route.split(">")})
    versions = ", ".join(f"{d} {native.version(d) or '?'}" for d in engines)
    native.close_all()
    reasons = Counter("no MonetDB engine" if "monetdb" in (x["skipped"] or "") else "not a query" if x["skipped"] == "not a query" else "not replayable" for x in replays if not x["compared"])
    return (
        f"With {native.ENV} set, this run also replayed {sum(routes.values())} of the {sum(r['outcome'] == 'proven' for r in results)} proofs on the "
        f"databases themselves ({', '.join(f'{route} {n}' for route, n in sorted(routes.items()))}; {versions}): "
        f"{sum(bool(r['wrong'] and r['replay']['compared']) for r in results if r.get('replay'))} differences. "
        f"Not replayed: {', '.join(f'{n} {reason}' for reason, n in sorted(reasons.items())) or 'none'}."
    )


def record_contradictions(results: list[dict]) -> int:
    """Add the proofs a native replay contradicted to ``native_contradictions.json`` (ids, route and engine versions only)."""

    import dlbench_native as native
    from bench_common import today

    found = [r for r in results if r["wrong"] and r.get("replay") and r["replay"]["compared"]]
    if not found:
        return 0
    listed = dict(contradicted())
    for r in found:
        source, target = r["replay"]["route"].split(">")
        listed[r["id"]] = {"route": r["replay"]["route"], "engines": {d: native.version(d) for d in (source, target)}, "recorded": today()}
    CONTRADICTIONS.write_text(json.dumps(dict(sorted(listed.items())), indent=1) + "\n", encoding="utf-8")
    contradicted.cache_clear()
    return len(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", help=f"a DLBench checkout (commit {SOURCE_COMMIT[:12]}) to run all pairs instead of the pinned subset")
    parser.add_argument("--write-subset", action="store_true", help="rebuild tests/fixtures/dlbench/pairs.jsonl from --data")
    parser.add_argument("--split", choices=("all", "dev", "held-out"), default="all")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--show", default="", help="comma-separated outcomes to list (proven, unknown, unsupported, wrong)")
    parser.add_argument("--json", help="write every outcome to this file")
    parser.add_argument("--write-results", action="store_true", help="update benchmarks/results/dlbench.json (pinned subset, --split all)")
    parser.add_argument("--record-contradictions", action="store_true", help="with DLBENCH_NATIVE set: add every proof the native replay contradicts to tests/fixtures/dlbench/native_contradictions.json, so it is never counted again")
    args = parser.parse_args(argv)
    from bench_common import quiet, write_results

    quiet()
    if args.write_subset:
        if not args.data:
            parser.error("--write-subset needs --data")
        print(f"wrote {write_subset(Path(args.data))} pairs")
        return 0
    pairs = read_checkout(Path(args.data)) if args.data else load_pairs()
    if args.split != "all":
        pairs = [p for p in pairs if p.held_out == (args.split == "held-out")]
    started = time.time()
    results = run(pairs, args.jobs)
    print("\n".join(table(results)))
    print(f"{len(results)} pairs in {time.time() - started:.0f}s")
    show = {s.strip() for s in args.show.split(",") if s.strip()}
    for pair, result in zip(pairs, results):
        if result["outcome"] in show or (result["wrong"] and "wrong" in show) or result["wrong"]:
            print(f"\n{result['id']} [{result['label']}] {result['outcome']} {'(held out)' if pair.held_out else result['detail'][:200]}{' WRONG' if result['wrong'] else ''}")
            if not pair.held_out:  # a held-out pair is known by its id and verdict only
                print(f"  S: {pair.source_query}\n  T: {pair.target_query}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    if args.record_contradictions:
        print(f"recorded {record_contradictions(results)} contradicted proofs in {CONTRADICTIONS.name}")
    if args.write_results:
        if args.data or args.split != "all":
            parser.error("--write-results scores the pinned subset with --split all")
        for name, row in results_rows(results).items():
            write_results(name, row)
    return 1 if any(r["wrong"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
