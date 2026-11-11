"""Replay DLBench pairs on the databases themselves: the source on its engine, the translation on its own.

``tools/dlbench_bench.py`` proves a translation equal to its source under sqlglot's reading of the two
dialects. This module checks such a proof where it can be checked: it loads the same random typed
data into the source's database and the translation's and runs each query where it belongs. A
difference contradicts the proof.

It is off unless ``DLBENCH_NATIVE`` is set. The value is a JSON object naming the servers to use;
SQLite and DuckDB run in process and are always available once it is set (``{}`` uses only them)::

    export DLBENCH_NATIVE='{
      "mysql":      {"unix_socket": "/path/mysql.sock", "user": "root"},
      "mariadb":    {"unix_socket": "/path/mariadb.sock", "user": "root"},
      "postgresql": {"host": "/path/pg-socket-dir", "port": 5432, "user": "postgres", "dbname": "postgres"},
      "clickhouse": "chdb"
    }'

The MySQL and MariaDB entries are keyword arguments of ``pymysql.connect``, the PostgreSQL entry of
``psycopg.connect``; ``"chdb"`` is the ClickHouse engine embedded in the ``chdb`` package. Point it only
at throwaway servers: every replay creates a scratch database (``dlbench_<pid>``) or schema and drops it when it is done.
There is no MonetDB entry (no engine was available when this was written).

What a replay can and cannot say:

* The tables are made from the *source's* schema text, under the source's names, because the fixture
  does not carry the translated schemas. Columns are typed by what the declaration says (text, whole
  number, fraction, date, date and time); a pair that reads a column of any other declared type, or a
  table with no published definition, is not replayed.
* Both queries pass through sqlglot once, each in its own dialect, to lower-case and quote names (and
  to give the translation the source's names). The engines then run them as written.
* A query an engine refuses is not a difference: the pair is then not replayed (no trial ran) or the
  trial is skipped (some data raised an error, such as a division by zero in PostgreSQL).
* Values are compared after a light normalisation: a decimal is a float, floats are rounded to six
  places, TRUE is 1, dates are ISO text. A text ``'2017'`` is not the number 2017.
* An ordered source is compared as a list, and only on data where its ORDER BY keys tell every row
  apart: the source runs once more with the keys as extra columns and no LIMIT, and a trial with two
  rows on one key is skipped (either engine may put them in either order). A source that cuts rows
  with LIMIT and no ORDER BY is not replayed.
"""

from __future__ import annotations

from collections import Counter
import datetime
import decimal
import json
import math
import os
import random
import re

from sqlglot import exp

ENV = "DLBENCH_NATIVE"
TRIALS = 40
GIVE_UP_AFTER = 6  # trials in a row that an engine refused before any ran: the pair is not replayed
IN_PROCESS = ("sqlite", "duckdb")
TYPES = {
    "sqlite": {"text": "TEXT", "int": "INTEGER", "real": "REAL", "date": "DATE", "datetime": "DATETIME"},
    "duckdb": {"text": "VARCHAR", "int": "BIGINT", "real": "DOUBLE", "date": "DATE", "datetime": "TIMESTAMP"},
    "mysql": {"text": "TEXT", "int": "BIGINT", "real": "DOUBLE", "date": "DATE", "datetime": "DATETIME"},
    "mariadb": {"text": "TEXT", "int": "BIGINT", "real": "DOUBLE", "date": "DATE", "datetime": "DATETIME"},
    "postgresql": {"text": "TEXT", "int": "BIGINT", "real": "DOUBLE PRECISION", "date": "DATE", "datetime": "TIMESTAMP"},
    "clickhouse": {
        "text": "Nullable(String)", "int": "Nullable(Int64)", "real": "Nullable(Float64)",
        "date": "Nullable(Date32)", "datetime": "Nullable(DateTime64(0))",
    },
}
DATES = ["2016-04-30", "2017-01-01", "1999-12-31", "2018-05-31"]
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def configured() -> dict | None:
    """The servers named in ``DLBENCH_NATIVE``, or ``None`` when the replay is off."""

    text = os.environ.get(ENV)
    if not text:
        return None
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{ENV} must be a JSON object")
    return value


def kind_of(declared: str) -> str | None:
    """text, int, real, date or datetime for a declared column type; ``None`` for anything else."""

    d = declared.lower().strip()
    if not d:
        return "text"
    if any(w in d for w in ("blob", "binary", "bit", "enum", "set", "json", "year", "bool", "geometry", "interval", "[]", "array")):
        return None
    if "datetime" in d or "timestamp" in d:
        return "datetime"
    if "date" in d:
        return "date"
    if "time" in d:
        return None
    if any(w in d for w in ("char", "text", "clob", "string")):
        return "text"
    if any(w in d for w in ("real", "float", "double", "dec", "num")):
        return "real"
    if re.search(r"\b(tiny|small|medium|big)?int(eger)?\d*\b", d) or d == "serial":
        return "int"  # not "point"
    return None


# -- engines --------------------------------------------------------------------------


def _quote(name: str, dbms: str) -> str:
    return f"`{name.replace('`', '``')}`" if dbms in ("mysql", "mariadb", "clickhouse") else '"' + name.replace('"', '""') + '"'


def _create(table: str, columns: dict[str, str], dbms: str) -> str:
    body = ", ".join(f"{_quote(c, dbms)} {TYPES[dbms][k]}" for c, k in columns.items())
    return f"CREATE TABLE {_quote(table, dbms)} ({body})" + (" ENGINE = Memory" if dbms == "clickhouse" else "")


class _Sqlite:
    dbms = "sqlite"

    def __init__(self, config=None):
        self.db = None

    def reset(self, tables):
        import sqlite3

        self.close()
        self.db = sqlite3.connect(":memory:")
        for table, columns in tables.items():
            self.db.execute(_create(table, columns, self.dbms))

    def load(self, tables, data):
        for table, columns in tables.items():
            self.db.execute(f"DELETE FROM {_quote(table, self.dbms)}")
            if data[table]:
                self.db.executemany(f"INSERT INTO {_quote(table, self.dbms)} VALUES ({', '.join('?' * len(columns))})", data[table])

    def run(self, sql):
        return self.db.execute(sql).fetchall()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    drop = close


class _Duck(_Sqlite):
    dbms = "duckdb"

    def reset(self, tables):
        import duckdb

        self.close()
        self.db = duckdb.connect()
        self.db.execute("SET TimeZone = 'UTC'")
        for table, columns in tables.items():
            self.db.execute(_create(table, columns, self.dbms))


class _MySQL:
    """MySQL or MariaDB through pymysql; one scratch database per process."""

    def __init__(self, config, dbms="mysql"):
        import pymysql

        self.dbms = dbms
        self.db = pymysql.connect(**{"autocommit": True, "charset": "utf8mb4", **config})
        self.name = f"dlbench_{os.getpid()}"
        with self.db.cursor() as cursor:
            cursor.execute("SET time_zone = '+00:00'")

    def reset(self, tables):
        with self.db.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS {self.name}")
            cursor.execute(f"CREATE DATABASE {self.name} CHARACTER SET utf8mb4")
            cursor.execute(f"USE {self.name}")
            for table, columns in tables.items():
                cursor.execute(_create(table, columns, self.dbms))

    def load(self, tables, data):
        with self.db.cursor() as cursor:
            for table, columns in tables.items():
                cursor.execute(f"DELETE FROM {_quote(table, self.dbms)}")
                if data[table]:
                    cursor.executemany(f"INSERT INTO {_quote(table, self.dbms)} VALUES ({', '.join(['%s'] * len(columns))})", data[table])

    def run(self, sql):
        with self.db.cursor() as cursor:
            cursor.execute(sql)  # no arguments: a % in the query is not a parameter
            return list(cursor.fetchall())

    def drop(self):
        with self.db.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS {self.name}")

    def close(self):
        try:
            self.drop()
            self.db.close()
        except Exception:
            pass


class _MariaDB(_MySQL):
    def __init__(self, config):
        super().__init__(config, "mariadb")


class _Postgres:
    dbms = "postgresql"

    def __init__(self, config):
        import psycopg

        self.db = psycopg.connect(**config, autocommit=True)
        self.name = f"dlbench_{os.getpid()}"
        self.db.execute("SET TIME ZONE 'UTC'")

    def reset(self, tables):
        self.db.execute(f"DROP SCHEMA IF EXISTS {self.name} CASCADE")
        self.db.execute(f"CREATE SCHEMA {self.name}")
        self.db.execute(f"SET search_path TO {self.name}, pg_catalog")
        for table, columns in tables.items():
            self.db.execute(_create(table, columns, self.dbms))

    def load(self, tables, data):
        with self.db.cursor() as cursor:
            for table, columns in tables.items():
                cursor.execute(f"DELETE FROM {_quote(table, self.dbms)}")
                if data[table]:
                    cursor.executemany(f"INSERT INTO {_quote(table, self.dbms)} VALUES ({', '.join(['%s'] * len(columns))})", data[table])

    def run(self, sql):
        with self.db.cursor() as cursor:
            cursor.execute(sql.encode("utf-8"))  # bytes and no arguments: a % in the query is not a parameter
            return list(cursor.fetchall())

    def drop(self):
        self.db.execute(f"DROP SCHEMA IF EXISTS {self.name} CASCADE")

    def close(self):
        try:
            self.drop()
            self.db.close()
        except Exception:
            pass


class _ClickHouse:
    """ClickHouse embedded in the ``chdb`` package."""

    dbms = "clickhouse"
    SETTINGS = " SETTINGS output_format_json_quote_64bit_integers = 0, output_format_json_quote_denormals = 1"

    def __init__(self, config):
        if config != "chdb":
            raise ValueError('the only ClickHouse engine supported is "chdb"')
        from chdb import session

        self.session = session.Session()
        self.name = f"dlbench_{os.getpid()}"

    def reset(self, tables):
        self.session.query(f"DROP DATABASE IF EXISTS {self.name}")
        self.session.query(f"CREATE DATABASE {self.name} ENGINE = Memory")
        self.session.query(f"USE {self.name}")
        for table, columns in tables.items():
            self.session.query(_create(table, columns, self.dbms))

    @staticmethod
    def _literal(value):
        if value is None:
            return "NULL"
        if isinstance(value, str):
            return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
        return repr(value)

    def load(self, tables, data):
        for table in tables:
            self.session.query(f"TRUNCATE TABLE {_quote(table, self.dbms)}")
            if data[table]:
                rows = ", ".join("(" + ", ".join(self._literal(v) for v in row) + ")" for row in data[table])
                self.session.query(f"INSERT INTO {_quote(table, self.dbms)} VALUES {rows}")

    def run(self, sql):
        text = str(self.session.query(sql.rstrip().rstrip(";") + self.SETTINGS, "JSONCompactEachRowWithNamesAndTypes"))
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return []
        types = lines[1]
        rows = []
        for line in lines[2:]:
            rows.append(tuple(
                float(v.replace("-nan", "nan")) if isinstance(v, str) and "Float" in t and v in ("inf", "-inf", "nan", "-nan") else v
                for v, t in zip(line, types)
            ))
        return rows

    def drop(self):
        self.session.query(f"DROP DATABASE IF EXISTS {self.name}")

    def close(self):
        try:
            self.drop()
            self.session.close()
        except Exception:
            pass


_ADAPTERS = {"sqlite": _Sqlite, "duckdb": _Duck, "mysql": _MySQL, "mariadb": _MariaDB, "postgresql": _Postgres, "clickhouse": _ClickHouse}
_open: dict[str, object] = {}  # one connection per engine per worker process


def engine(dbms: str):
    """The engine for ``dbms``, or ``None`` when it is not configured (or cannot be reached)."""

    config = configured()
    if config is None or dbms not in _ADAPTERS or (dbms not in IN_PROCESS and dbms not in config):
        return None
    if dbms not in _open:
        try:
            _open[dbms] = _ADAPTERS[dbms](config.get(dbms))
        except Exception as error:  # a server that is down is reported once per replay, never a difference
            _open[dbms] = f"{type(error).__name__}: {str(error)[:120]}"
    return _open[dbms]


VERSION_QUERY = {
    "sqlite": "SELECT sqlite_version()", "duckdb": "SELECT version()", "mysql": "SELECT version()", "mariadb": "SELECT version()",
    "postgresql": "SHOW server_version", "clickhouse": "SELECT version()",
}


def version(dbms: str) -> str | None:
    adapter = engine(dbms)
    if adapter is None or isinstance(adapter, str):
        return None
    try:
        if isinstance(adapter, (_Sqlite, _Duck)) and adapter.db is None:
            adapter.reset({})
        text = str(adapter.run(VERSION_QUERY[dbms])[0][0]).split()[0]
        number = re.match(r"v?(\d+(\.\d+)*)", text)  # 8.0.46, not the distribution's build suffix
        return number.group(1) if number else text
    except Exception:
        return None


def close_all() -> None:
    for adapter in _open.values():
        if not isinstance(adapter, str):
            adapter.close()
    _open.clear()


# -- data and comparison --------------------------------------------------------------


def random_rows(tables: dict[str, dict[str, str]], rng: random.Random, literals: list) -> dict[str, list[tuple]]:
    """Rows of the declared kinds: whole numbers, fractions, short texts, ISO dates; one value in ten NULL."""

    numbers = [0, 1, 2, 3, -1] + [v for v in literals if isinstance(v, (int, float)) and abs(v) < 2**53]
    fractions = numbers + [0.5, 2.5, -1.5, 2.675]  # halves: engines round them differently
    texts = ["a", "b", "A", "a ", "abcdef"] + [v for v in literals if isinstance(v, str)]  # 'a ': collations that pad differ on it
    dates = DATES + [v for v in literals if isinstance(v, str) and _ISO_DATE.fullmatch(v)]
    pools = {
        "text": texts, "real": fractions, "int": [v for v in numbers if isinstance(v, int)], "date": dates,
        "datetime": [d + t for d in dates for t in (" 00:00:00", " 13:45:10")],
    }
    data = {}
    for table, columns in tables.items():
        rows = []
        for _ in range(rng.choice([0, 1, 2, 3, 4, 6])):
            rows.append(tuple(None if rng.random() < 0.1 else rng.choice(pools[k]) for k in columns.values()))
        data[table] = rows
    return data


def normal(value):
    """A result value in the form it is compared in."""

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, decimal.Decimal):
        value = float(value) if value.is_finite() else str(value)
    if isinstance(value, float):
        return "nan" if math.isnan(value) else value if math.isinf(value) else round(value, 6) + 0.0
    if isinstance(value, int):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return "0x" + raw.hex()
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=None).isoformat(sep=" ")
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return str(value)
    if isinstance(value, (list, tuple)):
        return repr([normal(v) for v in value])
    if isinstance(value, dict):
        return repr({k: normal(v) for k, v in value.items()})
    return repr(value)


def _rows(rows) -> list[tuple]:
    return [tuple(normal(v) for v in row) for row in rows]


def _tables_read(tree) -> set[str]:
    local = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in local}


def tie_probe(tree) -> tuple[object, list[int]] | None:
    """``tree`` with its ORDER BY keys as extra output columns and no LIMIT, and where the keys are.

    ``None`` when the keys cannot be read out this way (not a plain SELECT, or ``SELECT *``).
    """

    probe = tree.copy()
    node = probe
    while isinstance(node, exp.Subquery):
        node = node.this
    if not isinstance(node, exp.Select) or node.args.get("order") is None or any(e.find(exp.Star) for e in node.expressions):
        return None
    names = {e.alias.lower(): i for i, e in enumerate(node.expressions) if isinstance(e, exp.Alias) and e.alias}
    positions = []
    for index, ordered in enumerate(list(node.args["order"].expressions)):
        key = ordered.this
        if isinstance(key, exp.Literal) and not key.is_string and key.name.isdigit():
            positions.append(int(key.name) - 1)
        elif isinstance(key, exp.Column) and not key.table and key.name.lower() in names:
            positions.append(names[key.name.lower()])
        else:
            positions.append(len(node.expressions))
            node.append("expressions", exp.alias_(key.copy(), f"dlbench_key_{index}"))
    node.set("limit", None)
    node.set("offset", None)
    return probe, positions


def _limited_without_order(tree) -> bool:
    return any((n.args.get("limit") or n.args.get("offset")) and not n.args.get("order") for n in tree.walk())


def replay(source_dbms: str, target_dbms: str, source_tree, target_tree, schema: dict[str, dict[str, str]],
           dialects: dict[str, str], ordered: bool, literals: list, trials: int = TRIALS, seed: int = 11) -> dict:
    """Run both queries where they belong on the same data.

    ``source_tree`` and ``target_tree`` are the two queries with the source's names, lower-cased.
    Returns ``{"route", "compared", "difference", "skipped"}``: ``difference`` describes the first data
    on which the results differ; ``skipped`` says why nothing could be compared.
    """

    route = f"{source_dbms}>{target_dbms}"
    result = {"route": route, "compared": 0, "difference": None, "skipped": None}
    if not isinstance(source_tree, exp.Query) or not isinstance(target_tree, exp.Query):
        result["skipped"] = "not a query"
        return result
    engines = []
    for dbms in (source_dbms, target_dbms):
        adapter = engine(dbms)
        if adapter is None or isinstance(adapter, str):
            result["skipped"] = f"no {dbms} engine" + (f" ({adapter})" if adapter else "")
            return result
        engines.append(adapter)
    read = _tables_read(source_tree) | _tables_read(target_tree)
    tables = {}
    for table in sorted(read):
        if table not in schema:
            result["skipped"] = f"no definition of table {table}"
            return result
        kinds = {column: kind_of(declared) for column, declared in schema[table].items()}
        if None in kinds.values():
            result["skipped"] = f"a column type of {table} is not replayed"
            return result
        tables[table] = kinds
    if _limited_without_order(source_tree) or _limited_without_order(target_tree):
        result["skipped"] = "LIMIT without ORDER BY"
        return result
    queries = [tree.sql(dialect=dialects[dbms], identify=True) for tree, dbms in ((source_tree, source_dbms), (target_tree, target_dbms))]
    probe = tie_probe(source_tree) if ordered else None
    if ordered and probe is None:
        result["skipped"] = "the ORDER BY keys cannot be read out"
        return result
    probe_sql = probe[0].sql(dialect=dialects[source_dbms], identify=True) if probe else None
    try:
        for adapter in engines:
            adapter.reset(tables)
    except Exception as error:
        result["skipped"] = f"tables not created: {type(error).__name__}: {str(error)[:120]}"
        return result
    try:
        return _trials(result, engines, queries, probe, probe_sql, tables, ordered, literals, trials, seed)
    finally:  # nothing is left on a server between two replays
        for adapter in engines:
            try:
                adapter.drop()
            except Exception:
                pass


def _trials(result: dict, engines: list, queries: list[str], probe, probe_sql, tables: dict, ordered: bool, literals: list, trials: int, seed: int) -> dict:
    route = result["route"]
    rng = random.Random(seed)
    refused = 0
    for _ in range(trials if tables else 1):
        data = random_rows(tables, rng, literals)
        try:
            answers = []
            for adapter, query in zip(engines, queries):
                adapter.load(tables, data)
                answers.append(_rows(adapter.run(query)))
            keys = [tuple(row[p] for p in probe[1]) for row in _rows(engines[0].run(probe_sql))] if probe else []
        except Exception as error:
            refused += 1
            if result["compared"] == 0 and refused >= GIVE_UP_AFTER:
                result["skipped"] = f"refused: {type(error).__name__}: {' '.join(str(error).split())[:160]}"
                return result
            if not tables:
                result["skipped"] = f"refused: {type(error).__name__}: {' '.join(str(error).split())[:160]}"
                return result
            _recover(engines)
            continue
        left, right = answers
        if len(set(keys)) != len(keys):
            continue  # two rows on one ORDER BY key: the order is not determined
        result["compared"] += 1
        if (left != right) if ordered else (Counter(left) != Counter(right)):
            result["difference"] = f"{route}: {left[:5]} against {right[:5]} on {data}"[:1500]
            return result
    if result["compared"] == 0:
        result["skipped"] = "every trial was refused or tied"
    return result


def _recover(engines) -> None:
    for adapter in engines:
        if isinstance(adapter, _Postgres):
            try:
                adapter.db.rollback()
            except Exception:
                pass
