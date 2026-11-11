"""A constant timestamp plus or minus a constant interval is folded to its value (``kumo_sql_prover.timestamp_shifts``)."""

import datetime
import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.timestamp_shifts import fold_timestamp_shifts

SCHEMA = {"e": ["id", "ts", "n"]}
TYPES = {"e": {"id": "INT", "ts": "TIMESTAMP", "n": "INT"}}
BASE = "CAST('2024-10-01 12:34:56' AS TIMESTAMP)"


def folded(sql, dialect="postgres"):
    return fold_timestamp_shifts(sqlglot.parse_one(sql, read=dialect), dialect).sql(dialect=dialect)


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


@pytest.mark.parametrize("expression, value", [
    (f"{BASE} - INTERVAL '1 year'", "2023-10-01 12:34:56"),
    (f"{BASE} - INTERVAL '30' DAY", "2024-09-01 12:34:56"),
    (f"{BASE} + INTERVAL '2 months'", "2024-12-01 12:34:56"),
    (f"({BASE}) - INTERVAL '1 YEAR' - INTERVAL '1 MONTH'", "2023-09-01 12:34:56"),
    ("TIMESTAMP '2024-03-31 00:00:00' - INTERVAL '1 month'", "2024-02-29 00:00:00"),  # the day the month does not have
    ("TIMESTAMP '2024-02-29 08:00:00' - INTERVAL '1 year'", "2023-02-28 08:00:00"),
    ("TIMESTAMP '2024-01-31 08:00:00' + INTERVAL '1 month'", "2024-02-29 08:00:00"),
    ("TIMESTAMP '2024-12-31 23:59:59' + INTERVAL '1 second'", "2025-01-01 00:00:00"),
    ("TIMESTAMP '2024-10-01' - INTERVAL '36 hours'", "2024-09-29 12:00:00"),
    ("TIMESTAMP '2024-10-01T12:34:56.5' + INTERVAL '90 minutes'", "2024-10-01 14:04:56.5"),
    ("TIMESTAMP '2024-10-01 12:34:56' + INTERVAL '-2 weeks'", "2024-09-17 12:34:56"),
])
def test_constant_shifts_fold(expression, value):
    assert folded(f"SELECT e.id FROM e WHERE e.ts < {expression}") == f"SELECT e.id FROM e WHERE e.ts < CAST('{value}' AS TIMESTAMP)"


@pytest.mark.parametrize("expression", [
    f"{BASE} - INTERVAL '1 year 2 months'",  # several fields
    f"{BASE} - INTERVAL '1.5 days'",
    f"{BASE} - INTERVAL '1' QUARTER",
    f"{BASE} - e.n * INTERVAL '1 day'",
    "e.ts - INTERVAL '1 year'",  # not a constant
    "CAST('2024-10-01 12:34:56' AS TIMESTAMPTZ) - INTERVAL '1 day'",  # depends on the session's time zone
    "CAST('2024-10-01 12:34:56' AS TIMESTAMP(0)) - INTERVAL '1 day'",
    "CAST('2024-02-30 12:34:56' AS TIMESTAMP) - INTERVAL '1 day'",  # not a date
    "CAST('yesterday' AS TIMESTAMP) - INTERVAL '1 day'",
    "CAST('0001-01-01 00:00:00' AS TIMESTAMP) - INTERVAL '1 day'",  # out of range
    "CAST('9999-12-31 00:00:00' AS TIMESTAMP) + INTERVAL '1 month'",
    f"INTERVAL '1 day' + {BASE}",
    "CAST('2024-10-01' AS DATE) - INTERVAL '1 hour'",  # a date: another rule's
])
def test_other_expressions_are_left_alone(expression):
    sql = f"SELECT e.id FROM e WHERE e.ts < {expression}"
    assert folded(sql) == sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres")


def test_other_dialects_are_left_alone():
    # BigQuery's TIMESTAMP is an instant with a zone, and MySQL's follows the session's
    for dialect in ("bigquery", "mysql", "snowflake"):
        sql = "SELECT e.id FROM e WHERE e.ts < CAST('2024-10-01 12:34:56' AS DATETIME) - INTERVAL 1 DAY"
        assert folded(sql, dialect) == sqlglot.parse_one(sql, read=dialect).sql(dialect=dialect)


def test_duckdb_computes_the_same_instants():
    duckdb = pytest.importorskip("duckdb")
    db = duckdb.connect()
    rng = random.Random(20261111)
    units = ["year", "years", "month", "months", "week", "day", "days", "hour", "minute", "second"]
    for _ in range(300):
        moment = datetime.datetime(rng.randint(1900, 2100), rng.randint(1, 12), 1) + datetime.timedelta(days=rng.randint(0, 30), seconds=rng.randint(0, 86399))  # noqa: DTZ001
        text = moment.strftime("%Y-%m-%d %H:%M:%S")
        expression = f"TIMESTAMP '{text}' {rng.choice('+-')} INTERVAL '{rng.randint(0, 400)} {rng.choice(units)}'"
        out = folded(f"SELECT {expression} AS t")
        assert "INTERVAL" not in out
        assert db.execute(sqlglot.transpile(out, read="postgres", write="duckdb")[0]).fetchall() == db.execute(sqlglot.transpile(f"SELECT {expression} AS t", read="postgres", write="duckdb")[0]).fetchall()


TARGETS = [
    (f"SELECT e.id FROM e WHERE e.ts < {BASE} - INTERVAL '1 year'", "SELECT e.id FROM e WHERE e.ts < TIMESTAMP '2023-10-01 12:34:56'"),
    (f"SELECT e.id FROM e WHERE e.ts >= {BASE} - INTERVAL '30' DAY", "SELECT e.id FROM e WHERE e.ts >= CAST('2024-09-01 12:34:56' AS TIMESTAMP)"),
    (f"SELECT COUNT(*) AS n FROM e WHERE e.ts >= ({BASE} - INTERVAL '30 days') GROUP BY e.n", "SELECT COUNT(*) AS n FROM e WHERE e.ts >= TIMESTAMP '2024-09-01 12:34:56' GROUP BY e.n"),
    (f"SELECT e.id FROM e WHERE e.ts < {BASE} - INTERVAL '1 year'", f"SELECT e.id FROM e WHERE e.ts < {BASE} - INTERVAL '12 months'"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_folded_constants_prove(left, right):
    assert proved(left, right)


ROWS = [(1, "2023-10-01 12:34:55"), (2, "2023-10-01 12:34:56"), (3, "2023-10-01 06:00:00"), (4, "2023-09-30 23:00:00"), (5, "2024-09-01 12:34:56"), (6, "2023-10-02 00:00:00"), (7, None)]
YEAR_AGO = f"SELECT e.id FROM e WHERE e.ts < {BASE} - INTERVAL '1 year'"
NEAR_MISSES = [
    (YEAR_AGO, "SELECT e.id FROM e WHERE e.ts < TIMESTAMP '2023-10-01 12:34:57'"),  # another instant
    (YEAR_AGO, "SELECT e.id FROM e WHERE e.ts <= TIMESTAMP '2023-10-01 12:34:56'"),  # another comparison
    (YEAR_AGO, "SELECT e.id FROM e WHERE e.ts < DATE '2023-10-01'"),  # the day's midnight, not the instant
    (YEAR_AGO, "SELECT e.id FROM e WHERE e.ts < TIMESTAMP '2023-10-02 12:34:56'"),  # 365 days back from a date after a leap day
    (YEAR_AGO, f"SELECT e.id FROM e WHERE e.ts < {BASE} - INTERVAL '1 month'"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    duckdb = pytest.importorskip("duckdb")
    assert not proved(left, right)
    db = duckdb.connect()
    db.execute("CREATE TABLE e (id BIGINT, ts TIMESTAMP, n BIGINT)")
    db.executemany("INSERT INTO e VALUES (?, CAST(? AS TIMESTAMP), 1)", ROWS)
    a, b = (sorted(db.execute(sqlglot.transpile(q, read="postgres", write="duckdb")[0]).fetchall()) for q in (left, right))
    assert a != b
