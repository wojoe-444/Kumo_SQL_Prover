"""A nested test of the outer row, read when its query names its columns bare or the test is its whole ``WHERE``
(``named_nested_correlation``).

Every reading is run as written and as read on DuckDB and SQLite over made-up rows with NULL keys.
"""

import sqlite3

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.named_nested_correlation import flatten_named_correlation
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "shops": ("shop_id INT, city VARCHAR", [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (None, "e")]),
    "calls": ("call_id INT, shop_id INT, seen DATE, score INT, weight DOUBLE", [(1, 1, "2024-01-02", 3, 1.0), (2, 1, "2024-01-05", 9, 1.0), (3, 2, "2024-01-03", 2, 2.0), (4, 2, "2024-01-04", None, 2.0), (5, 3, "2024-02-01", 7, 3.0), (6, None, "2024-01-06", 1, None), (7, 4, "2024-01-01", 4, 4.0)]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
HEAD = "SELECT s.shop_id, s.city FROM shops AS s WHERE "

READ = [
    # bare names in the nested queries
    ("s.shop_id IN (SELECT shop_id FROM calls WHERE NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND shop_id = calls.shop_id))", "s.shop_id IN (SELECT calls.shop_id FROM calls) AND NOT EXISTS(SELECT 1 FROM calls AS k WHERE k.score > 5 AND k.shop_id = s.shop_id)"),
    ("s.shop_id IN (SELECT shop_id FROM calls WHERE EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND shop_id = calls.shop_id))", "s.shop_id IN (SELECT calls.shop_id FROM calls) AND EXISTS(SELECT 1 FROM calls AS k WHERE k.score > 5 AND k.shop_id = s.shop_id)"),
    ("s.shop_id IN (SELECT shop_id FROM calls WHERE score < 9 AND NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND shop_id = calls.shop_id))", "s.shop_id IN (SELECT calls.shop_id FROM calls WHERE calls.score < 9) AND NOT EXISTS(SELECT 1 FROM calls AS k WHERE k.score > 5 AND k.shop_id = s.shop_id)"),
    ("EXISTS (SELECT 1 FROM calls WHERE shop_id = s.shop_id AND NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND shop_id = calls.shop_id))", "EXISTS(SELECT 1 FROM calls WHERE calls.shop_id = s.shop_id) AND NOT EXISTS(SELECT 1 FROM calls AS k WHERE k.score > 5 AND k.shop_id = s.shop_id)"),
    ("s.shop_id IN (SELECT shop_id FROM calls WHERE shop_id NOT IN (SELECT shop_id FROM calls AS k WHERE score > 5)) AND s.city <> 'd'", "s.shop_id IN (SELECT calls.shop_id FROM calls) AND NOT s.shop_id IN (SELECT k.shop_id FROM calls AS k WHERE k.score > 5) AND s.city <> 'd'"),
    # every name is there, and the test is the subquery's whole WHERE
    ("s.shop_id IN (SELECT c.shop_id FROM calls AS c WHERE NOT EXISTS (SELECT 1 FROM calls AS k WHERE k.score > 5 AND k.shop_id = c.shop_id))", "s.shop_id IN (SELECT c.shop_id FROM calls AS c) AND NOT EXISTS(SELECT 1 FROM calls AS k WHERE k.score > 5 AND k.shop_id = s.shop_id)"),
    ("s.shop_id IN (SELECT c.shop_id FROM calls AS c WHERE c.shop_id IN (SELECT k.shop_id FROM calls AS k WHERE k.score > 5) AND c.shop_id IN (SELECT k2.shop_id FROM calls AS k2 WHERE k2.score < 4))", "s.shop_id IN (SELECT c.shop_id FROM calls AS c) AND s.shop_id IN (SELECT k.shop_id FROM calls AS k WHERE k.score > 5) AND s.shop_id IN (SELECT k2.shop_id FROM calls AS k2 WHERE k2.score < 4)"),
]

LEFT_ALONE = [
    # nested_correlation reads these by itself: every name is there and a condition stays inside
    "s.shop_id IN (SELECT c.shop_id FROM calls AS c WHERE c.score < 9 AND NOT EXISTS (SELECT 1 FROM calls AS k WHERE k.score > 5 AND k.shop_id = c.shop_id))",
    # no nested test of the row
    "s.shop_id IN (SELECT shop_id FROM calls WHERE score > 5)",
    "s.shop_id IN (SELECT shop_id FROM calls)",
    # the nested test reads another column of the subquery's row
    "s.shop_id IN (SELECT shop_id FROM calls WHERE NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND call_id = calls.call_id))",
    # a bare name the nested table does not have reads a row further out
    "s.shop_id IN (SELECT shop_id FROM calls WHERE NOT EXISTS (SELECT 1 FROM calls AS k WHERE city = 'a' AND shop_id = calls.shop_id))",
    # the two tables under one name: which row a name means could change
    "s.shop_id IN (SELECT shop_id FROM calls WHERE shop_id NOT IN (SELECT shop_id FROM calls WHERE score > 5))",
    # not a top-level conjunct, or the other way round
    "s.city = 'a' OR s.shop_id IN (SELECT shop_id FROM calls WHERE NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND shop_id = calls.shop_id))",
    "s.shop_id NOT IN (SELECT shop_id FROM calls WHERE NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND shop_id = calls.shop_id))",
    "NOT EXISTS (SELECT 1 FROM calls WHERE shop_id = s.shop_id AND NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND shop_id = calls.shop_id))",
    # a whole number against a DOUBLE: equal values are not one value
    "s.shop_id IN (SELECT weight FROM calls WHERE NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND weight = calls.weight))",
    # grouping, an aggregate, a row limit, a join in the subquery
    "s.shop_id IN (SELECT shop_id FROM calls WHERE NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND shop_id = calls.shop_id) GROUP BY shop_id)",
    "s.shop_id IN (SELECT MAX(shop_id) FROM calls WHERE NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND shop_id = calls.shop_id))",
    "s.shop_id IN (SELECT shop_id FROM calls WHERE NOT EXISTS (SELECT 1 FROM calls AS k WHERE score > 5 AND shop_id = calls.shop_id) LIMIT 2)",
]


def _read(test: str, schema=SCHEMA):
    select = sqlglot.parse_one(HEAD + test, read="mysql")
    before = select.sql()
    read = flatten_named_correlation(select, TYPES, schema)
    assert select.sql() == before  # the input is not changed
    return None if read is None else read.sql(dialect="mysql")


def _rows(sql: str) -> list:
    answers = []
    for connect, marks in ((duckdb.connect, "?"), (lambda: sqlite3.connect(":memory:"), "?")):
        database = connect()
        for name, (columns, rows) in TABLES.items():
            database.execute(f"CREATE TABLE {name} ({columns})")
            database.executemany(f"INSERT INTO {name} VALUES ({', '.join(marks * len(rows[0]))})", rows)
        answers.append(sorted(database.execute(sql).fetchall(), key=repr))
    assert answers[0] == answers[1]
    return answers[0]


@pytest.mark.parametrize("test, expected", READ)
def test_the_nested_test_is_asked_once_outside(test, expected):
    read = _read(test)
    assert read == HEAD + expected
    assert _rows(read) == _rows(HEAD + test)


def test_the_readings_keep_and_drop_rows():
    kept = [len(_rows(HEAD + test)) for test, _ in READ]
    assert all(0 < n < 5 for n in kept) and len(set(kept)) > 1  # the made-up rows tell the tests apart


@pytest.mark.parametrize("test", LEFT_ALONE)
def test_other_selects_are_left_alone(test):
    assert _read(test) is None


def test_without_the_column_lists_nothing_is_named():
    assert _read(READ[0][0], schema=None) is None


def test_moving_the_test_out_of_a_not_in_would_change_rows():
    # why NOT IN is left alone: the list of shops with a high call, against "no call at all, or a high one"
    inside = HEAD + "s.shop_id NOT IN (SELECT shop_id FROM calls WHERE shop_id IS NOT NULL AND EXISTS (SELECT 1 FROM calls AS k WHERE k.score > 5 AND k.shop_id = calls.shop_id))"
    outside = HEAD + "s.shop_id NOT IN (SELECT shop_id FROM calls WHERE shop_id IS NOT NULL) AND EXISTS (SELECT 1 FROM calls AS k WHERE k.score > 5 AND k.shop_id = s.shop_id)"
    assert _rows(inside) != _rows(outside)


OPTIONS = dict(
    schema={"shops": SCHEMA["shops"], "calls": SCHEMA["calls"]},
    types=TYPES,
    constraints={
        "shops": TableConstraints(not_null=frozenset({"shop_id", "city"}), keys=(("shop_id",),)),
        "calls": TableConstraints(not_null=frozenset({"call_id", "shop_id", "seen", "score"}), keys=(("call_id",),)),
    },
    compare_names=False,
    dialect="mysql",
    exact_arithmetic=True,
    timeout_ms=3000,
)
# shops whose every call was in the first week of 2024: the bounds of the group, and a join to the list of shops with a
# call and none outside the week
BOUNDS = "SELECT s.shop_id, s.city FROM shops AS s JOIN (SELECT shop_id FROM calls GROUP BY shop_id HAVING MIN(seen) >= '2024-01-01' AND MAX(seen) <= '2024-01-07') AS v ON s.shop_id = v.shop_id"
LISTED = "SELECT shops.shop_id, shops.city FROM shops JOIN (SELECT DISTINCT shop_id FROM calls WHERE shop_id {} (SELECT shop_id FROM calls WHERE seen NOT BETWEEN '2024-01-01' AND '2024-01-{}')) AS t ON shops.shop_id = t.shop_id"


def test_a_pair_is_proved_and_only_with_the_rule():
    left, right = BOUNDS, LISTED.format("NOT IN", "07")
    assert prove_equivalent_algebraic(left, right, **OPTIONS).proven
    with shape_fallback.without("named_nested_correlation"):
        assert not prove_equivalent_algebraic(left, right, **OPTIONS).proven


@pytest.mark.parametrize("test, last", [("NOT IN", "08"), ("NOT IN", "06"), ("IN", "07")])
def test_pairs_that_differ_stay_unproved(test, last):
    assert not prove_equivalent_algebraic(BOUNDS, LISTED.format(test, last), **OPTIONS).proven
