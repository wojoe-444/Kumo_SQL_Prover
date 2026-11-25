"""Two absence tests of one table, asked as one (``merged_absence_tests``).

Every reading is run as written and as read on DuckDB and SQLite over made-up rows with NULLs in the tested columns.
"""

import sqlite3

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.merged_absence_tests import merge_absence_tests
from kumo_sql_prover.smt_equivalence import TableConstraints

TABLES = {
    "shops": ("shop_id INT, city VARCHAR", [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e"), (None, "f")]),
    "calls": ("call_id INT, shop_id INT, seen DATE, score INT", [(1, 1, "2024-01-02", 3), (2, 1, "2024-01-05", 9), (3, 2, "2024-01-03", 2), (4, 2, "2024-01-04", None), (5, 3, "2024-02-01", 7), (6, None, "2024-01-06", 1), (7, 4, "2024-01-01", 4), (8, 4, None, 6)]),
    "visits": ("visit_id INT, shop_id INT, score INT", [(1, 1, 1), (2, 5, 8), (3, 3, None)]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
HEAD = "SELECT s.shop_id, s.city FROM shops AS s WHERE "
LOW = "NOT EXISTS (SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id AND a.score < 3)"
HIGH = "NOT EXISTS (SELECT 1 FROM calls AS b WHERE b.shop_id = s.shop_id AND b.score > 6)"

READ = [
    (f"{LOW} AND {HIGH}", "NOT EXISTS(SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id AND (a.score < 3 OR a.score > 6))"),
    (f"{HIGH} AND s.city <> 'e' AND {LOW}", "NOT EXISTS(SELECT 1 FROM calls AS b WHERE b.shop_id = s.shop_id AND (b.score > 6 OR b.score < 3)) AND s.city <> 'e'"),
    # one name in both, the table's own name, a star, a column in the select list
    ("NOT EXISTS (SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id AND a.score < 3) AND NOT EXISTS (SELECT * FROM calls AS a WHERE a.shop_id = s.shop_id AND a.seen > '2024-01-31')", "NOT EXISTS(SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id AND (a.score < 3 OR a.seen > '2024-01-31'))"),
    ("NOT EXISTS (SELECT calls.call_id FROM calls WHERE calls.shop_id = s.shop_id AND calls.score < 3) AND NOT EXISTS (SELECT 1 FROM calls AS b WHERE b.shop_id = s.shop_id AND b.score > 6)", "NOT EXISTS(SELECT calls.call_id FROM calls WHERE calls.shop_id = s.shop_id AND (calls.score < 3 OR calls.score > 6))"),
    # no shared part; one test that is the other's shared part; three tests
    ("NOT EXISTS (SELECT 1 FROM calls AS a WHERE a.score = s.shop_id) AND NOT EXISTS (SELECT 1 FROM calls AS b WHERE b.shop_id = s.shop_id AND b.score > 6)", "NOT EXISTS(SELECT 1 FROM calls AS a WHERE (a.score = s.shop_id OR (a.shop_id = s.shop_id AND a.score > 6)))"),
    (f"NOT EXISTS (SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id) AND {HIGH}", "NOT EXISTS(SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id)"),
    (f"{LOW} AND {HIGH} AND NOT EXISTS (SELECT 1 FROM calls AS c WHERE c.seen IS NULL AND c.shop_id = s.shop_id)", "NOT EXISTS(SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id AND (a.score < 3 OR a.score > 6 OR a.seen IS NULL))"),
    # two tables: each table's tests are merged
    (f"{LOW} AND NOT EXISTS (SELECT 1 FROM visits AS v WHERE v.shop_id = s.shop_id AND v.score > 5) AND {HIGH} AND NOT EXISTS (SELECT 1 FROM visits AS w WHERE w.shop_id = s.shop_id AND w.score IS NULL)", "NOT EXISTS(SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id AND (a.score < 3 OR a.score > 6)) AND NOT EXISTS(SELECT 1 FROM visits AS v WHERE v.shop_id = s.shop_id AND (v.score > 5 OR v.score IS NULL))"),
]

LEFT_ALONE = [
    LOW,
    # EXISTS without NOT: two calls may each pass one test and no call both
    LOW.replace("NOT ", "") + " AND " + HIGH.replace("NOT ", ""),
    f"{LOW} AND " + HIGH.replace("NOT ", ""),
    f"{LOW} OR {HIGH}",
    f"NOT ({LOW} AND {HIGH})",
    # different tables
    f"{LOW} AND NOT EXISTS (SELECT 1 FROM visits AS b WHERE b.shop_id = s.shop_id AND b.score > 6)",
    # a join, grouping, an aggregate (one row out of none), a row limit, DISTINCT, a nested query
    f"{LOW} AND NOT EXISTS (SELECT 1 FROM calls AS b JOIN visits AS v ON v.shop_id = b.shop_id WHERE b.shop_id = s.shop_id AND b.score > 6)",
    f"{LOW} AND NOT EXISTS (SELECT 1 FROM calls AS b WHERE b.shop_id = s.shop_id AND b.score > 6 GROUP BY b.shop_id)",
    f"{LOW} AND NOT EXISTS (SELECT MAX(b.score) FROM calls AS b WHERE b.shop_id = s.shop_id AND b.score > 6)",
    f"{LOW} AND NOT EXISTS (SELECT 1 FROM calls AS b WHERE b.shop_id = s.shop_id AND b.score > 6 LIMIT 1)",
    f"{LOW} AND NOT EXISTS (SELECT DISTINCT 1 FROM calls AS b WHERE b.shop_id = s.shop_id AND b.score > 6)",
    f"{LOW} AND NOT EXISTS (SELECT 1 FROM calls AS b WHERE b.shop_id = s.shop_id AND b.score > (SELECT MAX(v.score) FROM visits AS v))",
    # a bare column
    f"{LOW} AND NOT EXISTS (SELECT 1 FROM calls AS b WHERE b.shop_id = s.shop_id AND score > 6)",
    # no WHERE in one
    f"{LOW} AND NOT EXISTS (SELECT 1 FROM calls AS b)",
]


def _read(test: str, head: str = HEAD):
    select = sqlglot.parse_one(head + test, read="mysql")
    before = select.sql()
    read = merge_absence_tests(select)
    assert select.sql() == before  # the input is not changed
    return None if read is None else read.sql(dialect="mysql")


def _rows(sql: str) -> list:
    answers = []
    for connect in (duckdb.connect, lambda: sqlite3.connect(":memory:")):
        database = connect()
        for name, (columns, rows) in TABLES.items():
            database.execute(f"CREATE TABLE {name} ({columns})")
            database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
        answers.append(sorted((tuple(row) for row in database.execute(sql).fetchall()), key=repr))
    assert answers[0] == answers[1]
    return answers[0]


@pytest.mark.parametrize("test, expected", READ)
def test_the_absence_tests_of_one_table_are_one_test(test, expected):
    read = _read(test)
    assert read == HEAD + expected
    assert _rows(read) == _rows(HEAD + test)


def test_the_readings_keep_and_drop_rows():
    kept = {len(_rows(HEAD + test)) for test, _ in READ}
    assert len(kept) > 2 and 0 not in kept and 6 not in kept  # the made-up rows tell the tests apart


@pytest.mark.parametrize("test", LEFT_ALONE)
def test_other_tests_are_left_alone(test):
    assert _read(test) is None


def test_a_name_that_means_an_outer_row_in_the_second_test_is_left_alone():
    # the outer table is read as "a", which is also the first subquery's name for calls
    head = "SELECT a.shop_id FROM shops AS a WHERE "
    test = "NOT EXISTS (SELECT 1 FROM calls AS a WHERE a.score < 3) AND NOT EXISTS (SELECT 1 FROM calls AS b WHERE b.shop_id = a.shop_id AND b.score > 6)"
    assert _read(test, head) is None


def test_merging_two_presence_tests_would_change_rows():
    both = HEAD + "EXISTS (SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id AND a.score <= 3) AND EXISTS (SELECT 1 FROM calls AS b WHERE b.shop_id = s.shop_id AND b.score > 6)"
    one = HEAD + "EXISTS (SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id AND (a.score <= 3 AND a.score > 6))"
    either = HEAD + "EXISTS (SELECT 1 FROM calls AS a WHERE a.shop_id = s.shop_id AND (a.score <= 3 OR a.score > 6))"
    assert _rows(both) not in (_rows(one), _rows(either))


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
# shops whose every call was in the first week of 2024: the bounds of the group, and "a call in the week, none listed
# before it and none listed after it", each list a join to the shops
BOUNDS = "SELECT s.shop_id, s.city FROM shops AS s JOIN (SELECT shop_id FROM calls GROUP BY shop_id HAVING MIN(seen) >= '2024-01-01' AND MAX(seen) <= '2024-01-07') AS v ON s.shop_id = v.shop_id"
LIST = "(SELECT p.shop_id FROM calls AS c JOIN shops AS p ON c.shop_id = p.shop_id WHERE c.seen {})"
TWO_LISTS = "SELECT DISTINCT p.shop_id, p.city FROM calls AS c JOIN shops AS p ON c.shop_id = p.shop_id WHERE c.seen BETWEEN '2024-01-01' AND '2024-01-07' AND p.shop_id NOT IN " + LIST + " AND p.shop_id NOT IN " + LIST


def test_a_pair_is_proved_and_only_with_the_rule():
    left, right = BOUNDS, TWO_LISTS.format("< '2024-01-01'", "> '2024-01-07'")
    assert prove_equivalent_algebraic(left, right, **OPTIONS).proven
    with shape_fallback.without("merged_absence_tests"):
        assert not prove_equivalent_algebraic(left, right, **OPTIONS).proven


@pytest.mark.parametrize("before, after", [("< '2024-01-01'", "> '2024-01-08'"), ("<= '2024-01-01'", "> '2024-01-07'"), ("< '2024-01-01'", "< '2024-01-01'")])
def test_pairs_that_differ_stay_unproved(before, after):
    assert not prove_equivalent_algebraic(BOUNDS, TWO_LISTS.format(before, after), **OPTIONS).proven
