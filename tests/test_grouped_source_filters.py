"""Filters around a grouped derived table that is joined to other sources (``grouped_source_filters``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover import grouped_source_filters as rules

GROUPED = "(SELECT k, c, SUM(x) AS s FROM t{where} GROUP BY k, c) AS d"


def rewritten(rule, sql: str) -> str | None:
    result = rule(sqlglot.parse_one(sql, read="bigquery"))
    return None if result is None else result.sql(dialect="bigquery")


def test_a_group_key_filter_moves_into_the_grouped_source():
    sql = f"SELECT u.y FROM {GROUPED.format(where='')}, u WHERE d.k = u.k AND d.c BETWEEN 2 AND 5 AND d.s > 3"
    assert rewritten(rules.push_filter_into_grouped_source, sql) == (
        "SELECT u.y FROM (SELECT k, c, SUM(x) AS s FROM t WHERE c BETWEEN 2 AND 5 GROUP BY k, c) AS d CROSS JOIN u WHERE d.k = u.k AND d.s > 3"
    )
    # from an inner join's ON, next to a filter the source already has; an OR keeps its parentheses
    sql = f"SELECT u.y FROM u JOIN {GROUPED.format(where=' WHERE x > 0')} ON (d.k = 1 OR d.k IS NULL)"
    assert rewritten(rules.push_filter_into_grouped_source, sql) == (
        "SELECT u.y FROM u CROSS JOIN (SELECT k, c, SUM(x) AS s FROM t WHERE x > 0 AND (k = 1 OR k IS NULL) GROUP BY k, c) AS d"
    )


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT u.y FROM u JOIN {GROUPED.format(where='')} ON d.k = u.k WHERE d.s > 3",  # an aggregate, not a key
        f"SELECT u.y FROM u LEFT JOIN {GROUPED.format(where='')} ON d.k = u.k WHERE d.c = 1",  # an outer join
        f"SELECT u.y FROM {GROUPED.format(where='')} LEFT JOIN u ON d.k = u.k WHERE d.c = 1",
        f"SELECT u.y FROM u JOIN {GROUPED.format(where='')} ON d.k = u.k WHERE d.c = u.y",  # reads another source
        f"SELECT u.y FROM u JOIN {GROUPED.format(where='')} ON d.k = u.k WHERE d.c + 1 > 2",  # arithmetic can fail
        f"SELECT u.y FROM u JOIN {GROUPED.format(where='')} ON d.k = u.k WHERE d.c IN (SELECT k FROM t)",
        "SELECT u.y FROM u JOIN (SELECT k, c, SUM(x) AS s FROM t GROUP BY k, c LIMIT 3) AS d ON d.k = u.k WHERE d.c = 1",
        "SELECT u.y FROM u JOIN (SELECT k, c, x AS s FROM t) AS d ON d.k = u.k WHERE d.c = 1",  # not grouped: nothing to cross
        "SELECT u.y FROM u JOIN (SELECT k, c, SUM(x) OVER (PARTITION BY k) AS s FROM t) AS d ON d.k = u.k WHERE d.c = 1",
        "SELECT u.y FROM u JOIN (SELECT k, c, SUM(x) AS s FROM t GROUP BY ROLLUP (k, c)) AS d ON d.k = u.k WHERE d.c = 1",
    ],
)
def test_other_filters_stay(sql):
    assert rewritten(rules.push_filter_into_grouped_source, sql) is None


def test_a_null_filter_the_join_repeats_is_dropped():
    sql = f"SELECT u.y FROM u JOIN {GROUPED.format(where=' WHERE x > 0 AND k IS NOT NULL')} ON u.k = d.k"
    assert rewritten(rules.drop_rejected_null_filter, sql) == "SELECT u.y FROM u JOIN (SELECT k, c, SUM(x) AS s FROM t WHERE x > 0 GROUP BY k, c) AS d ON u.k = d.k"
    # one level further in: the rows dropped reach only the group the join rejects
    nested = "(SELECT d2.k, AVG(d2.s) AS a FROM (SELECT k, c, SUM(x) AS s FROM t WHERE NOT k IS NULL GROUP BY k, c) AS d2 GROUP BY d2.k) AS d"
    assert rewritten(rules.drop_rejected_null_filter, f"SELECT u.y FROM u, {nested} WHERE u.k = d.k AND u.y > d.a") == (
        "SELECT u.y FROM u CROSS JOIN (SELECT d2.k, AVG(d2.s) AS a FROM (SELECT k, c, SUM(x) AS s FROM t GROUP BY k, c) AS d2 GROUP BY d2.k) AS d WHERE u.k = d.k AND u.y > d.a"
    )


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT u.y FROM u JOIN {GROUPED.format(where=' WHERE k IS NOT NULL')} ON u.k = d.s",  # the join reads another column
        f"SELECT u.y FROM u JOIN {GROUPED.format(where=' WHERE k IS NOT NULL')} ON u.k = d.k OR d.k IS NULL",  # NULL passes
        f"SELECT u.y FROM u JOIN {GROUPED.format(where=' WHERE k IS NOT NULL')} ON u.k IS NOT DISTINCT FROM d.k",
        f"SELECT u.y FROM u JOIN {GROUPED.format(where=' WHERE k IS NOT NULL')} ON u.k = d.k + 1",  # not the column itself
        f"SELECT u.y FROM u LEFT JOIN {GROUPED.format(where=' WHERE k IS NOT NULL')} ON u.k = d.k",  # an outer join
        "SELECT u.y FROM u JOIN (SELECT k, SUM(x) AS s FROM t WHERE k IS NOT NULL GROUP BY k ORDER BY k LIMIT 3) AS d ON u.k = d.k",
        "SELECT u.y FROM u JOIN (SELECT MAX(k) AS k, c FROM t WHERE k IS NOT NULL GROUP BY c) AS d ON u.k = d.k",  # not a key
        f"SELECT u.y FROM u JOIN {GROUPED.format(where=' WHERE x IS NOT NULL')} ON u.k = d.k",  # another column's filter
    ],
)
def test_other_null_filters_stay(sql):
    assert rewritten(rules.drop_rejected_null_filter, sql) is None


def _database(rng):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE t (k INT, c INT, x INT)")
    con.execute("CREATE TABLE u (k INT, y INT)")
    value = lambda: rng.choice([None, None, 0, 1, 2, 3])  # noqa: E731
    for _ in range(rng.randint(0, 9)):
        con.execute("INSERT INTO t VALUES (?, ?, ?)", [value(), value(), value()])
    for _ in range(rng.randint(0, 5)):
        con.execute("INSERT INTO u VALUES (?, ?)", [value(), value()])
    return con


SOURCES = [
    "(SELECT k, c, SUM(x) AS s FROM t {w} GROUP BY k, c) AS d",
    "(SELECT DISTINCT k, c, 1 AS s FROM t {w}) AS d",
    "(SELECT k, MIN(c) AS c, SUM(x) AS s FROM t {w} GROUP BY k HAVING COUNT(*) > 1) AS d",
    "(SELECT d2.k, d2.k AS c, MAX(d2.s) AS s FROM (SELECT k, c, SUM(x) AS s FROM t {w} GROUP BY k, c) AS d2 GROUP BY d2.k) AS d",
    "(SELECT k + 1 AS k, c, MIN(x) AS s FROM t {w} GROUP BY k + 1, c) AS d",
]
FILTERS = ["", "WHERE k IS NOT NULL", "WHERE x > 0 AND NOT k IS NULL", "WHERE c IS NOT NULL AND k IS NOT NULL"]
CONDITIONS = [
    "d.k = u.k", "d.k > 1", "d.c BETWEEN 1 AND 2", "d.k IN (0, 2)", "(d.k = 1 OR d.k IS NULL)", "d.c IS NOT NULL", "u.y < d.k",
    "d.s > 1", "d.k IS NULL", "NOT (d.c > 1)", "d.k = d.c", "d.c <> u.y",
]


@pytest.mark.parametrize("seed", range(3))
def test_both_rules_keep_the_rows_duckdb_returns(seed):
    """Made-up queries over NULL-heavy data: a rewritten query returns the rows the original returns."""

    rng = random.Random(seed)
    fired = 0
    for _ in range(30):
        source = rng.choice(SOURCES).format(w=rng.choice(FILTERS))
        first, *rest = rng.sample(CONDITIONS, rng.randint(1, 3))
        where = f" WHERE {' AND '.join(rest)}" if rest else ""
        sql = rng.choice([f"SELECT u.k, u.y, d.k, d.s FROM u JOIN {source} ON {first}{where}", f"SELECT COUNT(*) FROM {source}, u WHERE {' AND '.join([first, *rest])}"])
        tree = sqlglot.parse_one(sql, read="duckdb")
        for rule in (rules.push_filter_into_grouped_source, rules.drop_rejected_null_filter):
            result = rule(tree)
            if result is None:
                continue
            fired += 1
            for _ in range(3):
                con = _database(rng)
                rows = lambda text: sorted(con.execute(text).fetchall(), key=repr)  # noqa: E731
                assert rows(sql) == rows(result.sql(dialect="duckdb")), (sql, result.sql(dialect="duckdb"))
    assert fired > 5


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402

SCHEMA = {"t": ["k", "c", "x"], "u": ["k", "y"]}
TYPES = {"t": {"k": "INT64", "c": "INT64", "x": "NUMERIC"}, "u": {"k": "INT64", "y": "NUMERIC"}}


def proven(left: str, right: str) -> bool:
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="bigquery").proven


OUTSIDE = "SELECT u.y FROM (SELECT k, c, SUM(x) AS s FROM t GROUP BY k, c) AS d, u WHERE d.k = u.k AND d.c BETWEEN 2 AND 5 AND u.y > d.s"
INSIDE = "SELECT u.y FROM (SELECT k, c, SUM(x) AS s FROM t WHERE c >= 2 AND c <= 5 GROUP BY k, c) AS d JOIN u ON d.k = u.k AND u.y > d.s"


def test_a_filter_on_either_side_of_the_grouping_is_one_query():
    assert proven(OUTSIDE, INSIDE)
    assert not proven(OUTSIDE, INSIDE.replace("c <= 5", "c <= 6"))
    assert not proven(OUTSIDE, INSIDE.replace("c >= 2 AND c <= 5", "x >= 2 AND x <= 5"))  # a filter on the summed column
    assert not proven(OUTSIDE.replace("d.c BETWEEN 2 AND 5", "d.s BETWEEN 2 AND 5"), INSIDE.replace("c >= 2 AND c <= 5", "x >= 2 AND x <= 5"))


def test_an_average_of_group_sums_compared_with_a_decimal_factor():
    # the decorrelated form of ``u.y > (SELECT 1.2 * AVG(s) FROM (grouped) WHERE k = u.k)`` and an optimizer's version
    plain = "SELECT u.y FROM u, (SELECT d2.k, AVG(d2.s) AS a FROM (SELECT k, c, SUM(x) AS s FROM t WHERE c BETWEEN 1 AND 9 GROUP BY k, c) AS d2 GROUP BY d2.k) AS d WHERE u.k = d.k AND u.y > d.a * 1.2"
    inferred = "SELECT u.y FROM u JOIN (SELECT d3.k, AVG(d3.s) AS a FROM (SELECT k, c, SUM(x) AS s FROM t WHERE c >= 1 AND c <= 9 AND k IS NOT NULL GROUP BY k, c) AS d3 GROUP BY d3.k) AS d ON u.k = d.k AND u.y > d.a * 1.2"
    assert proven(plain, inferred)
    assert not proven(plain, inferred.replace("k IS NOT NULL", "c IS NOT NULL").replace("c >= 1 AND c <= 9 AND ", ""))
    assert not proven(plain, inferred.replace("u.k = d.k AND", "u.k = d.a AND"))
    assert not proven(plain, inferred.replace("1.2", "1.25"))
