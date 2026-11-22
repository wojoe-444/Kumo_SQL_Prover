"""Constant rows outer-joined to a table and aggregated per row are one aggregate query per row (``constant_bins``)."""

import random

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.constant_bins import NAME, read_constant_bins
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"visits": ["id", "secs", "kind"], "people": ["id", "name"]}
TYPES = {"visits": {"id": "INT", "secs": "INT", "kind": "VARCHAR"}, "people": {"id": "INT", "name": "VARCHAR"}}
CONSTRAINTS = {
    "visits": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "people": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
}
BINS = "(SELECT 'short' AS bin UNION ALL SELECT 'medium' UNION ALL SELECT 'long')"
RANGES = "(SELECT 'short' AS bin, 0 AS low, 300 AS high UNION ALL SELECT 'medium', 300, 900 UNION ALL SELECT 'long', 900, NULL)"
LABELLED = "(SELECT id, secs, CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' WHEN secs >= 900 THEN 'long' END AS bin FROM visits)"
COUNTED = "(SELECT CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' WHEN secs >= 900 THEN 'long' END AS bin, COUNT(*) AS total, SUM(secs) AS secs, MAX(id) AS newest FROM visits GROUP BY CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' WHEN secs >= 900 THEN 'long' END)"

# each query has the form the rule reads; the rewritten query must return the same rows
READ = [
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",
    f"SELECT b.bin, COUNT(v.id) AS total FROM {LABELLED} AS v RIGHT JOIN {BINS} AS b ON b.bin = v.bin GROUP BY b.bin",
    f"SELECT b.bin, COALESCE(SUM(v.secs), 0) AS secs, MIN(v.secs) AS least, COUNT(DISTINCT v.secs) AS n FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",
    f"SELECT bin, COUNT(id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v USING (bin) GROUP BY bin",
    f"SELECT b.bin, COUNT(v.id) AS total FROM {RANGES} AS b LEFT JOIN visits AS v ON v.secs >= b.low AND (b.high IS NULL OR v.secs < b.high) GROUP BY b.bin",
    f"SELECT b.bin, b.low, AVG(v.secs) AS mean FROM {RANGES} AS b LEFT JOIN visits AS v ON v.secs >= b.low AND v.secs < b.high GROUP BY b.bin",
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN visits AS v ON v.kind = b.bin GROUP BY b.bin",
    f"SELECT b.bin, COALESCE(t.total, 0) AS total FROM {BINS} AS b LEFT JOIN {COUNTED} AS t ON t.bin = b.bin",
    f"SELECT b.bin, IFNULL(t.total, 0) AS total, t.secs, t.newest FROM {COUNTED} AS t RIGHT JOIN {BINS} AS b ON b.bin = t.bin",
    f"SELECT bin, COALESCE(total, 0) + 1 AS total FROM {BINS} AS b LEFT JOIN {COUNTED} AS t USING (bin)",
    "SELECT b.n, COUNT(v.id) AS total FROM (SELECT 1 AS n UNION SELECT 2 UNION SELECT 3) AS b LEFT JOIN visits AS v ON v.secs = b.n GROUP BY b.n",
    "SELECT b.n, COALESCE(t.total, 0) AS total FROM (SELECT 1 AS n UNION SELECT 2) AS b LEFT JOIN (SELECT secs, COUNT(*) AS total FROM visits GROUP BY secs) AS t ON t.secs = b.n",
    f"SELECT b.bin, COUNT(v.id) AS total FROM (SELECT * FROM {BINS} AS inner_rows) AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",
    "SELECT b.bin, COALESCE(t.total, 0) AS total FROM (SELECT 'a' AS bin UNION ALL SELECT 'b') AS b LEFT JOIN (SELECT CASE WHEN secs < 300 THEN 'a' ELSE 'b' END AS bin, COUNT(id) AS total FROM visits GROUP BY bin) AS t ON t.bin = b.bin",
]


def written(sql: str, dialect: str = "mysql") -> str | None:
    tree, changed = read_constant_bins(sqlglot.parse_one(sql, read=dialect), SCHEMA, TYPES, dialect)
    return tree.sql(dialect=dialect) if changed else None


def prove(left: str, right: str, dialect: str = "postgres"):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=5000
    )


def test_the_grouped_form_is_one_aggregate_select_per_row():
    sql = f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON b.bin = v.bin GROUP BY b.bin"
    branch = "SELECT '{0}' AS bin, COUNT(v.id) AS total FROM " + LABELLED + " AS v WHERE v.bin = '{0}'"
    assert written(sql) == " UNION ALL ".join(branch.format(label) for label in ("short", "medium", "long"))


def test_a_range_join_is_written_with_the_bounds_of_each_row():
    sql = f"SELECT b.bin, COUNT(v.id) AS total FROM {RANGES} AS b LEFT JOIN visits AS v ON v.secs >= b.low AND (b.high IS NULL OR v.secs < b.high) GROUP BY b.bin"
    assert written(sql) == (
        "SELECT 'short' AS bin, COUNT(v.id) AS total FROM visits AS v WHERE v.secs >= 0 AND (300 IS NULL OR v.secs < 300) UNION ALL "
        "SELECT 'medium' AS bin, COUNT(v.id) AS total FROM visits AS v WHERE v.secs >= 300 AND (900 IS NULL OR v.secs < 900) UNION ALL "
        "SELECT 'long' AS bin, COUNT(v.id) AS total FROM visits AS v WHERE v.secs >= 900 AND (NULL IS NULL OR v.secs < NULL)"
    )


def test_the_form_counted_first_loses_its_grouping():
    sql = "SELECT b.bin, COALESCE(t.total, 0) AS total, t.most FROM (SELECT 'a' AS bin UNION SELECT 'b') AS b LEFT JOIN (SELECT kind, COUNT(*) AS total, MAX(secs) AS most FROM visits WHERE secs > 0 GROUP BY kind) AS t ON b.bin = t.kind"
    assert written(sql) == (
        "SELECT 'a' AS bin, COALESCE(COUNT(*), 0) AS total, MAX(secs) AS most FROM visits WHERE secs > 0 AND kind = 'a' UNION ALL "
        "SELECT 'b' AS bin, COALESCE(COUNT(*), 0) AS total, MAX(secs) AS most FROM visits WHERE secs > 0 AND kind = 'b'"
    )


def test_on_the_right_of_a_set_operation_the_selects_are_glued_the_way_it_is():
    # Written without parentheses, ``A UNION b1 UNION ALL b2`` is ``(A UNION b1) UNION ALL b2``: a row of A that the
    # second select repeats would come out twice. So under UNION the selects are glued with UNION.
    grouped = "SELECT b.bin, COUNT(v.id) AS total FROM (SELECT 'a' AS bin UNION ALL SELECT 'b') AS b LEFT JOIN visits AS v ON v.kind = b.bin GROUP BY b.bin"
    one = "SELECT '{0}' AS bin, COUNT(v.id) AS total FROM visits AS v WHERE v.kind = '{0}'"
    first = "SELECT 'b' AS bin, 0 AS total"
    assert written(f"{first} UNION {grouped}") == f"{first} UNION {one.format('a')} UNION {one.format('b')}"
    assert written(f"{first} UNION ALL {grouped}") == f"{first} UNION ALL {one.format('a')} UNION ALL {one.format('b')}"
    assert written(f"{grouped} UNION {first}") == f"{one.format('a')} UNION ALL {one.format('b')} UNION {first}"
    assert written(f"{first} EXCEPT {grouped}") is None and written(f"{first} INTERSECT {grouped}") is None
    assert written(f"{grouped} EXCEPT {first}") is None
    con = duckdb.connect()
    con.execute("CREATE TABLE visits (id INT, secs INT, kind VARCHAR)")
    con.execute("INSERT INTO visits VALUES (1, 10, 'a')")
    for glue in ("UNION", "UNION ALL"):
        for sql in (f"{first} {glue} {grouped}", f"{grouped} {glue} {first}", f"{first} {glue} {grouped} {glue} {first}"):
            new = written(sql)
            assert new is not None and rows(con, sql) == rows(con, new), sql
            assert sqlglot.parse_one(new, read="mysql").sql(dialect="mysql") == new


def rows(con, sql: str) -> list:
    return sorted(con.execute(sqlglot.transpile(sql, read="mysql", write="duckdb")[0]).fetchall(), key=repr)


def test_the_rewritten_queries_return_the_same_rows_in_duckdb():
    rng = random.Random(16)
    for trial in range(40):
        con = duckdb.connect()
        con.execute("CREATE TABLE visits (id INT, secs INT, kind VARCHAR)")
        con.execute("CREATE TABLE people (id INT, name VARCHAR)")
        for n in range(rng.choice([0, 0, 1, 3, 8, 15])):
            secs = rng.choice([None, 1, 2, 3, 299, 300, 301, 899, 900, 901, 5000])
            con.execute("INSERT INTO visits VALUES (?, ?, ?)", [n + 1, secs, rng.choice([None, "short", "medium", "long", "a", "b", "other"])])
        for sql in READ:
            new = written(sql)
            assert new is not None, sql
            # DuckDB wants a column of the bin row that is not the grouping column said to be any value of its group
            asked = sql.replace("b.low, AVG", "ANY_VALUE(b.low), AVG")
            assert rows(con, asked) == rows(con, new), (trial, sql)


# each query breaks one condition and is left as it is
LEFT_ALONE = [
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # an inner join shows no empty bin
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b RIGHT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # the rows are kept, not the bins
    f"SELECT b.bin, COUNT(*) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # counts the NULL row of an empty bin
    f"SELECT b.bin, COUNT(COALESCE(v.id, 0)) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # so does this
    f"SELECT b.bin, SUM(v.secs + 1) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # only a plain column is read
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin WHERE v.secs > 5 GROUP BY b.bin",
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin HAVING COUNT(v.id) > 0",
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin ORDER BY 1 LIMIT 2",
    f"SELECT DISTINCT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",
    f"SELECT v.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY v.bin",  # grouped by the other side
    f"SELECT b.bin, v.secs, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # a column outside an aggregate
    f"SELECT b.bin FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # no aggregate: not one row per bin once written
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin LEFT JOIN people AS p ON p.id = v.id GROUP BY b.bin",
    f"SELECT b.bin, COUNT(v.id) AS total FROM (SELECT 'x' AS bin UNION ALL SELECT 'x') AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # one group, two rows
    f"SELECT b.bin, COUNT(v.id) AS total FROM (SELECT 'x' AS bin UNION ALL SELECT 'X') AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # one group under a collation
    f"SELECT b.bin, COUNT(v.id) AS total FROM (SELECT 'x' AS bin UNION ALL SELECT 'x ') AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # padded
    f"SELECT b.bin, COUNT(v.id) AS total FROM (SELECT 'x' AS bin UNION ALL SELECT NULL) AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",
    f"SELECT b.bin, COUNT(v.id) AS total FROM (SELECT 'x' AS bin UNION ALL SELECT 1) AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # text and number
    f"SELECT b.bin, COUNT(v.id) AS total FROM (SELECT 'short' AS bin UNION ALL SELECT 'SHORT ') AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",
    f"SELECT b.bin, COUNT(v.id) AS total FROM (SELECT 'Short' AS bin UNION ALL SELECT 'long') AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",  # 'Short' may equal the 'short' of the CASE
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN visits AS v ON v.secs = b.bin GROUP BY b.bin",  # a text against a number: compared as numbers
    "SELECT b.n, COUNT(v.id) AS total FROM (SELECT 1 AS n UNION SELECT 2) AS b LEFT JOIN visits AS v ON v.kind = b.n GROUP BY b.n",  # a number against a text
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN visits AS v ON LOWER(v.kind) = b.bin GROUP BY b.bin",  # not seen to be text
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN visits AS v ON v.kind = UPPER(b.bin) GROUP BY b.bin",  # the constant under a function
    f"SELECT b.bin, COUNT(v.id) AS total FROM {RANGES} AS b LEFT JOIN visits AS v ON v.secs BETWEEN b.low AND b.high GROUP BY b.bin",  # not one side of a comparison
    f"SELECT b.bin, COUNT(v.id) AS total FROM {RANGES} AS b LEFT JOIN visits AS v ON v.secs >= b.low AND b.low < b.high GROUP BY b.bin",
    f"SELECT b.bin, COUNT(id) AS total FROM {BINS} AS b LEFT JOIN unknown_rows AS v ON v.kind = b.bin GROUP BY b.bin",  # whose id, whose kind
    f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin, v.kind",
    f"SELECT b.bin, (SELECT COUNT(*) FROM people) AS total, COUNT(v.id) FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",
    f"SELECT b.bin, COUNT(v.id) OVER () AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin",
    f"SELECT b.bin, t.total FROM {BINS} AS b LEFT JOIN {COUNTED} AS t ON t.bin = b.bin",  # a count of no rows is 0, the join shows NULL
    f"SELECT b.bin, COALESCE(t.total, 1) AS total FROM {BINS} AS b LEFT JOIN {COUNTED} AS t ON t.bin = b.bin",
    f"SELECT b.bin, t.bin AS seen, COALESCE(t.total, 0) AS total FROM {BINS} AS b LEFT JOIN {COUNTED} AS t ON t.bin = b.bin",  # the key is NULL for an empty bin
    f"SELECT b.bin, COALESCE(t.total, 0) AS total FROM {BINS} AS b LEFT JOIN {COUNTED} AS t ON t.bin = b.bin AND t.total > 1",
    f"SELECT b.bin, COALESCE(t.total, 0) AS total FROM {BINS} AS b LEFT JOIN {COUNTED} AS t ON t.bin <> b.bin",
    f"SELECT b.bin, COALESCE(t.total, 0) AS total FROM {BINS} AS b LEFT JOIN (SELECT kind, secs, COUNT(*) AS total FROM visits GROUP BY kind, secs) AS t ON t.kind = b.bin",  # several groups per bin
    f"SELECT b.bin, COALESCE(t.total, 0) AS total FROM {BINS} AS b LEFT JOIN (SELECT kind, COUNT(*) AS total FROM visits GROUP BY kind HAVING COUNT(*) > 1) AS t ON t.kind = b.bin",
    f"SELECT b.bin, COALESCE(t.total, 0) AS total FROM {BINS} AS b LEFT JOIN (SELECT kind, COUNT(*) + 1 AS total FROM visits GROUP BY kind) AS t ON t.kind = b.bin",
    f"SELECT b.bin, COALESCE(t.total, 0) AS total FROM (SELECT 'a' AS bin FROM people UNION SELECT 'b' FROM people) AS b LEFT JOIN {COUNTED} AS t ON t.bin = b.bin",  # no bins when people is empty
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_selects_are_left_alone(sql):
    assert written(sql) is None


def test_the_readings_that_are_refused_would_change_rows():
    # Why COUNT(*) and a bare count column are not read: the bin with no rows shows 1, or NULL, where the scalar
    # query shows 0.
    con = duckdb.connect()
    con.execute("CREATE TABLE visits (id INT, secs INT, kind VARCHAR)")
    con.execute("INSERT INTO visits VALUES (1, 10, 'a')")
    bins = "(SELECT 'a' AS bin UNION ALL SELECT 'b')"
    assert rows(con, f"SELECT b.bin, COUNT(*) AS total FROM {bins} AS b LEFT JOIN visits AS v ON v.kind = b.bin GROUP BY b.bin") == [("a", 1), ("b", 1)]
    assert rows(con, "SELECT 'b' AS bin, COUNT(*) AS total FROM visits AS v WHERE v.kind = 'b'") == [("b", 0)]
    counted = "(SELECT kind, COUNT(*) AS total FROM visits GROUP BY kind)"
    assert rows(con, f"SELECT b.bin, t.total FROM {bins} AS b LEFT JOIN {counted} AS t ON t.kind = b.bin") == [("a", 1), ("b", None)]


PAIRS = [
    (
        f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",
        "SELECT 'short' AS bin, COUNT(id) AS total FROM visits WHERE secs < 300 UNION ALL SELECT 'medium', COUNT(id) FROM visits WHERE secs >= 300 AND secs < 900 "
        "UNION ALL SELECT 'long', COUNT(id) FROM visits WHERE secs >= 900",
    ),
    (
        f"SELECT b.bin, COALESCE(t.total, 0) AS total FROM {BINS} AS b LEFT JOIN {COUNTED} AS t ON t.bin = b.bin",
        "SELECT 'short' AS bin, COUNT(*) AS total FROM visits WHERE secs < 300 UNION SELECT 'medium', COUNT(*) FROM visits WHERE secs BETWEEN 300 AND 899 "
        "UNION SELECT 'long', COUNT(*) FROM visits WHERE secs >= 900",
    ),
    (
        f"SELECT b.bin, COUNT(v.id) AS total FROM {RANGES} AS b LEFT JOIN visits AS v ON v.secs >= b.low AND (b.high IS NULL OR v.secs < b.high) GROUP BY b.bin",
        "SELECT 'short' AS bin, COUNT(id) AS total FROM visits WHERE secs >= 0 AND secs < 300 UNION ALL SELECT 'medium', COUNT(id) FROM visits WHERE secs >= 300 AND secs < 900 "
        "UNION ALL SELECT 'long', COUNT(id) FROM visits WHERE secs >= 900",
    ),
]


@pytest.mark.parametrize("left, right", PAIRS)
def test_pairs_proved_only_with_the_rule(left, right):
    assert prove(left, right).proven
    with shape_fallback.without(NAME):
        assert not prove(left, right).proven


@pytest.mark.parametrize(
    "left, right",
    [
        (  # the last bin misses 900
            f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",
            "SELECT 'short' AS bin, COUNT(id) AS total FROM visits WHERE secs < 300 UNION ALL SELECT 'medium', COUNT(id) FROM visits WHERE secs >= 300 AND secs < 900 "
            "UNION ALL SELECT 'long', COUNT(id) FROM visits WHERE secs > 900",
        ),
        (  # ELSE takes the rows whose secs is NULL
            f"SELECT b.bin, COUNT(v.id) AS total FROM {BINS} AS b LEFT JOIN (SELECT id, CASE WHEN secs < 300 THEN 'short' WHEN secs < 900 THEN 'medium' ELSE 'long' END AS bin FROM visits) AS v ON v.bin = b.bin GROUP BY b.bin",
            "SELECT 'short' AS bin, COUNT(id) AS total FROM visits WHERE secs < 300 UNION ALL SELECT 'medium', COUNT(id) FROM visits WHERE secs >= 300 AND secs < 900 "
            "UNION ALL SELECT 'long', COUNT(id) FROM visits WHERE secs >= 900",
        ),
        (  # SUM of no rows is NULL, COUNT is 0
            f"SELECT b.bin, SUM(v.id) AS total FROM {BINS} AS b LEFT JOIN {LABELLED} AS v ON v.bin = b.bin GROUP BY b.bin",
            "SELECT 'short' AS bin, COUNT(id) AS total FROM visits WHERE secs < 300 UNION ALL SELECT 'medium', COUNT(id) FROM visits WHERE secs >= 300 AND secs < 900 "
            "UNION ALL SELECT 'long', COUNT(id) FROM visits WHERE secs >= 900",
        ),
    ],
)
def test_near_misses_stay_unproved(left, right):
    assert not prove(left, right).proven
