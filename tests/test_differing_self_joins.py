"""A self-join for rows that agree on some columns and differ on another counts distinct values (``differing_self_joins``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.differing_self_joins import regroup_differing_self_join

JOINED = "SELECT DISTINCT a.viewer FROM views AS a JOIN views AS b ON a.viewer = b.viewer AND a.day = b.day AND a.article <> b.article"
GROUPED = "SELECT DISTINCT a.viewer FROM views AS a WHERE NOT a.viewer IS NULL AND NOT a.day IS NULL GROUP BY a.viewer, a.day HAVING COUNT(DISTINCT a.article) >= 2"


def rewritten(sql: str, types=None) -> str | None:
    result = regroup_differing_self_join(sqlglot.parse_one(sql, read="mysql"), types)
    return None if result is None else result.sql(dialect="mysql")


def test_two_rows_that_differ_are_two_values_in_a_group():
    assert rewritten(JOINED) == GROUPED
    # a comma join with the conditions in WHERE, the comparison the other way round, and the other row listed
    assert rewritten("SELECT DISTINCT b.viewer FROM views AS a, views AS b WHERE b.article <> a.article AND a.day = b.day AND a.viewer = b.viewer") == GROUPED.replace("a.viewer IS NULL AND NOT a.day IS NULL GROUP BY a.viewer, a.day", "a.day IS NULL AND NOT a.viewer IS NULL GROUP BY a.day, a.viewer")
    # of two different values one is the smaller
    assert rewritten(JOINED.replace("a.article <> b.article", "a.article < b.article")) == GROUPED
    assert rewritten(JOINED.replace("a.article <> b.article", "a.article > b.article")) == GROUPED
    assert rewritten(JOINED.replace("a.article <> b.article", "a.article < b.article AND b.article > a.article")) == GROUPED
    # an output name is kept; every equated column may be listed
    assert rewritten(JOINED.replace("a.viewer FROM", "a.viewer AS id FROM")) == GROUPED.replace("a.viewer FROM", "a.viewer AS id FROM")
    assert rewritten(JOINED.replace("a.viewer FROM", "a.viewer, b.day FROM")) == GROUPED.replace("a.viewer FROM", "a.viewer, a.day FROM")


def test_the_normalizers_distinct_projection_of_the_table_is_read():
    """Under ``DISTINCT`` the normalizer reads a table as its distinct projection onto the columns in use."""

    one = "(SELECT DISTINCT s0.article AS article, s0.day AS day, s0.viewer AS viewer FROM views AS s0) AS a"
    two = "(SELECT DISTINCT s1.article AS article, s1.day AS day, s1.viewer AS viewer FROM views AS s1) AS b"
    sql = f"SELECT DISTINCT a.viewer FROM {one} JOIN {two} ON a.viewer = b.viewer AND a.day = b.day AND a.article <> b.article"
    assert rewritten(sql) == GROUPED
    # a projection that hides a column in use, renames one, filters, or is not DISTINCT is not that table
    assert rewritten(sql.replace("s1.article AS article, ", "")) is None
    assert rewritten(sql.replace("s1.day AS day", "s1.other AS day")) is None
    assert rewritten(sql.replace("FROM views AS s1", "FROM views AS s1 WHERE s1.day > 3")) is None
    assert rewritten(sql.replace("(SELECT DISTINCT s1.", "(SELECT s1.")) is None
    assert rewritten(sql.replace("FROM views AS s1", "FROM visits AS s1")) is None


@pytest.mark.parametrize(
    "sql",
    [
        JOINED.replace("SELECT DISTINCT", "SELECT"),  # every pair is listed, twice
        JOINED.replace("a.article <> b.article", "a.article <= b.article"),  # a row pairs with itself
        JOINED.replace("a.article <> b.article", "a.article < b.article AND a.article > b.article"),  # never TRUE
        JOINED.replace("a.article <> b.article", "a.article < b.article AND b.article < a.article"),
        JOINED.replace(" AND a.article <> b.article", ""),
        JOINED.replace("a.article <> b.article", "a.article <> b.article AND a.author = 7"),  # another condition
        JOINED.replace("a.article <> b.article", "a.article <> b.article AND a.author <> b.author"),  # two columns must differ
        JOINED.replace("a.article <> b.article", "a.article <> b.author"),  # not the same column
        JOINED.replace("a.article <> b.article", "a.day <> b.day"),  # equal and different
        JOINED.replace("a.day = b.day", "a.day = b.viewer"),
        JOINED.replace("a.day = b.day", "a.day = a.day"),  # not across the two rows
        JOINED.replace("DISTINCT a.viewer", "DISTINCT a.viewer, a.article"),  # the column that differs is not shared
        JOINED.replace("DISTINCT a.viewer", "DISTINCT a.viewer, a.author"),
        JOINED.replace("DISTINCT a.viewer", "DISTINCT a.viewer, b.viewer"),  # one column listed twice
        JOINED.replace("DISTINCT a.viewer", "DISTINCT UPPER(a.viewer) AS viewer"),
        JOINED.replace("views AS b", "visits AS b"),  # another table
        JOINED.replace("views AS b", "VIEWS AS b"),  # a name that differs by case is another table to some engines
        JOINED.replace("JOIN views AS b", "LEFT JOIN views AS b"),
        JOINED + " ORDER BY a.viewer LIMIT 3",
        JOINED + " AND EXISTS (SELECT 1 FROM views AS c WHERE c.viewer = a.viewer)",
        JOINED.replace("JOIN views AS b ON", "JOIN views AS b JOIN views AS c ON a.viewer = c.viewer AND"),  # three rows
    ],
)
def test_other_selects_are_left_alone(sql):
    assert rewritten(sql) is None


def test_a_floating_point_column_is_left_alone():
    assert rewritten(JOINED, types={"views": {"article": "DOUBLE"}}) is None
    assert rewritten(JOINED, types={"views": {"day": "FLOAT"}}) is None
    assert rewritten(JOINED, types={"views": {"article": "INT", "day": "DATE", "viewer": "INT"}}) == GROUPED


SPELLINGS = [
    JOINED,
    "SELECT DISTINCT b.viewer FROM views AS a, views AS b WHERE a.article > b.article AND a.viewer = b.viewer AND a.day = b.day",
    "SELECT DISTINCT a.viewer, b.day FROM views AS a JOIN views AS b ON a.viewer = b.viewer AND a.day = b.day AND a.article < b.article",
    "SELECT DISTINCT a.day FROM views AS a JOIN views AS b ON a.day = b.day WHERE a.viewer <> b.viewer",
]


@pytest.mark.parametrize("seed", range(4))
def test_the_grouping_returns_the_rows_duckdb_returns(seed):
    """No declared fact is needed: NULLs anywhere, repeated rows, empty tables."""

    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    nonempty = 0
    for sql in SPELLINGS:
        grouped = rewritten(sql)
        assert grouped is not None, sql
        for _ in range(12):
            con = duckdb.connect()
            con.execute("CREATE TABLE views (viewer INT, day INT, article INT, author INT)")
            for _row in range(rng.randint(0, 8)):
                con.execute("INSERT INTO views VALUES (?, ?, ?, ?)", [rng.choice([None, 1, 1, 2]), rng.choice([None, 5, 5, 6]), rng.choice([None, 7, 7, 8, 9]), rng.choice([None, 3])])
            expected = sorted(con.execute(sql).fetchall(), key=repr)
            assert expected == sorted(con.execute(grouped).fetchall(), key=repr), sql
            nonempty += bool(expected)
    assert nonempty > 5


def test_the_prover_reads_the_join_as_the_grouping():
    pytest.importorskip("z3")
    from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
    from kumo_sql_prover.smt_equivalence import TableConstraints

    schema = {"views": ["article", "author", "viewer", "day"]}
    joined = "SELECT DISTINCT v1.viewer AS id FROM views v1 JOIN views v2 ON v1.viewer = v2.viewer AND v1.day = v2.day AND v1.article != v2.article"
    counted = "SELECT DISTINCT viewer AS id FROM views GROUP BY day, viewer HAVING COUNT(DISTINCT article) > 1"
    facts = {"views": TableConstraints(not_null=frozenset({"viewer", "day"}))}
    assert prove_equivalent_algebraic(joined, counted, schema=schema, constraints=facts, compare_names=False, dialect="mysql").proven
    # without the NOT NULL facts the grouped spelling lists a NULL viewer's group, and the join does not
    assert not prove_equivalent_algebraic(joined, counted, schema=schema, compare_names=False, dialect="mysql").proven
    # counting rows is another question: two rows may show one article
    rows = counted.replace("COUNT(DISTINCT article)", "COUNT(*)")
    assert not prove_equivalent_algebraic(joined, rows, schema=schema, constraints=facts, compare_names=False, dialect="mysql").proven
