"""A ``UNION`` with an operand that can never hold a row (``kumo_sql_prover.empty_union_operands``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.empty_union_operands import drop_empty_union_operand

SCHEMA = {"s": ["ck", "p"], "w": ["ck", "p"], "c": ["k", "n", "y"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}

FROM_S = "SELECT c.n FROM s JOIN c ON s.ck = c.k"
FROM_W = "SELECT c.n FROM w JOIN c ON w.ck = c.k"
NONE_OF_W = f"{FROM_W} WHERE FALSE"
TAGGED = "(SELECT 's' AS src, c.n FROM s JOIN c ON s.ck = c.k UNION ALL SELECT 'w' AS src, c.n FROM w JOIN c ON w.ck = c.k) AS sd"


def dropped(sql):
    out = drop_empty_union_operand(sqlglot.parse_one(sql, read="postgres"))
    return None if out is None else out.sql(dialect="postgres")


def normal(sql):
    return normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_the_other_operand_remains():
    assert dropped(f"{FROM_S} UNION ALL {NONE_OF_W}") == FROM_S
    assert dropped(f"{FROM_S} UNION {NONE_OF_W}") == FROM_S.replace("SELECT", "SELECT DISTINCT")
    assert dropped(f"{FROM_W} WHERE 1 = 2 UNION ALL {FROM_S}") == FROM_S
    # an ordered operand stays inside its union: on its own, the ORDER BY would order the whole result
    assert dropped(f"({FROM_W} LIMIT 0) UNION ALL ({FROM_S} ORDER BY c.y LIMIT 3)") is None
    # through derived tables that pass the column on by name
    assert dropped("SELECT q.n FROM (SELECT n, y FROM c WHERE y > 0) AS q UNION ALL SELECT n FROM c WHERE FALSE") == (
        "SELECT q.n FROM (SELECT n, y FROM c WHERE y > 0) AS q"
    )
    # the first operand names the columns
    assert dropped(f"SELECT c.n AS z FROM c WHERE FALSE UNION ALL {FROM_S}") == FROM_S.replace("c.n", "c.n AS z")
    # what remains under a duplicate-removing UNION is a set
    assert dropped(f"({FROM_S} UNION ALL {FROM_W}) UNION SELECT n FROM c WHERE FALSE") == f"{FROM_S} UNION {FROM_W}"
    assert dropped(f"({FROM_S} EXCEPT {FROM_W}) UNION SELECT n FROM c WHERE FALSE") == f"{FROM_S} EXCEPT {FROM_W}"
    assert dropped(f"SELECT DISTINCT n FROM c UNION {NONE_OF_W}") == "SELECT DISTINCT n FROM c"
    assert dropped(f"SELECT n FROM c GROUP BY n UNION {NONE_OF_W}") == "SELECT DISTINCT n FROM c GROUP BY n"
    # an operand that is empty as a union of empty selects
    assert dropped(f"({NONE_OF_W} UNION ALL SELECT n FROM c WHERE FALSE) UNION ALL {FROM_S}") == FROM_S


def test_inside_a_set_operation_it_keeps_its_parentheses():
    tree = sqlglot.parse_one(f"({FROM_S} UNION ALL {NONE_OF_W}) EXCEPT SELECT n FROM c", read="postgres")
    assert isinstance(drop_empty_union_operand(tree.this.this), sqlglot.exp.Select)  # its parent is the parentheses
    tree = sqlglot.parse_one(f"{FROM_S} UNION ALL {NONE_OF_W} UNION ALL SELECT n FROM c", read="postgres")
    assert isinstance(drop_empty_union_operand(tree.this), sqlglot.exp.Subquery)


@pytest.mark.parametrize(
    "sql",
    [
        f"{FROM_S} UNION ALL {FROM_W}",
        f"{FROM_S} WHERE FALSE UNION ALL {NONE_OF_W}",  # empty as a whole: another rule's
        f"{FROM_S} UNION ALL {NONE_OF_W} ORDER BY n LIMIT 2",
        f"{FROM_S} INTERSECT {NONE_OF_W}",
        f"{FROM_S} EXCEPT {NONE_OF_W}",
        # an empty operand still gives the column its type: only the same table column is dropped
        f"{FROM_S} UNION ALL SELECT c.y FROM w JOIN c ON w.ck = c.k WHERE FALSE",
        f"{FROM_S} UNION ALL SELECT w.p FROM w WHERE FALSE",
        f"{FROM_S} UNION ALL SELECT CAST(c.n AS DOUBLE PRECISION) FROM c WHERE FALSE",
        f"{FROM_S} UNION ALL SELECT c.n + 0 FROM c WHERE FALSE",
        f"{FROM_S} UNION ALL SELECT 1 WHERE FALSE",
        f"{FROM_S} UNION ALL SELECT COUNT(*) FROM c WHERE FALSE",  # one row over no input
        "SELECT n FROM c UNION ALL SELECT x.n FROM c AS x(n, k, y) WHERE FALSE",  # renamed: another column
        "SELECT n FROM c UNION ALL SELECT q.n FROM (SELECT y AS n FROM c) AS q WHERE FALSE",
        "SELECT n FROM c, s UNION ALL SELECT n FROM c WHERE FALSE",  # a bare column of two sources
        f"({FROM_S} LIMIT 3) UNION {NONE_OF_W}",  # DISTINCT would come before the LIMIT
        f"({FROM_S} ORDER BY c.y OFFSET 1) UNION {NONE_OF_W}",
        f"SELECT DISTINCT ON (c.n) c.n FROM c UNION {NONE_OF_W}",
        f"({FROM_S} INTERSECT ALL {FROM_W}) UNION {NONE_OF_W}",
        f"({FROM_S} EXCEPT ALL {FROM_W}) UNION {NONE_OF_W}",
        "SELECT * FROM c WHERE FALSE UNION ALL SELECT k, n, y FROM c",  # the names are the star's
        "SELECT c.n AS z FROM c WHERE FALSE UNION ALL SELECT n FROM c GROUP BY n",  # GROUP BY could read the old name
        "SELECT c.n AS z FROM c WHERE FALSE UNION ALL SELECT n FROM c ORDER BY n",
        f"SELECT c.n AS z FROM c WHERE FALSE UNION ALL ({FROM_S} UNION ALL {FROM_W})",
        "SELECT c.n AS z, c.k FROM c WHERE FALSE UNION ALL SELECT n FROM c",
    ],
)
def test_what_is_left_alone(sql):
    assert dropped(sql) is None, sql


def test_a_tag_filter_reads_one_branch():
    assert normal(f"SELECT n FROM {TAGGED} WHERE src = 'w'") == normal(f"SELECT n FROM ({FROM_W}) AS sd")
    assert proved(f"SELECT n FROM {TAGGED} WHERE src = 'w'", FROM_W)
    assert proved(f"SELECT DISTINCT n FROM {TAGGED} WHERE src = 's'", FROM_S.replace("SELECT", "SELECT DISTINCT"))
    # inside another set operation, where the prover has no reading of the union form
    tagged = f"SELECT COUNT(*) FROM (SELECT DISTINCT n FROM {TAGGED} WHERE src = 's' EXCEPT SELECT DISTINCT n FROM {TAGGED} WHERE src = 'w') AS q"
    assert proved(tagged, f"SELECT COUNT(*) FROM ({FROM_S.replace('SELECT', 'SELECT DISTINCT')} EXCEPT {FROM_W.replace('SELECT', 'SELECT DISTINCT')}) AS q")


@pytest.mark.parametrize(
    "left, right",
    [
        (f"SELECT n FROM {TAGGED} WHERE src = 'w'", FROM_S),
        (f"SELECT n FROM {TAGGED} WHERE src <> 'w'", FROM_W),
        (f"SELECT n FROM {TAGGED} WHERE src = 'w'", FROM_W.replace("SELECT", "SELECT DISTINCT")),
        (f"{FROM_S} UNION {NONE_OF_W}", FROM_S),  # the duplicates still go
        (f"{FROM_S} UNION ALL {FROM_W} WHERE c.n IS NULL AND c.y = 1", f"{FROM_S} UNION ALL {FROM_W}"),
        (f"{FROM_S} UNION ALL SELECT COUNT(*) FROM c WHERE FALSE", FROM_S),
    ],
)
def test_near_misses_stay_unproven(left, right):
    assert not proved(left, right)


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261117)
    value = lambda: rng.choice([None, *range(0, 4)])  # noqa: E731
    queries = [
        f"{FROM_S} UNION ALL {NONE_OF_W}",
        f"{FROM_S} UNION {NONE_OF_W}",
        f"SELECT c.n AS z FROM c WHERE FALSE UNION ALL {FROM_S}",
        f"({FROM_S} UNION ALL {FROM_W}) UNION SELECT n FROM c WHERE FALSE",
        f"({FROM_S} EXCEPT {FROM_W}) UNION SELECT n FROM c WHERE FALSE",
        f"SELECT n FROM c GROUP BY n UNION {NONE_OF_W}",
        f"({NONE_OF_W} UNION ALL SELECT n FROM c WHERE FALSE) UNION ALL {FROM_S}",
    ]
    readings = [dropped(sql) for sql in queries]
    assert all(readings)
    rows_seen = 0
    for trial in range(30):
        db = duckdb.connect()
        for table, columns in SCHEMA.items():
            db.execute(f"CREATE TABLE {table} ({', '.join(c + ' BIGINT' for c in columns)})")
            rows = [tuple(value() for _ in columns) for _ in range(rng.choice([0, 1, 4, 9]) if trial % 5 else 6)]
            if rows:
                db.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})", rows)
        for sql, after in zip(queries, readings):
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before = db.execute(sql)
            names = [c[0] for c in before.description]
            before_rows = sorted(before.fetchall(), key=key)
            reread = db.execute(after)
            assert names == [c[0] for c in reread.description], (sql, after)
            assert before_rows == sorted(reread.fetchall(), key=key), (sql, after)
            rows_seen += len(before_rows)
    assert rows_seen > 200


def test_off_on_the_second_attempt():
    sql = f"{FROM_S} UNION ALL {NONE_OF_W}"
    assert "UNION" not in normal(sql)
    with shape_fallback.disabled():
        assert "UNION ALL" in normal(sql)
