"""Operands of ``INTERSECT`` / ``EXCEPT`` over filtered derived tables (``kumo_sql_prover.setop_operand_sources``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.setop_operand_sources import merge_operand_sources

SCHEMA = {"s": ["ck", "dk", "p"], "w": ["ck", "dk", "p"], "c": ["k", "n", "y"], "t": ["k", "x", "m"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}

FLAT = "SELECT DISTINCT c.n, t.x FROM {f} AS f, c, t WHERE f.ck = c.k AND f.dk = t.k AND t.m > 3 AND f.p < 9"
STEPS = (
    "SELECT {d} c.n, d.x FROM (SELECT ck, dk FROM {f} WHERE p < 9) AS f JOIN c ON f.ck = c.k "
    "JOIN (SELECT k, x FROM t WHERE m > 3) AS d ON f.dk = d.k"
)


def operand(sql, side="expression"):
    """The rule's reading of one operand of the set operation ``sql``, as text (``None`` when left alone)."""

    tree = sqlglot.parse_one(sql, read="postgres")
    node = tree.args[side]
    while not isinstance(node, sqlglot.exp.Select):
        node = node.this
    out = merge_operand_sources(node)
    return None if out is None else out.sql(dialect="postgres")


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres").proven


def test_a_filtered_derived_table_is_read_as_its_table():
    # one derived table per call, the first one first
    once = operand(f"SELECT n, x FROM c, t EXCEPT {STEPS.format(d='DISTINCT', f='s')}")
    assert once == (
        "SELECT DISTINCT c.n, d.x FROM s AS kumosql_m0_s, c, (SELECT k, x FROM t WHERE m > 3) AS d "
        "WHERE kumosql_m0_s.p < 9 AND kumosql_m0_s.ck = c.k AND kumosql_m0_s.dk = d.k"
    ).replace("kumosql_m0_s", once.split(" AS ")[1].split(",")[0])
    twice = operand(f"SELECT n, x FROM c, t EXCEPT {once}")
    assert twice is not None and "(SELECT" not in twice and twice.startswith("SELECT DISTINCT c.n, ") and twice.endswith(" AS x FROM " + twice.split(" FROM ", 1)[1])
    # the first operand too, and through parentheses
    assert operand(f"({STEPS.format(d='', f='s')}) INTERSECT SELECT n, x FROM c, t", "this") is not None


KEPT = "SELECT g.n FROM (SELECT n, COUNT(*) AS z FROM c GROUP BY n) AS g WHERE g.z > 1"  # not a select of tables: the set operation stays


def test_an_operand_the_normalizer_keeps_is_read_as_a_set():
    assert operand(f"{KEPT} EXCEPT SELECT c.n FROM s, c WHERE s.ck = c.k") == "SELECT DISTINCT c.n FROM s, c WHERE s.ck = c.k"
    assert operand(f"{KEPT} INTERSECT SELECT c.n FROM s, c WHERE s.ck = c.k") == "SELECT DISTINCT c.n FROM s, c WHERE s.ck = c.k"
    assert operand(f"{KEPT} EXCEPT SELECT x FROM t", "this") == KEPT.replace("SELECT g.n", "SELECT DISTINCT g.n")
    # ORDER BY and LIMIT after the second operand belong to the set operation
    assert operand(f"{KEPT} EXCEPT SELECT x FROM t ORDER BY 1 LIMIT 3") == "SELECT DISTINCT x FROM t"


@pytest.mark.parametrize(
    "second",
    [
        "SELECT d.x FROM (SELECT x FROM t WHERE m > 3 LIMIT 2) AS d",
        "SELECT d.x FROM (SELECT DISTINCT x FROM t WHERE m > 3) AS d, s",  # the join counts the derived table's rows
        "SELECT d.x FROM (SELECT x, COUNT(*) AS q FROM t GROUP BY x) AS d",
        "SELECT d.x FROM s LEFT JOIN (SELECT k, x FROM t WHERE m > 3) AS d ON s.dk = d.k",  # padded side
        "SELECT d.x FROM (SELECT x FROM t WHERE m > 3 UNION ALL SELECT n FROM c) AS d",
    ],
)
def test_derived_tables_that_stay(second):
    after = operand(f"SELECT n FROM c EXCEPT {second}")
    assert after == second.replace("SELECT d.x", "SELECT DISTINCT d.x"), second  # read as a set, sources as written


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT n FROM c UNION SELECT d.x FROM (SELECT x FROM t WHERE m > 3) AS d",  # a UNION is not read here
        "SELECT n FROM c UNION ALL SELECT d.x FROM (SELECT x FROM t WHERE m > 3) AS d",
        "SELECT n FROM c EXCEPT SELECT DISTINCT ON (d.x) d.x FROM (SELECT x FROM t WHERE m > 3) AS d",
        "SELECT n FROM c EXCEPT SELECT COUNT(d.x) FROM (SELECT x FROM t WHERE m > 3) AS d",  # the grouping rule's
        "SELECT n FROM c EXCEPT SELECT d.x FROM (SELECT x FROM t WHERE m > 3) AS d GROUP BY d.x",
        f"{KEPT} EXCEPT ALL SELECT x FROM t",  # EXCEPT ALL counts rows: no DISTINCT
        f"{KEPT} INTERSECT ALL SELECT x FROM t",
        "SELECT n FROM c EXCEPT SELECT x FROM t",  # read as an EXISTS test on this pass: left as written
        f"{KEPT} EXCEPT (SELECT x FROM t LIMIT 3)",  # DISTINCT would come before the LIMIT
        f"{KEPT} EXCEPT (SELECT x FROM t ORDER BY m OFFSET 1)",
        f"{KEPT} EXCEPT SELECT ROW_NUMBER() OVER (ORDER BY x) FROM t",
        f"{KEPT} EXCEPT SELECT DISTINCT x FROM t",  # already a set
        "SELECT n FROM c EXCEPT SELECT d.x FROM (SELECT x, ROW_NUMBER() OVER (ORDER BY k) AS i FROM t) AS d",
    ],
)
def test_what_is_left_alone(sql):
    assert operand(sql) is None, sql


def test_not_an_operand_not_read():
    select = sqlglot.parse_one("SELECT d.x FROM (SELECT x FROM t WHERE m > 3) AS d", read="postgres")
    assert merge_operand_sources(select) is None
    derived = sqlglot.parse_one("SELECT q.x FROM (SELECT d.x FROM (SELECT x FROM t WHERE m > 3) AS d) AS q EXCEPT SELECT n FROM c", read="postgres")
    inner = derived.this.args["from_" if "from_" in derived.this.args else "from"].this.this
    assert merge_operand_sources(inner) is None  # behind an alias: a derived table, not the operand


@pytest.mark.parametrize("operator", ["EXCEPT", "INTERSECT"])
def test_pairs_it_proves(operator):
    flat = f"SELECT COUNT(*) FROM ({FLAT.format(f='s')} {operator} {FLAT.format(f='w')}) AS q"
    assert proved(flat, f"SELECT COUNT(*) FROM ({STEPS.format(d='DISTINCT', f='s')} {operator} {STEPS.format(d='DISTINCT', f='w')}) AS q")
    # no DISTINCT of their own: the set operation removes the duplicates
    assert proved(flat, f"SELECT COUNT(*) FROM ({STEPS.format(d='', f='s')} {operator} {STEPS.format(d='', f='w')}) AS q")


def test_three_operands():
    # the second set operation stays one, and its operands are read as sets
    three = "SELECT COUNT(*) FROM ({} EXCEPT {} EXCEPT {}) AS q"
    flat = three.format(FLAT.format(f="s"), FLAT.format(f="w"), "SELECT DISTINCT c.n, t.x FROM c, t WHERE c.k = t.k")
    assert proved(flat, three.format(STEPS.format(d="", f="s"), STEPS.format(d="", f="w"), "SELECT c.n, t.x FROM c JOIN t ON c.k = t.k"))
    assert proved(flat, f"SELECT COUNT(*) FROM (SELECT * FROM ({STEPS.format(d='', f='s')} EXCEPT {STEPS.format(d='', f='w')}) AS e EXCEPT SELECT c.n, t.x FROM c JOIN t ON c.k = t.k) AS q")


PAIR = "SELECT COUNT(*) FROM ({} EXCEPT {}) AS q"
PAIR_FLAT = PAIR.format(FLAT.format(f="s"), FLAT.format(f="w"))
PAIR_STEPS = PAIR.format(STEPS.format(d="DISTINCT", f="s"), STEPS.format(d="DISTINCT", f="w"))


@pytest.mark.parametrize(
    "other",
    [
        PAIR_STEPS.replace("m > 3", "m >= 3"),
        PAIR.format(STEPS.format(d="DISTINCT", f="w"), STEPS.format(d="DISTINCT", f="s")),  # operands swapped
        PAIR_STEPS.replace("JOIN (SELECT k, x FROM t WHERE m > 3) AS d ON f.dk = d.k", "LEFT JOIN (SELECT k, x FROM t WHERE m > 3) AS d ON f.dk = d.k"),
        PAIR_STEPS.replace("EXCEPT", "INTERSECT"),
        PAIR.format(STEPS.format(d="", f="s"), STEPS.format(d="", f="w")).replace("EXCEPT", "EXCEPT ALL"),  # each operand's rows are counted
    ],
)
def test_near_misses_stay_unproven(other):
    assert not proved(PAIR_FLAT, other)


def test_duplicates_count_where_nothing_removes_them():
    assert not proved("SELECT c.n FROM s, c WHERE s.ck = c.k", "SELECT c.n FROM s, c WHERE s.ck = c.k EXCEPT SELECT c.n FROM c WHERE FALSE")
    assert not proved(
        "SELECT COUNT(*) FROM (SELECT c.n FROM s, c WHERE s.ck = c.k EXCEPT ALL SELECT x FROM t) AS q",
        "SELECT COUNT(*) FROM (SELECT DISTINCT c.n FROM s, c WHERE s.ck = c.k EXCEPT ALL SELECT x FROM t) AS q",
    )


def test_the_engine_agrees_on_random_tables():
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(20261116)
    value = lambda: rng.choice([None, *range(0, 5)])  # noqa: E731
    three = "SELECT n, x FROM ({} EXCEPT {} EXCEPT {}) AS q"
    queries = [
        f"{STEPS.format(d='DISTINCT', f='s')} EXCEPT {STEPS.format(d='', f='w')}",
        f"{STEPS.format(d='', f='s')} INTERSECT {STEPS.format(d='DISTINCT', f='w')}",
        three.format(STEPS.format(d="", f="s"), STEPS.format(d="", f="w"), "SELECT c.n, t.x FROM c JOIN t ON c.k = t.k"),
        f"{KEPT} EXCEPT SELECT c.n FROM s, c WHERE s.ck = c.k",
        f"SELECT x FROM t INTERSECT {KEPT}",
    ]
    readings = [normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES) for sql in queries]
    assert all(after != sql for sql, after in zip(queries, readings))
    rows_seen = 0
    for trial in range(20):
        db = duckdb.connect()
        for table, columns in SCHEMA.items():
            db.execute(f"CREATE TABLE {table} ({', '.join(c + ' BIGINT' for c in columns)})")
            rows = [tuple(value() for _ in columns) for _ in range(rng.choice([0, 1, 4, 9]) if trial % 5 else 6)]
            if rows:
                db.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})", rows)
        for sql, after in zip(queries, readings):
            key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
            before_rows = sorted(db.execute(sql).fetchall(), key=key)
            assert before_rows == sorted(db.execute(after).fetchall(), key=key), (sql, after)
            rows_seen += len(before_rows)
    assert rows_seen > 15


def test_off_on_the_second_attempt():
    sql = f"SELECT COUNT(*) FROM ({STEPS.format(d='DISTINCT', f='s')} EXCEPT {STEPS.format(d='DISTINCT', f='w')}) AS q"
    assert "EXCEPT" not in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
    with shape_fallback.disabled():
        assert "EXCEPT" in normalize(sql, schema=SCHEMA, dialect="postgres", types=TYPES)
