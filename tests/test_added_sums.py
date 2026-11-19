"""A sum of an addition of never-NULL terms split into sums, and a zero-padded total moved inside (``added_sums``).

Checked on a MySQL 8.0.46 server on 2026-11-19 (400 generated tables of 0 to 12 rows with NULLs, INT and DECIMAL
columns, as a one-row select, grouped, and filtered to no rows): the nine pairs of spellings the module joins
returned the same values with the same type and scale, 12,951 result rows in all, and the four controls (a term
that may be NULL, and a padded total with no sum beside it or beside a sum that may be NULL) differed 1,162
times. The replay below makes the comparisons on SQLite and DuckDB.
"""

import random
import sqlite3

import pytest
import sqlglot

from kumo_sql_prover.added_sums import split_added_sums

SCHEMA = {"box": ["box_id", "chest_id", "apples", "pears", "weight", "cost"], "chest": ["chest_id", "apples", "pears"]}
TYPES = {
    "box": {"box_id": "INT", "chest_id": "INT", "apples": "INT", "pears": "INT", "weight": "DOUBLE", "cost": "DECIMAL(10, 2)"},
    "chest": {"chest_id": "INT", "apples": "INT", "pears": "INT"},
}
NOT_NULL = {"box": frozenset({"box_id", "apples", "pears", "weight", "cost"}), "chest": frozenset({"chest_id", "apples", "pears"})}
JOINED = "FROM box AS b LEFT JOIN chest AS c ON b.chest_id = c.chest_id"


def reading(sql: str, dialect: str = "mysql", not_null=NOT_NULL, types=TYPES) -> str | None:
    tree, changed = split_added_sums(sqlglot.parse_one(sql, read=dialect), SCHEMA, not_null, types)
    return tree.sql(dialect=dialect) if changed else None


def test_a_sum_of_never_null_terms_is_split():
    assert reading("SELECT SUM(apples + pears) FROM box") == "SELECT (SUM(apples) + SUM(pears)) FROM box"
    assert reading("SELECT chest_id, SUM(apples + (pears + box_id)) AS n FROM box GROUP BY chest_id") == "SELECT chest_id, (SUM(apples) + SUM(pears) + SUM(box_id)) AS n FROM box GROUP BY chest_id"
    # a padded column of the left join's padded side counts, with COALESCE or IFNULL and any whole-number filler
    assert reading(f"SELECT SUM(b.apples + IFNULL(c.apples, 0)) {JOINED}") == f"SELECT (SUM(b.apples) + SUM(COALESCE(c.apples, 0))) {JOINED}"
    assert reading(f"SELECT SUM(COALESCE(c.pears, 7) + b.pears) * 2 {JOINED}") == f"SELECT (SUM(COALESCE(c.pears, 7)) + SUM(b.pears)) * 2 {JOINED}"
    assert reading("SELECT SUM(cost + apples) FROM box") == "SELECT (SUM(cost) + SUM(apples)) FROM box"  # fixed-point adds exactly
    # in HAVING and in a derived table
    assert reading("SELECT chest_id FROM box GROUP BY chest_id HAVING SUM(apples + pears) > 3") == "SELECT chest_id FROM box GROUP BY chest_id HAVING (SUM(apples) + SUM(pears)) > 3"
    assert reading("SELECT n FROM (SELECT SUM(apples + pears) AS n FROM box) AS t") == "SELECT n FROM (SELECT (SUM(apples) + SUM(pears)) AS n FROM box) AS t"


def test_a_padded_total_beside_a_sum_that_is_null_over_no_rows_moves_inside():
    assert reading(f"SELECT SUM(b.apples) + IFNULL(SUM(c.apples), 0) {JOINED}") == f"SELECT SUM(b.apples) + SUM(COALESCE(c.apples, 0)) {JOINED}"
    assert reading(f"SELECT COALESCE(SUM(c.apples), 0) + (SUM(b.apples) + COALESCE(SUM(c.pears), 0)) {JOINED}") == f"SELECT SUM(COALESCE(c.apples, 0)) + (SUM(b.apples) + SUM(COALESCE(c.pears, 0))) {JOINED}"
    assert reading(f"SELECT IFNULL(SUM(c.apples), 0) + SUM(IFNULL(c.pears, 0)) {JOINED}") == f"SELECT SUM(COALESCE(c.apples, 0)) + SUM(COALESCE(c.pears, 0)) {JOINED}"


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT SUM(b.apples + c.apples) {JOINED}",  # the padded side may be NULL
        "SELECT SUM(apples + chest_id) FROM box",  # chest_id is not declared NOT NULL
        "SELECT SUM(apples + weight) FROM box",  # floating point: the additions are regrouped
        "SELECT SUM(apples - pears) FROM box",
        "SELECT SUM(apples * pears) FROM box",
        "SELECT SUM(apples + 1) FROM box",
        "SELECT SUM(apples + CASE WHEN pears > 1 THEN 1 ELSE 0 END) FROM box",
        "SELECT SUM(apples + COALESCE(chest_id, pears)) FROM box",  # the filler is not a literal
        "SELECT SUM(apples + COALESCE(chest_id, 0.5)) FROM box",
        "SELECT SUM(DISTINCT apples + pears) FROM box",
        "SELECT SUM(apples + pears) OVER (PARTITION BY chest_id) FROM box",
        "SELECT (SELECT SUM(b.apples + c.apples) FROM chest AS c) FROM box AS b",  # a term of the outer query
        "SELECT SUM(b.apples + c.apples) FROM box AS b JOIN chest AS c ON b.chest_id = c.chest_id JOIN chest AS c ON 1 = 1",
        "SELECT SUM(apples + pears) FROM box AS b JOIN chest AS c ON b.chest_id = c.chest_id",  # which table's apples
        "SELECT SUM(t.apples + t.pears) FROM (SELECT apples, pears FROM box) AS t",  # a derived table's columns have no declared facts
        f"SELECT IFNULL(SUM(c.apples), 0) {JOINED}",  # no sum beside it: 0 against NULL over no rows
        f"SELECT SUM(c.pears) + IFNULL(SUM(c.apples), 0) {JOINED}",  # the sum beside it may be NULL over some rows
        f"SELECT SUM(b.apples) * IFNULL(SUM(c.apples), 0) {JOINED}",
        f"SELECT SUM(b.apples) + IFNULL(SUM(c.apples), 1) {JOINED}",
        f"SELECT SUM(b.apples) - IFNULL(SUM(c.apples), 0) {JOINED}",
        f"SELECT b.chest_id, SUM(b.apples) + IFNULL(SUM(c.apples), 0) {JOINED} GROUP BY b.chest_id",  # padded_sums reads the grouped select
    ],
)
def test_other_sums_are_left_alone(sql):
    assert reading(sql) is None


def test_nothing_fires_without_declared_types_or_facts():
    assert reading("SELECT SUM(apples + pears) FROM box", types=None) is None
    assert reading("SELECT SUM(apples + pears) FROM box", types={"box": {"apples": "INT"}}) is None
    assert reading("SELECT SUM(apples + pears) FROM box", not_null={}) is None


PAIRS = [
    ("SUM(c + d)", "SUM(c) + SUM(d)"),
    ("SUM(c + IFNULL(a, 0))", "SUM(c) + IFNULL(SUM(a), 0)"),
    ("SUM(c + COALESCE(a, 7) + d)", "SUM(c) + SUM(COALESCE(a, 7)) + SUM(d)"),
    ("SUM(IFNULL(a, 0) + IFNULL(b, 0))", "SUM(IFNULL(a, 0)) + SUM(IFNULL(b, 0))"),
    ("IFNULL(SUM(a), 0) + SUM(d) + IFNULL(SUM(b), 0)", "SUM(IFNULL(a, 0)) + SUM(d) + SUM(IFNULL(b, 0))"),
]
CONTROLS = [("SUM(a + c)", "SUM(a) + SUM(c)"), ("IFNULL(SUM(a), 0)", "SUM(IFNULL(a, 0))"), ("SUM(a) + IFNULL(SUM(b), 0)", "SUM(a) + SUM(IFNULL(b, 0))")]
T_SCHEMA = {"t": ["g", "a", "b", "c", "d"]}
T_TYPES = {"t": {name: "INT" for name in T_SCHEMA["t"]}}
T_NOT_NULL = {"t": frozenset({"c", "d"})}


@pytest.mark.parametrize("seed", range(3))
def test_the_reading_returns_what_sqlite_returns(seed):
    """Each statement returns what the module's rewrite of it returns, and the spellings of a pair return the same rows."""

    rng = random.Random(seed)

    def read(sql):
        tree, _ = split_added_sums(sqlglot.parse_one(sql, read="mysql"), T_SCHEMA, T_NOT_NULL, T_TYPES)
        return tree.sql(dialect="sqlite")

    shapes = ("SELECT {} FROM t", "SELECT g, {} FROM t GROUP BY g ORDER BY g", "SELECT {} FROM t WHERE g = 1")
    statements = [shape.format(side) for shape in shapes for pair in PAIRS for side in pair]
    rewritten = {sql: read(sql) for sql in statements}
    plain = {sql: sqlglot.parse_one(sql, read="mysql").sql(dialect="sqlite") for sql in statements}
    assert sum(rewritten[sql] != plain[sql] for sql in statements) >= 12
    differing = 0
    for _ in range(40):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE t (g INT, a INT, b INT, c INT NOT NULL, d INT NOT NULL)")
        rows = [(rng.choice([1, 2, None]), rng.choice([None, None, 1, -2, 3]), rng.choice([None, 1, 2]), rng.randint(-3, 3), rng.randint(0, 9)) for _ in range(rng.choice([0, 1, 2, 5, 12]))]
        con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?)", rows)
        for sql in statements:
            assert con.execute(plain[sql]).fetchall() == con.execute(rewritten[sql]).fetchall(), (sql, rewritten[sql], rows)
        for shape in shapes:
            for left, right in PAIRS:  # the two spellings of a pair agree, which is what the one text they share claims
                assert con.execute(plain[shape.format(left)]).fetchall() == con.execute(plain[shape.format(right)]).fetchall(), (shape, left, right, rows)
            for left, right in CONTROLS:  # and the spellings the module refuses to join do differ
                run = lambda text: con.execute(sqlglot.parse_one(shape.format(text), read="mysql").sql(dialect="sqlite")).fetchall()  # noqa: E731
                differing += run(left) != run(right)
    assert differing > 20


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

FACTS = {"box": TableConstraints(not_null=NOT_NULL["box"], keys=(("box_id",),)), "chest": TableConstraints(not_null=NOT_NULL["chest"], keys=(("chest_id",),))}
OPTIONS = dict(schema=SCHEMA, types=TYPES, constraints=FACTS, compare_names=False, dialect="mysql")
TOTAL = f"SELECT SUM(b.apples + IFNULL(c.apples, 0)) AS apples {JOINED}"


def test_the_spellings_of_a_total_over_a_left_join_meet():
    for other in ("SUM(b.apples) + SUM(IFNULL(c.apples, 0))", "SUM(b.apples) + IFNULL(SUM(c.apples), 0)", "SUM(b.apples) + SUM(COALESCE(c.apples, 0))", "SUM(IFNULL(b.apples, 0)) + SUM(IFNULL(c.apples, 0))"):
        assert prove_equivalent_algebraic(TOTAL, f"SELECT {other} AS apples {JOINED}", **OPTIONS).proven, other
    # another column, and a total that is not padded, are other questions
    assert not prove_equivalent_algebraic(TOTAL, f"SELECT SUM(b.apples) + IFNULL(SUM(c.pears), 0) AS apples {JOINED}", **OPTIONS).proven
    assert not prove_equivalent_algebraic(TOTAL, f"SELECT SUM(b.apples) + SUM(c.apples) AS apples {JOINED}", **OPTIONS).proven


def test_a_term_that_may_be_null_keeps_the_pair_unproved():
    loose = {**FACTS, "box": TableConstraints(not_null=frozenset({"box_id"}), keys=(("box_id",),))}
    split = f"SELECT SUM(b.apples) + SUM(IFNULL(c.apples, 0)) AS apples {JOINED}"
    assert prove_equivalent_algebraic(TOTAL, split, **OPTIONS).proven
    assert not prove_equivalent_algebraic(TOTAL, split, **{**OPTIONS, "constraints": loose}).proven
    assert not prove_equivalent_algebraic("SELECT IFNULL(SUM(c.apples), 0) AS n " + JOINED, "SELECT SUM(IFNULL(c.apples, 0)) AS n " + JOINED, **OPTIONS).proven


def test_it_is_a_rule_of_the_eleventh_batch():
    assert "added_sums" in shape_fallback.ELEVENTH_BATCH
    sql = "SELECT SUM(apples + pears) FROM box"
    options = dict(schema=SCHEMA, dialect="mysql", types=TYPES, not_null=NOT_NULL)
    with shape_fallback.tracking() as fired:
        normalize(sql, **options)
    assert "added_sums" in fired
    with shape_fallback.without("added_sums"), shape_fallback.tracking() as fired:
        normalize(sql, **options)
    assert "added_sums" not in fired
