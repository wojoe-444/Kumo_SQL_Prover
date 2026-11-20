"""A MySQL sum of a truth test written as a sum of ones (``boolean_sums``).

Checked on a MySQL 8.0.46 server on 2026-11-19 (400 generated tables of 0 to 12 rows, with NULLs, and with every
nullable column NULL in three tables of ten): for 14 tests over NOT NULL columns and ``IS [NOT] NULL``, ``SUM(p)`` and
``SUM(CASE WHEN p THEN 1 ELSE 0 END)`` returned the same values of the same type as a grouped output, as a one-row
output, under ``HAVING .. = 0`` and as a term beside ``SUM((c = 1) + (d = 2))``; for 9 tests over nullable columns the
two spellings kept the same groups under ``> 0``, ``>= 1``, ``<> 0``, ``= 1``, ``>= 2``, ``> 1``, ``0 <`` and ``1 <=``
in a ``HAVING`` and its ``AND``; 45,228 result rows in all. The controls (``= 0``, ``< 1``, the sum as an output,
``NOT (.. > 0)``) differed 7,456 times. For "none beside at least" (``HAVING SUM(p) > 0 AND SUM(q) = 0`` where ``p`` TRUE
leaves no column of ``q`` NULL): 9 pairs of tests, four spellings of each comparison, all columns nullable, 57,600
statements on 400 tables, every one equal to its ``CASE`` spelling; five control pairs in which ``q`` reads a column
``p`` does not require differed 4,404 times. The replays below make the same comparisons on SQLite, which also reads
a comparison as 1, 0 or NULL.
"""

import random
import sqlite3

import pytest
import sqlglot

from kumo_sql_prover.boolean_sums import write_boolean_sums

SCHEMA = {"sale": ["id", "buyer", "item_id", "paid", "note"], "item": ["item_id", "label", "price"]}
TYPES = {"sale": {"id": "INT", "buyer": "INT", "item_id": "INT", "paid": "INT", "note": "VARCHAR"}, "item": {"item_id": "INT", "label": "VARCHAR", "price": "INT"}}
NOT_NULL = {"sale": frozenset({"id", "buyer", "item_id"}), "item": frozenset({"item_id", "label"})}
JOINED = "FROM sale AS s JOIN item AS i ON s.item_id = i.item_id"


def reading(sql: str, dialect: str = "mysql", not_null=NOT_NULL) -> str | None:
    tree, changed = write_boolean_sums(sqlglot.parse_one(sql, read=dialect), SCHEMA, not_null, TYPES, dialect)
    return tree.sql(dialect=dialect) if changed else None


def flag(test: str) -> str:
    return f"CASE WHEN {test} THEN 1 ELSE 0 END"


def test_a_test_that_is_never_unknown_is_a_sum_of_ones_wherever_it_stands():
    assert reading(f"SELECT s.buyer, SUM(i.label = 'S') AS n {JOINED} GROUP BY s.buyer") == f"SELECT s.buyer, SUM({flag("i.label = 'S'")}) AS n {JOINED} GROUP BY s.buyer"
    # an unqualified column that one table has, both tables' columns, every operator read, and IS NULL of a nullable column
    assert reading(f"SELECT SUM(label <> 'S' AND buyer >= 3) {JOINED}") == f"SELECT SUM({flag("label <> 'S' AND buyer >= 3")}) {JOINED}"
    for test in ("buyer BETWEEN 1 AND 3", "buyer IN (1, -2)", "i.label LIKE 'S%'", "NOT (s.buyer = i.item_id)", "(buyer < 2 OR label = 'T')", "s.paid IS NULL", "NOT s.note IS NULL", "s.paid IS NULL OR buyer = 1"):
        assert reading(f"SELECT SUM({test}) {JOINED}") == f"SELECT SUM({flag(test)}) {JOINED}".replace("CASE WHEN (buyer < 2 OR label = 'T')", "CASE WHEN buyer < 2 OR label = 'T'"), test
    # under HAVING .. = 0 too, and in a derived table
    assert reading(f"SELECT s.buyer {JOINED} GROUP BY s.buyer HAVING SUM(i.label = 'T') = 0") == f"SELECT s.buyer {JOINED} GROUP BY s.buyer HAVING SUM({flag("i.label = 'T'")}) = 0"
    assert reading("SELECT n FROM (SELECT SUM(buyer = 1) AS n FROM sale) AS t") == f"SELECT n FROM (SELECT SUM({flag('buyer = 1')}) AS n FROM sale) AS t"


def test_the_terms_of_an_addition_inside_the_sum_are_read_one_by_one():
    assert reading("SELECT SUM((buyer = 1) + (item_id = 2)) FROM sale") == f"SELECT SUM(({flag('buyer = 1')}) + ({flag('item_id = 2')})) FROM sale"
    # a term that may be unknown stays, the others are read
    assert reading("SELECT SUM((buyer = 1) + (paid = 2) + id) FROM sale") == f"SELECT SUM(({flag('buyer = 1')}) + (paid = 2) + id) FROM sale"


def test_an_at_least_test_in_having_needs_no_not_null_fact():
    grouped = "SELECT buyer FROM sale GROUP BY buyer HAVING "
    for test, read in [
        ("SUM(paid = 3) > 0", f"SUM({flag('paid = 3')}) > 0"),
        ("SUM(paid = 3) >= 1 AND COUNT(*) > 2", f"SUM({flag('paid = 3')}) >= 1 AND COUNT(*) > 2"),
        ("(SUM(note = 'x') <> 0) AND (0 < SUM(paid > 1))", f"(SUM({flag("note = 'x'")}) <> 0) AND (0 < SUM({flag('paid > 1')}))"),
        ("SUM(paid = 3) = 2", f"SUM({flag('paid = 3')}) = 2"),
        ("2 <= SUM(paid = 3)", f"2 <= SUM({flag('paid = 3')})"),
        ("SUM(paid = 3) > 0 AND SUM(note = 'x') = 0", f"SUM({flag('paid = 3')}) > 0 AND SUM(note = 'x') = 0"),  # only the first is 'at least'
    ]:
        assert reading(grouped + test, not_null={}) == grouped + read, test


def test_a_none_test_beside_an_at_least_test_that_makes_it_known_needs_no_fact_either():
    grouped = "SELECT s.buyer FROM sale AS s LEFT JOIN item AS i USING (item_id) GROUP BY 1 HAVING "
    for test, read in [
        ("SUM(label = 'S') > 0 AND SUM(label = 'T') = 0", f"SUM({flag("label = 'S'")}) > 0 AND SUM({flag("label = 'T'")}) = 0"),
        ("(SUM(label = 'T') < 1) AND (SUM(label = 'S') >= 1)", f"(SUM({flag("label = 'T'")}) < 1) AND (SUM({flag("label = 'S'")}) >= 1)"),
        ("0 = SUM(label IN ('T', 'U')) AND SUM(label LIKE 'S%') > 0 AND SUM(label = 'V') <= 0", f"0 = SUM({flag("label IN ('T', 'U')")}) AND SUM({flag("label LIKE 'S%'")}) > 0 AND SUM({flag("label = 'V'")}) <= 0"),
        # two columns required by the AND, both read by the other test; and an at-least test already written with CASE
        ("SUM(label = 'S' AND paid < 9) > 0 AND SUM(label = 'T' OR paid > 3) = 0", f"SUM({flag("label = 'S' AND paid < 9")}) > 0 AND SUM({flag("label = 'T' OR paid > 3")}) = 0"),
        (f"SUM({flag("label = 'S'")}) > 0 AND SUM(label = 'T') = 0", f"SUM({flag("label = 'S'")}) > 0 AND SUM({flag("label = 'T'")}) = 0"),
        # what is not made known stays: another column, a column one side of an OR compares, a NOT, another qualifier
        ("SUM(label = 'S') > 0 AND SUM(paid = 3) = 0", f"SUM({flag("label = 'S'")}) > 0 AND SUM(paid = 3) = 0"),
        ("SUM(label = 'S' OR paid > 3) > 0 AND SUM(label = 'T') = 0", f"SUM({flag("label = 'S' OR paid > 3")}) > 0 AND SUM(label = 'T') = 0"),
        ("SUM(NOT label = 'S') > 0 AND SUM(label = 'T') = 0", f"SUM({flag("NOT label = 'S'")}) > 0 AND SUM(label = 'T') = 0"),
        ("SUM(i.label = 'S') > 0 AND SUM(label = 'T') = 0", f"SUM({flag("i.label = 'S'")}) > 0 AND SUM(label = 'T') = 0"),
        ("SUM(label = 'S') > 0 AND SUM(label = 'T' AND paid > 3) = 0", f"SUM({flag("label = 'S'")}) > 0 AND SUM(label = 'T' AND paid > 3) = 0"),
    ]:
        assert reading(grouped + test, not_null={}) == grouped + read, test
    # under an OR the group may pass without the at-least test: nothing is read
    assert reading(grouped + "SUM(label = 'T') = 0 AND (SUM(label = 'S') > 0 OR COUNT(*) > 3)", not_null={}) is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT buyer, SUM(paid = 3) FROM sale GROUP BY buyer",  # may be unknown, and an output: NULL against 0
        "SELECT buyer FROM sale GROUP BY buyer HAVING SUM(paid = 3) = 0",
        "SELECT buyer FROM sale GROUP BY buyer HAVING SUM(paid = 3) < 1",
        "SELECT buyer FROM sale GROUP BY buyer HAVING SUM(paid = 3) >= 0",  # TRUE for a count of 0 and unknown for NULL
        "SELECT buyer FROM sale GROUP BY buyer HAVING SUM(paid = 3) > -1",
        "SELECT buyer FROM sale GROUP BY buyer HAVING NOT (SUM(paid = 3) > 0)",
        "SELECT buyer FROM sale GROUP BY buyer HAVING SUM(paid = 3) > 0 OR buyer = 1",
        "SELECT buyer FROM sale GROUP BY buyer HAVING SUM(paid = 3) > 0.5",
        "SELECT buyer FROM (SELECT buyer, SUM(paid = 3) AS n FROM sale GROUP BY buyer) AS t WHERE n > 0",  # the test is not in the HAVING of the sum's select
        "SELECT s.buyer, SUM(i.label = 'S') FROM sale AS s LEFT JOIN item AS i ON s.item_id = i.item_id GROUP BY s.buyer",  # the padded side
        "SELECT s.buyer, SUM(i.label = 'S') FROM item AS i RIGHT JOIN sale AS s ON s.item_id = i.item_id GROUP BY s.buyer",
        "SELECT SUM(item_id = 2) FROM sale AS s JOIN item AS i ON s.item_id = i.item_id",  # two tables have an item_id
        "SELECT SUM(t.buyer = 1) FROM (SELECT buyer FROM sale) AS t",  # a derived table's column has no declared fact
        "SELECT (SELECT SUM(s.buyer = 1) FROM item AS i) FROM sale AS s",  # the outer query's aggregate
        "SELECT SUM(DISTINCT buyer = 1) FROM sale",
        "SELECT SUM(buyer = 1) OVER (PARTITION BY item_id) FROM sale",
        "SELECT SUM(buyer) FROM sale",
        "SELECT SUM(buyer + 1 = 2) FROM sale",  # arithmetic inside the test is not read
        "SELECT SUM(buyer = NULL) FROM sale",
        "SELECT SUM(buyer IN (SELECT item_id FROM item)) FROM sale",
        "SELECT SUM(CASE WHEN buyer = 1 THEN 1 ELSE 0 END) FROM sale",
        "SELECT COUNT(buyer = 1), MAX(buyer = 1), AVG(buyer = 1) FROM sale",
    ],
)
def test_other_sums_are_left_alone(sql):
    assert reading(sql) is None


def test_other_dialects_are_left_alone():
    for dialect in ("sqlite", "postgres", "bigquery", "duckdb"):
        assert reading("SELECT SUM(buyer = 1) FROM sale", dialect) is None


NEVER = ["c = d", "c <> 2", "c < d", "c >= d AND d > 1", "c = 1 OR d = 2", "NOT (c = d)", "c BETWEEN 1 AND 2", "c IN (1, 3)", "s2 = 'a'", "s2 LIKE 'a%'", "a IS NULL", "a IS NOT NULL", "c = d OR a IS NULL"]
MAYBE = ["a = b", "a > 1", "a = c", "s = 'a'", "a IN (1, 2)", "a BETWEEN 1 AND 2", "NOT (a = b)", "a = 1 OR b = 2", "s LIKE 'a%'"]
AT_LEAST = ["{} > 0", "{} >= 1", "{} <> 0", "{} = 1", "{} >= 2", "{} > 1", "0 < {}", "1 <= {}", "{} > 0 AND COUNT(*) > 1", "({} >= 1) AND g IS NOT NULL"]
CONTROLS = ["SELECT g FROM t GROUP BY g HAVING {} = 0 ORDER BY g", "SELECT g, {} FROM t GROUP BY g ORDER BY g", "SELECT g FROM t GROUP BY g HAVING {} < 1 ORDER BY g", "SELECT g FROM t GROUP BY g HAVING NOT ({} > 0) ORDER BY g"]
T_SCHEMA = {"t": ["g", "a", "b", "c", "d", "s", "s2"]}
T_NOT_NULL = {"t": frozenset({"c", "d", "s2"})}


@pytest.mark.parametrize("seed", range(3))
def test_both_readings_return_what_sqlite_returns(seed):
    """The module's own rewrite of each statement (asked as MySQL, run on SQLite) returns the rows the statement returns."""

    rng = random.Random(seed)

    def read(sql):
        tree, changed = write_boolean_sums(sqlglot.parse_one(sql, read="mysql"), T_SCHEMA, T_NOT_NULL, None, "mysql")
        assert changed, sql
        return tree.sql(dialect="sqlite")

    statements = []
    for test in NEVER:
        statements += [f"SELECT g, SUM({test}) FROM t GROUP BY g ORDER BY g", f"SELECT SUM({test}) FROM t", f"SELECT g FROM t GROUP BY g HAVING SUM({test}) = 0 ORDER BY g", f"SELECT g, SUM(({test}) + (c = 1)) FROM t GROUP BY g ORDER BY g"]
    for test in MAYBE:
        statements += ["SELECT g FROM t GROUP BY g HAVING " + shape.format(f"SUM({test})") + " ORDER BY g" for shape in AT_LEAST]
    pairs = [(sql, read(sql)) for sql in statements]
    differing = 0
    for _ in range(25):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE t (g INT, a INT, b INT, c INT NOT NULL, d INT NOT NULL, s TEXT, s2 TEXT NOT NULL)")
        rows = [(rng.choice([1, 2, 3, None]), rng.choice([None, None, 1, 2, 3]), rng.choice([None, 1, 2]), rng.randint(1, 3), rng.randint(1, 3), rng.choice([None, "a", "ab", "b"]), rng.choice(["a", "ab", "b"])) for _ in range(rng.choice([0, 1, 2, 5, 12]))]
        if rng.random() < 0.3:
            rows = [(g, None, None, c, d, None, s2) for g, a, b, c, d, s, s2 in rows]
        con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        for sql, rewritten in pairs:
            assert con.execute(sql).fetchall() == con.execute(rewritten).fetchall(), (sql, rewritten, rows)
        for test in MAYBE:  # what the module refuses does differ
            for control in CONTROLS:
                differing += con.execute(control.format(f"SUM({test})")).fetchall() != con.execute(control.format(f"SUM({flag(test)})")).fetchall()
    assert differing > 20


BESIDE = [("s = 'a'", "s = 'b'"), ("s = 'a'", "s IN ('b', 'ab')"), ("a = 1", "a > 1"), ("a = b", "a < b"), ("a = 1 AND b = 2", "a = 2 OR b = 1"), ("a BETWEEN 1 AND 2", "a = 3"), ("s LIKE 'a%'", "s = 'b'"), ("a = 1 OR a = 2", "a = 3"), ("a = 1", "a <> 1 OR b IS NULL")]
NOT_BESIDE = [("s = 'a'", "a = 1"), ("a = 1 OR b = 2", "a = 2"), ("NOT (a = 1)", "b = 2"), ("a IS NULL", "a = 1"), ("a = 1", "b = 1")]


@pytest.mark.parametrize("seed", range(3))
def test_a_none_test_beside_an_at_least_test_returns_what_sqlite_returns(seed):
    rng = random.Random(seed)

    def read(sql):
        tree, _ = write_boolean_sums(sqlglot.parse_one(sql, read="mysql"), T_SCHEMA, {}, None, "mysql")
        return tree.sql(dialect="sqlite")

    def statements(pairs):
        return [
            "SELECT g FROM t GROUP BY g HAVING " + least.format(f"SUM({p})") + " AND " + none.format(f"SUM({q})") + " ORDER BY g"
            for p, q in pairs for least in ("{} > 0", "{} >= 1", "1 <= {}", "{} = 2") for none in ("{} = 0", "{} < 1", "{} <= 0", "0 = {}")
        ]

    read_pairs = [(sql, read(sql)) for sql in statements(BESIDE)]
    assert all(rewritten.count("CASE WHEN") == 2 for _, rewritten in read_pairs)
    refused = [(sql, read(sql)) for sql in statements(NOT_BESIDE)]
    assert all(rewritten.count("CASE WHEN") == 1 for _, rewritten in refused)  # only the at-least test is read
    differing = 0
    for _ in range(25):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE t (g INT, a INT, b INT, c INT, d INT, s TEXT, s2 TEXT)")
        rows = [(rng.choice([1, 2, 3, None]), rng.choice([None, None, 1, 2, 3]), rng.choice([None, 1, 2]), 1, 1, rng.choice([None, None, "a", "ab", "b"]), "a") for _ in range(rng.choice([0, 1, 2, 5, 12]))]
        con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        for sql, rewritten in read_pairs + refused:
            assert con.execute(sql).fetchall() == con.execute(rewritten).fetchall(), (sql, rewritten, rows)
        for p, q in NOT_BESIDE:  # reading the refused test too would change the answer
            sql = f"SELECT g FROM t GROUP BY g HAVING SUM({p}) > 0 AND {{}} = 0 ORDER BY g"
            differing += con.execute(sql.format(f"SUM({q})")).fetchall() != con.execute(sql.format(f"SUM({flag(q)})")).fetchall()
    assert differing > 5


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

FACTS = {"sale": TableConstraints(not_null=frozenset({"id", "buyer", "item_id"}), keys=(("id",),)), "item": TableConstraints(not_null=frozenset({"item_id", "label"}), keys=(("item_id",),))}
OPTIONS = dict(schema=SCHEMA, types=TYPES, constraints=FACTS, compare_names=False, dialect="mysql")
BY_SUMS = f"SELECT s.buyer {JOINED} GROUP BY s.buyer HAVING SUM(i.label = 'S') > 0 AND SUM(i.label = 'T') = 0"
BY_LIST = f"SELECT DISTINCT s.buyer {JOINED} WHERE i.label = 'S' AND s.buyer NOT IN (SELECT s2.buyer FROM sale AS s2 JOIN item AS i2 ON s2.item_id = i2.item_id WHERE i2.label = 'T')"


def test_buyers_of_one_label_and_not_another_meet_the_list_spelling():
    assert prove_equivalent_algebraic(BY_SUMS, BY_LIST, **OPTIONS).proven
    assert prove_equivalent_algebraic(BY_SUMS, BY_SUMS.replace("SUM(i.label = 'S')", "SUM(CASE WHEN i.label = 'S' THEN 1 ELSE 0 END)"), **OPTIONS).proven
    # another label, and the two tests swapped, are other questions
    assert not prove_equivalent_algebraic(BY_SUMS, BY_LIST.replace("i2.label = 'T'", "i2.label = 'U'"), **OPTIONS).proven
    assert not prove_equivalent_algebraic(BY_SUMS, BY_LIST.replace("i.label = 'S'", "i.label = 'T'").replace("i2.label = 'T'", "i2.label = 'S'"), **OPTIONS).proven


def test_a_label_that_may_be_null_keeps_the_output_pair_unproved():
    """With ``label`` nullable a buyer whose rows all lack a label has ``SUM(label = 'T')`` NULL, and the ``CASE`` spelling 0."""

    loose = {**FACTS, "item": TableConstraints(not_null=frozenset({"item_id"}), keys=(("item_id",),))}
    counted = f"SELECT s.buyer, SUM(i.label = 'T') AS n {JOINED} GROUP BY s.buyer"
    by_case = f"SELECT s.buyer, SUM(CASE WHEN i.label = 'T' THEN 1 ELSE 0 END) AS n {JOINED} GROUP BY s.buyer"
    assert prove_equivalent_algebraic(counted, by_case, **OPTIONS).proven
    assert not prove_equivalent_algebraic(counted, by_case, **{**OPTIONS, "constraints": loose}).proven


def test_a_left_join_and_no_facts_still_meet_the_list_spelling():
    """With nothing declared about ``label`` the at-least test makes the none-test known; the buyer must be NOT NULL for the list."""

    loose = {"sale": TableConstraints(not_null=frozenset({"buyer"})), "item": TableConstraints()}
    options = {**OPTIONS, "constraints": loose}
    assert prove_equivalent_algebraic(BY_SUMS, BY_LIST, **options).proven
    assert not prove_equivalent_algebraic(BY_SUMS.replace("SUM(i.label = 'T') = 0", "SUM(s.paid = 3) = 0"), BY_LIST.replace("i2.label = 'T'", "s2.paid = 3"), **options).proven


def test_sums_of_tests_added_meet_one_sum_of_flags():
    one = f"SELECT s.buyer, SUM(CASE WHEN s.buyer = i.item_id THEN 1 ELSE 0 END + CASE WHEN i.label = 'S' THEN 1 ELSE 0 END) AS n {JOINED} GROUP BY s.buyer"
    added = f"SELECT s.buyer, SUM(s.buyer = i.item_id) + SUM(i.label = 'S') AS n {JOINED} GROUP BY s.buyer"
    assert prove_equivalent_algebraic(one, added, **OPTIONS).proven
    assert not prove_equivalent_algebraic(one, added.replace("'S'", "'T'"), **OPTIONS).proven


def test_it_is_a_rule_of_the_eleventh_batch():
    assert "boolean_sums" in shape_fallback.ELEVENTH_BATCH
    sql = "SELECT buyer, SUM(buyer = 1) FROM sale GROUP BY buyer"
    options = dict(schema=SCHEMA, dialect="mysql", types=TYPES, not_null=NOT_NULL)
    with shape_fallback.tracking() as fired:
        normalize(sql, **options)
    assert "boolean_sums" in fired
    with shape_fallback.without("boolean_sums"), shape_fallback.tracking() as fired:
        normalize(sql, **options)
    assert "boolean_sums" not in fired
