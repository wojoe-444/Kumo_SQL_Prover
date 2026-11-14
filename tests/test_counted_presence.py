"""A ``HAVING`` that counts the rows of a group meeting a condition is a filter and a membership test (``counted_presence``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.counted_presence import read_counted_presence

NOT_NULL = {"sales": frozenset({"buyer"})}
SCHEMA = {"sales": ["id", "buyer", "item", "day"], "items": ["item", "name"]}
COUNTED = (
    "SELECT s.buyer FROM sales AS s JOIN items AS i ON s.item = i.item GROUP BY s.buyer "
    "HAVING SUM(CASE WHEN i.name = 'pen' THEN 1 ELSE 0 END) > 0 AND SUM(CASE WHEN i.name = 'ink' THEN 1 ELSE 0 END) = 0"
)
READ = (
    "SELECT DISTINCT s.buyer FROM sales AS s JOIN items AS i ON s.item = i.item WHERE (i.name = 'pen') AND NOT EXISTS("
    "SELECT 1 FROM sales AS kqc1_s JOIN items AS kqc1_i ON kqc1_s.item = kqc1_i.item WHERE (kqc1_i.name = 'ink') AND kqc1_s.buyer = s.buyer)"
)


def rewritten(sql: str, not_null=NOT_NULL, types=None, schema=SCHEMA) -> str | None:
    result = read_counted_presence(sqlglot.parse_one(sql, read="mysql"), not_null, types, schema)
    return None if result is None else result.sql(dialect="mysql")


def test_a_count_above_zero_is_a_filter_and_a_count_of_zero_is_not_exists():
    assert rewritten(COUNTED) == READ


@pytest.mark.parametrize(
    "has, lacks",
    [
        ("SUM(CASE WHEN i.name = 'pen' THEN 1 ELSE 0 END) >= 1", "SUM(CASE WHEN i.name = 'ink' THEN 1 ELSE 0 END) < 1"),
        ("SUM(IF(i.name = 'pen', 1, 0)) <> 0", "SUM(IF(i.name = 'ink', 1, 0)) <= 0"),
        ("0 < SUM(CASE WHEN i.name = 'pen' THEN 2 ELSE 0 END)", "0 = SUM(CASE WHEN i.name = 'ink' THEN 3 ELSE 0 END)"),
        ("COUNT(CASE WHEN i.name = 'pen' THEN 1 END) > 0", "COUNT(CASE WHEN i.name = 'ink' THEN 'x' ELSE NULL END) = 0"),
        ("COUNT(IF(i.name = 'pen', 1, NULL)) >= 1", "1 > COUNT(IF(i.name = 'ink', 1, NULL))"),
        ("MAX(CASE WHEN i.name = 'pen' THEN 1 ELSE 0 END) = 1", "MAX(CASE WHEN i.name = 'ink' THEN 1 ELSE 0 END) = 0"),
        ("MAX(IF(i.name = 'pen', 1, 0)) <> 0", "MAX(IF(i.name = 'ink', 1, 0)) <> 1"),
        ("(SUM(CASE WHEN (i.name = 'pen') THEN 1 ELSE 0 END) > 0)", "(SUM((CASE WHEN i.name = 'ink' THEN 1 ELSE 0 END)) = 0)"),
    ],
)
def test_the_spellings_of_has_a_row_and_has_none(has, lacks):
    head = COUNTED.split(" HAVING ")[0]
    assert rewritten(f"{head} HAVING {has} AND {lacks}") == READ
    # the order of the two conditions does not matter: the first "has a row" is the filter
    assert rewritten(f"{head} HAVING {lacks} AND {has}") == READ.replace("kqc1_", "kqc0_")


def test_the_other_shapes_it_reads():
    head = COUNTED.split(" HAVING ")[0]
    # only "has none": every group is a candidate
    assert rewritten(f"{head} HAVING SUM(CASE WHEN i.name = 'ink' THEN 1 ELSE 0 END) = 0") == (
        "SELECT DISTINCT s.buyer FROM sales AS s JOIN items AS i ON s.item = i.item WHERE NOT EXISTS("
        "SELECT 1 FROM sales AS kqc0_s JOIN items AS kqc0_i ON kqc0_s.item = kqc0_i.item WHERE (kqc0_i.name = 'ink') AND kqc0_s.buyer = s.buyer)"
    )
    # a second "has a row" is a membership test, and the select's own WHERE is asked of the other rows too
    both = rewritten(f"{head.replace(' GROUP BY', ' WHERE s.day > 3 OR s.day < 1 GROUP BY')} HAVING COUNTIF(i.name = 'pen') > 0 AND COUNTIF(i.name = 'ink') > 0")
    assert both == (
        "SELECT DISTINCT s.buyer FROM sales AS s JOIN items AS i ON s.item = i.item WHERE (s.day > 3 OR s.day < 1) AND (i.name = 'pen') AND EXISTS("
        "SELECT 1 FROM sales AS kqc1_s JOIN items AS kqc1_i ON kqc1_s.item = kqc1_i.item WHERE (kqc1_s.day > 3 OR kqc1_s.day < 1) AND (kqc1_i.name = 'ink') AND kqc1_s.buyer = s.buyer)"
    )
    # bare names are given their table, a left join is kept, an output name is kept, DISTINCT changes nothing
    bare = rewritten(
        "SELECT DISTINCT buyer AS b FROM sales LEFT JOIN items ON sales.item = items.item GROUP BY buyer "
        "HAVING SUM(CASE WHEN name = 'pen' THEN 1 ELSE 0 END) > 0 AND SUM(CASE WHEN name = 'ink' THEN 1 ELSE 0 END) = 0"
    )
    assert bare == (
        "SELECT DISTINCT sales.buyer AS b FROM sales LEFT JOIN items ON sales.item = items.item WHERE (name = 'pen') AND NOT EXISTS("
        "SELECT 1 FROM sales AS kqc1_sales LEFT JOIN items AS kqc1_items ON kqc1_sales.item = kqc1_items.item WHERE (name = 'ink') AND kqc1_sales.buyer = sales.buyer)"
    )
    # two grouping columns, listed the other way round
    two = rewritten("SELECT s.day, s.buyer FROM sales AS s GROUP BY s.buyer, s.day HAVING SUM(CASE WHEN s.item = 1 THEN 1 ELSE 0 END) = 0", {"sales": frozenset({"buyer", "day"})})
    assert two == "SELECT DISTINCT s.day, s.buyer FROM sales AS s WHERE NOT EXISTS(SELECT 1 FROM sales AS kqc0_s WHERE (kqc0_s.item = 1) AND kqc0_s.buyer = s.buyer AND kqc0_s.day = s.day)"


# the form a left join takes once the normalizer has set it apart, and the form of a CTE read once
DERIVED = (
    "SELECT d.b FROM (SELECT s.buyer AS b, i.name AS n FROM sales AS s LEFT JOIN items AS i ON s.item = i.item) AS d GROUP BY d.b "
    "HAVING SUM(CASE WHEN d.n = 'pen' THEN 1 ELSE 0 END) > 0 AND SUM(CASE WHEN d.n = 'ink' THEN 1 ELSE 0 END) = 0"
)
HEAD = COUNTED.split(" HAVING ")[0]
TAIL = COUNTED.split(" HAVING ")[1]


@pytest.mark.parametrize(
    "sql, not_null",
    [
        (COUNTED, None),
        (COUNTED, {"sales": frozenset({"item"})}),  # two rows with a NULL buyer are one group and never equal
        (COUNTED, {"items": frozenset({"buyer"})}),
        (COUNTED.replace("= 0", "= 1"), NOT_NULL),  # exactly one row is a count, not a presence
        (COUNTED.replace("> 0", "> 1"), NOT_NULL),
        (COUNTED.replace("> 0", ">= 0"), NOT_NULL),
        (COUNTED.replace("THEN 1 ELSE 0 END) > 0", "THEN 1 ELSE 2 END) > 0"), NOT_NULL),
        (COUNTED.replace("THEN 1 ELSE 0 END) > 0", "THEN 1 END) > 0"), NOT_NULL),  # a SUM with no ELSE is NULL for a group with no such row: other rules read it
        (COUNTED.replace("THEN 1 ELSE 0 END) = 0", "THEN -1 ELSE 0 END) = 0"), NOT_NULL),
        (COUNTED.replace("SUM(CASE WHEN i.name = 'pen' THEN 1 ELSE 0 END)", "MAX(CASE WHEN i.name = 'pen' THEN 2 ELSE 0 END)"), NOT_NULL),
        (COUNTED.replace("SUM(CASE WHEN i.name = 'pen' THEN 1 ELSE 0 END)", "COUNT(DISTINCT CASE WHEN i.name = 'pen' THEN 1 END)"), NOT_NULL),
        (COUNTED.replace("SUM(CASE WHEN i.name = 'pen' THEN 1 ELSE 0 END)", "COUNT(CASE WHEN i.name = 'pen' THEN s.day END)"), NOT_NULL),  # counts the days that are not NULL
        (COUNTED.replace("SUM(CASE WHEN i.name = 'pen' THEN 1 ELSE 0 END) > 0", "COUNT(*) > 1"), NOT_NULL),
        (COUNTED + " AND s.buyer > 3", NOT_NULL),
        (COUNTED.replace(" AND SUM", " OR SUM"), NOT_NULL),
        (COUNTED.replace("SELECT s.buyer FROM", "SELECT s.buyer, COUNT(*) AS n FROM"), NOT_NULL),
        (COUNTED.replace("GROUP BY s.buyer", "GROUP BY s.buyer, s.day"), {"sales": frozenset({"buyer", "day"})}),  # a group for every day, listed without it
        (COUNTED.replace("GROUP BY s.buyer", "GROUP BY s.buyer, i.name"), NOT_NULL),
        (COUNTED.replace("SELECT s.buyer FROM", "SELECT s.buyer + 1 AS buyer FROM"), NOT_NULL),
        (COUNTED.replace("GROUP BY s.buyer", "GROUP BY s.buyer + 1"), NOT_NULL),
        (COUNTED.replace("JOIN items", "RIGHT JOIN items"), NOT_NULL),  # a padded buyer is NULL
        (COUNTED.replace("JOIN items", "FULL JOIN items"), NOT_NULL),
        (COUNTED.replace("items AS i ON", "(SELECT item, name FROM items LIMIT 3) AS i ON"), NOT_NULL),  # read twice, it may keep other rows
        (DERIVED.replace("s.buyer AS b,", "s.buyer + 0 AS b,"), NOT_NULL),  # not a column that is declared NOT NULL
        (DERIVED.replace("s.buyer AS b,", "i.item AS b,"), {"items": frozenset({"item"})}),  # the padded side of the join inside
        (DERIVED.replace("ON s.item = i.item)", "ON s.item = i.item GROUP BY s.buyer, i.name)"), NOT_NULL),
        (DERIVED.replace("s.buyer AS b,", "s.buyer AS b, s.day AS b,"), NOT_NULL),
        (COUNTED + " ORDER BY s.buyer LIMIT 2", NOT_NULL),
        (COUNTED.replace("i.name = 'ink'", "i.name = 'ink' AND RAND() < 0.5"), NOT_NULL),
        (COUNTED.replace("i.name = 'ink'", "i.item IN (SELECT item FROM items)"), NOT_NULL),
        (COUNTED.replace("i.name = 'ink'", "i.name = 'ink' AND COUNT(*) > 1"), NOT_NULL),
        (COUNTED.replace("sales AS s", "shop.sales AS s"), NOT_NULL),
        ("SELECT i.name FROM sales AS s LEFT JOIN items AS i ON s.item = i.item GROUP BY i.name HAVING SUM(CASE WHEN s.day = 1 THEN 1 ELSE 0 END) = 0", {"items": frozenset({"name"})}),  # the padded side
        ("SELECT buyer FROM sales AS s JOIN sales AS t ON s.id = t.id GROUP BY buyer HAVING SUM(CASE WHEN s.day = 1 THEN 1 ELSE 0 END) = 0", NOT_NULL),  # whose buyer
    ],
)
def test_other_selects_are_left_alone(sql, not_null):
    assert rewritten(sql, not_null) is None


def test_a_derived_table_is_read_as_a_source():
    inner = "(SELECT s.buyer AS b, i.name AS n FROM sales AS s LEFT JOIN items AS i ON s.item = i.item)"
    assert rewritten(DERIVED) == (
        f"SELECT DISTINCT d.b FROM {inner} AS d WHERE (d.n = 'pen') AND NOT EXISTS(SELECT 1 FROM {inner} AS kqc1_d WHERE (kqc1_d.n = 'ink') AND kqc1_d.b = d.b)"
    )
    # a bare name is the derived table's output, and a source name used inside the derived table is not renamed there
    bare = rewritten("SELECT b FROM (SELECT s.buyer AS b, s.item AS n FROM sales AS s) AS s GROUP BY b HAVING COUNT(CASE WHEN n = 1 THEN 1 END) = 0")
    assert bare == (
        "SELECT DISTINCT s.b FROM (SELECT s.buyer AS b, s.item AS n FROM sales AS s) AS s "
        "WHERE NOT EXISTS(SELECT 1 FROM (SELECT s.buyer AS b, s.item AS n FROM sales AS s) AS kqc0_s WHERE (n = 1) AND kqc0_s.b = s.b)"
    )
    # through two levels, and DISTINCT inside changes nothing about NULLs
    nested = "SELECT e.b FROM (SELECT DISTINCT d.b, d.n FROM (SELECT s.buyer AS b, s.item AS n FROM sales AS s WHERE s.day > 1) AS d) AS e GROUP BY e.b HAVING SUM(IF(e.n = 1, 1, 0)) = 0"
    assert rewritten(nested) is not None


def test_a_bare_grouping_column_needs_the_schema_to_name_its_table():
    bare = COUNTED.replace("s.buyer", "buyer")
    assert rewritten(bare) is not None
    assert rewritten(bare, schema=None) is None
    assert rewritten(bare, schema={"sales": SCHEMA["sales"]}) is None  # items may have a buyer column too


def test_a_floating_point_grouping_column_is_left_alone():
    assert rewritten(COUNTED, types={"sales": {"buyer": "DOUBLE"}}) is None
    assert rewritten(COUNTED, types={"sales": {"buyer": "INT"}}) == READ


SPELLINGS = [
    COUNTED,
    f"{HEAD} HAVING SUM(CASE WHEN i.name = 'ink' THEN 1 ELSE 0 END) = 0",
    f"{HEAD.replace(' GROUP BY', ' WHERE s.day > 1 OR s.day IS NULL GROUP BY')} HAVING COUNT(CASE WHEN i.name = 'pen' THEN 1 END) > 0 AND MAX(CASE WHEN s.day = 2 OR i.name = 'ink' THEN 1 ELSE 0 END) = 1",
    "SELECT s.buyer FROM sales AS s LEFT JOIN items AS i ON s.item = i.item GROUP BY s.buyer HAVING SUM(CASE WHEN i.name = 'pen' THEN 1 ELSE 0 END) = 0 AND SUM(CASE WHEN i.name IS NULL THEN 1 ELSE 0 END) >= 1",
    "SELECT s.day, s.buyer FROM sales AS s GROUP BY s.buyer, s.day HAVING SUM(CASE WHEN s.item = 1 THEN 1 ELSE 0 END) = 0 AND SUM(CASE WHEN s.item = 2 THEN 1 ELSE 0 END) > 0 AND COUNT(CASE WHEN s.item > 2 THEN 1 END) < 1",
    DERIVED,
    "SELECT b FROM (SELECT s.buyer AS b, s.item AS n FROM sales AS s) AS s GROUP BY b HAVING COUNT(CASE WHEN n = 1 THEN 1 END) = 0 AND MAX(IF(n > 2, 1, 0)) = 1",
]


@pytest.mark.parametrize("seed", range(3))
def test_the_reading_returns_the_rows_duckdb_returns_when_the_columns_are_not_null(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    differs_with_nulls = 0
    for sql in SPELLINGS:
        facts = {"sales": frozenset({"buyer", "day"})} if "s.day, s.buyer" in sql else NOT_NULL
        read = rewritten(sql, facts)
        assert read is not None, sql
        for filled in (True, True, True, False, False):
            con = duckdb.connect()
            con.execute("CREATE TABLE sales (id INT, buyer INT, item INT, day INT)")
            con.execute("CREATE TABLE items (item INT, name VARCHAR)")
            for row in range(rng.randint(0, 10)):
                buyer = rng.choice([1, 2, 3] if filled else [None, None, 1, 2])
                day = rng.choice([1, 2, 3]) if filled or "s.day, s.buyer" not in sql else rng.choice([None, 1, 2])
                con.execute("INSERT INTO sales VALUES (?, ?, ?, ?)", [row, buyer, rng.choice([None, 1, 2, 3, 4]), day if "s.day, s.buyer" in sql else rng.choice([None, 1, 2, 3])])
            for item in range(1, 5):
                for _ in range(rng.choice([0, 1, 1, 2])):
                    con.execute("INSERT INTO items VALUES (?, ?)", [item, rng.choice([None, "pen", "ink", "cap"])])
            same = sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(read).fetchall(), key=repr)
            if filled:
                assert same, sql
            else:
                differs_with_nulls += not same
    assert differs_with_nulls > 0  # NOT NULL is what makes it true


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

FILTERED = "SELECT DISTINCT s.buyer FROM items AS i JOIN sales AS s ON s.item = i.item WHERE i.name = 'pen' AND s.buyer NOT IN (SELECT t.buyer FROM items AS j JOIN sales AS t ON t.item = j.item WHERE j.name = 'ink')"
FACTS = {"sales": TableConstraints(not_null=frozenset({"buyer"}))}
TYPES = {"items": {"name": "VARCHAR", "item": "INT"}, "sales": {"buyer": "INT", "item": "INT", "day": "INT", "id": "INT"}}


def prove(left, right, **options):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="mysql", compare_names=False, **options)


def test_the_counted_and_the_filtered_spelling_are_one_query_when_the_buyer_is_not_null():
    assert prove(COUNTED, FILTERED, constraints=FACTS).proven
    assert not prove(COUNTED, FILTERED).proven  # a NULL buyer of ink empties the NOT IN and no group
    assert not prove(COUNTED.replace("= 0", "= 1"), FILTERED, constraints=FACTS).proven
    assert not prove(COUNTED.replace("'ink'", "'cap'"), FILTERED, constraints=FACTS).proven
    assert not prove(COUNTED, FILTERED.replace("NOT IN", "IN"), constraints=FACTS).proven


def test_the_conditional_verdict_names_the_column():
    # in the slow tier (tests/order.json): the search runs the prover several times
    result = prove(COUNTED, FILTERED, conditional=True)
    assert result.status.value == "proven_conditionally"
    assert sorted((c.kind, tuple(c.columns)) for c in result.conditions) == [("not_null", ("buyer",))]



def test_the_rule_is_one_of_the_narrow_retry_rules():
    """Rewriting the SUM(CASE ..) > 0 side took a conditional LeetCode pair (a MIN/MAX date range against counts of
    in-range and out-of-range rows) out of the shape main proves; a pair left unproven is tried again with the rule off."""

    from kumo_sql_prover import shape_fallback

    with shape_fallback.tracking() as fired:
        assert rewritten(COUNTED) == READ
    assert fired == {"counted_presence"}
    with shape_fallback.without("counted_presence"):
        assert rewritten(COUNTED) is None
    with shape_fallback.tracking() as fired:
        assert rewritten("SELECT s.buyer FROM sales AS s GROUP BY s.buyer") is None
    assert not fired
