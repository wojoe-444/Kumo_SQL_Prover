"""The directed witness search of the QUITE harness (tools/witness_search.py), on made-up tables.

Each test gives the search two queries and checks either that it finds a database the harness accepts
(different bags, legal rows) or that it finds none: for an equivalent pair, for a difference that is only a
division by zero, and for one that is only tie-breaking under a LIMIT.
"""

import datetime
import importlib.util
import random
import sys
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

# The file loads the QUITE bench script, so it is selected with the evals (pytest -m eval).
pytestmark = pytest.mark.eval

_TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load(name: str, registered: str):
    spec = importlib.util.spec_from_file_location(registered, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[registered] = module
    spec.loader.exec_module(module)
    return module


bench = _load("quite_bench", "quite_bench_witness_tests")
search = _load("witness_search", "witness_search")

DDL = """
CREATE TABLE shop (shop_id integer NOT NULL, region char(2), tier varchar(10), PRIMARY KEY (shop_id));
CREATE TABLE buyer (buyer_id integer NOT NULL, status char(1), level varchar(12), born integer, name varchar(8), PRIMARY KEY (buyer_id));
CREATE TABLE sale (
    sale_id integer NOT NULL, line integer NOT NULL, shop_id integer, buyer_id integer, qty integer,
    amount decimal(7,2), paid decimal(7,2), day date, reason integer,
    PRIMARY KEY (sale_id, line), FOREIGN KEY (shop_id) REFERENCES shop (shop_id)
);
"""
SECONDS = 20.0  # an upper bound: a search returns as soon as it has a witness


@pytest.fixture
def tables():
    return bench.load_schema(DDL)


def find(tables, left, right, seconds=SECONDS):
    left, right = (bench.adapt(sql)[0] for sql in (left, right))
    kinds = {bench.determinism(search.sqlglot.parse_one(sql, read="postgres")) for sql in (left, right)}
    kind = "random" if "random" in kinds else "ties" if "ties" in kinds else "fixed"
    replayer = bench.Replayer(tables)
    try:
        return search.find_witness(left, right, tables, replayer, kind, seconds, bench)
    finally:
        replayer.close()


def check(tables, witness):
    """The witness is a database of the schema: keys, NOT NULL, foreign keys, lengths and precisions."""

    data = {name: [tuple(row) for row in rows] for name, rows in witness["tables"].items()}
    repaired = {name: rows for name, rows in bench.repair(data, tables).items() if rows}
    assert bench.legal(data, tables) and repaired == data  # nothing to repair
    for name, rows in data.items():
        for row in rows:
            for value, pg_type in zip(row, tables[name].columns.values()):
                assert value is None or search.fit(value, pg_type) == value, (name, value, pg_type)
    assert witness["left"] != witness["right"]
    return data


JOINED = "FROM sale a, shop s, buyer b WHERE a.shop_id = s.shop_id AND a.buyer_id = b.buyer_id"
EITHER = "((b.status = 'M' AND b.level = 'Unknown') OR (b.status = 'W' AND b.level = 'Advanced'))"


def test_a_dropped_grouping_key(tables):
    left = f"SELECT s.region, SUM(a.amount) {JOINED} AND {EITHER} AND s.tier LIKE 'gold%' GROUP BY s.region, b.status, b.level"
    right = f"SELECT s.region, SUM(a.amount) {JOINED} AND {EITHER} AND s.tier LIKE 'gold%' GROUP BY s.region"
    witness, label = find(tables, left, right)
    assert witness and label.startswith("directed: ")
    data = check(tables, witness)
    assert len({(row[1], row[2]) for row in data["buyer"]}) >= 2  # two buyers that differ in the dropped keys


def test_a_dropped_list_member_and_a_moved_bound(tables):
    left = f"SELECT a.sale_id {JOINED} AND b.status IN ('M', 'S', 'U') AND b.born BETWEEN 1950 AND 1956"
    witness, _ = find(tables, left, left.replace("'M', 'S', 'U'", "'M', 'S'"))
    assert "U" in {row[1] for row in check(tables, witness)["buyer"]}
    witness, _ = find(tables, left, left.replace("BETWEEN 1950 AND 1956", "BETWEEN 1950 AND 1955"))
    assert 1956 in {row[3] for row in check(tables, witness)["buyer"]}
    witness, _ = find(tables, left, left.replace("b.born BETWEEN 1950 AND 1956", "b.born > 1950 AND b.born <= 1956"))
    assert 1950 in {row[3] for row in check(tables, witness)["buyer"]}


def test_a_dropped_coalesce(tables):
    left = "SELECT b.buyer_id, COALESCE(b.name, '?') || '/' || COALESCE(b.level, '?') FROM buyer b WHERE b.status = 'M'"
    right = "SELECT b.buyer_id, b.name || '/' || b.level FROM buyer b WHERE b.status = 'M'"
    witness, _ = find(tables, left, right)
    assert any(row[2] is None or row[4] is None for row in check(tables, witness)["buyer"])


def test_a_ratio_rewritten_as_a_product_needs_a_negative_divisor_never_zero(tables):
    left = "SELECT a.sale_id FROM sale a WHERE a.amount / a.qty BETWEEN 236 AND 295"
    right = "SELECT a.sale_id FROM sale a WHERE a.amount BETWEEN 236 * a.qty AND 295 * a.qty"
    witness, _ = find(tables, left, right)
    rows = check(tables, witness)["sale"]
    assert all(row[4] != 0 for row in rows) and any(row[4] is not None and row[4] < 0 for row in rows)


def test_a_bounded_divisor_holds_a_power_of_two_inside_its_bounds(tables):
    duckdb = pytest.importorskip("duckdb")
    left = "SELECT a.sale_id FROM sale a WHERE a.amount / a.paid BETWEEN 2 AND 5 AND a.paid BETWEEN 236 AND 265"
    plan = search.read_queries(left, left, tables, duckdb.connect())
    paid, amount = plan.columns[("sale", "paid")], plan.columns[("sale", "amount")]
    assert paid.usual == [Decimal("256.00")] and Decimal("0") not in paid.usual + paid.rare
    assert Decimal("896.00") in amount.usual  # 3.5 * 256: the ratio lands inside the bounds
    # with nothing holding the divisor positive the pair differs, on a negative divisor
    right = "SELECT a.sale_id FROM sale a WHERE a.amount BETWEEN 2 * a.paid AND 5 * a.paid"
    witness, _ = find(tables, "SELECT a.sale_id FROM sale a WHERE a.amount / a.paid BETWEEN 2 AND 5", right)
    assert any(row[6] is not None and row[6] < 0 for row in check(tables, witness)["sale"])


def test_a_filter_pushed_into_the_table_an_average_reads(tables):
    totals = "SELECT buyer_id, shop_id, reason, SUM(paid) AS total FROM sale {} GROUP BY buyer_id, shop_id, reason"
    outer = (
        "WITH t AS ({}) SELECT t1.buyer_id FROM t t1, shop s WHERE t1.total > (SELECT AVG(t2.total) * 1.2 FROM t t2 "
        "WHERE t2.shop_id = t1.shop_id) AND s.shop_id = t1.shop_id AND s.region = 'NC' {}"
    )
    left = outer.format(totals.format(""), "AND t1.reason BETWEEN 28 AND 31")
    right = outer.format(totals.format("WHERE reason BETWEEN 28 AND 31"), "")
    witness, _ = find(tables, left, right)
    reasons = {row[8] for row in check(tables, witness)["sale"]}
    assert any(r is None or not 28 <= r <= 31 for r in reasons) and any(r is not None and 28 <= r <= 31 for r in reasons)


def test_membership_replaced_by_a_join_with_groups_that_repeat(tables):
    groups = "SELECT shop_id, day FROM sale GROUP BY shop_id, day HAVING COUNT(*) > 2"
    left = f"SELECT s.shop_id, s.region FROM shop s WHERE s.shop_id IN (SELECT g.shop_id FROM ({groups}) g)"
    right = f"SELECT s.shop_id, s.region FROM shop s JOIN ({groups}) g ON g.shop_id = s.shop_id"
    witness, _ = find(tables, left, right)
    assert len(check(tables, witness)["sale"]) >= 6  # two groups of three in one shop


EQUIVALENT = [
    (f"SELECT s.region, SUM(a.amount) {JOINED} AND b.status = 'M' GROUP BY s.region",
     "SELECT s.region, SUM(a.amount) FROM sale a JOIN shop s ON a.shop_id = s.shop_id JOIN buyer b ON a.buyer_id = b.buyer_id "
     "WHERE b.status = 'M' GROUP BY s.region"),
    ("SELECT a.sale_id FROM sale a WHERE a.qty BETWEEN 2 AND 5", "SELECT a.sale_id FROM sale a WHERE a.qty >= 2 AND a.qty <= 5"),
    ("SELECT s.shop_id FROM shop s WHERE s.shop_id IN (SELECT shop_id FROM sale WHERE qty > 1)",
     "SELECT s.shop_id FROM shop s WHERE EXISTS (SELECT 1 FROM sale a WHERE a.shop_id = s.shop_id AND a.qty > 1)"),
    # equal wherever PostgreSQL returns rows at all: the only difference is a division by zero
    ("SELECT a.sale_id, a.amount / a.qty FROM sale a", "SELECT a.sale_id, CASE WHEN a.qty = 0 THEN 0 ELSE a.amount / a.qty END FROM sale a"),
    ("SELECT a.sale_id, a.amount / (a.qty - 1) FROM sale a", "SELECT a.sale_id, CASE WHEN a.qty = 1 THEN 7 ELSE a.amount / (a.qty - 1) END FROM sale a"),
    ("SELECT a.sale_id, a.line % a.qty FROM sale a", "SELECT a.sale_id, CASE WHEN a.qty = 0 THEN NULL ELSE a.line % a.qty END FROM sale a"),
    # a ratio rewritten as a product is the same when the divisor is held positive
    ("SELECT a.sale_id FROM sale a WHERE a.amount / a.paid BETWEEN 2 AND 5 AND a.paid BETWEEN 236 AND 265",
     "SELECT a.sale_id FROM sale a WHERE a.amount BETWEEN 2 * a.paid AND 5 * a.paid AND a.paid BETWEEN 236 AND 265"),
    # the same rows unless the LIMIT has to choose among ties
    ("SELECT a.sale_id FROM sale a ORDER BY a.amount LIMIT 1", "SELECT a.sale_id FROM sale a ORDER BY a.amount, a.sale_id DESC LIMIT 1"),
    ("SELECT a.buyer_id, a.amount FROM sale a ORDER BY a.amount LIMIT 2", "SELECT a.buyer_id, a.amount FROM sale a ORDER BY a.amount, a.day LIMIT 2"),
]


@pytest.mark.parametrize("left, right", EQUIVALENT)
def test_no_witness(tables, left, right):
    assert find(tables, left, right, seconds=0.7) == (None, "")


def test_a_real_difference_under_a_limit_that_cuts_nothing(tables):
    left = "SELECT a.buyer_id FROM sale a WHERE a.qty > 1 ORDER BY a.amount LIMIT 5"
    witness, _ = find(tables, left, left.replace("a.qty > 1", "a.qty >= 1"))
    assert len(check(tables, witness)["sale"]) <= 5


def test_pairs_that_are_not_searched(tables):
    numbered = "SELECT sale_id, ROW_NUMBER() OVER (ORDER BY amount) FROM sale WHERE qty {} 1"
    assert find(tables, numbered.format(">"), numbered.format(">="), seconds=0.7) == (None, "")
    nested = "SELECT COUNT(*) FROM (SELECT sale_id FROM sale WHERE qty {} 1 ORDER BY amount LIMIT 1) t"
    assert find(tables, nested.format(">"), nested.format(">="), seconds=0.7) == (None, "")
    assert find(tables, "SELECT sale_id FROM sale WHERE RANDOM() < 2", "SELECT sale_id FROM sale WHERE qty > 1", seconds=0.7) == (None, "")
    offset = "SELECT sale_id FROM sale WHERE qty {} 1 ORDER BY amount LIMIT 5 OFFSET 1"
    assert find(tables, offset.format(">"), offset.format(">="), seconds=0.7) == (None, "")


def test_a_zero_divisor_raises_in_the_replayed_text():
    duckdb = pytest.importorskip("duckdb")
    db = duckdb.connect()
    db.execute("CREATE TABLE t (x BIGINT, y BIGINT)")
    db.execute("INSERT INTO t VALUES (6, 3), (5, 0)")
    guarded = search.guard_divisions("SELECT x / y, x % y, x / 2, x // (y + 1) FROM t WHERE y <> 0")
    assert db.execute(guarded).fetchall() == [(2.0, 0, 3.0, 1)]
    with pytest.raises(duckdb.Error, match="division by zero"):
        db.execute(search.guard_divisions("SELECT x / y FROM t")).fetchall()
    with pytest.raises(duckdb.Error, match="division by zero"):
        db.execute(search.guard_divisions("SELECT SUM(x) / (SUM(y) - 3) FROM t")).fetchall()
    assert "error" not in search.guard_divisions("SELECT x / 2, x * y FROM t").lower()


def test_the_outermost_limit_alone():
    text, count = search.outer_limits("SELECT a FROM t ORDER BY b LIMIT 3", bench.determinism)
    assert count == 3 and "LIMIT" not in text
    assert search.outer_limits("SELECT a FROM t", bench.determinism) == ("SELECT a FROM t", None)
    assert search.outer_limits("SELECT a FROM (SELECT a FROM t ORDER BY b LIMIT 3) AS s LIMIT 2", bench.determinism) is None
    assert search.outer_limits("SELECT a, ROW_NUMBER() OVER (ORDER BY b) FROM t LIMIT 2", bench.determinism) is None
    assert search.outer_limits("SELECT a FROM t ORDER BY b LIMIT 3 OFFSET 1", bench.determinism) is None


def test_values_fit_the_column():
    assert search.fit(Decimal("0.57"), "DECIMAL(7, 2)") == Decimal("0.57")
    assert search.fit(Decimal("0.575"), "DECIMAL(7, 2)") is None  # more digits than the column keeps
    assert search.fit(123456, "DECIMAL(7, 2)") is None
    assert search.fit(Decimal("3.0"), "INT") == 3 and search.fit(Decimal("3.5"), "INT") is None
    assert search.fit("Advanced Degree", "CHAR(20)") == "Advanced Degree" and search.fit("Advanced Degree", "CHAR(10)") is None
    assert search.fit("padded ", "VARCHAR(20)") is None
    assert search.fit(datetime.datetime(1999, 3, 2), "DATE") == datetime.date(1999, 3, 2)
    assert search.fit(datetime.datetime(1999, 3, 2, 5), "DATE") is None
    assert search.fit("1999-03-02", "DATE") == datetime.date(1999, 3, 2)
    assert search.fit(True, "INT") is None and search.fit(None, "INT") is None


def test_what_the_queries_say_becomes_the_columns_values(tables):
    duckdb = pytest.importorskip("duckdb")
    left = (
        "SELECT a.sale_id FROM sale a, shop s WHERE a.shop_id = s.shop_id AND s.region IN ('NC', 'MI') AND a.reason > 3 "
        "AND a.day BETWEEN CAST('1999-02-01' AS DATE) AND CAST('1999-02-01' AS DATE) + INTERVAL '30 day' AND a.amount / a.qty >= 57 * 0.01 "
        "GROUP BY a.sale_id HAVING COUNT(*) > 4"
    )
    plan = search.read_queries(left, left, tables, duckdb.connect())
    column = lambda table, name: plan.columns[(table, name)]  # noqa: E731
    assert column("shop", "region").usual == ["NC", "MI"]
    assert column("sale", "reason").usual == [4] and 3 in column("sale", "reason").rare
    assert column("sale", "day").usual[0] == datetime.date(1999, 2, 1) and datetime.date(1999, 3, 3) in column("sale", "day").usual
    assert datetime.date(1999, 1, 31) in column("sale", "day").rare
    assert column("sale", "qty").divisor and column("sale", "qty").usual == [1, 2, -1, -2] and not column("sale", "qty").rare
    assert Decimal("0.57") in column("sale", "amount").usual and Decimal("-0.57") in column("sale", "amount").usual
    assert plan.group == 4 and plan.anchors[("sale", "shop_id")] == ("shop", "shop_id")
    assert not column("buyer", "status").read if ("buyer", "status") in plan.columns else True
    assert plan.order.index("shop") < plan.order.index("sale")


def test_which_columns_share_values_and_which_do_not(tables):
    duckdb = pytest.importorskip("duckdb")
    left = (
        "WITH t AS (SELECT a.shop_id, a.buyer_id AS who, SUM(a.amount) AS total, SUM(a.paid) / COUNT(*) AS mean FROM sale a GROUP BY a.shop_id, a.buyer_id) "
        "SELECT t.who, t.total / t.mean FROM t, buyer b, sale c WHERE t.who = b.buyer_id AND t.total > t.mean AND c.line = c.sale_id - 52 "
        "AND b.born IN (SELECT reason FROM sale)"
    )
    plan = search.read_queries(left, left, tables, duckdb.connect())
    # a renamed column joins like the column; a name for a sum does not tie its columns to anything
    assert plan.anchors[("sale", "buyer_id")] == ("buyer", "buyer_id")
    assert ("sale", "amount") not in plan.shared and ("sale", "paid") not in plan.shared
    # a divisor that is a name for an expression is checked when replayed, not held to powers of two
    assert not plan.columns[("sale", "paid")].divisor and not plan.columns[("sale", "amount")].divisor
    # ``line = sale_id - 52``: values 52 apart, in one shared set
    assert plan.shared[("sale", "line")] == plan.shared[("sale", "sale_id")]
    assert {1, 53} <= set(plan.columns[("sale", "line")].usual)
    assert plan.shared[("buyer", "born")] == plan.shared[("sale", "reason")]  # IN (subquery)


def test_a_value_a_scalar_subquery_filters_on_is_in_one_row_only(tables):
    duckdb = pytest.importorskip("duckdb")
    left = "SELECT a.sale_id FROM sale a WHERE a.reason = (SELECT c.reason FROM sale c WHERE c.day = CAST('1999-02-11' AS DATE))"
    plan = search.read_queries(left, left, tables, duckdb.connect())
    day = datetime.date(1999, 2, 11)
    assert plan.once == {("sale", "day"): [day]}
    position = list(tables["sale"].columns).index("day")
    rng = random.Random(3)
    held = 0
    for profile in range(40):
        _, data = search.database(plan, rng, profile)
        days = [row[position] for row in data["sale"]]
        assert days.count(day) <= 1, days
        held += days.count(day)
    assert held > 20  # and it is there most of the time


def test_sums_compared_within_a_factor_hold_values_that_close(tables):
    duckdb = pytest.importorskip("duckdb")
    sums = (
        "WITH x AS (SELECT shop_id, SUM(amount) AS total FROM sale GROUP BY shop_id), y AS (SELECT shop_id, SUM(paid) AS total FROM sale GROUP BY shop_id) "
        "SELECT x.shop_id FROM x, y WHERE x.shop_id = y.shop_id"
    )
    left = f"{sums} AND x.total BETWEEN 0.9 * y.total AND 1.1 * y.total AND y.total BETWEEN 0.9 * x.total AND 1.1 * x.total"
    right = f"{sums} AND x.total BETWEEN 0.9 * y.total AND 1.1 * y.total"
    plan = search.read_queries(left, right, tables, duckdb.connect())
    for name in ("amount", "paid"):
        assert plan.columns[("sale", name)].usual == [Decimal("10.00"), Decimal("9.00"), Decimal("11.00")]
    witness, label = find(tables, left, right)
    assert witness is not None and label.startswith("directed")
    check(tables, witness)


def test_every_drawn_database_is_legal(tables):
    duckdb = pytest.importorskip("duckdb")
    left = f"SELECT s.region, SUM(a.amount) {JOINED} AND {EITHER} AND a.amount / a.qty > 2 GROUP BY s.region HAVING COUNT(*) > 3"
    plan = search.read_queries(left, left, tables, duckdb.connect())
    rng = random.Random(3)
    labels = set()
    for index in range(70):
        label, data = search.database(plan, rng, index)
        labels.add(label)
        assert bench.legal(data, tables), (label, data)
        assert bench.repair(data, tables) == data  # nothing to repair: legal as drawn
        assert all(row[4] in (None, 1, 2, -1, -2) for row in data["sale"])  # a divisor is never zero
        for name, rows in data.items():
            for row in rows:
                for value, pg_type in zip(row, tables[name].columns.values()):
                    assert value is None or search.fit(value, pg_type) == value
    assert len(labels) == 7


def test_the_harness_search_runs_it_last(tables, monkeypatch):
    monkeypatch.setenv("KUMOSQL_TARGETED", "0")  # only the directed search is left
    left = bench.adapt(f"SELECT a.sale_id {JOINED} AND b.status IN ('M', 'S', 'U')")[0]
    right = left.replace("'M', 'S', 'U'", "'M', 'S'")
    replayer = bench.Replayer(tables)
    try:
        assert bench.search(left, right, tables, replayer, "fixed", [], None) == (None, "")  # no budget, no search
        witness, source = bench.search(left, right, tables, replayer, "fixed", [], None, 10.0)
        assert witness and source.startswith("directed: ")
        monkeypatch.setenv("KUMOSQL_DIRECTED", "0")
        assert bench.search(left, right, tables, replayer, "fixed", [], None, 10.0) == (None, "")
    finally:
        replayer.close()
