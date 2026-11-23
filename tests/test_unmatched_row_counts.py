"""Rows without a match, counted with a flag over a left join, are read as a count over the padded rows
(``unmatched_row_counts``). Every reading is run on SQLite over made-up tables with NULLs."""

import random
import sqlite3

import pytest
from sqlglot import parse_one

from kumo_sql_prover.unmatched_row_counts import NAME, read_unmatched_row_counts

SCHEMA = {"visits": ["visit", "customer", "site"], "purchases": ["id", "visit", "amount"], "refunds": ["id", "visit"]}
TYPES = {
    "visits": {"visit": "INT", "customer": "INT", "site": "VARCHAR(20)"},
    "purchases": {"id": "INT", "visit": "INT", "amount": "INT"},
    "refunds": {"id": "INT", "visit": "INT"},
}
NOT_NULL = {"visits": ["visit"], "purchases": ["id"], "refunds": ["id"]}
JOIN = "FROM visits AS v LEFT JOIN purchases AS p ON p.visit = v.visit"
FLAG = "SUM(CASE WHEN p.id IS NULL THEN 1 ELSE 0 END)"
PLAIN = f"SELECT v.customer, {FLAG} AS n {JOIN} GROUP BY v.customer HAVING n > 0"
COUNTED = f"SELECT v.customer, COUNT(*) AS n {JOIN} WHERE p.id IS NULL GROUP BY v.customer"


def read(sql: str, dialect: str = "mysql", schema=SCHEMA, not_null=NOT_NULL, types=TYPES) -> str | None:
    tree = parse_one(sql, read=dialect)
    text = tree.sql(dialect=dialect)
    new, wrote = read_unmatched_row_counts(tree, schema, not_null, types, dialect)
    assert tree.sql(dialect=dialect) == text  # the tree given is not changed
    if not wrote:
        assert new is tree
        return None
    assert read_unmatched_row_counts(new, schema, not_null, types, dialect)[1] is False  # once is enough
    again = parse_one(new.sql(dialect=dialect), read=dialect)
    assert read_unmatched_row_counts(again, schema, not_null, types, dialect)[1] is False  # also read from its text
    return new.sql(dialect=dialect)


def test_the_name():
    assert NAME == "unmatched_row_counts"


def test_a_flag_sum_is_read_as_a_count_over_the_padded_rows():
    assert read(PLAIN) == COUNTED
    # a WHERE clause stays, and the test is added to it
    assert read(PLAIN.replace(" GROUP BY", " WHERE v.visit > 2 AND v.site = 'a' GROUP BY")) == COUNTED.replace(
        "WHERE p.id IS NULL", "WHERE v.visit > 2 AND v.site = 'a' AND p.id IS NULL"
    )
    # more than one flag, and a flag without a name
    assert read(f"SELECT v.customer, {FLAG} AS n, COUNT(*) - COUNT(p.id) AS m {JOIN} GROUP BY v.customer HAVING m <> 0") == (
        f"SELECT v.customer, COUNT(*) AS n, COUNT(*) AS m {JOIN} WHERE p.id IS NULL GROUP BY v.customer"
    )
    assert read(f"SELECT v.customer, {FLAG} {JOIN} GROUP BY v.customer HAVING {FLAG} > 0") == (
        f"SELECT v.customer, COUNT(*) {JOIN} WHERE p.id IS NULL GROUP BY v.customer"
    )
    # names without a table in front, where the declared columns settle whose they are
    assert read("SELECT customer, SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS n FROM visits LEFT JOIN purchases ON purchases.visit = visits.visit GROUP BY customer HAVING n > 0") == (
        "SELECT customer, COUNT(*) AS n FROM visits LEFT JOIN purchases ON purchases.visit = visits.visit WHERE purchases.id IS NULL GROUP BY customer"
    )
    # LEFT OUTER JOIN, and a column the ON compares with = (not NULL on a matched pair)
    assert read(PLAIN.replace("LEFT JOIN", "LEFT OUTER JOIN").replace("p.id IS NULL", "p.visit IS NULL")) == COUNTED.replace("LEFT JOIN", "LEFT OUTER JOIN").replace(
        "p.id IS NULL", "p.visit IS NULL"
    )


FLAGS = {
    "sum of a case": FLAG,
    "sum of an if": "SUM(IF(p.id IS NULL, 1, 0))",
    "sum of a test": "SUM(p.id IS NULL)",
    "sum of a test on the compared column": "SUM(ISNULL(p.visit))",
    "count of a case": "COUNT(CASE WHEN p.visit IS NULL THEN 1 END)",
    "count of a case with else null": "COUNT(CASE WHEN p.id IS NULL THEN 'x' ELSE NULL END)",
    "count of an if": "COUNT(IF(p.id IS NULL, 7, NULL))",
    "all less the matched": "COUNT(*) - COUNT(p.id)",
    "ones less the matched": "COUNT(1) - COUNT(p.visit)",
    "a left column less the matched": "(COUNT(v.visit) - COUNT(p.id))",
    "all less a sum of a case": "COUNT(v.visit) - SUM(CASE WHEN p.id IS NOT NULL THEN 1 ELSE 0 END)",
    "all less a sum of a test": "COUNT(*) - SUM(p.id IS NOT NULL)",
}
FRAMES = {
    "by name": "SELECT v.customer, {flag} AS n {join} GROUP BY v.customer HAVING n > 0",
    "written again": "SELECT v.customer, {flag} AS n {join} GROUP BY v.customer HAVING {flag} <> 0",
    "at least one": "SELECT v.customer, {flag} AS n {join} GROUP BY v.customer HAVING n >= 1",
    "the other way round": "SELECT v.customer, {flag} AS n {join} GROUP BY v.customer HAVING 0 < n",
    "one at most": "SELECT {flag} AS n, v.customer {join} GROUP BY v.customer HAVING 1 <= {flag}",
    "with a filter": "SELECT v.customer, {flag} AS n {join} WHERE v.visit > 2 GROUP BY v.customer HAVING n > 0",
    "with a filter on the joined table": "SELECT v.customer, {flag} AS n {join} WHERE p.amount IS NULL OR p.amount > 1 GROUP BY v.customer HAVING n > 0",
    "by position": "SELECT v.customer, {flag} AS n {join} GROUP BY 1 HAVING n > 0",
    "by a name of the select list": "SELECT v.customer AS who, {flag} AS n {join} GROUP BY who HAVING n > 0",
    "two keys": "SELECT v.customer, v.visit, {flag} AS n {join} GROUP BY v.customer, v.visit HAVING n > 0",
    "a key that is not selected": "SELECT v.customer, {flag} AS n {join} GROUP BY v.site, v.customer HAVING n > 0",
    "a key of the joined table": "SELECT p.id, {flag} AS n {join} GROUP BY p.id HAVING n > 0",
    "more in the ON": "SELECT v.customer, {flag} AS n FROM visits AS v LEFT JOIN purchases AS p ON p.visit = v.visit AND p.amount > 1 GROUP BY v.customer HAVING n > 0",
    "tested outside": "SELECT d.customer, d.n FROM (SELECT v.customer, {flag} AS n {join} GROUP BY v.customer) AS d WHERE d.n > 0",
    "tested outside with more": "SELECT * FROM (SELECT v.customer, {flag} AS n {join} GROUP BY v.customer) AS d WHERE n <> 0 AND customer > 1",
    "tested outside and grouped again": "SELECT n, COUNT(*) FROM (SELECT v.customer, {flag} AS n {join} GROUP BY v.customer) AS d WHERE n >= 1 GROUP BY n",
}


def _tables(rng: random.Random) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE visits (visit INT NOT NULL, customer INT, site TEXT)")
    con.execute("CREATE TABLE purchases (id INT NOT NULL, visit INT, amount INT)")
    for visit in rng.sample(range(1, 9), rng.randint(0, 6)):
        for _ in range(rng.choice((1, 1, 1, 2))):  # a visit number is no key
            con.execute("INSERT INTO visits VALUES (?, ?, ?)", (visit, rng.choice((1, 2, 3, None)), rng.choice(("a", "b"))))
    for number in range(rng.randint(0, 7)):
        con.execute("INSERT INTO purchases VALUES (?, ?, ?)", (number, rng.choice((1, 2, 3, 4, 5, 9, None)), rng.choice((0, 1, 2, 3, None))))
    return con


def _rows(con: sqlite3.Connection, sql: str) -> list:
    return sorted(con.execute(parse_one(sql, read="mysql").sql(dialect="sqlite")).fetchall(), key=repr)


@pytest.mark.parametrize("flag", sorted(FLAGS))
@pytest.mark.parametrize("frame", sorted(FRAMES))
def test_the_reading_returns_the_same_rows(flag, frame):
    sql = FRAMES[frame].format(flag=FLAGS[flag], join=JOIN)
    new = read(sql)
    assert new is not None and "COUNT(*)" in new and "HAVING" not in new and " IS NULL GROUP BY" in new
    rng = random.Random(f"{flag} / {frame}")
    seen = 0
    for _ in range(40):
        con = _tables(rng)
        rows = _rows(con, sql)
        assert rows == _rows(con, new)
        seen += len(rows)
        con.close()
    assert seen or "joined table" in frame  # the tables do hold visits without a purchase


def test_a_test_is_a_number_only_in_some_dialects():
    lax = PLAIN.replace(FLAG, "SUM(p.id IS NULL)")
    assert read(lax, "mysql") is not None and read(lax, "sqlite") is not None
    assert read(lax, "postgres") is None and read(lax, "duckdb") is None
    assert read(PLAIN.replace("HAVING n > 0", f"HAVING {FLAG} > 0"), "postgres") is not None


LEFT_ALONE = {
    "no GROUP BY": f"SELECT {FLAG} AS n {JOIN} HAVING n > 0",
    "no HAVING": f"SELECT v.customer, {FLAG} AS n {JOIN} GROUP BY v.customer",
    "more than one row asked for": PLAIN.replace("n > 0", "n > 1"),
    "none asked for": PLAIN.replace("n > 0", "n >= 0"),
    "a second test": PLAIN.replace("n > 0", "n > 0 AND COUNT(*) > 1"),
    "a test of something else": PLAIN.replace("n > 0", "COUNT(*) > 0"),
    "an inner join": PLAIN.replace("LEFT JOIN", "JOIN"),
    "a right join": PLAIN.replace("LEFT JOIN", "RIGHT JOIN"),
    "a full join": PLAIN.replace("LEFT JOIN", "FULL JOIN"),
    "USING": PLAIN.replace("ON p.visit = v.visit", "USING (visit)"),
    "a third table": PLAIN.replace(" GROUP BY", " LEFT JOIN refunds AS r ON r.visit = v.visit GROUP BY"),
    "a derived table": PLAIN.replace("purchases AS p", "(SELECT * FROM purchases) AS p"),
    "a table that is not declared": PLAIN.replace("purchases", "orders"),
    "a table of another schema": PLAIN.replace("purchases", "old.purchases"),
    "one name for both": "SELECT visits.customer, COUNT(*) - COUNT(visits.visit) AS n FROM visits LEFT JOIN visits ON visits.visit = visits.visit GROUP BY visits.customer HAVING n > 0",
    "a column that can be NULL on a matched pair": PLAIN.replace("p.id IS NULL", "p.amount IS NULL"),
    "a column the ON compares under OR": f"SELECT v.customer, SUM(CASE WHEN p.visit IS NULL THEN 1 ELSE 0 END) AS n FROM visits AS v LEFT JOIN purchases AS p ON p.visit = v.visit OR p.amount = 1 GROUP BY v.customer HAVING n > 0",
    "a column the ON compares with <=>": f"SELECT v.customer, SUM(CASE WHEN p.visit IS NULL THEN 1 ELSE 0 END) AS n FROM visits AS v LEFT JOIN purchases AS p ON p.visit <=> v.visit GROUP BY v.customer HAVING n > 0",
    "a column of the left table": PLAIN.replace("p.id IS NULL", "v.customer IS NULL"),
    "a column both tables have": PLAIN.replace("p.id IS NULL", "visit IS NULL"),
    "the matched rows": PLAIN.replace("p.id IS NULL", "p.id IS NOT NULL"),
    "another value for the rest": PLAIN.replace("ELSE 0", "ELSE 2"),
    "another value for the padded": PLAIN.replace("THEN 1", "THEN 2"),
    "a sum without ELSE": PLAIN.replace(" ELSE 0", ""),
    "a count with ELSE": PLAIN.replace(FLAG, "COUNT(CASE WHEN p.id IS NULL THEN 1 ELSE 0 END)"),
    "a count of a column": PLAIN.replace(FLAG, "COUNT(CASE WHEN p.id IS NULL THEN v.customer END)"),
    "a distinct sum": PLAIN.replace("SUM(", "SUM(DISTINCT "),
    "a distinct count": PLAIN.replace(FLAG, "COUNT(DISTINCT CASE WHEN p.id IS NULL THEN 1 END)"),
    "a left column that can be NULL": PLAIN.replace(FLAG, "COUNT(v.customer) - COUNT(p.id)"),
    "distinct visits less the matched": PLAIN.replace(FLAG, "COUNT(DISTINCT v.visit) - COUNT(p.id)"),
    "all less a column that can be NULL": PLAIN.replace(FLAG, "COUNT(*) - COUNT(p.amount)"),
    "the matched less all": PLAIN.replace(FLAG, "COUNT(p.id) - COUNT(*)"),
    "another aggregate": PLAIN.replace(" AS n", " AS n, MAX(p.amount) AS most"),
    "another count": PLAIN.replace(" AS n", " AS n, COUNT(*) AS every"),
    "a flag with something added": PLAIN.replace(" AS n", " + 1 AS n"),
    "an item that is not grouped": PLAIN.replace("SELECT v.customer,", "SELECT v.customer, v.visit,"),
    "a text shown per group": PLAIN.replace("v.customer", "v.site"),
    "a computed value shown per group": PLAIN.replace("v.customer", "v.customer + 1"),
    "a name given twice": PLAIN.replace("v.customer,", "v.customer AS n,"),
    "the flag grouped by": PLAIN.replace("GROUP BY v.customer", "GROUP BY v.customer, n"),
    "a name a table has too": PLAIN.replace("AS n", "AS amount").replace("n > 0", "amount > 0"),
    "WITH ROLLUP": PLAIN.replace("GROUP BY v.customer", "GROUP BY v.customer WITH ROLLUP"),
    "DISTINCT": PLAIN.replace("SELECT", "SELECT DISTINCT"),
    "ORDER BY": PLAIN + " ORDER BY n",
    "LIMIT": PLAIN + " LIMIT 3",
    "a window": PLAIN.replace(" AS n", " AS n, ROW_NUMBER() OVER (ORDER BY v.customer) AS place"),
    "a subquery in the filter": PLAIN.replace(" GROUP BY", " WHERE v.visit IN (SELECT visit FROM refunds) GROUP BY"),
    "a star": PLAIN.replace("SELECT v.customer,", "SELECT v.*,"),
    "tested outside over a join": f"SELECT d.customer FROM (SELECT v.customer, {FLAG} AS n {JOIN} GROUP BY v.customer) AS d JOIN refunds AS r ON r.id = d.customer WHERE d.n > 0",
    "tested outside under OR": f"SELECT d.customer FROM (SELECT v.customer, {FLAG} AS n {JOIN} GROUP BY v.customer) AS d WHERE d.n > 0 OR d.customer = 1",
    "tested outside for more": f"SELECT d.customer FROM (SELECT v.customer, {FLAG} AS n {JOIN} GROUP BY v.customer) AS d WHERE d.n > 1",
    "tested outside without a name": f"SELECT d.customer FROM (SELECT v.customer, {FLAG} AS n, COUNT(*) - COUNT(p.id) {JOIN} GROUP BY v.customer) AS d WHERE d.n > 0",
    "tested outside under new names": f"SELECT d.a FROM (SELECT v.customer, {FLAG} AS n {JOIN} GROUP BY v.customer) AS d (a, n) WHERE d.n > 0",
    "tested outside on the grouping column": f"SELECT d.customer FROM (SELECT v.customer, {FLAG} AS n {JOIN} GROUP BY v.customer) AS d WHERE d.customer > 0",
}


@pytest.mark.parametrize("why", sorted(LEFT_ALONE))
def test_left_alone(why):
    assert read(LEFT_ALONE[why]) is None


def test_left_alone_inside_a_with_clause_and_without_declared_tables_or_types():
    assert read(f"WITH purchases AS (SELECT 1 AS id, 1 AS visit) {PLAIN}") is None
    assert read(f"WITH w AS ({PLAIN}) SELECT * FROM w") is None
    assert read(PLAIN, schema=None) is None and read(PLAIN, types=None) is None
    assert read(PLAIN, types={"visits": {"visit": "INT", "customer": "DECIMAL(5, 2)"}}) is None
    # without the NOT NULL declaration, the column the ON compares still reads
    assert read(PLAIN, not_null=None) is None
    assert read(PLAIN.replace("p.id IS NULL", "p.visit IS NULL"), not_null=None) is not None
    assert read(PLAIN.replace(FLAG, "COUNT(v.visit) - COUNT(p.visit)"), not_null=None) is None


def _demo(*purchases) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE visits (visit INT NOT NULL, customer INT, site TEXT)")
    con.execute("CREATE TABLE purchases (id INT NOT NULL, visit INT, amount INT)")
    con.executemany("INSERT INTO visits VALUES (?, ?, ?)", [(1, 1, "a"), (2, 1, "a"), (3, 2, "b")])
    con.executemany("INSERT INTO purchases VALUES (?, ?, ?)", list(purchases))
    return con


def test_why_the_refused_shapes_are_refused():
    # each by-hand reading of a refused shape, on a table where it returns other rows
    con = _demo((1, 1, None), (2, 3, 5))
    assert _rows(con, PLAIN) == _rows(con, COUNTED) == [(1, 1)]
    # a column that can be NULL on a matched pair: visit 1 has a purchase without an amount
    nullable = PLAIN.replace("p.id IS NULL", "p.amount IS NULL")
    assert _rows(con, nullable) == [(1, 2)] and _rows(con, COUNTED) == [(1, 1)]
    # without the HAVING, the customers whose visits all have a purchase are rows too
    assert _rows(con, PLAIN.replace(" HAVING n > 0", "")) == [(1, 1), (2, 0)]
    # another aggregate reads every row of its group, not the padded ones alone
    assert _rows(con, PLAIN.replace(" AS n", " AS n, COUNT(*) AS every")) == [(1, 1, 2)]
    assert _rows(con, COUNTED.replace(" AS n", " AS n, COUNT(*) AS every")) == [(1, 1, 1)]
    # no GROUP BY: over no rows at all the flag is one row that fails the test, and the count is a 0
    empty = sqlite3.connect(":memory:")
    empty.execute("CREATE TABLE visits (visit INT NOT NULL, customer INT, site TEXT)")
    empty.execute("CREATE TABLE purchases (id INT NOT NULL, visit INT, amount INT)")
    assert _rows(empty, f"SELECT {FLAG} AS n {JOIN} HAVING n > 0") == [] and _rows(empty, f"SELECT COUNT(*) AS n {JOIN} WHERE p.id IS NULL") == [(0,)]
    # tested outside over a join that pads: the count's own rows decide which outer rows are padded
    # (here the outer select has the derived table alone, which is why that is asked for)
    alone = f"SELECT d.customer FROM (SELECT v.customer, {FLAG} AS n {JOIN} GROUP BY v.customer) AS d WHERE d.n > 0"
    assert _rows(con, alone) == _rows(con, read(alone)) == [(1,)]


def test_a_group_of_texts_may_show_another_of_its_texts():
    # why a shown grouping column must be a whole number: with a collation that calls 'a' and 'A' equal, the group
    # shows one of its texts, and the padded rows alone may show the other
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE visits (visit INT NOT NULL, customer INT, site TEXT COLLATE NOCASE)")
    con.execute("CREATE TABLE purchases (id INT NOT NULL, visit INT, amount INT)")
    con.executemany("INSERT INTO visits VALUES (?, ?, ?)", [(1, 1, "a"), (2, 1, "A")])
    con.execute("INSERT INTO purchases VALUES (1, 1, 5)")
    by_site = PLAIN.replace("v.customer", "v.site")
    assert read(by_site) is None
    assert _rows(con, by_site) == [("a", 1)] and _rows(con, COUNTED.replace("v.customer", "v.site")) == [("A", 1)]
