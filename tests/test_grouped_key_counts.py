"""``COUNT(DISTINCT j.k)`` in a group that cannot hold a row of ``j`` twice loses its ``DISTINCT``
(``grouped_key_counts``). Every reading is run on SQLite over made-up tables with NULLs."""

import random
import sqlite3

import pytest
from sqlglot import parse_one

from kumo_sql_prover.grouped_key_counts import NAME, grouped_key_counts

TYPES = {
    "staff": {"id": "INT", "name": "VARCHAR(20)", "boss": "INT", "age": "INT"},
    "teams": {"team": "INT", "lead": "INT"},
    "badges": {"staff": "INT", "kind": "VARCHAR(5)", "score": "DOUBLE"},
}
KEYS = {"staff": [("id",)], "teams": [("team",)], "badges": [("staff", "kind")]}
NOT_NULL = {"staff": frozenset({"id"}), "teams": frozenset({"team"}), "badges": frozenset({"staff", "kind"})}
REPORTS = "FROM staff AS m JOIN staff AS e ON e.boss = m.id"
PLAIN = f"SELECT m.id, m.name, COUNT(DISTINCT e.id) AS reports {REPORTS} GROUP BY m.id, m.name"


def read(sql: str, keys=KEYS, not_null=NOT_NULL, types=TYPES) -> str | None:
    tree = parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    new = grouped_key_counts(tree, keys, not_null, types)
    assert tree.sql(dialect="mysql") == text  # the select given is not changed
    if new is None:
        return None
    assert grouped_key_counts(new, keys, not_null, types) is None  # once is enough
    return new.sql(dialect="mysql")


def test_the_name():
    assert NAME == "grouped_key_counts"


def test_the_count_loses_its_distinct():
    assert read(PLAIN) == PLAIN.replace("DISTINCT ", "")
    # only the counts of such a column: the ages of a group may repeat
    both = PLAIN.replace(" AS reports", " AS reports, COUNT(DISTINCT e.age) AS ages, COUNT(DISTINCT (e.id)) AS again")
    assert read(both) == both.replace("COUNT(DISTINCT e.id)", "COUNT(e.id)").replace("COUNT(DISTINCT (e.id))", "COUNT(e.id)")
    # in HAVING and ORDER BY as well
    more = f"{PLAIN} HAVING COUNT(DISTINCT e.id) > 1 ORDER BY COUNT(DISTINCT e.id) DESC"
    assert read(more) == more.replace("DISTINCT ", "")


READ = {
    "by the key and more": PLAIN,
    "by the key alone": f"SELECT m.id, COUNT(DISTINCT e.id) AS reports {REPORTS} GROUP BY m.id",
    "by position": f"SELECT m.id, m.name, COUNT(DISTINCT e.id) AS reports {REPORTS} GROUP BY 1, 2",
    "by a name of the select list": f"SELECT m.id AS manager, COUNT(DISTINCT e.id) AS reports {REPORTS} GROUP BY manager",
    "by a computed value too": f"SELECT m.id, e.age > 30, COUNT(DISTINCT e.id) AS reports {REPORTS} GROUP BY m.id, e.age > 30",
    "with a filter and a test": f"SELECT m.id, COUNT(DISTINCT e.id) AS reports {REPORTS} WHERE e.age > 25 GROUP BY m.id HAVING COUNT(DISTINCT e.id) > 1",
    "any condition": "SELECT m.id, COUNT(DISTINCT e.id) AS n FROM staff AS m JOIN staff AS e ON e.age > m.age GROUP BY m.id",
    "a cross join": "SELECT m.id, COUNT(DISTINCT e.id) AS n FROM staff AS m, staff AS e WHERE e.age > m.age GROUP BY m.id",
    "a left join": "SELECT m.id, COUNT(DISTINCT e.id) AS n FROM staff AS m LEFT JOIN staff AS e ON e.boss = m.id GROUP BY m.id",
    "the padded side pinned": "SELECT m.id, COUNT(DISTINCT e.id) AS n FROM staff AS e LEFT JOIN staff AS m ON e.boss = m.id GROUP BY m.id",
    "three tables": "SELECT t.team, m.id, COUNT(DISTINCT e.id) AS n FROM teams AS t JOIN staff AS m ON m.id = t.lead JOIN staff AS e ON e.boss = m.id GROUP BY t.team, m.id",
    "a key of two columns": "SELECT b.staff, b.kind, COUNT(DISTINCT e.id) AS n FROM badges AS b JOIN staff AS e ON e.boss = b.staff GROUP BY b.staff, b.kind",
    "one table": "SELECT boss, COUNT(DISTINCT id) AS n FROM staff GROUP BY boss",
    "names without a table": "SELECT team, COUNT(DISTINCT id) AS n FROM teams JOIN staff ON boss = lead GROUP BY team",
}


def _tables(rng: random.Random) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE staff (id INT PRIMARY KEY NOT NULL, name TEXT, boss INT, age INT)")
    con.execute("CREATE TABLE teams (team INT PRIMARY KEY NOT NULL, lead INT)")
    con.execute("CREATE TABLE badges (staff INT NOT NULL, kind TEXT NOT NULL, score DOUBLE, PRIMARY KEY (staff, kind))")
    people = rng.sample(range(1, 9), rng.randint(0, 7))
    for person in people:
        con.execute("INSERT INTO staff VALUES (?, ?, ?, ?)", (person, rng.choice(("ann", "bo", None)), rng.choice((1, 2, 3, 4, None)), rng.choice((20, 30, 40, None))))
    for team in rng.sample(range(1, 6), rng.randint(0, 4)):
        con.execute("INSERT INTO teams VALUES (?, ?)", (team, rng.choice((1, 2, 3, None))))
    for person, kind in rng.sample([(p, k) for p in (1, 2, 3) for k in ("x", "y")], rng.randint(0, 5)):
        con.execute("INSERT INTO badges VALUES (?, ?, ?)", (person, kind, rng.choice((0.5, 1.5, None))))
    return con


def _rows(con: sqlite3.Connection, sql: str) -> list:
    return sorted(con.execute(parse_one(sql, read="mysql").sql(dialect="sqlite")).fetchall(), key=repr)


@pytest.mark.parametrize("which", sorted(READ))
def test_the_reading_returns_the_same_rows(which):
    sql = READ[which]
    new = read(sql)
    assert new is not None and "DISTINCT" not in new
    rng = random.Random(which)
    seen = 0
    for _ in range(60):
        con = _tables(rng)
        rows = _rows(con, sql)
        assert rows == _rows(con, new)
        seen += len(rows)
        con.close()
    assert seen


AGE = "FROM staff AS m JOIN staff AS e ON e.age > m.age"
LEFT_ALONE = {
    "the other table grouped by a column that is no key": f"SELECT m.name, COUNT(DISTINCT e.id) AS n {AGE} GROUP BY m.name",
    "the other table not grouped by": f"SELECT e.age, COUNT(DISTINCT e.id) AS n {AGE} GROUP BY e.age",
    "half a key": "SELECT b.staff, COUNT(DISTINCT e.id) AS n FROM badges AS b JOIN staff AS e ON e.boss = b.staff GROUP BY b.staff",
    "one of three tables not pinned": "SELECT m.id, COUNT(DISTINCT e.id) AS n FROM teams AS t JOIN staff AS m ON m.id = t.lead JOIN staff AS e ON e.boss = m.id GROUP BY m.id",
    "a column that is no key": f"SELECT m.id, COUNT(DISTINCT e.boss) AS n {AGE} GROUP BY m.id",
    "half a key counted": "SELECT e.id, COUNT(DISTINCT b.staff) AS n FROM badges AS b JOIN staff AS e ON e.boss <> b.staff GROUP BY e.id",
    "two columns counted": f"SELECT m.id, COUNT(DISTINCT e.id, e.age) AS n {AGE} GROUP BY m.id",
    "a computed value counted": f"SELECT m.id, COUNT(DISTINCT e.id + 1) AS n {AGE} GROUP BY m.id",
    "a name both tables have": f"SELECT m.id, COUNT(DISTINCT id) AS n {AGE} GROUP BY m.id",
    "grouped by a name both tables have": f"SELECT COUNT(DISTINCT e.id) AS n {AGE} GROUP BY id",
    "grouped by the key plus one": f"SELECT m.id + 1, COUNT(DISTINCT e.id) AS n {AGE} GROUP BY m.id + 1",
    "grouped by a name of something else": f"SELECT m.age AS who, COUNT(DISTINCT e.id) AS n {AGE} GROUP BY who",
    "no GROUP BY": f"SELECT COUNT(DISTINCT e.id) AS n {AGE}",
    "WITH ROLLUP": f"SELECT m.id, COUNT(DISTINCT e.id) AS n {AGE} GROUP BY m.id WITH ROLLUP",
    "a right join": f"SELECT m.id, COUNT(DISTINCT e.id) AS n {AGE} GROUP BY m.id".replace("JOIN", "RIGHT JOIN"),
    "a full join": f"SELECT m.id, COUNT(DISTINCT e.id) AS n {AGE} GROUP BY m.id".replace("JOIN", "FULL JOIN"),
    "USING": "SELECT m.id, COUNT(DISTINCT e.id) AS n FROM staff AS m JOIN staff AS e USING (age) GROUP BY m.id",
    "a natural join": "SELECT m.id, COUNT(DISTINCT e.id) AS n FROM staff AS m NATURAL JOIN staff AS e GROUP BY m.id",
    "a derived table": f"SELECT m.id, COUNT(DISTINCT e.id) AS n FROM (SELECT * FROM staff) AS m JOIN staff AS e ON e.age > m.age GROUP BY m.id",
    "a table that is not declared": "SELECT m.id, COUNT(DISTINCT e.id) AS n FROM staff AS m JOIN guests AS e ON e.age > m.age GROUP BY m.id",
    "a table of another schema": "SELECT m.id, COUNT(DISTINCT e.id) AS n FROM staff AS m JOIN old.staff AS e ON e.age > m.age GROUP BY m.id",
    "a floating-point column": "SELECT m.id, COUNT(DISTINCT b.score) AS n FROM staff AS m JOIN badges AS b ON b.staff = m.id GROUP BY m.id",
    "a window count": f"SELECT m.id, COUNT(DISTINCT e.id) OVER (PARTITION BY m.id) AS n {AGE}",
    "a count of a subquery": f"SELECT m.id, (SELECT COUNT(DISTINCT e.id) FROM staff AS e WHERE e.age > m.age) AS n FROM staff AS m JOIN teams AS t ON t.lead = m.id GROUP BY m.id, t.team",
    "a plain count": f"SELECT m.id, COUNT(e.id) AS n {AGE} GROUP BY m.id",
    "a distinct sum": f"SELECT m.id, SUM(DISTINCT e.id) AS n {AGE} GROUP BY m.id",
}


@pytest.mark.parametrize("why", sorted(LEFT_ALONE))
def test_left_alone(why):
    assert read(LEFT_ALONE[why]) is None


def test_left_alone_without_what_the_tables_declare_and_inside_a_with_clause():
    age = f"SELECT m.id, COUNT(DISTINCT e.id) AS n {AGE} GROUP BY m.id"
    assert read(age) is not None
    assert read(age, keys=None) is None and read(age, types=None) is None
    # a key that admits NULLs does not pin its table: two rows without a number are one group
    assert read(age, not_null=None) is None and read(age, not_null={"staff": frozenset({"age"})}) is None
    assert read(age, keys={"staff": [("id", "name")]}) is None  # the counted column alone is no key
    assert read(age, types={"staff": {"id": "FLOAT", "name": "VARCHAR", "boss": "INT", "age": "INT"}}) is None
    assert read(f"WITH staff AS (SELECT 1 AS id, 2 AS age UNION ALL SELECT 1, 3) {age}") is None
    assert read(f"WITH w AS ({age}) SELECT * FROM w") is None
    # a float key of two columns as the one that pins is read; the counted column decides
    assert read("SELECT b.staff, b.kind, COUNT(DISTINCT e.id) AS n FROM badges AS b JOIN staff AS e ON e.boss = b.staff GROUP BY b.staff, b.kind") is not None


def _demo() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE staff (id INT PRIMARY KEY NOT NULL, name TEXT, boss INT, age INT)")
    con.execute("CREATE TABLE badges (staff INT NOT NULL, kind TEXT NOT NULL, score DOUBLE, PRIMARY KEY (staff, kind))")
    con.executemany("INSERT INTO staff VALUES (?, ?, ?, ?)", [(1, "ann", None, 20), (2, "ann", None, 30), (3, "bo", 1, 40)])
    con.executemany("INSERT INTO badges VALUES (?, ?, ?)", [(1, "x", 0.5), (1, "y", 0.5)])
    return con


def test_why_the_refused_shapes_are_refused():
    con = _demo()
    # two managers of one name: the colleague older than both is in the group twice
    by_name = LEFT_ALONE["the other table grouped by a column that is no key"]
    assert _rows(con, by_name) == [("ann", 2)] and _rows(con, by_name.replace("DISTINCT ", "")) == [("ann", 3)]
    # two badges of one person: each report is there once per badge
    half = LEFT_ALONE["half a key"]
    assert _rows(con, half) == [(1, 1)] and _rows(con, half.replace("DISTINCT ", "")) == [(1, 2)]
    # a column that is no key repeats within the group of one row
    no_key = LEFT_ALONE["a column that is no key"]
    con.execute("INSERT INTO staff VALUES (4, 'cy', 1, 50)")
    assert _rows(con, no_key) == [(1, 1), (2, 1), (3, 1)] and _rows(con, no_key.replace("DISTINCT ", "")) == [(1, 2), (2, 2), (3, 1)]
    # and with the manager pinned the two agree on the same tables
    pinned = f"SELECT m.id, COUNT(DISTINCT e.id) AS n {AGE} GROUP BY m.id"
    assert _rows(con, pinned) == _rows(con, read(pinned)) == [(1, 3), (2, 2), (3, 1)]
