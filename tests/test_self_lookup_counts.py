"""Counts over a grouped table joined back to at most one of its table's own rows (``self_lookup_counts``)."""

import random

import duckdb
import pytest
import sqlglot
from sqlglot import exp

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.self_lookup_counts import NAME, read_self_lookup_counts, waits_for_lookup
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"logins": ["player", "day", "device", "score"], "steps": ["walker", "step", "note"]}
TYPES = {
    "logins": {"player": "INT", "day": "DATE", "device": "INT", "score": "INT"},
    "steps": {"walker": "INT", "step": "INT", "note": "VARCHAR(20)"},
}
KEYS = {"logins": [("player", "day")], "steps": [("walker", "step")]}
NOT_NULL = {"logins": {"player", "day"}, "steps": {"walker", "step"}}
CONSTRAINTS = {
    "logins": TableConstraints(not_null=frozenset({"player", "day"}), keys=(("player", "day"),)),
    "steps": TableConstraints(not_null=frozenset({"walker", "step"}), keys=(("walker", "step"),)),
}
P = "(SELECT player, MIN(day) AS first FROM logins GROUP BY player)"
W = "(SELECT walker, MIN(step) AS first, MAX(step) AS last, COUNT(*) AS steps FROM steps GROUP BY walker)"
JOINED = f"{P} AS p JOIN logins AS a ON p.player = a.player"
NEXT = f"{P} AS p LEFT JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day"
BACK = f"(SELECT f.player FROM {P} AS f JOIN logins AS a ON f.player = a.player AND f.first + 1 = a.day)"  # the players who came back


def written(sql: str, dialect: str = "mysql", keys=KEYS, not_null=NOT_NULL, types=TYPES) -> str | None:
    tree = sqlglot.parse_one(sql, read=dialect)
    before = tree.sql(dialect=dialect)
    new = read_self_lookup_counts(tree, SCHEMA, keys, not_null, types, dialect)
    assert tree.sql(dialect=dialect) == before  # the select it was given is not changed
    return None if new is None else new.sql(dialect=dialect)


def prove(left: str, right: str, dialect: str = "mysql"):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, types=TYPES, constraints=CONSTRAINTS, compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=5000
    )


# --- what is written -------------------------------------------------------------------------------------------------

TEXTS = [
    (  # the form the normalizer hands over: the outer join in a derived table of plain columns
        f"SELECT d.c0 AS first, COUNT(d.c1) AS players, COUNT(d.c2) AS back FROM (SELECT p.first AS c0, p.player AS c1, a.player AS c2 FROM {NEXT}) AS d GROUP BY d.c0",
        f"SELECT p.first AS first, COUNT(DISTINCT p.player) AS players, COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) AS back FROM {JOINED} GROUP BY p.first",
    ),
    (  # COUNT(*), a flag summed, a distinct count of the grouped column, the shift written on the other side
        f"SELECT first, COUNT(*) AS players, SUM(CASE WHEN next IS NULL THEN 0 ELSE 1 END) AS back, COUNT(DISTINCT player) AS again FROM (SELECT p.first, p.player, a.day AS next FROM {P} AS p LEFT JOIN logins AS a ON p.player = a.player AND a.day - 1 = p.first) AS d GROUP BY first",
        f"SELECT p.first, COUNT(DISTINCT p.player) AS players, SUM(CASE WHEN a.day - 1 = p.first THEN 1 ELSE 0 END) AS back, COUNT(DISTINCT p.player) AS again FROM {JOINED} GROUP BY p.first",
    ),
    (  # no GROUP BY; the distinct players that came back are the rows that pass
        f"SELECT ROUND(COUNT(DISTINCT d.back) / COUNT(DISTINCT d.player), 2) AS share FROM (SELECT p.player, a.player AS back FROM {P} AS p LEFT JOIN logins AS a ON p.player = a.player AND a.day - p.first = 1) AS d",
        f"SELECT ROUND(COUNT(CASE WHEN a.day - p.first = 1 THEN 1 END) / COUNT(DISTINCT p.player), 2) AS share FROM {JOINED}",
    ),
    (  # IS NOT NULL and IF; a flag that is 1 or NULL under COUNT; a further test on the looked-up row
        f"SELECT first, SUM(IF(next IS NOT NULL, 1, 0)) AS back, COUNT(IF(next IS NULL, NULL, 1)) AS again, COUNT(first) AS players FROM (SELECT p.first, a.day AS next FROM {P} AS p LEFT JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day AND a.score > 3) AS d GROUP BY first",
        f"SELECT p.first, SUM(CASE WHEN p.first + 1 = a.day AND a.score > 3 THEN 1 ELSE 0 END) AS back, COUNT(CASE WHEN p.first + 1 = a.day AND a.score > 3 THEN 1 END) AS again, COUNT(DISTINCT p.player) AS players FROM {JOINED} GROUP BY p.first",
    ),
    (  # the inner join with the test, no GROUP BY
        f"SELECT ROUND(COUNT(p.player) / (SELECT COUNT(DISTINCT player) FROM logins), 2) AS share FROM {P} AS p JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day",
        f"SELECT ROUND(COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) / (SELECT COUNT(DISTINCT player) FROM logins), 2) AS share FROM {JOINED}",
    ),
    (  # .. written with a comma and WHERE, the table first
        f"SELECT COUNT(DISTINCT a.player) AS back, COUNT(*) AS again FROM logins AS a, {P} AS p WHERE p.player = a.player AND a.day = p.first + 1",
        f"SELECT COUNT(CASE WHEN a.day = p.first + 1 THEN 1 END) AS back, COUNT(CASE WHEN a.day = p.first + 1 THEN 1 END) AS again FROM {JOINED}",
    ),
    (  # the whole join: the number of distinct values in the table is the number of grouped rows
        f"SELECT ROUND(COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) / (SELECT COUNT(DISTINCT player) FROM logins), 2) AS share FROM {JOINED}",
        f"SELECT ROUND(COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) / COUNT(DISTINCT p.player), 2) AS share FROM {JOINED}",
    ),
    (  # the same row looked up: every grouped row finds its first day
        f"SELECT COUNT(d.c1) AS players, COUNT(d.c2) AS found FROM (SELECT p.player AS c1, a.device AS c2, a.day AS c3 FROM {P} AS p LEFT JOIN logins AS a ON p.player = a.player AND a.day = p.first) AS d",
        None,  # device may be NULL: COUNT(a.device) is not the number of rows found
    ),
    (
        f"SELECT COUNT(d.c1) AS players, COUNT(d.c3) AS found FROM (SELECT p.player AS c1, a.device AS c2, a.day AS c3 FROM {P} AS p LEFT JOIN logins AS a ON p.player = a.player AND a.day = p.first) AS d",
        f"SELECT COUNT(DISTINCT p.player) AS players, COUNT(CASE WHEN a.day = p.first THEN 1 END) AS found FROM {JOINED}",
    ),
    (  # the outer join read where it stands
        f"SELECT p.first, COUNT(p.player) AS players, COUNT(a.player) AS back FROM {NEXT} GROUP BY p.first",
        f"SELECT p.first, COUNT(DISTINCT p.player) AS players, COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) AS back FROM {JOINED} GROUP BY p.first",
    ),
    (  # .. grouped by a name of the select list that is no column of either source; COUNT(1) counts rows
        f"SELECT p.first AS started, COUNT(1) AS players, COUNT(a.day) AS back FROM {NEXT} GROUP BY started",
        f"SELECT p.first AS started, COUNT(DISTINCT p.player) AS players, COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) AS back FROM {JOINED} GROUP BY p.first",
    ),
    (  # .. written as a right join, the table first
        f"SELECT COUNT(*) AS players, COUNT(a.player) AS back FROM logins AS a RIGHT JOIN {P} AS p ON p.player = a.player AND p.first + 1 = a.day",
        f"SELECT COUNT(DISTINCT p.player) AS players, COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) AS back FROM {JOINED}",
    ),
    (  # .. and with the join in parentheses
        f"SELECT p.first, COUNT(*) AS players, COUNT(DISTINCT a.player) AS back FROM ({NEXT}) GROUP BY p.first",
        f"SELECT p.first, COUNT(DISTINCT p.player) AS players, COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) AS back FROM {JOINED} GROUP BY p.first",
    ),
    (  # the players who came back listed first, then looked up
        f"SELECT p.first, COUNT(*) AS players, COUNT(r.player) AS back FROM {P} AS p LEFT JOIN {BACK} AS r ON p.player = r.player GROUP BY p.first",
        f"SELECT p.first, COUNT(DISTINCT p.player) AS players, COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) AS back FROM {JOINED} GROUP BY p.first",
    ),
    (  # .. from a grouped table that shows the player only; inside, the table first and the test in WHERE
        f"SELECT COUNT(d.c0) AS players, SUM(CASE WHEN d.c1 IS NULL THEN 0 ELSE 1 END) AS back FROM (SELECT p.player AS c0, r.back AS c1 FROM (SELECT player FROM logins GROUP BY player) AS p LEFT JOIN (SELECT a.player AS back FROM logins AS a, {P} AS f WHERE a.player = f.player AND a.day - 1 = f.first) AS r ON r.back = p.player) AS d",
        f"SELECT COUNT(DISTINCT p.player) AS players, SUM(CASE WHEN a.day - 1 = p.first THEN 1 ELSE 0 END) AS back FROM {JOINED}",
    ),
]


@pytest.mark.parametrize("sql, expected", TEXTS)
def test_the_counts_are_written_over_the_whole_join(sql, expected):
    assert written(sql) == expected


def test_a_test_summed_as_a_number_is_read_in_mysql_only():
    # in MySQL a test is the number 1 or 0, so SUM(a.player IS NOT NULL) adds one for each row found
    sql = f"SELECT p.first, SUM(a.player IS NOT NULL) AS back FROM {NEXT} GROUP BY p.first"
    assert written(sql) == f"SELECT p.first, SUM(CASE WHEN p.first + 1 = a.day THEN 1 ELSE 0 END) AS back FROM {JOINED} GROUP BY p.first"
    steps = f"SELECT SUM(s.step IS NOT NULL) AS onward FROM {W} AS w LEFT JOIN steps AS s ON w.walker = s.walker AND w.first + 1 = s.step"
    assert written(steps) is not None
    for dialect in ("postgres", "duckdb", "bigquery"):
        assert written(steps, dialect) is None
    assert written(sql.replace("a.player IS NOT NULL", "a.device IS NOT NULL")) is None  # device may be NULL in a row found
    assert written(sql.replace("a.player IS NOT NULL", "a.player IS NULL")) is None  # the rows that found nothing: not read
    assert written(sql.replace("SUM(", "COUNT(")) is None


def test_the_found_rows_of_a_lookup_are_known_before_the_join_is_read():
    # another rule would write the found rows differently; the last step leaves them alone until the join is read
    def waiting(sql: str) -> list[bool]:
        tree = sqlglot.parse_one(sql, read="mysql")
        return [waits_for_lookup(select, SCHEMA, KEYS, NOT_NULL, TYPES, "mysql") for select in tree.find_all(exp.Select)]

    looked_up = f"SELECT p.first, COUNT(r.player) AS back FROM {P} AS p LEFT JOIN {BACK} AS r ON p.player = r.player GROUP BY p.first"
    assert waiting(looked_up) == [False, False, True, False]  # the select, the grouped table, the found rows, their grouped table
    assert waiting(f"SELECT d.c0, COUNT(d.c1) AS back FROM (SELECT p.first AS c0, r.player AS c1 FROM {P} AS p LEFT JOIN {BACK} AS r ON p.player = r.player) AS d GROUP BY d.c0") == [False, False, False, True, False]
    assert not any(waiting(f"SELECT COUNT(r.player) AS back FROM {BACK} AS r"))  # not looked up
    assert not any(waiting(looked_up.replace("LEFT JOIN", "JOIN")))
    assert not any(waiting(looked_up.replace("f.first + 1 = a.day", "f.first < a.day")))
    assert not any(waiting(TEXTS[0][0]))
    tree = sqlglot.parse_one(looked_up, read="mysql")
    before = tree.sql()
    assert not waits_for_lookup(tree, {}, KEYS, NOT_NULL, TYPES, "mysql") and not waits_for_lookup(tree, SCHEMA, {}, NOT_NULL, TYPES, "mysql") and tree.sql() == before


def test_whole_numbers_shift_in_every_dialect_and_dates_only_in_mysql():
    steps = f"SELECT COUNT(d.c1) AS walkers, COUNT(d.c2) AS onward FROM (SELECT w.walker AS c1, s.step AS c2 FROM {W} AS w LEFT JOIN steps AS s ON w.walker = s.walker AND w.first + 1 = s.step) AS d"
    for dialect in ("mysql", "postgres", "duckdb", "bigquery"):
        assert written(steps, dialect) is not None
    dated = TEXTS[0][0]
    assert written(dated, "mysql") is not None
    for dialect in ("postgres", "duckdb", "bigquery"):
        assert written(dated, dialect) is None  # what a date plus a number is there is not this module's knowledge
    plain = dated.replace("p.first + 1 = a.day", "p.first = a.day")
    for dialect in ("mysql", "postgres", "duckdb"):
        assert written(plain, dialect) is not None


# --- the reading returns the same rows -------------------------------------------------------------------------------

DAYS = ["2023-02-27", "2023-02-28", "2023-03-01", "2023-03-02", "2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02"]


def _tables(seed: int, count: int = 40):
    rng = random.Random(seed)
    yield [], []
    for _ in range(count):
        logins = {(rng.randint(1, 4), rng.choice(DAYS)) for _ in range(rng.randint(1, 10))}
        steps = {(rng.randint(1, 3), rng.randint(1, 6)) for _ in range(rng.randint(1, 9))}
        yield (
            [(player, day, rng.choice([None, 1, 2]), rng.randint(0, 6)) for player, day in sorted(logins)],
            [(walker, step, rng.choice([None, "a", "b"])) for walker, step in sorted(steps)],
        )


def _database(logins, steps):
    con = duckdb.connect()
    con.execute("CREATE TABLE logins (player INT, day DATE, device INT, score INT)")
    con.execute("CREATE TABLE steps (walker INT, step INT, note VARCHAR)")
    if logins:
        con.executemany("INSERT INTO logins VALUES (?, ?, ?, ?)", logins)
    if steps:
        con.executemany("INSERT INTO steps VALUES (?, ?, ?)", steps)
    return con


def _bag(con, sql: str):
    return sorted(repr(row) for row in con.execute(sqlglot.transpile(sql, read="mysql", write="duckdb")[0]).fetchall())


READ = [sql for sql, expected in TEXTS if expected is not None] + [
    f"SELECT d.last, COUNT(*) AS walkers, COUNT(d.onward) AS onward, SUM(CASE WHEN d.onward IS NOT NULL THEN 1 ELSE 0 END) AS again, COUNT(d.steps) AS counted FROM (SELECT w.last, w.steps, s.step AS onward FROM {W} AS w LEFT JOIN steps AS s ON w.walker = s.walker AND s.step = w.first + 2) AS d GROUP BY d.last",
    f"SELECT COUNT(*) AS walkers, COUNT(DISTINCT d.back) AS back FROM (SELECT w.walker, s.walker AS back FROM {W} AS w LEFT JOIN steps AS s ON w.walker = s.walker AND w.last - s.step = 1 AND s.note IS NOT NULL) AS d",
    f"SELECT COUNT(*) AS pairs, COUNT(s.step) AS again, COUNT(DISTINCT w.walker) AS walkers FROM {W} AS w JOIN steps AS s ON w.walker = s.walker WHERE s.step - 1 = w.first",
    f"SELECT COUNT(CASE WHEN s.step > w.first THEN 1 END) / (SELECT COUNT(DISTINCT walker) FROM steps) AS later FROM {W} AS w JOIN steps AS s ON w.walker = s.walker",
]


@pytest.mark.parametrize("position", range(len(READ)))
def test_the_reading_returns_the_same_rows(position):
    sql = READ[position]
    new = written(sql)
    assert new is not None and new != sql
    for logins, steps in _tables(1097 + position):
        con = _database(logins, steps)
        assert _bag(con, sql) == _bag(con, new), (logins, steps)


# --- what is left alone, and why --------------------------------------------------------------------------------------


def _outer(test: str, items: str = "COUNT(d.c1) AS players, COUNT(d.c2) AS back", grouped: str = P, group: str = "") -> str:
    return f"SELECT {items} FROM (SELECT p.first AS c0, p.player AS c1, a.player AS c2, a.device AS c3 FROM {grouped} AS p LEFT JOIN logins AS a ON {test}) AS d{group}"


LEFT_ALONE = [
    _outer("p.player = a.player"),  # no test: every grouped row finds all its rows
    _outer("p.player = a.player AND a.day > p.first"),  # the test gives no day: several rows may pass
    _outer("p.player = a.player AND a.score = p.first"),  # another column than the key's, of another type
    _outer("p.player = a.player AND a.day = '2023-03-01'"),  # a constant is not read as a value of the grouped row
    _outer("p.player = a.player AND a.day + 1.5 = p.first"),  # not a whole number
    _outer("p.player = a.player AND a.day * 2 = p.first"),
    _outer("p.player = a.player AND DATEDIFF(a.day, p.first) = 1"),  # not read (another module may)
    _outer("p.first + 1 = a.day"),  # no equality on the grouped column
    _outer("p.player = a.device AND p.first + 1 = a.day"),  # joined on another column
    _outer("p.player = a.player AND p.first + 1 = a.day", grouped="(SELECT player, MIN(day) AS first FROM logins WHERE score > 2 GROUP BY player)"),
    _outer("p.player = a.player AND p.first + 1 = a.day", grouped="(SELECT player, MIN(day) AS first FROM logins GROUP BY player HAVING COUNT(*) > 1)"),
    _outer("p.player = a.player AND p.first + 1 = a.day", grouped="(SELECT player, MIN(day) AS first FROM logins GROUP BY player, device)"),
    _outer("p.player = a.player AND p.first + 1 = a.day", grouped="(SELECT device AS player, MIN(day) AS first FROM logins GROUP BY device)"),  # device may be NULL
    _outer("p.player = a.player AND p.first + 1 = a.day", grouped="(SELECT player, device, MIN(day) AS first FROM logins GROUP BY player)"),
    _outer("p.player = a.player AND p.first + 1 = a.day", grouped="(SELECT walker AS player, MIN(step) AS first FROM steps GROUP BY walker)"),  # another table
    _outer("p.player = a.player AND p.first + 1 = a.day", grouped="(SELECT DISTINCT player, MIN(day) AS first FROM logins GROUP BY player)"),
    _outer("p.player = a.player AND p.first + 1 = a.day", items="COUNT(d.c3) AS devices"),  # device may be NULL
    _outer("p.player = a.player AND p.first + 1 = a.day", items="COUNT(DISTINCT d.c3) AS devices"),
    _outer("p.player = a.player AND p.first + 1 = a.day", items="SUM(d.c2) AS total"),
    _outer("p.player = a.player AND p.first + 1 = a.day", items="MAX(d.c0) AS latest, COUNT(*) AS players"),
    _outer("p.player = a.player AND p.first + 1 = a.day", items="SUM(CASE WHEN d.c3 IS NULL THEN 0 ELSE 1 END) AS back"),  # device may be NULL in a row found
    _outer("p.player = a.player AND p.first + 1 = a.day", items="SUM(CASE WHEN d.c2 IS NULL THEN 1 ELSE 2 END) AS odd"),
    _outer("p.player = a.player AND p.first + 1 = a.day", items="COUNT(*) FILTER (WHERE d.c2 IS NOT NULL) AS back"),
    _outer("p.player = a.player AND p.first + 1 = a.day", items="d.c1, COUNT(d.c2) AS back", group=" GROUP BY d.c0"),  # a column that is not grouped
    _outer("p.player = a.player AND p.first + 1 = a.day", items="d.c2, COUNT(*) AS n", group=" GROUP BY d.c2"),  # grouped by the looked-up row
    _outer("p.player = a.player AND p.first + 1 = a.day", items="d.c0, d.c2"),  # no aggregate: one row per row of the join
    _outer("p.player = a.player AND p.first + 1 = a.day", items="COUNT(d.c1) AS players", group=" GROUP BY d.c0 HAVING COUNT(d.c2) > 0"),
    _outer("p.player = a.player AND p.first + 1 = a.day", items="COUNT(d.c1) AS players", group=" GROUP BY d.c0 ORDER BY 1 LIMIT 2"),
    _outer("p.player = a.player AND p.first + 1 = a.day", items="DISTINCT COUNT(d.c1) AS players", group=" GROUP BY d.c0"),
    _outer("p.player = a.player AND p.first + 1 = a.day", items="COUNT(d.c1) OVER () AS players"),
    _outer("p.player = a.player AND p.first + 1 = a.day AND a.score > RAND()"),
    _outer("p.player = a.player AND p.first + 1 = a.day AND a.score > (SELECT MIN(score) FROM logins)"),
    f"SELECT COUNT(d.c1) AS players FROM (SELECT p.player AS c1, a.player AS c2 FROM {NEXT} WHERE a.score > 1) AS d",
    f"SELECT COUNT(d.c1) AS players FROM (SELECT p.player AS c1, a.player AS c2 FROM logins AS a LEFT JOIN {P} AS p ON p.player = a.player AND p.first + 1 = a.day) AS d",  # the table drives
    f"SELECT COUNT(d.c1) AS players FROM (SELECT p.player AS c1, a.player AS c2 FROM {P} AS p RIGHT JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day) AS d",
    f"SELECT COUNT(d.c1) AS players FROM (SELECT p.player AS c1, a.player AS c2 FROM {P} AS p FULL JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day) AS d",
    f"SELECT COUNT(d.c1) AS players FROM (SELECT p.player AS c1, a.player AS c2 FROM {NEXT} LEFT JOIN steps AS s ON s.walker = p.player) AS d",
    f'SELECT COUNT(d.c1) AS players FROM (SELECT p.player AS c1, a.player AS c2 FROM {P} AS p LEFT JOIN logins AS a ON p.player = a.player AND p.first + 1 = a."day") AS d',  # a quoted name
    f"SELECT first, COUNT(player) AS players FROM (SELECT p.first, p.player, a.player FROM {NEXT}) AS d GROUP BY first",  # two outputs named player
    # the inner join with the test keeps only the rows that found one: a group may be lost
    f"SELECT p.first, COUNT(*) AS back FROM {P} AS p JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day GROUP BY p.first",
    f"SELECT COUNT(*) AS n FROM {P} AS p JOIN logins AS a ON p.player = a.player AND a.day > p.first",  # several rows may pass
    f"SELECT SUM(a.score) AS total FROM {P} AS p JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day",
    f"SELECT COUNT(a.device) AS n FROM {P} AS p JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day",
    f"SELECT p.player FROM {P} AS p JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day",
    # the total is read only beside aggregates over the whole join, without GROUP BY
    f"SELECT (SELECT COUNT(DISTINCT player) FROM logins) AS players FROM {JOINED}",  # no aggregate: one row per row
    f"SELECT COUNT(*) / (SELECT COUNT(DISTINCT player) FROM logins) AS mean FROM {P} AS p JOIN logins AS a ON p.player = a.player GROUP BY p.first",
    f"SELECT COUNT(*) / (SELECT COUNT(DISTINCT device) FROM logins) AS mean FROM {JOINED}",
    f"SELECT COUNT(*) / (SELECT COUNT(player) FROM logins) AS mean FROM {JOINED}",
    f"SELECT COUNT(*) / (SELECT COUNT(DISTINCT player) FROM logins WHERE score > 1) AS mean FROM {JOINED}",
    f"SELECT COUNT(*) / (SELECT COUNT(DISTINCT walker) FROM steps) AS mean FROM {JOINED}",
    f"SELECT SUM((SELECT COUNT(DISTINCT player) FROM logins)) AS total FROM {JOINED}",
    f"SELECT COUNT(*) AS n FROM {JOINED}",  # nothing to read
    # a first source that is not grouped, and a total that reads no table
    "SELECT COUNT(*) AS n FROM (SELECT player, day AS first FROM logins) AS p LEFT JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day",
    "SELECT COUNT(*) AS n FROM (SELECT 1 AS player, 2 AS first) AS p LEFT JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day",
    "SELECT COUNT(*) AS n FROM (SELECT player, MIN(day) AS first FROM logins) AS p JOIN logins AS a ON p.player = a.player",
    f"SELECT COUNT(*) / (SELECT COUNT(DISTINCT 1)) AS mean FROM {JOINED}",
    f"SELECT COUNT(*) / (SELECT 2) AS mean FROM {JOINED}",
    # the outer join read where it stands has nothing else on the select
    f"SELECT COUNT(*) AS n FROM {NEXT} WHERE a.score > 1",
    f"SELECT COUNT(*) AS n FROM {P} AS p LEFT JOIN logins AS a ON p.player = a.player",  # no test
    f"SELECT a.day, COUNT(*) AS n FROM {NEXT} GROUP BY a.day",  # grouped by the looked-up row
    f"SELECT a.day AS seen, COUNT(*) AS n FROM {NEXT} GROUP BY seen",
    f"SELECT p.first AS device, COUNT(*) AS n FROM {NEXT} GROUP BY device",  # the name is a column of the table: MySQL groups by that
    f"SELECT p.first AS started, COUNT(*) AS n FROM {NEXT} GROUP BY started, a.device",
    f"SELECT p.first, COUNT(*) AS n FROM {NEXT} GROUP BY p.first HAVING COUNT(a.player) > 0",
    f"SELECT p.first, COUNT(*) AS n FROM {NEXT} GROUP BY p.first ORDER BY n LIMIT 1",
    f"SELECT p.first, COUNT(NULL) AS n FROM {NEXT} GROUP BY p.first",
    f"SELECT COUNT(*) AS n FROM ({NEXT}) AS j",  # a named parenthesis is another thing
    f"SELECT COUNT(*) AS n FROM logins AS a RIGHT JOIN {P} AS p ON p.player = a.player AND p.first + 1 = a.day WHERE a.score > 1",
    # the found rows listed first: one row at most for a player, of the same grouped table, joined on the player alone
    f"SELECT COUNT(*) AS n FROM {P} AS p LEFT JOIN (SELECT f.player FROM {P} AS f JOIN logins AS a ON f.player = a.player AND a.day > f.first) AS r ON p.player = r.player",
    f"SELECT COUNT(*) AS n FROM {P} AS p LEFT JOIN (SELECT f.player FROM {P} AS f JOIN logins AS a ON f.player = a.player) AS r ON p.player = r.player",
    f"SELECT COUNT(*) AS n FROM (SELECT player, MAX(day) AS first FROM logins GROUP BY player) AS p LEFT JOIN {BACK} AS r ON p.player = r.player",  # another first
    f"SELECT COUNT(*) AS n FROM (SELECT player, MIN(day) AS first FROM logins WHERE score > 1 GROUP BY player) AS p LEFT JOIN {BACK} AS r ON p.player = r.player",
    f"SELECT COUNT(*) AS n FROM {P} AS p LEFT JOIN (SELECT f.player, a.device FROM {P} AS f JOIN logins AS a ON f.player = a.player AND f.first + 1 = a.day) AS r ON p.player = r.player",
    f"SELECT COUNT(*) AS n FROM {P} AS p LEFT JOIN (SELECT f.first AS player FROM {P} AS f JOIN logins AS a ON f.player = a.player AND f.first + 1 = a.day) AS r ON p.player = r.player",
    f"SELECT COUNT(*) AS n FROM {P} AS p LEFT JOIN {BACK} AS r ON p.player = r.player AND p.first > '2023-03-01'",
    f"SELECT COUNT(*) AS n FROM {P} AS p LEFT JOIN {BACK} AS r ON p.first = r.player",
    f"SELECT COUNT(*) AS n FROM {P} AS p LEFT JOIN (SELECT DISTINCT f.player FROM {P} AS f JOIN logins AS a ON f.player = a.player AND f.first + 1 = a.day) AS r ON p.player = r.player",
    f"SELECT COUNT(*) AS n FROM {P} AS p LEFT JOIN (SELECT f.player FROM {P} AS f LEFT JOIN logins AS a ON f.player = a.player AND f.first + 1 = a.day) AS r ON p.player = r.player",
    f"SELECT COUNT(*) AS n FROM {P} AS p LEFT JOIN (SELECT f.player FROM {P} AS f JOIN logins AS a ON f.player = a.player AND f.first + 1 = a.day LIMIT 1) AS r ON p.player = r.player",
    f"SELECT COUNT(*) AS n FROM {P} AS p JOIN {BACK} AS r ON p.player = r.player",  # an inner join to the found rows
    f"SELECT COUNT(*) AS n FROM {W} AS p LEFT JOIN {BACK} AS r ON p.walker = r.player",  # another table
    f"WITH logins AS (SELECT walker AS player, step AS day, 1 AS device, 1 AS score FROM steps) SELECT COUNT(d.c1) AS players, COUNT(d.c2) AS back FROM (SELECT p.player AS c1, a.player AS c2 FROM {NEXT}) AS d",  # a WITH name hides the table
]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_other_selects_are_left_alone(sql):
    assert written(sql) is None


def test_without_a_declared_key_or_not_null_nothing_is_read():
    sql = TEXTS[0][0]
    assert written(sql) is not None
    assert written(sql, keys={"logins": [("player", "day", "device")]}) is None  # device may be NULL, and the test does not give it
    assert written(sql, keys={"logins": [("device",)]}) is None
    assert written(sql, not_null={"logins": {"day"}}) is None  # a NULL player is a grouped row that finds nothing
    assert written(sql, not_null={"logins": {"player"}}) is None  # a key with a column that may be NULL tells no rows apart
    assert written(sql, types={"logins": {"player": "INT", "day": "VARCHAR(10)", "device": "INT", "score": "INT"}}) is None  # text plus one is no shift
    assert written(sql, types={}) is None


def _differs(sql: str, new: str, logins) -> bool:
    con = _database(logins, [])
    return _bag(con, sql) != _bag(con, new)


def test_each_condition_is_needed():
    read = TEXTS[0][0]
    # two rows for one player and day (no key): the outer join holds the grouped row twice
    twice = [(1, "2023-03-01", 1, 0), (1, "2023-03-02", 1, 0), (1, "2023-03-02", 2, 0)]
    assert _differs(read, written(read), twice)
    # a test that gives no day: the outer join holds the grouped row once for each later row
    later = read.replace("p.first + 1 = a.day", "a.day > p.first")
    forced = written(later, keys={"logins": [("player",)]})  # as if the player alone told rows apart
    assert forced is not None and _differs(later, forced, [(1, "2023-03-01", 1, 0), (1, "2023-03-02", 1, 0), (1, "2023-12-31", 1, 0)])
    # a grouped column that may be NULL: its grouped row finds nothing, and the whole join loses it
    by_device = read.replace(P, "(SELECT device AS player, MIN(day) AS first FROM logins GROUP BY device)").replace("p.player = a.player", "p.player = a.device").replace("COUNT(d.c1)", "COUNT(*)")
    forced = written(by_device, keys={"logins": [("device", "day")]}, not_null={"logins": {"device", "day", "player"}})
    assert forced is not None and _differs(by_device, forced, [(1, "2023-03-01", None, 0), (2, "2023-03-01", 1, 0)])
    # a filtered grouped table: the total counts players it does not hold
    total = f"SELECT COUNT(*) / (SELECT COUNT(DISTINCT player) FROM logins) AS mean FROM {JOINED}"
    assert written(total) is not None
    filtered = total.replace("FROM logins GROUP BY", "FROM logins WHERE score > 2 GROUP BY")
    as_read = written(total).replace("FROM logins GROUP BY", "FROM logins WHERE score > 2 GROUP BY")
    assert written(filtered) is None and _differs(filtered, as_read, [(1, "2023-03-01", 1, 5), (2, "2023-03-01", 1, 0)])
    # the found rows listed first, of another grouped table (the last day, not the first)
    last = f"SELECT COUNT(*) AS players, COUNT(r.player) AS back FROM (SELECT player, MAX(day) AS first FROM logins GROUP BY player) AS p LEFT JOIN {BACK} AS r ON p.player = r.player"
    as_read = f"SELECT COUNT(DISTINCT p.player) AS players, COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) AS back FROM (SELECT player, MAX(day) AS first FROM logins GROUP BY player) AS p JOIN logins AS a ON p.player = a.player"
    assert written(last) is None and _differs(last, as_read, [(1, "2023-03-01", 1, 0), (1, "2023-03-02", 1, 0)])
    # found rows that may hold a player twice: the outer join holds that player twice
    often = f"SELECT COUNT(*) AS players FROM {P} AS p LEFT JOIN (SELECT f.player FROM {P} AS f JOIN logins AS a ON f.player = a.player AND a.day > f.first) AS r ON p.player = r.player"
    assert written(often) is None and _differs(often, f"SELECT COUNT(DISTINCT p.player) AS players FROM {JOINED}", [(1, "2023-03-01", 1, 0), (1, "2023-03-02", 1, 0), (1, "2023-12-31", 1, 0)])
    # a name in GROUP BY that is a column of the table groups by that column in MySQL, not by the item of that name
    named = f"SELECT COUNT(*) AS n FROM {NEXT} GROUP BY a.device"  # what GROUP BY device means beside p.first AS device
    assert written(f"SELECT p.first AS device, COUNT(*) AS n FROM {NEXT} GROUP BY device") is None
    assert _differs(named, f"SELECT COUNT(DISTINCT p.player) AS n FROM {JOINED} GROUP BY p.first", [(1, "2023-03-01", 1, 0), (1, "2023-03-02", 2, 0), (2, "2023-03-01", 1, 0), (2, "2023-03-02", 3, 0)])
    # the inner join with the test under GROUP BY: the first days nobody came back after are lost
    inner = f"SELECT p.first, COUNT(*) AS back FROM {P} AS p JOIN logins AS a ON p.player = a.player AND p.first + 1 = a.day GROUP BY p.first"
    whole = f"SELECT p.first, COUNT(CASE WHEN p.first + 1 = a.day THEN 1 END) AS back FROM {JOINED} GROUP BY p.first"
    assert written(inner) is None and _differs(inner, whole, [(1, "2023-03-01", 1, 0)])


# --- proofs ----------------------------------------------------------------------------------------------------------

REFERENCE = f"SELECT p.first AS day, COUNT(DISTINCT p.player) AS players, ROUND(SUM(CASE WHEN p.first + 1 = a.day THEN 1 ELSE 0 END) / COUNT(DISTINCT p.player), 2) AS share FROM {JOINED} GROUP BY p.first"
FRACTION = f"SELECT ROUND(SUM(CASE WHEN p.first + 1 = a.day THEN 1 ELSE 0 END) / COUNT(DISTINCT p.player), 2) AS share FROM {JOINED}"
PAIRS = [
    (REFERENCE, f"SELECT t.first, COUNT(t.player) AS players, ROUND(COUNT(b.player) / COUNT(t.player), 2) AS share FROM {P} AS t LEFT JOIN logins AS b ON t.player = b.player AND t.first = b.day - 1 GROUP BY t.first"),
    (
        REFERENCE,
        f"WITH firsts AS {P}, seen AS (SELECT f.player, f.first, b.day AS next FROM firsts AS f LEFT JOIN logins AS b ON f.player = b.player AND b.day = f.first + 1) "
        "SELECT first, COUNT(*) AS players, ROUND(SUM(IF(next IS NULL, 0, 1)) / COUNT(*), 2) AS share FROM seen GROUP BY first",
    ),
    (FRACTION, f"SELECT ROUND(COUNT(DISTINCT b.player) / COUNT(DISTINCT t.player), 2) AS share FROM {P} AS t LEFT JOIN logins AS b ON t.player = b.player AND b.day - t.first = 1"),
    (FRACTION, f"SELECT ROUND(COUNT(t.player) / (SELECT COUNT(DISTINCT player) FROM logins), 2) AS share FROM {P} AS t JOIN logins AS b ON t.player = b.player AND t.first + 1 = b.day"),
    (FRACTION, f"SELECT ROUND(COUNT(*) / (SELECT COUNT(DISTINCT player) FROM logins), 2) AS share FROM logins AS b, {P} AS t WHERE t.player = b.player AND b.day - 1 = t.first"),
    (REFERENCE, f"SELECT t.first AS started, COUNT(1) AS players, ROUND(SUM(b.player IS NOT NULL) / COUNT(1), 2) AS share FROM logins AS b RIGHT JOIN {P} AS t ON t.player = b.player AND t.first = b.day - 1 GROUP BY started"),
    (
        REFERENCE,
        f"WITH firsts AS {P}, back AS (SELECT b.player FROM logins AS b JOIN firsts AS f ON b.player = f.player AND b.day = f.first + 1) "
        "SELECT t.first, COUNT(*) AS players, ROUND(COUNT(r.player) / COUNT(*), 2) AS share FROM firsts AS t LEFT JOIN back AS r ON t.player = r.player GROUP BY t.first",
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
        # the day after as a date and as a number are different tests in MySQL (the 31st plus one is no date)
        (REFERENCE, f"SELECT t.first, COUNT(t.player) AS players, ROUND(COUNT(b.player) / COUNT(t.player), 2) AS share FROM {P} AS t LEFT JOIN logins AS b ON t.player = b.player AND DATEDIFF(b.day, t.first) = 1 GROUP BY t.first"),
        (REFERENCE, f"SELECT t.first, COUNT(t.player) AS players, ROUND(COUNT(b.player) / COUNT(t.player), 2) AS share FROM {P} AS t LEFT JOIN logins AS b ON t.player = b.player AND t.first = b.day - 2 GROUP BY t.first"),
        (REFERENCE, f"SELECT t.first, COUNT(t.player) AS players, ROUND(COUNT(b.device) / COUNT(t.player), 2) AS share FROM {P} AS t LEFT JOIN logins AS b ON t.player = b.player AND t.first = b.day - 1 GROUP BY t.first"),
        (REFERENCE, f"SELECT t.first, COUNT(t.player) AS players, ROUND(COUNT(b.player) / COUNT(t.player), 2) AS share FROM {P} AS t JOIN logins AS b ON t.player = b.player AND t.first = b.day - 1 GROUP BY t.first"),
        (FRACTION, f"SELECT ROUND(COUNT(DISTINCT b.player) / COUNT(DISTINCT t.player), 2) AS share FROM {P} AS t LEFT JOIN logins AS b ON t.player = b.player AND b.day > t.first"),
        (FRACTION, f"SELECT ROUND(COUNT(*) / (SELECT COUNT(player) FROM logins), 2) AS share FROM logins AS b, {P} AS t WHERE t.player = b.player AND b.day - 1 = t.first"),
    ],
)
def test_near_misses_stay_unproved(left, right):
    assert not prove(left, right).proven
