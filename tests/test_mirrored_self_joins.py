"""A self join whose two rows may change places is read both ways round (``mirrored_self_joins``)."""

import random

import pytest
from sqlglot import parse_one

duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover.mirrored_self_joins import NAME, _key, mirrored_self_joins

SCHEMA = {"sessions": ["account", "ip", "login", "logout", "site"], "accounts": ["account", "site"]}
TYPES = {
    "sessions": {"account": "INT", "ip": "INT", "login": "INT", "logout": "INT", "site": "VARCHAR"},
    "accounts": {"account": "INT", "site": "VARCHAR"},
}
HEAD = "SELECT DISTINCT a.account FROM sessions AS a JOIN sessions AS b ON a.account = b.account AND a.ip <> b.ip"
INSIDE = "b.login BETWEEN a.login AND a.logout"
INSIDE_MIRRORED = "a.login BETWEEN b.login AND b.logout"


def read(sql: str, schema=SCHEMA, types=TYPES) -> str | None:
    tree = parse_one(sql, read="mysql")
    text = tree.sql(dialect="mysql")
    new = mirrored_self_joins(tree, schema, types)
    assert tree.sql(dialect="mysql") == text  # the select given is not changed
    if new is None:
        return None
    assert mirrored_self_joins(new, schema, types) is None  # once is enough
    again = parse_one(new.sql(dialect="mysql"), read="mysql")
    assert mirrored_self_joins(again, schema, types) is None  # also when it is read from its text
    return new.sql(dialect="mysql")


def test_the_name():
    assert NAME == "mirrored_self_joins"


def test_the_condition_is_read_both_ways_round():
    assert read(f"{HEAD} AND {INSIDE}") == f"{HEAD} WHERE ({INSIDE}) OR ({INSIDE_MIRRORED})"
    # terms of ON and of WHERE are one condition; the terms that read the same both ways stay in ON
    assert read(f"{HEAD} WHERE {INSIDE}") == f"{HEAD} WHERE ({INSIDE}) OR ({INSIDE_MIRRORED})"
    assert read(f"SELECT DISTINCT a.account FROM sessions AS a JOIN sessions AS b ON {INSIDE} WHERE a.account = b.account AND b.ip <> a.ip") == (
        f"SELECT DISTINCT a.account FROM sessions AS a JOIN sessions AS b ON a.account = b.account AND b.ip <> a.ip WHERE ({INSIDE}) OR ({INSIDE_MIRRORED})"
    )
    # several terms that do not read the same both ways are taken together
    assert read(f"{HEAD} AND a.login < b.login AND b.login <= a.logout") == (
        f"{HEAD} WHERE (a.login < b.login AND b.login <= a.logout) OR (b.login < a.login AND a.login <= b.logout)"
    )
    # one address before the other
    assert read("SELECT DISTINCT a.account FROM sessions AS a JOIN sessions AS b ON a.account = b.account AND a.ip < b.ip") == (
        "SELECT DISTINCT a.account FROM sessions AS a JOIN sessions AS b ON a.account = b.account WHERE (a.ip < b.ip) OR (b.ip < a.ip)"
    )


def test_other_spellings_of_the_join_and_of_the_select_list():
    comma = "SELECT DISTINCT a.account FROM sessions AS a, sessions AS b WHERE a.account = b.account AND a.ip <> b.ip"
    assert read(f"{comma} AND {INSIDE}") == f"{HEAD} WHERE ({INSIDE}) OR ({INSIDE_MIRRORED})"
    assert read(f"{comma} AND {INSIDE}".replace(", sessions", " CROSS JOIN sessions")) == f"{HEAD} WHERE ({INSIDE}) OR ({INSIDE_MIRRORED})"
    assert read(f"{HEAD} AND {INSIDE}".replace("JOIN", "INNER JOIN")) == f"{HEAD} WHERE ({INSIDE}) OR ({INSIDE_MIRRORED})".replace("JOIN", "INNER JOIN")
    # the second row's column, both, another name for it, and names in capitals
    for items in ("b.account", "a.account, b.account", "a.account AS who", "(a.account)"):
        assert read(f"{HEAD} AND {INSIDE}".replace("a.account FROM", f"{items} FROM")) is not None
    capitals = "SELECT DISTINCT L1.ACCOUNT FROM SESSIONS AS L1 JOIN SESSIONS AS L2 ON L1.ACCOUNT = L2.ACCOUNT AND L2.LOGIN BETWEEN L1.LOGIN AND L1.LOGOUT"
    assert read(capitals) == (
        "SELECT DISTINCT L1.ACCOUNT FROM SESSIONS AS L1 JOIN SESSIONS AS L2 ON L1.ACCOUNT = L2.ACCOUNT "
        "WHERE (L2.LOGIN BETWEEN L1.LOGIN AND L1.LOGOUT) OR (L1.LOGIN BETWEEN L2.LOGIN AND L2.LOGOUT)"
    )
    # a table read once under its own name
    assert read("SELECT DISTINCT b.account FROM sessions JOIN sessions AS b ON sessions.account = b.account AND b.login > sessions.logout") is not None


SAME_BOTH_WAYS = [
    "a.login <= b.logout AND b.login <= a.logout",
    "b.logout >= a.login AND a.logout >= b.login",
    "GREATEST(a.login, b.login) <= LEAST(b.logout, a.logout)",
    f"({INSIDE} OR {INSIDE_MIRRORED})",
    "a.login + b.login > 4",
    "a.login = b.login",
    "a.login <> b.logout AND b.login <> a.logout",
    "(a.login < b.login OR b.login < a.login)",
    "a.site = 'x' AND b.site = 'x'",
]


@pytest.mark.parametrize("rest", SAME_BOTH_WAYS)
def test_a_condition_that_reads_the_same_both_ways_round_is_left_alone(rest):
    assert read(f"{HEAD} AND {rest}") is None
    assert read(HEAD) is None


def test_the_key_tells_only_orders_that_keep_the_value_apart():
    def key(text: str):
        return _key(parse_one(text, read="mysql"))

    assert key("a.x = b.x") == key("b.x = a.x") == key("(B.X = A.X)")
    assert key("a.x < b.y") == key("b.y > a.x") != key("a.x > b.y")
    assert key("a.x <= b.y") == key("b.y >= a.x") != key("a.x < b.y")
    assert key("a.x = 1 AND (b.y = 2 AND a.z > 3)") == key("a.z > 3 AND 2 = b.y AND a.x = 1")
    assert key("a.x = 1 OR b.y = 2") == key("b.y = 2 OR a.x = 1") != key("a.x = 1 AND b.y = 2")
    assert key("a.x - b.x = 1") != key("b.x - a.x = 1")
    assert key("a.x BETWEEN b.lo AND b.hi") != key("b.x BETWEEN a.lo AND a.hi")
    assert key("a.x BETWEEN b.lo AND b.hi") == key("A.X BETWEEN B.LO AND B.HI")
    assert key("a.s = 'x'") != key("a.s = 'X'")  # a text is kept as it is written
    assert key("LEAST(a.x, b.x)") == key("LEAST(b.x, a.x)") != key("GREATEST(a.x, b.x)")
    assert key("CONCAT(a.s, b.s)") != key("CONCAT(b.s, a.s)")


LEFT_ALONE = {
    "no DISTINCT": f"{HEAD} AND {INSIDE}".replace("DISTINCT ", ""),
    "GROUP BY": f"{HEAD} AND {INSIDE} GROUP BY a.account",
    "HAVING": f"{HEAD} AND {INSIDE} HAVING a.account > 1",
    "ORDER BY": f"{HEAD} AND {INSIDE} ORDER BY a.account",
    "LIMIT": f"{HEAD} AND {INSIDE} LIMIT 3",
    "an aggregate": f"{HEAD} AND {INSIDE}".replace("a.account FROM", "COUNT(a.account) FROM"),
    "a window": f"{HEAD} AND {INSIDE}".replace("a.account FROM", "a.account, ROW_NUMBER() OVER () FROM"),
    "a column the condition does not equate": f"{HEAD} AND {INSIDE}".replace("a.account FROM", "a.account, a.ip FROM"),
    "a column equated with another column": f"{HEAD} AND a.login = b.logout AND {INSIDE}".replace("a.account FROM", "a.login FROM"),
    "a column equated under OR": f"SELECT DISTINCT a.account FROM sessions AS a JOIN sessions AS b ON (a.account = b.account OR a.ip = b.ip) AND {INSIDE}",
    "a text column": f"{HEAD} AND a.site = b.site AND {INSIDE}".replace("a.account FROM", "a.account, a.site FROM"),
    "an expression": f"{HEAD} AND {INSIDE}".replace("a.account FROM", "a.account + 1 FROM"),
    "a constant beside it": f"{HEAD} AND {INSIDE}".replace("a.account FROM", "a.account, 1 FROM"),
    "a star": f"{HEAD} AND {INSIDE}".replace("a.account FROM", "* FROM"),
    "a left join": f"{HEAD} AND {INSIDE}".replace("JOIN", "LEFT JOIN"),
    "a right join": f"{HEAD} AND {INSIDE}".replace("JOIN", "RIGHT JOIN"),
    "USING": f"SELECT DISTINCT a.account FROM sessions AS a JOIN sessions AS b USING (account) WHERE {INSIDE}",
    "NATURAL": f"SELECT DISTINCT a.account FROM sessions AS a NATURAL JOIN sessions AS b WHERE {INSIDE}",
    "a third source": f"{HEAD} AND {INSIDE} JOIN accounts AS c ON c.account = a.account",
    "two tables": f"SELECT DISTINCT a.account FROM sessions AS a JOIN accounts AS b ON a.account = b.account AND a.site < b.site",
    "one source": f"SELECT DISTINCT a.account FROM sessions AS a WHERE a.login < a.logout",
    "a derived table": f"SELECT DISTINCT a.account FROM (SELECT * FROM sessions) AS a JOIN sessions AS b ON a.account = b.account AND {INSIDE}",
    "a table of another schema": f"{HEAD} AND {INSIDE}".replace("sessions AS b", "other.sessions AS b"),
    "a subquery in the condition": f"{HEAD} AND {INSIDE} AND a.login > (SELECT MIN(c.login) FROM sessions AS c)",
    "EXISTS in the condition": f"{HEAD} AND {INSIDE} AND EXISTS (SELECT 1 FROM accounts AS c WHERE c.account = a.account)",
    "a column without a name in front": f"{HEAD} AND {INSIDE} AND site = 'x'",
    "a column of a query around it": f"{HEAD} AND {INSIDE} AND a.site = o.site",
    "a call that changes": f"{HEAD} AND {INSIDE} AND RAND() < 0.5",
    "a clock": f"{HEAD} AND b.login < NOW() AND {INSIDE}",
    "a function the prover does not know": f"{HEAD} AND {INSIDE} AND MYSTERY(a.ip) = 1",
    "no condition": "SELECT DISTINCT a.account FROM sessions AS a CROSS JOIN sessions AS b",
    "a WITH clause on it": f"WITH x AS (SELECT 1 AS one) {HEAD} AND {INSIDE}",
}


@pytest.mark.parametrize("why", sorted(LEFT_ALONE))
def test_left_alone(why):
    assert read(LEFT_ALONE[why]) is None, why


def test_left_alone_inside_a_with_clause_and_without_a_declared_table_or_type():
    sql = f"{HEAD} AND {INSIDE}"
    assert read(sql) is not None
    inside = parse_one(f"WITH sessions AS (SELECT * FROM accounts) SELECT * FROM ({sql}) AS d", read="mysql")
    select = next(node for node in inside.find_all(type(parse_one("SELECT 1"))) if node.args.get("distinct"))
    assert mirrored_self_joins(select, SCHEMA, TYPES) is None  # the name may be the WITH table's
    assert read(sql, schema={"accounts": ["account"]}) is None
    assert read(sql, schema=None) is None
    assert read(sql, types=None) is None
    assert read(sql, types={"sessions": {"account": "VARCHAR"}}) is None
    assert read(sql, types={"sessions": {"account": "DECIMAL(10, 2)"}}) is None
    assert read(sql, types={"SESSIONS": {"ACCOUNT": "bigint"}}) is not None


# --- the reading returns the same rows --------------------------------------------------

RESTS = [
    INSIDE,
    "a.login < b.login",
    "a.login < b.login AND b.login <= a.logout",
    "a.logout + 1 = b.login",
    "b.login BETWEEN a.login AND a.logout OR b.logout BETWEEN a.login AND a.logout",
    "a.login <= b.logout",
    "a.logout IS NULL",
    "COALESCE(a.login, 0) < COALESCE(b.logout, 2)",
    "NOT a.login < b.login",
    "a.site = 'x'",
    "a.login - b.login = 1 AND a.site <> b.site",
]
HEADS = [
    HEAD,
    "SELECT DISTINCT a.account FROM sessions AS a JOIN sessions AS b ON a.account = b.account",
    "SELECT DISTINCT a.account, b.ip FROM sessions AS a JOIN sessions AS b ON a.account = b.account AND b.ip = a.ip",
    "SELECT DISTINCT b.account AS who FROM sessions AS a JOIN sessions AS b ON a.account = b.account AND a.ip < b.ip",
]


def _tables(rng: random.Random):
    con = duckdb.connect()
    con.execute("CREATE TABLE sessions (account INT, ip INT, login INT, logout INT, site VARCHAR)")
    rows = []
    for _ in range(rng.randint(0, 6)):
        rows.append(
            (
                rng.choice([1, 1, 2, 3, None]),
                rng.choice([1, 2, 2, None]),
                rng.choice([0, 1, 2, 3, None]),
                rng.choice([1, 2, 3, 4, None]),
                rng.choice(["x", "y", None]),
            )
        )
    if rows:
        con.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?, ?)", rows)
    return con


def _rows(con, sql: str):
    return sorted(con.execute(parse_one(sql, read="mysql").sql(dialect="duckdb")).fetchall(), key=repr)


@pytest.mark.parametrize("head", range(len(HEADS)))
@pytest.mark.parametrize("rest", range(len(RESTS)))
def test_the_reading_returns_the_same_rows(head, rest):
    sql = f"{HEADS[head]} AND ({RESTS[rest]})"
    new = read(sql)
    assert new is not None
    rng = random.Random(head * 100 + rest)
    for _ in range(40):
        con = _tables(rng)
        assert _rows(con, sql) == _rows(con, new)
        con.close()


def test_the_reading_is_about_the_rows_distinct_keeps():
    # the pairs themselves are others: without DISTINCT the two return different rows
    sql = f"{HEAD} AND {INSIDE}"
    new = read(sql)
    con = duckdb.connect()
    con.execute("CREATE TABLE sessions (account INT, ip INT, login INT, logout INT, site VARCHAR)")
    con.execute("INSERT INTO sessions VALUES (1, 1, 0, 9, 'x'), (1, 2, 3, 4, 'x'), (2, 1, 0, 1, 'x'), (2, 2, 5, 6, 'x')")
    assert _rows(con, sql) == _rows(con, new) == [(1,)]
    assert _rows(con, sql.replace("DISTINCT ", "")) == [(1,)] and _rows(con, new.replace("DISTINCT ", "")) == [(1,), (1,)]


def test_a_column_the_condition_does_not_equate_would_change_the_rows():
    # why the rule asks for it: read both ways round by hand, the row 2's address appears though no session of the
    # pair starts inside a session from address 2
    sql = "SELECT DISTINCT a.ip FROM sessions AS a JOIN sessions AS b ON a.account = b.account AND b.login BETWEEN a.login AND a.logout AND a.ip <> b.ip"
    assert read(sql) is None
    by_hand = sql.replace("b.login BETWEEN a.login AND a.logout", "(b.login BETWEEN a.login AND a.logout OR a.login BETWEEN b.login AND b.logout)")
    con = duckdb.connect()
    con.execute("CREATE TABLE sessions (account INT, ip INT, login INT, logout INT, site VARCHAR)")
    con.execute("INSERT INTO sessions VALUES (1, 1, 0, 9, 'x'), (1, 2, 3, 4, 'x')")
    assert _rows(con, sql) == [(1,)] and _rows(con, by_hand) == [(1,), (2,)]
