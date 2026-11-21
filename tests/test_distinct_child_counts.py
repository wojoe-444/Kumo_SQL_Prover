"""Counts of the distinct keys of joined child tables are one counted table per child (``distinct_child_counts``)."""

import random

import duckdb
import pytest
import sqlglot

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.distinct_child_counts import NAME, read_distinct_child_counts
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {
    "notes": ["id", "title", "kind", "author", "code"],
    "people": ["id", "name", "team"],
    "replies": ["id", "note", "kind", "tag"],
    "votes": ["id", "note", "weight"],
    "loose": ["id", "note"],
}
TYPES = {
    "notes": {"id": "INT", "title": "TEXT", "kind": "INT", "author": "INT", "code": "TEXT"},
    "people": {"id": "INT", "name": "TEXT", "team": "INT"},
    "replies": {"id": "INT", "note": "INT", "kind": "INT", "tag": "TEXT"},
    "votes": {"id": "INT", "note": "INT", "weight": "INT"},
    "loose": {"id": "INT", "note": "INT"},
}
KEYS = {"notes": [("id",)], "people": [("id",)], "replies": [("id",)], "votes": [("id",)], "loose": [("id",)]}
NOT_NULL = {"notes": {"id"}, "people": {"id"}, "replies": {"id"}, "votes": {"id"}}  # ``loose.id`` may be NULL
CONSTRAINTS = {table: TableConstraints(not_null=frozenset(NOT_NULL.get(table, ())), keys=tuple(KEYS[table])) for table in SCHEMA}

TWO = (
    "SELECT n.id, n.title, COUNT(DISTINCT r.id) AS replies, COUNT(DISTINCT v.id) AS votes "
    "FROM notes AS n LEFT JOIN replies AS r ON n.id = r.note LEFT JOIN votes AS v ON n.id = v.note "
    "GROUP BY n.id, n.title"
)
TWO_FIRST = (
    "SELECT n.id, n.title, COALESCE(a.cnt, 0) AS replies, COALESCE(b.cnt, 0) AS votes FROM notes AS n "
    "LEFT JOIN (SELECT note, COUNT(*) AS cnt FROM replies GROUP BY note) AS a ON n.id = a.note "
    "LEFT JOIN (SELECT note, COUNT(*) AS cnt FROM votes GROUP BY note) AS b ON n.id = b.note"
)
# every select here is read; the DuckDB replay checks each against its rewritten form
READ = [
    TWO,
    TWO.replace("GROUP BY", "WHERE n.kind = 1 AND n.title IS NOT NULL GROUP BY"),
    # tests on the child's own columns beside the link
    "SELECT n.id, COUNT(DISTINCT r.id) AS a, COUNT(DISTINCT v.id) AS b FROM notes AS n "
    "LEFT JOIN replies AS r ON r.note = n.id AND r.kind = 2 LEFT JOIN votes AS v ON n.id = v.note AND v.weight > 0 GROUP BY n.id",
    # lookups, inner and left, and a child linked through a lookup's column
    "SELECT n.id, p.name, COUNT(DISTINCT r.id) AS a, COUNT(DISTINCT v.id) + 1 AS b FROM notes AS n JOIN people AS p ON p.id = n.author "
    "LEFT JOIN replies AS r ON n.id = r.note LEFT JOIN votes AS v ON v.note = p.team WHERE p.team < 3 GROUP BY n.id, p.name",
    "SELECT n.title, p.name, COUNT(DISTINCT r.id) AS a, COUNT(DISTINCT v.id) AS b FROM notes AS n LEFT JOIN replies AS r ON n.id = r.note "
    "LEFT JOIN people AS p ON n.author = p.id LEFT JOIN votes AS v ON v.note = p.id GROUP BY n.id, n.title, p.name, p.id",
    # one child: a plain COUNT of its key counts its rows too, and a WHERE over the parents stays
    "SELECT n.id, n.title, COUNT(r.id) AS a FROM notes AS n LEFT JOIN replies AS r ON n.id = r.note WHERE n.kind = 1 GROUP BY n.id, n.title",
    # the same child table twice, and one child counted twice
    "SELECT n.id, COUNT(DISTINCT r.id) AS a, COUNT(DISTINCT q.id) AS b, COUNT(DISTINCT r.id) * 2 AS c FROM notes AS n "
    "LEFT JOIN replies AS r ON n.id = r.note AND r.kind = 1 LEFT JOIN replies AS q ON n.id = q.note AND q.kind = 2 GROUP BY n.id",
    # a child linked on a column that is no key of the parent, and may be NULL
    "SELECT n.id, n.kind, COUNT(DISTINCT v.id) AS a, COUNT(DISTINCT r.id) AS b FROM notes AS n "
    "LEFT JOIN votes AS v ON v.weight = n.kind LEFT JOIN replies AS r ON r.note = n.id GROUP BY n.id, n.kind",
]
# each breaks one condition of the reading, and is left as it is
LEFT_ALONE = [
    TWO.replace("COUNT(DISTINCT v.id)", "COUNT(v.id)"),  # a plain COUNT with two children counts pairs
    TWO.replace("COUNT(DISTINCT v.id)", "COUNT(*)"),
    TWO.replace("COUNT(DISTINCT v.id)", "COUNT(DISTINCT v.weight)"),  # not a key of the child
    TWO.replace("COUNT(DISTINCT v.id)", "SUM(v.weight)"),
    TWO.replace("COUNT(DISTINCT v.id)", "COUNT(DISTINCT v.id) FILTER (WHERE v.weight > 1)"),
    TWO.replace("votes AS v", "loose AS v"),  # the counted key may be NULL
    TWO.replace("GROUP BY n.id, n.title", "GROUP BY n.title"),  # two parent rows may be one group
    TWO.replace("GROUP BY", "WHERE r.kind = 1 GROUP BY"),  # a test on a child above the joins drops the padded row
    TWO + " HAVING COUNT(DISTINCT r.id) > 1",
    TWO + " ORDER BY n.id LIMIT 3",
    TWO.replace("SELECT n.id", "SELECT DISTINCT n.id"),
    TWO.replace("LEFT JOIN votes", "JOIN votes"),  # an inner join to a child drops parent rows
    TWO.replace("n.id = v.note", "n.id = v.note AND n.kind = 1"),  # a test on the parent inside a child's ON
    TWO.replace("n.id = v.note", "r.id = v.note"),  # a child of a child
    TWO.replace("n.id = v.note", "n.id < v.note"),
    TWO.replace("n.id = v.note", "n.code = v.note"),  # a text compared with a number
    TWO.replace("n.title, COUNT", "n.title, v.weight, COUNT"),  # a child column outside the counts
    TWO.replace("n.title, COUNT", "n.title, RANDOM() AS chance, COUNT"),
    TWO.replace("n.title, COUNT", "n.title, (SELECT MAX(id) FROM people) AS top, COUNT"),
    TWO.replace("LEFT JOIN votes", "JOIN people AS p ON p.team = n.author LEFT JOIN votes"),  # a join that is no lookup
    TWO.replace("LEFT JOIN votes", "LEFT JOIN people AS p ON p.id = n.author AND p.team = 1 LEFT JOIN votes"),
    "SELECT n.kind AS id, n.title, COUNT(DISTINCT r.id) AS a FROM notes AS n LEFT JOIN replies AS r ON n.id = r.note GROUP BY id, n.title",
    f"SELECT * FROM ({TWO.replace(' AS replies', '')}) AS t",  # the count's own name is read from outside
    "SELECT n.id, COUNT(DISTINCT r.id) AS a FROM notes AS n, replies AS r WHERE n.id = r.note GROUP BY n.id",
    "SELECT n.id, COUNT(DISTINCT r.id) AS a FROM notes AS n LEFT JOIN replies AS r USING (id) GROUP BY n.id",
]


def read(sql: str, types=TYPES):
    tree = sqlglot.parse_one(sql, read="postgres")
    before = tree.sql("postgres")
    new, fired = read_distinct_child_counts(tree, SCHEMA, KEYS, NOT_NULL, types)
    assert tree.sql("postgres") == before  # the tree handed in is never changed
    return new.sql("postgres"), fired


def prove(left: str, right: str):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, constraints=CONSTRAINTS, types=TYPES, compare_names=False, dialect="postgres", exact_arithmetic=True, timeout_ms=5000
    )


def test_two_children_are_written_as_one_counted_table_each():
    sql, fired = read(TWO.replace("GROUP BY", "WHERE n.kind = 1 GROUP BY"))
    assert fired
    assert sql == (
        "SELECT n.id, n.title, COALESCE(kqd0.kqn0, 0) AS replies, COALESCE(kqd1.kqn0, 0) AS votes FROM notes AS n "
        "LEFT JOIN (SELECT kqs.note AS note, COUNT(*) AS kqn0 FROM replies AS kqs GROUP BY kqs.note) AS kqd0 ON n.id = kqd0.note "
        "LEFT JOIN (SELECT kqs.note AS note, COUNT(*) AS kqn0 FROM votes AS kqs GROUP BY kqs.note) AS kqd1 ON n.id = kqd1.note "
        "WHERE n.kind = 1"
    )


def test_tests_on_the_child_move_into_its_counted_table():
    sql, fired = read("SELECT n.id, COUNT(r.id) AS a FROM notes AS n LEFT JOIN replies AS r ON r.note = n.id AND (r.kind = 2 OR tag = 'x') GROUP BY n.id")
    assert fired
    assert sql == (
        "SELECT n.id, COALESCE(kqd0.kqn0, 0) AS a FROM notes AS n LEFT JOIN (SELECT kqs.note AS note, COUNT(*) AS kqn0 FROM replies AS kqs "
        "WHERE (kqs.kind = 2 OR kqs.tag = 'x') GROUP BY kqs.note) AS kqd0 ON n.id = kqd0.note"
    )


@pytest.mark.parametrize("sql", READ)
def test_every_readable_select_is_read(sql):
    assert read(sql)[1]


@pytest.mark.parametrize("sql", LEFT_ALONE)
def test_a_select_that_breaks_a_condition_is_left_alone(sql):
    new, fired = read(sql)
    assert not fired
    assert new == sqlglot.parse_one(sql, read="postgres").sql("postgres")


def test_a_select_inside_a_larger_query_is_read_when_its_counts_are_named():
    sql, fired = read(f"SELECT t.replies FROM ({TWO}) AS t WHERE t.votes > 1")
    assert fired and "kqd1" in sql and "GROUP BY n.id" not in sql


def test_undeclared_types_do_not_stop_the_reading_and_declared_ones_must_agree():
    assert read(TWO, types=None)[1]
    assert not read(TWO, types={**TYPES, "votes": {**TYPES["votes"], "note": "TEXT"}})[1]


def test_nothing_is_read_without_declared_keys():
    tree = sqlglot.parse_one(TWO, read="postgres")
    assert not read_distinct_child_counts(tree, SCHEMA, None, NOT_NULL, TYPES)[1]
    assert not read_distinct_child_counts(tree, None, KEYS, NOT_NULL, TYPES)[1]
    assert not read_distinct_child_counts(tree, SCHEMA, KEYS, None, TYPES)[1]  # no key is known to hold values


def _database(seed: int):
    rng = random.Random(seed)
    con = duckdb.connect()
    con.execute("CREATE TABLE notes (id INT, title TEXT, kind INT, author INT, code TEXT)")
    con.execute("CREATE TABLE people (id INT, name TEXT, team INT)")
    con.execute("CREATE TABLE replies (id INT, note INT, kind INT, tag TEXT)")
    con.execute("CREATE TABLE votes (id INT, note INT, weight INT)")

    def maybe(value):
        return None if rng.random() < 0.2 else value

    for i in range(rng.randint(0, 6)):
        con.execute("INSERT INTO notes VALUES (?, ?, ?, ?, ?)", [i, maybe(rng.choice("ab")), maybe(rng.randint(1, 3)), maybe(rng.randint(0, 4)), "c"])
    for i in range(rng.randint(0, 4)):
        con.execute("INSERT INTO people VALUES (?, ?, ?)", [i, maybe(rng.choice("xy")), maybe(rng.randint(0, 4))])
    for i in range(rng.randint(0, 12)):
        con.execute("INSERT INTO replies VALUES (?, ?, ?, ?)", [i, maybe(rng.randint(0, 6)), maybe(rng.randint(1, 2)), maybe(rng.choice("xz"))])
    for i in range(rng.randint(0, 12)):
        con.execute("INSERT INTO votes VALUES (?, ?, ?)", [i, maybe(rng.randint(0, 6)), maybe(rng.randint(0, 3))])
    return con


def _rows(con, sql: str):
    return sorted(con.execute(sqlglot.transpile(sql, read="postgres", write="duckdb")[0]).fetchall(), key=repr)


def test_the_rewritten_select_returns_the_same_rows_in_duckdb():
    # DuckDB wants every selected column in the GROUP BY; the selects in READ name theirs
    written = [(sql, read(sql)[0]) for sql in READ]
    counted = 0
    for seed in range(24):
        con = _database(seed)
        for sql, new in written:
            rows = _rows(con, sql)
            assert rows == _rows(con, new), (seed, sql)
            counted += any(isinstance(value, int) and value > 1 for row in rows for value in row[2:])
        con.close()
    assert counted > 40  # the databases do hold parents with several child rows


def test_duckdb_tells_the_refused_spellings_apart():
    # why two of the conditions are there: these rewrites, done by hand, give other rows
    pairs = [
        (TWO.replace("COUNT(DISTINCT v.id)", "COUNT(v.id)"), TWO_FIRST),  # a plain COUNT with two children
        (TWO.replace("GROUP BY", "WHERE r.kind = 1 GROUP BY"), TWO_FIRST.replace("FROM replies", "FROM replies WHERE kind = 1")),
    ]
    for left, right in pairs:
        assert any(_rows(con, left) != _rows(con, right) for con in map(_database, range(40))), left


@pytest.mark.parametrize(
    "left, right",
    [
        (TWO, TWO_FIRST),
        (TWO.replace("GROUP BY", "WHERE n.kind = 1 GROUP BY"), TWO_FIRST + " WHERE n.kind = 1"),
        (TWO + " ORDER BY n.title LIMIT 5", TWO_FIRST + " ORDER BY n.title LIMIT 5"),
    ],
)
def test_the_pair_is_proved_and_only_with_the_rule(left, right):
    assert prove(left, right).proven
    with shape_fallback.without(NAME):
        assert not prove(left, right).proven


def test_other_counts_are_not_proved():
    assert not prove(TWO, TWO_FIRST.replace("FROM votes", "FROM votes WHERE weight > 0")).proven
    assert not prove(TWO.replace("COUNT(DISTINCT v.id)", "COUNT(v.id)"), TWO_FIRST).proven


def test_the_rule_fires_only_in_the_last_step_of_a_proof():
    from kumo_sql_prover.algebraic_equivalence import normalize

    options = {"schema": SCHEMA, "keys": KEYS, "not_null": {table: frozenset(columns) for table, columns in NOT_NULL.items()}, "types": TYPES, "dialect": "postgres"}
    assert NAME in shape_fallback.FIFTEENTH_BATCH
    assert not shape_fallback.fifteenth_batch_enabled() and not shape_fallback.fifteenth_staged() and shape_fallback.stage_key()[5] is False
    with shape_fallback.tracking() as fired:
        outside = normalize(TWO, **options)  # outside a proof the rule is off
    assert NAME not in fired
    with shape_fallback.fifteenth_deferred() as seen:
        assert shape_fallback.fifteenth_staged() and shape_fallback.fifteenth_watching() and not shape_fallback.fifteenth_batch_enabled()
        assert normalize(TWO, **options) == outside  # an earlier step asks the rule and changes nothing
        assert seen == [True] and not shape_fallback.fifteenth_watching()
    with shape_fallback.fifteenth_deferred() as seen:
        normalize(TWO_FIRST, **options)
        with shape_fallback.disabled():
            normalize(TWO, **options)
        assert seen == []  # nothing to read, and the shape rules off
    with shape_fallback.fifteenth_on(), shape_fallback.tracking() as fired:
        assert shape_fallback.fifteenth_batch_enabled() and shape_fallback.stage_key()[5] is True
        assert normalize(TWO, **options) != outside
        for off in (shape_fallback.disabled, shape_fallback.late):
            with off():
                assert not shape_fallback.fifteenth_batch_enabled()
    assert NAME in fired
    assert not shape_fallback.fifteenth_staged()


def test_a_pair_the_rule_reads_and_does_not_prove_costs_one_more_run(monkeypatch):
    from kumo_sql_prover import algebraic_equivalence

    runs = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    depth = []

    def counted(left, right, **kwargs):
        if not depth:  # a rule may start a proof inside the proof: that is part of the run it starts in
            runs.append(shape_fallback.stage_key()[5])
        depth.append(1)
        try:
            return whole(left, right, **kwargs)
        finally:
            depth.pop()

    monkeypatch.setattr(algebraic_equivalence, "_prove_equivalent_algebraic_whole", counted)
    assert not prove(TWO, TWO_FIRST.replace("FROM votes", "FROM votes WHERE weight > 0")).proven
    assert runs == [False, True]
    runs.clear()
    assert not prove(TWO_FIRST, TWO_FIRST.replace("FROM votes", "FROM votes WHERE weight > 0")).proven
    assert True not in runs  # no rule of the batch would fire: no last step
    runs.clear()
    assert prove(TWO, TWO.replace("AS replies", "AS answers")).proven
    assert True not in runs  # a pair proved before the batch never sees it
