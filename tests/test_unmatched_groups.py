"""A LEFT JOIN grouped by the kept table's key with a zero count, read as NOT EXISTS (``kumo_sql_prover.unmatched_groups``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import canonical_negation
from kumo_sql_prover.smt_equivalence import TableConstraints
from kumo_sql_prover.unmatched_groups import read_unmatched_groups

# c.id and o.id are NOT NULL keys; c.code is a key that may be NULL; o.cid and o.note may be NULL; t has no key
SCHEMA = {"c": ["id", "code", "name"], "o": ["id", "cid", "note"], "t": ["id", "v"]}
NOT_NULL = {"c": frozenset({"id"}), "o": frozenset({"id"})}
KEYS = {"c": [("id",), ("code",)], "o": [("id",)]}
CONSTRAINTS = {
    "c": TableConstraints(not_null=frozenset({"id"}), keys=(("id",), ("code",))),
    "o": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
}


def rewritten(sql, not_null=NOT_NULL, keys=KEYS, schema=SCHEMA):
    before = canonical_negation(sqlglot.parse_one(sql, read="mysql"))
    text = before.sql(dialect="mysql")
    after = read_unmatched_groups(before, schema, not_null, keys).sql(dialect="mysql")
    return after if after != text else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=constraints, compare_names=False, dialect="mysql").proven


def run(tables, *queries):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    db.execute("CREATE TABLE c (id BIGINT, code BIGINT, name BIGINT)")
    db.execute("CREATE TABLE o (id BIGINT, cid BIGINT, note BIGINT)")
    db.execute("CREATE TABLE t (id BIGINT, v BIGINT)")
    for name, rows in tables.items():
        insert_rows(db, name, rows)
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *[any_value(sql) for sql in queries])]


def any_value(sql):
    """The query for DuckDB, which wants ``ANY_VALUE`` around an output column that is not grouped (MySQL returns
    the group's value for a column the grouped key determines without it)."""

    tree = sqlglot.parse_one(sql, read="mysql")
    for select in tree.find_all(sqlglot.exp.Select):
        group = select.args.get("group")
        grouped = {item.sql() for item in group.expressions} if group is not None else None
        for item in select.expressions if grouped is not None else []:
            value = item.unalias()
            if isinstance(value, sqlglot.exp.Column) and value.sql() not in grouped:
                item.replace(sqlglot.exp.alias_(sqlglot.exp.AnyValue(this=value.copy()), item.alias_or_name))
    return tree.sql(dialect="duckdb")


def test_a_zero_count_over_a_left_join_becomes_not_exists():
    assert rewritten("SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0") == (
        "SELECT c.name FROM c WHERE NOT EXISTS(SELECT 1 FROM o WHERE c.id = o.cid)"
    )
    # the counted column is not NULL on a match because the ON condition compares it
    assert rewritten("SELECT x.name AS n FROM c AS x LEFT OUTER JOIN o AS y ON x.id = y.cid GROUP BY x.id, x.name HAVING COUNT(y.cid) < 1") == (
        "SELECT x.name AS n FROM c AS x WHERE NOT EXISTS(SELECT 1 FROM o AS y WHERE x.id = y.cid)"
    )
    # the kept table second, a filter on it, bare names that only one table has, the number first
    assert rewritten("SELECT name FROM o RIGHT JOIN c ON c.id = cid AND note > 1 WHERE name <> 3 OR name IS NULL GROUP BY c.id HAVING 1 > COUNT(cid)") == (
        "SELECT name FROM c WHERE (name <> 3 OR name IS NULL) AND NOT EXISTS(SELECT 1 FROM o WHERE c.id = cid AND note > 1)"
    )
    assert rewritten("SELECT c.id, c.name FROM c LEFT JOIN o ON o.cid = c.id WHERE c.name = 1 GROUP BY c.name, c.id HAVING COUNT(o.id) <= 0") == (
        "SELECT c.id, c.name FROM c WHERE c.name = 1 AND NOT EXISTS(SELECT 1 FROM o WHERE o.cid = c.id)"
    )
    # inside a derived table
    assert rewritten("SELECT d.name FROM (SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING 0 = COUNT(o.id)) AS d") == (
        "SELECT d.name FROM (SELECT c.name FROM c WHERE NOT EXISTS(SELECT 1 FROM o WHERE c.id = o.cid)) AS d"
    )


@pytest.mark.parametrize("sql", [
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(*) = 0",  # the padded row counts
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.note) = 0",  # NULL on a match
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(DISTINCT o.id) = 0",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(c.id) = 0",  # the kept table's column
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 1",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) < 2",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) >= 0",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0 AND MAX(c.name) > 1",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING SUM(o.id) = 0",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.name HAVING COUNT(o.id) = 0",  # not a key: rows share a group
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.code HAVING COUNT(o.id) = 0",  # a key that may be NULL
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id + 0 HAVING COUNT(o.id) = 0",
    "SELECT o.note FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id, o.note HAVING COUNT(o.id) = 0",  # groups by the joined table
    "SELECT c.name, COUNT(o.id) FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT MAX(c.name) FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT c.name + 1 FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT o.note FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",  # the joined table's column
    "SELECT DISTINCT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0 LIMIT 1",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid WHERE o.note = 1 GROUP BY c.id HAVING COUNT(o.id) = 0",  # filters the joined rows
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid WHERE o.id IS NULL GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT c.name FROM c JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",  # an inner join keeps no unmatched row
    "SELECT c.name FROM c FULL JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT c.name FROM c RIGHT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",  # o is the kept table
    "SELECT c.name FROM c LEFT JOIN o USING (id) GROUP BY c.id HAVING COUNT(o.cid) = 0",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid LEFT JOIN t ON t.id = c.id GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT c.name FROM c LEFT JOIN (SELECT id, cid FROM o) AS o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid AND o.note IN (SELECT v FROM t) GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid AND RAND() > 0.5 GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY id HAVING COUNT(o.id) = 0",  # id is either table's
    "SELECT c.name AS id FROM c LEFT JOIN t ON c.id = t.id GROUP BY c.id HAVING COUNT(t.v) = 0",  # v may be NULL on a match
    "SELECT t.v FROM t LEFT JOIN o ON t.id = o.cid GROUP BY t.id HAVING COUNT(o.id) = 0",  # t has no key
    "SELECT a.name FROM c AS a LEFT JOIN c AS a ON a.id = a.code GROUP BY a.id HAVING COUNT(a.id) = 0",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid HAVING COUNT(o.id) = 0",
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id WITH ROLLUP HAVING COUNT(o.id) = 0",
    "SELECT c.name FROM c, o WHERE c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",
])
def test_other_selects_stay(sql):
    assert rewritten(sql) is None


def test_a_bare_group_name_that_an_alias_could_capture_stays():
    assert rewritten("SELECT c.name AS code FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id, code HAVING COUNT(o.id) = 0") is None
    assert rewritten("SELECT c.code AS code FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id, code HAVING COUNT(o.id) = 0") is not None


def test_the_key_has_to_be_declared_and_not_null():
    sql = "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0"
    assert rewritten(sql) is not None
    assert rewritten(sql, keys=None) is None
    assert rewritten(sql, not_null=None) is None
    assert rewritten(sql, schema=None) is None
    assert rewritten(sql, keys={"o": [("id",)]}) is None
    assert rewritten(sql, keys={"c": [("id", "name")]}) is None  # the whole key has to be grouped
    assert rewritten(sql, not_null={"o": frozenset({"id"})}) is None
    # without the declaration that o.id is not NULL the count may miss a match
    assert rewritten(sql, not_null={"c": frozenset({"id"})}) is None
    assert rewritten(sql.replace("COUNT(o.id)", "COUNT(o.cid)"), not_null={"c": frozenset({"id"})}) is not None
    # a key of two columns, both grouped and NOT NULL
    two = "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id, c.code HAVING COUNT(o.id) = 0"
    assert rewritten(two, keys={"c": [("id", "code")]}, not_null={"c": frozenset({"id", "code"}), "o": frozenset({"id"})}) is not None
    assert rewritten(two, keys={"c": [("id", "code")]}) is None  # code may be NULL
    assert rewritten("WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 3) " + sql) is None


PROOFS = [
    (
        "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",
        "SELECT c.name FROM c WHERE NOT EXISTS (SELECT 1 FROM o WHERE o.cid = c.id)",
    ),
    (
        "SELECT x.name FROM c AS x LEFT JOIN o AS y ON x.id = y.cid GROUP BY x.id, x.name HAVING COUNT(y.cid) < 1",
        "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid WHERE o.id IS NULL",
    ),
    (
        "SELECT c.id FROM o RIGHT JOIN c ON c.id = o.cid AND o.note > 1 WHERE c.name = 2 GROUP BY c.id HAVING COUNT(o.id) = 0",
        "SELECT c.id FROM c WHERE c.name = 2 AND NOT EXISTS (SELECT 1 FROM o WHERE o.cid = c.id AND o.note > 1)",
    ),
]


@pytest.mark.parametrize("left, right", PROOFS)
def test_pairs_that_are_proved(left, right):
    assert proved(left, right)
    assert proved(right, left)


NOT_PROVED = [
    # COUNT(*) counts the padded row: never 0
    ("SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(*) = 0", "SELECT c.name FROM c WHERE NOT EXISTS (SELECT 1 FROM o WHERE o.cid = c.id)"),
    # a matched row with a NULL note is not counted
    ("SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.note) = 0", "SELECT c.name FROM c WHERE NOT EXISTS (SELECT 1 FROM o WHERE o.cid = c.id)"),
    # grouped by a column that is no key: one output row for two unmatched rows
    ("SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.name HAVING COUNT(o.id) = 0", "SELECT c.name FROM c WHERE NOT EXISTS (SELECT 1 FROM o WHERE o.cid = c.id)"),
    # NOT IN is never TRUE when a cid is NULL
    ("SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0", "SELECT c.name FROM c WHERE c.id NOT IN (SELECT cid FROM o)"),
    # a filter on the joined rows after the join is not part of the match
    ("SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid WHERE o.note = 1 GROUP BY c.id HAVING COUNT(o.id) = 0", "SELECT c.name FROM c WHERE NOT EXISTS (SELECT 1 FROM o WHERE o.cid = c.id AND o.note = 1)"),
]


@pytest.mark.parametrize("left, right", NOT_PROVED)
def test_pairs_that_are_not_proved(left, right):
    assert not proved(left, right)
    assert not proved(right, left)


def test_without_the_key_nothing_is_proved():
    left, right = PROOFS[0]
    assert not proved(left, right, constraints=None)
    assert not proved(left, right, constraints={"o": CONSTRAINTS["o"]})


def test_the_near_misses_really_differ():
    c, o = [(1, 1, 5), (2, 2, 5)], [(7, 1, None)]
    left, right = NOT_PROVED[0]
    assert run({"c": c, "o": o}, left, right) == [[], ["(5,)"]]
    left, right = NOT_PROVED[1]
    assert run({"c": c, "o": o}, left, right) == [["(5,)", "(5,)"], ["(5,)"]]
    left, right = NOT_PROVED[2]
    assert run({"c": c, "o": []}, left, right) == [["(5,)"], ["(5,)", "(5,)"]]
    left, right = NOT_PROVED[3]
    assert run({"c": c, "o": [(7, None, 0)]}, left, right) == [["(5,)", "(5,)"], []]
    left, right = NOT_PROVED[4]
    assert run({"c": c, "o": o}, left, right) == [[], ["(5,)", "(5,)"]]


QUERIES = [
    "SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT x.name AS n FROM c AS x LEFT OUTER JOIN o AS y ON x.id = y.cid GROUP BY x.id, x.name HAVING COUNT(y.cid) < 1",
    "SELECT name FROM o RIGHT JOIN c ON c.id = cid AND note > 1 WHERE name <> 3 OR name IS NULL GROUP BY c.id HAVING 1 > COUNT(cid)",
    "SELECT c.id, c.name FROM c LEFT JOIN o ON o.cid = c.id WHERE c.name = 1 GROUP BY c.name, c.id HAVING COUNT(o.id) <= 0",
    "SELECT c.code, c.name FROM c LEFT JOIN o ON o.cid < c.id OR o.note = c.name GROUP BY c.id HAVING COUNT(o.id) = 0",
    "SELECT c.name FROM c LEFT JOIN o ON c.code = o.note AND c.name = o.cid GROUP BY c.id HAVING 0 >= COUNT(o.note)",
    "SELECT d.name FROM (SELECT c.name FROM c LEFT JOIN o ON c.id = o.cid GROUP BY c.id HAVING 0 = COUNT(o.id)) AS d",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_the_rewritten_query_returns_the_same_rows(sql):
    """The written query and the rewritten one on random tables that keep the declarations: c.id and o.id are keys
    without NULLs, c.code is a key that may be NULL more than once. Either table may be empty; o.cid and o.note
    may be NULL, repeat, or match no row of c."""

    after = rewritten(sql)
    assert after is not None
    rng = random.Random(len(sql))
    for _ in range(150):
        codes = rng.sample([10, 11, 12, 13, 14, 15], 6)
        tables = {
            "c": [(i, rng.choice([None, codes[n]]), rng.choice([None, 1, 2, 3])) for n, i in enumerate(rng.sample([1, 2, 3, 4, 5], rng.choice([0, 1, 2, 3, 5])))],
            "o": [(i, rng.choice([None, 1, 2, 3, 9]), rng.choice([None, 1, 2, 3, 10, 11])) for i in rng.sample(range(1, 9), rng.choice([0, 1, 2, 4, 7]))],
        }
        a, b = run(tables, sql, after)
        assert a == b, (sql, after, tables)
