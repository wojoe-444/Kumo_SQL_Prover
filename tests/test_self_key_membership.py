"""A row's key looked up through its own table again is a test on the row itself (``self_key_membership``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.self_key_membership import read_self_key_membership

SCHEMA = {"students": ["id", "name", "department_id"], "departments": ["id", "name"], "rooms": ["id", "department_id"]}
KEYS = {"students": [("id",)], "rooms": [("id", "department_id")]}
NOT_NULL = {"students": frozenset({"id"}), "rooms": frozenset({"id", "department_id"})}
ROUNDABOUT = "SELECT id, name FROM students WHERE NOT id IN (SELECT s.id FROM students AS s JOIN departments AS d ON s.department_id = d.id)"
DIRECT = "SELECT id, name FROM students WHERE NOT EXISTS(SELECT 1 FROM departments AS d WHERE (students.department_id = d.id))"


def read(sql: str, keys=KEYS, not_null=NOT_NULL, types=None, schema=SCHEMA) -> str:
    return read_self_key_membership(sqlglot.parse_one(sql, read="mysql"), schema, keys, not_null, types).sql(dialect="mysql")


def test_the_second_reading_of_the_table_is_the_row_itself():
    assert read(ROUNDABOUT) == DIRECT
    assert read(ROUNDABOUT.replace("NOT id IN", "id IN")) == DIRECT.replace("NOT EXISTS", "EXISTS")
    # DISTINCT and GROUP BY on the key change nothing an IN can see
    assert read(ROUNDABOUT.replace("SELECT s.id", "SELECT DISTINCT s.id")) == DIRECT
    assert read(ROUNDABOUT.replace("d.id)", "d.id GROUP BY s.id)")) == DIRECT
    assert read(ROUNDABOUT.replace("SELECT s.id", "SELECT DISTINCT(s.id)")) == DIRECT
    # a comma join with its conditions in WHERE, bare columns that one source owns, and an outer alias
    comma = "SELECT o.name FROM students AS o WHERE o.id IN (SELECT s.id FROM students AS s, departments AS d WHERE department_id = d.id AND d.name > s.name)"
    assert read(comma) == "SELECT o.name FROM students AS o WHERE EXISTS(SELECT 1 FROM departments AS d WHERE (s.department_id = d.id AND d.name > s.name))".replace("s.department_id", "o.department_id").replace("s.name", "o.name")
    # two other tables stay, cross-joined under the moved conditions
    three = ROUNDABOUT.replace("d.id)", "d.id JOIN rooms AS r ON r.department_id = d.id)")
    assert read(three) == "SELECT id, name FROM students WHERE NOT EXISTS(SELECT 1 FROM departments AS d CROSS JOIN rooms AS r WHERE (students.department_id = d.id) AND (r.department_id = d.id))"
    # under OR and inside a nested query the predicate has the same value
    assert "EXISTS" in read(ROUNDABOUT.replace("WHERE NOT id IN", "WHERE name = 'x' OR NOT id IN"))
    assert "EXISTS" in read(f"SELECT COUNT(*) FROM ({ROUNDABOUT}) AS q")


@pytest.mark.parametrize(
    "sql, keys, not_null",
    [
        (ROUNDABOUT, None, NOT_NULL),  # no key declared
        (ROUNDABOUT, KEYS, {}),  # the key may be NULL: NOT IN is then unknown where NOT EXISTS is true
        (ROUNDABOUT, {"students": [("id", "name")]}, {"students": frozenset({"id", "name"})}),  # half of a composite key
        (ROUNDABOUT.replace("WHERE NOT id IN", "WHERE NOT department_id IN"), KEYS, NOT_NULL),  # not the key
        (ROUNDABOUT.replace("SELECT s.id", "SELECT s.department_id"), KEYS, NOT_NULL),  # the subquery lists another column
        (ROUNDABOUT.replace("SELECT s.id", "SELECT d.id"), KEYS, NOT_NULL),  # the key of the other table
        (ROUNDABOUT.replace("SELECT s.id", "SELECT s.id + 0"), KEYS, NOT_NULL),  # not the column itself
        (ROUNDABOUT.replace("JOIN departments", "LEFT JOIN departments"), KEYS, NOT_NULL),  # an outer join keeps unmatched rows
        (ROUNDABOUT.replace("d.id)", "d.id LIMIT 1)"), KEYS, NOT_NULL),
        (ROUNDABOUT.replace("d.id)", "d.id GROUP BY s.id HAVING COUNT(*) > 1)"), KEYS, NOT_NULL),
        (ROUNDABOUT.replace("d.id)", "d.id GROUP BY s.name)"), KEYS, NOT_NULL),
        (ROUNDABOUT.replace("d.id)", "d.id AND EXISTS(SELECT 1 FROM rooms))"), KEYS, NOT_NULL),  # a subquery inside
        (ROUNDABOUT.replace("d.id)", "d.id JOIN students AS s2 ON s2.id = d.id)"), KEYS, NOT_NULL),  # the subquery reads the table twice
        ("SELECT id, name FROM students WHERE NOT id IN (SELECT s.id FROM students AS s)", KEYS, NOT_NULL),  # nothing else is read
        ("SELECT id, name FROM students WHERE NOT id IN (SELECT s.id FROM students AS s CROSS JOIN departments AS d)", KEYS, NOT_NULL),  # no condition
        (ROUNDABOUT.replace("d.id)", "d.id AND students.name = d.name)"), KEYS, NOT_NULL),  # reads the query around it
        (ROUNDABOUT.replace("departments AS d", "departments AS students").replace("d.id", "students.id"), KEYS, NOT_NULL),  # the outer name is taken
        ("SELECT o.id FROM departments AS q LEFT JOIN students AS o ON o.department_id = q.id WHERE NOT o.id IN (SELECT s.id FROM students AS s JOIN departments AS d ON s.department_id = d.id)", KEYS, NOT_NULL),  # o may be a NULL-extended row
        ("SELECT d.id FROM departments AS d JOIN students AS o ON NOT o.id IN (SELECT s.id FROM students AS s JOIN rooms AS r ON s.department_id = r.id)", KEYS, NOT_NULL),  # in a join condition
        ("SELECT id IN (SELECT s.id FROM students AS s JOIN departments AS d ON s.department_id = d.id) FROM students", KEYS, NOT_NULL),  # in the select list
        ("SELECT id FROM (SELECT id, department_id FROM students) AS o WHERE NOT o.id IN (SELECT s.id FROM students AS s JOIN departments AS d ON s.department_id = d.id)", KEYS, NOT_NULL),  # the outer source is not the table
        ("SELECT id FROM students WHERE NOT id IN (1, 2)", KEYS, NOT_NULL),
    ],
)
def test_other_memberships_are_left_as_written(sql, keys, not_null):
    assert read(sql, keys, not_null) == sqlglot.parse_one(sql, read="mysql").sql(dialect="mysql")


def test_a_floating_point_key_is_left_alone():
    assert read(ROUNDABOUT, types={"students": {"id": "FLOAT64"}}) == sqlglot.parse_one(ROUNDABOUT, read="mysql").sql(dialect="mysql")
    assert read(ROUNDABOUT, types={"students": {"id": "INT64"}}) == DIRECT
    assert read(ROUNDABOUT, schema=None) == sqlglot.parse_one(ROUNDABOUT, read="mysql").sql(dialect="mysql")  # columns cannot be attributed


CONDITIONS = [
    "s.department_id = d.id", "s.department_id > d.id", "s.name = d.name", "d.name IS NULL", "s.department_id IS NULL", "s.id <> d.id",
    "(s.name = 'a' OR d.name = 'b')", "NOT s.department_id = d.id", "s.department_id + 1 = d.id",
]
SHAPES = [
    "SELECT id, name FROM students WHERE {n} id IN (SELECT {d} s.id FROM students AS s JOIN departments AS d ON {c} {g})",
    "SELECT o.id, o.department_id FROM students AS o WHERE o.name IS NULL OR {n} o.id IN (SELECT {d} s.id FROM students AS s, departments AS d WHERE {c} {g})",
    "SELECT o.id, q.id FROM students AS o JOIN departments AS q ON o.department_id >= q.id WHERE {n} o.id IN (SELECT {d} s.id FROM departments AS d JOIN students AS s ON {c} {g})",
]


def _database(rng, keyed: bool):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE students (id INT, name VARCHAR, department_id INT)")
    con.execute("CREATE TABLE departments (id INT, name VARCHAR)")
    text = lambda: rng.choice([None, "a", "b"])  # noqa: E731
    number = lambda: rng.choice([None, None, 0, 1, 2])  # noqa: E731
    ids = rng.sample(range(8), rng.randint(0, 6)) if keyed else [rng.choice([None, 0, 0, 1]) for _ in range(rng.randint(2, 6))]
    for value in ids:
        con.execute("INSERT INTO students VALUES (?, ?, ?)", [value, text(), number()])
    for _ in range(rng.randint(0, 4)):
        con.execute("INSERT INTO departments VALUES (?, ?)", [number(), text()])
    return con


@pytest.mark.parametrize("seed", range(3))
def test_the_rewritten_query_returns_the_rows_duckdb_returns_when_the_key_holds(seed):
    rng = random.Random(seed)
    fired = differs_without_key = 0
    for _ in range(15):
        condition = " AND ".join(rng.sample(CONDITIONS, rng.randint(1, 2)))
        sql = rng.choice(SHAPES).format(n=rng.choice(["", "NOT"]), d=rng.choice(["", "DISTINCT"]), c=condition, g=rng.choice(["", "", "GROUP BY s.id"]))
        rewritten = read_self_key_membership(sqlglot.parse_one(sql, read="duckdb"), SCHEMA, KEYS, NOT_NULL).sql(dialect="duckdb")
        assert "EXISTS" in rewritten, sql
        fired += 1
        for _ in range(3):
            con = _database(rng, keyed=True)
            rows = lambda text: sorted(con.execute(text).fetchall(), key=repr)  # noqa: E731
            assert rows(sql) == rows(rewritten), (sql, rewritten)
        con = _database(rng, keyed=False)
        differs_without_key += sorted(con.execute(sql).fetchall(), key=repr) != sorted(con.execute(rewritten).fetchall(), key=repr)
    assert fired == 15 and differs_without_key > 0  # the key is what makes it true


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

PLAIN = "SELECT id, name FROM students WHERE department_id NOT IN (SELECT id FROM departments)"
KEYED = {
    "students": TableConstraints(not_null=frozenset({"id", "department_id"}), keys=(("id",),)),
    "departments": TableConstraints(not_null=frozenset({"id"})),
}


def test_the_roundabout_anti_join_is_the_plain_one_under_the_key():
    prove = lambda constraints: prove_equivalent_algebraic(ROUNDABOUT, PLAIN, schema=SCHEMA, dialect="mysql", constraints=constraints)  # noqa: E731
    assert prove(KEYED).proven
    # without the key two students can share an id, one with a department and one without
    unkeyed = {**KEYED, "students": TableConstraints(not_null=frozenset({"id", "department_id"}))}
    assert not prove(unkeyed).proven
    assert not prove(None).proven


def test_a_pair_main_proves_without_the_rule_is_still_proved():
    """The rule works on the second attempt only: on the first it rewrote the NOT IN of this pair to a NOT EXISTS,
    and the other rules no longer matched it with the CTE spelling (three conditional Singh pairs were lost)."""

    left = (
        "SELECT DISTINCT sales.buyer_id AS buyer_id FROM product INNER JOIN sales ON sales.product_id = product.product_id "
        "WHERE product_name = 'a' AND sales.buyer_id NOT IN (SELECT sales.buyer_id FROM product INNER JOIN sales ON sales.product_id = product.product_id WHERE product_name = 'b')"
    )
    right = (
        "WITH x AS (SELECT DISTINCT sales.buyer_id AS ba FROM sales LEFT JOIN product ON sales.product_id = product.product_id WHERE product_name = 'a'), "
        "y AS (SELECT DISTINCT sales.buyer_id AS bb FROM sales LEFT JOIN product ON sales.product_id = product.product_id WHERE product_name = 'b') "
        "SELECT x.ba AS buyer_id FROM x LEFT JOIN y ON y.bb = x.ba WHERE bb IS NULL"
    )
    schema = {"product": ["product_id", "product_name"], "sales": ["product_id", "buyer_id"]}
    constraints = {
        "product": TableConstraints(keys=(("product_id",),)),
        "sales": TableConstraints(not_null=frozenset({"buyer_id"}), keys=(("buyer_id",),)),
    }
    assert prove_equivalent_algebraic(left, right, schema=schema, dialect="mysql", constraints=constraints).proven


def test_a_not_in_proved_without_the_rule_is_not_rewritten_out_of_its_shape():
    """The second attempt (shape rules off) does not apply the rule either: it rewrote the NOT IN on one side only."""

    left = (
        "SELECT DISTINCT sales.buyer_id AS buyer_id FROM product INNER JOIN sales ON sales.product_id = product.product_id "
        "WHERE product_name = 'a' AND sales.buyer_id NOT IN (SELECT sales.buyer_id FROM product INNER JOIN sales ON sales.product_id = product.product_id WHERE product_name = 'b')"
    )
    right = (
        "WITH a_buyers AS (SELECT DISTINCT s.buyer_id, s.product_id FROM sales s JOIN product p ON s.product_id = p.product_id WHERE p.product_name = 'a') "
        "SELECT buyer_id FROM a_buyers WHERE buyer_id NOT IN (SELECT s.buyer_id FROM product p JOIN sales s ON p.product_id = s.product_id WHERE p.product_name = 'b')"
    )
    schema = {"product": ["product_id", "product_name"], "sales": ["product_id", "buyer_id"]}
    constraints = {
        "product": TableConstraints(keys=(("product_id",),)),
        "sales": TableConstraints(not_null=frozenset({"buyer_id"}), keys=(("buyer_id",),)),
    }
    assert prove_equivalent_algebraic(left, right, schema=schema, dialect="mysql", constraints=constraints).proven


def test_the_late_attempts_are_made_only_when_the_rule_would_change_a_query(monkeypatch):
    from kumo_sql_prover import shape_fallback

    attempts = []
    real = shape_fallback.late

    def counted():
        attempts.append(1)
        return real()

    monkeypatch.setattr(shape_fallback, "late", counted)
    prove = lambda sql: prove_equivalent_algebraic(sql, sql.replace("id = 1", "id = 2"), schema=SCHEMA, dialect="mysql", constraints=KEYED)  # noqa: E731
    assert not prove("SELECT id FROM students WHERE id = 1").proven
    assert not attempts  # nothing for the rule to rewrite: no extra attempt
    assert not prove(ROUNDABOUT.replace("SELECT s.id", "SELECT s.id").replace("WHERE NOT id IN", "WHERE id = 1 AND NOT id IN")).proven
    assert attempts  # the rule would rewrite the NOT IN: the late attempts are made
