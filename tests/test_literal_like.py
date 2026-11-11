"""``x LIKE 'abc'`` read as ``x = 'abc'`` (``literal_like``), where the two compare strings the same way."""

import pytest

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic

SCHEMA = {"s": ["id", "name", "code"]}
TYPES = {"s": {"id": "INTEGER", "name": "VARCHAR", "code": "CHAR(5)"}}


def proven(left, right, dialect="bigquery", types=None):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=types, dialect=dialect, compare_names=False).proven


@pytest.mark.parametrize("dialect, types", [("bigquery", None), ("duckdb", None), ("postgres", TYPES)])
def test_a_pattern_without_wildcards_is_an_equality(dialect, types):
    assert proven("SELECT id FROM s WHERE name LIKE 'abc'", "SELECT id FROM s WHERE name = 'abc'", dialect, types)
    assert proven("SELECT id FROM s WHERE name NOT LIKE 'abc'", "SELECT id FROM s WHERE name <> 'abc'", dialect, types)
    assert proven("SELECT id FROM s WHERE name LIKE ''", "SELECT id FROM s WHERE name = ''", dialect, types)
    assert proven(
        "SELECT a.id FROM s AS a JOIN s AS b ON a.id = b.id WHERE b.name LIKE 'x y'",
        "SELECT a.id FROM s AS a JOIN s AS b ON a.id = b.id WHERE b.name = 'x y'",
        dialect, types,
    )


NEAR_MISSES = [
    ("SELECT id FROM s WHERE name LIKE 'abc%'", "SELECT id FROM s WHERE name = 'abc%'"),
    ("SELECT id FROM s WHERE name LIKE 'a_c'", "SELECT id FROM s WHERE name = 'a_c'"),
    ("SELECT id FROM s WHERE name LIKE 'a\\\\%c'", "SELECT id FROM s WHERE name = 'a\\\\%c'"),
    ("SELECT id FROM s WHERE name LIKE 'abc'", "SELECT id FROM s WHERE name = 'ABC'"),
    ("SELECT id FROM s WHERE name LIKE 'abc'", "SELECT id FROM s WHERE name = 'abc '"),
    ("SELECT id FROM s WHERE name LIKE 'abc'", "SELECT id FROM s WHERE name = 'abc' OR name IS NULL"),
    ("SELECT id FROM s WHERE name NOT LIKE 'abc'", "SELECT id FROM s WHERE name <> 'abc' OR name IS NULL"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
@pytest.mark.parametrize("dialect, types", [("bigquery", None), ("postgres", TYPES)])
def test_near_misses_are_not_proved(left, right, dialect, types):
    assert not proven(left, right, dialect, types)


def test_the_rule_is_not_applied_where_like_and_equality_compare_differently():
    left, right = "SELECT id FROM s WHERE name LIKE 'abc'", "SELECT id FROM s WHERE name = 'abc'"
    assert not proven(left, right, "mysql", TYPES)  # = ignores trailing spaces under a PAD SPACE collation
    assert not proven(left, right, "sqlite", TYPES)  # LIKE ignores ASCII case
    assert not proven(left, right, "postgres", None)  # the column may be CHAR(n)
    padded = "SELECT id FROM s WHERE code LIKE 'abc'", "SELECT id FROM s WHERE code = 'abc'"
    assert not proven(*padded, "postgres", TYPES)
    derived = "SELECT id FROM (SELECT id, code AS name FROM s) AS d WHERE name LIKE 'abc'"
    assert "LIKE" in normalize(derived, schema=SCHEMA, types=TYPES, dialect="postgres").upper()


def test_ilike_and_escape_are_left_alone():
    assert "ILIKE" in normalize("SELECT id FROM s WHERE name ILIKE 'abc'", schema=SCHEMA, dialect="duckdb").upper()
    assert "LIKE" in normalize("SELECT id FROM s WHERE name LIKE 'a!%c' ESCAPE '!'", schema=SCHEMA, dialect="duckdb").upper()
    assert "LIKE" in normalize("SELECT id FROM s WHERE name LIKE code", schema=SCHEMA, dialect="duckdb").upper()


def test_duckdb_agrees_on_every_string_shape():
    duckdb = pytest.importorskip("duckdb")
    db = duckdb.connect()
    db.execute("CREATE TABLE s (id INT, name VARCHAR)")
    db.execute("INSERT INTO s VALUES (1, 'abc'), (2, 'ABC'), (3, 'abc '), (4, NULL), (5, ''), (6, 'abcd'), (7, ' abc')")
    for pattern in ("abc", "", "ABC", "abc "):
        like = db.execute(f"SELECT id FROM s WHERE name LIKE '{pattern}' ORDER BY id").fetchall()
        equal = db.execute(f"SELECT id FROM s WHERE name = '{pattern}' ORDER BY id").fetchall()
        assert like == equal
