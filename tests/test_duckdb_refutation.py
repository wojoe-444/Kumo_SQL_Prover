"""The executed search for DuckDB-dialect pairs: a counterexample is two runs of the queries as written."""

import pytest

pytest.importorskip("z3")
duckdb = pytest.importorskip("duckdb")

from kumo_sql_prover import duckdb_refutation
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, TableConstraints

SCHEMA = {"t": ["a", "b"], "u": ["a", "c"]}
TYPES = {"t": {"a": "INT", "b": "INT"}, "u": {"a": "INT", "c": "VARCHAR"}}


def search(left, right, **kwargs):
    kwargs.setdefault("schema", SCHEMA)
    kwargs.setdefault("types", TYPES)
    return duckdb_refutation.search(left, right, **kwargs)


def replay(found, left, right, types=TYPES):
    """Both queries' rows on the counterexample, on a fresh DuckDB with the declared types."""

    db = duckdb.connect()
    for table, rows in found.tables.items():
        db.execute(f"CREATE TABLE {table} ({', '.join(f'{c} {t}' for c, t in types[table].items())})")
        for row in rows:
            db.execute(f"INSERT INTO {table} VALUES ({', '.join('?' * len(row))})", [row[c] for c in types[table]])
    return sorted(db.execute(left).fetchall(), key=repr), sorted(db.execute(right).fetchall(), key=repr)


DIFFERENT = [
    ("SELECT a FROM t WHERE b > 1", "SELECT a FROM t WHERE b >= 1"),
    ("SELECT a FROM t WHERE a NOT IN (SELECT a FROM u)", "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.a = t.a)"),
    ("SELECT t.a FROM t JOIN u ON t.a = u.a", "SELECT t.a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a)"),
    ("SELECT t.a, u.c FROM t LEFT JOIN u ON t.a = u.a", "SELECT t.a, u.c FROM t JOIN u ON t.a = u.a"),
    ("SELECT COUNT(*) FROM t", "SELECT COUNT(b) FROM t"),
    ("SELECT a FROM t UNION SELECT a FROM u", "SELECT a FROM t UNION ALL SELECT a FROM u"),
    ("SELECT a, SUM(b) OVER (PARTITION BY a) FROM t", "SELECT a, SUM(b) OVER () FROM t"),
    ("SELECT a FROM t WHERE b = CAST(0.5 AS DECIMAL(4,2))", "SELECT a FROM t WHERE b = 0"),
    ("SELECT a FROM t WHERE a = b", "SELECT a FROM t WHERE a IS NOT DISTINCT FROM b"),
]


@pytest.mark.parametrize("left, right", DIFFERENT)
def test_a_pair_that_differs_is_refuted_by_a_database_that_replays(left, right):
    found = search(left, right)
    assert found is not None
    assert replay(found, left, right) == (found.left_rows, found.right_rows) and found.left_rows != found.right_rows


SAME = [
    ("SELECT a FROM t WHERE b > 1", "SELECT a FROM t WHERE NOT b <= 1"),
    ("SELECT a FROM t WHERE b > 1 AND a IS NOT NULL", "SELECT a FROM t WHERE a IS NOT NULL AND 1 < b"),
    ("SELECT t.a FROM t JOIN u ON t.a = u.a", "SELECT t.a FROM u JOIN t ON u.a = t.a"),
    ("SELECT a, COUNT(*) FROM t GROUP BY a", "SELECT a, COUNT(*) FROM t GROUP BY 1"),
    ("SELECT DISTINCT a FROM t", "SELECT a FROM t GROUP BY a"),
    ("SELECT a FROM t ORDER BY a", "SELECT a FROM t ORDER BY b"),  # the same bag: row order is not compared
    ("SELECT a FROM t EXCEPT SELECT a FROM u", "SELECT DISTINCT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.a IS NOT DISTINCT FROM t.a)"),
]


@pytest.mark.parametrize("left, right", SAME)
def test_an_equivalent_pair_is_never_refuted(left, right):
    assert search(left, right) is None


NOT_RUN = [
    # LIMIT, OFFSET and FETCH: without a total order the engine picks the rows
    ("SELECT a FROM t ORDER BY b LIMIT 1", "SELECT a FROM t ORDER BY b DESC LIMIT 1"),
    ("SELECT a FROM t ORDER BY b OFFSET 1 ROWS FETCH NEXT 1 ROWS ONLY", "SELECT a FROM t"),
    ("SELECT a FROM (SELECT a FROM t LIMIT 1)", "SELECT a FROM t WHERE FALSE"),
    # DISTINCT ON keeps an arbitrary row of each group
    ("SELECT DISTINCT ON (a) a, b FROM t", "SELECT a, b FROM t WHERE FALSE"),
    # a ROWS frame, a ranking or a navigation function depends on the order of tied rows
    ("SELECT SUM(b) OVER (ORDER BY a ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t", "SELECT b FROM t WHERE FALSE"),
    ("SELECT ROW_NUMBER() OVER (ORDER BY a) FROM t", "SELECT b FROM t WHERE FALSE"),
    ("SELECT FIRST_VALUE(b) OVER (PARTITION BY a) FROM t", "SELECT b FROM t WHERE FALSE"),
    # functions outside the list: random, the clock, an arbitrary pick, a file
    ("SELECT a FROM t WHERE random() < 2", "SELECT a FROM t WHERE FALSE"),
    ("SELECT a FROM t WHERE current_date > DATE '2000-01-01'", "SELECT a FROM t WHERE FALSE"),
    ("SELECT ANY_VALUE(b) FROM t", "SELECT b FROM t WHERE FALSE"),
    ("SELECT a FROM t", "SELECT a FROM read_csv('missing.csv')"),
    # a table that is not declared (DuckDB would look for a file or a variable of that name)
    ("SELECT a FROM t", "SELECT a FROM elsewhere"),
    ("SELECT a FROM t", "SELECT a FROM 'data.csv'"),
    ("SELECT a FROM t", "SELECT a FROM main.t WHERE FALSE"),
    # not one query
    ("SELECT a FROM t; SELECT 1", "SELECT a FROM t WHERE FALSE"),
    ("DELETE FROM t", "SELECT a FROM t"),
]


@pytest.mark.parametrize("left, right", NOT_RUN)
def test_a_query_whose_rows_the_engine_may_pick_is_not_run(left, right):
    assert search(left, right) is None
    assert search(right, left) is None


def test_no_search_without_a_declared_type_for_every_column_read():
    left, right = "SELECT a FROM t WHERE b > 1", "SELECT a FROM t"
    assert search(left, right, types={"t": {"a": "INT"}}) is None
    assert search(left, right, types=None) is None
    assert search(left, right, types={"t": {"a": "INT", "b": "GEOMETRY"}}) is None
    assert search(left, right, types={"t": {"a": "INT", "b": "INT); DROP TABLE t; --"}}) is None


def test_the_counterexample_respects_declared_keys_and_not_null_columns():
    left, right = "SELECT COUNT(*) FROM t", "SELECT COUNT(DISTINCT a) FROM t"
    assert search(left, right) is not None
    keyed = {"t": TableConstraints(not_null=frozenset({"a"}), keys=(("a",),))}
    assert search(left, right, constraints=keyed) is None
    left, right = "SELECT a FROM t WHERE b IS NULL", "SELECT a FROM t WHERE FALSE"
    assert search(left, right) is not None
    assert search(left, right, constraints={"t": TableConstraints(not_null=frozenset({"b"}))}) is None


def test_values_are_loaded_into_the_declared_types():
    # 300 does not fit a UTINYINT and no negative number does: the pair is equal on every database of that type
    types = {"t": {"a": "INT", "b": "UTINYINT"}}
    assert search("SELECT a FROM t WHERE b >= 0 AND b < 300", "SELECT a FROM t WHERE b IS NOT NULL", types=types) is None
    found = search("SELECT a FROM t WHERE b > 1", "SELECT a FROM t WHERE b IS NOT NULL", types=types)
    assert found is not None and all(row["b"] is None or row["b"] >= 0 for row in found.tables["t"])


def test_a_pair_that_reads_no_table_is_decided_by_one_run():
    assert search("SELECT 1 WHERE 1 = 2", "SELECT 1", schema=None, types=None) is not None
    assert search("SELECT 1 + 1", "SELECT 2", schema=None, types=None) is None


def test_a_difference_that_depends_on_row_order_is_not_a_refutation():
    types = {"t": {"a": "DOUBLE", "b": "INT"}}
    # a float sum can differ in its last digits between two plans, never beyond six
    assert search("SELECT SUM(a) FROM t", "SELECT SUM(a + 0) FROM t", types=types) is None


def test_the_prover_reports_the_database_as_a_refutation_only_when_asked():
    left, right = "SELECT t.a FROM t LEFT JOIN u ON t.a = u.a", "SELECT t.a FROM t JOIN u ON t.a = u.a"
    plain = prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="duckdb")
    assert plain.status is SmtStatus.NOT_PROVEN
    searched = prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="duckdb", search_counterexample=True)
    assert searched.status is SmtStatus.NOT_EQUIVALENT and searched.counterexample is not None
    assert replay(searched.counterexample, left, right)[0] != replay(searched.counterexample, left, right)[1]


def test_a_proof_is_not_searched_and_other_dialects_are_left_alone():
    left, right = "SELECT a FROM t WHERE b > 1", "SELECT a FROM t WHERE 1 < b"
    result = prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="duckdb", search_counterexample=True)
    assert result.status is SmtStatus.PROVEN_EQUIVALENT
    left, right = "SELECT t.a FROM t LEFT JOIN u ON t.a = u.a", "SELECT t.a FROM t JOIN u ON t.a = u.a"
    for dialect in ("mysql", "postgres", "sqlite"):
        result = prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect=dialect, search_counterexample=True)
        assert result.status is SmtStatus.NOT_PROVEN
