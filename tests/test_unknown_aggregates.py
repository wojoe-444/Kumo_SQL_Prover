"""An aggregate the parser does not know is refused, never read as a function of one row (issue 73).

sqlglot parses a call it has no class for as an anonymous function, and the provers read an anonymous call as a
deterministic function of the row. For ``STD``, ``TOTAL`` or ``HLL_COUNT.INIT`` that made ``SELECT STD(x) FROM t``
one value per row of ``t`` where the engine returns one row.
"""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from kumo_sql_prover import ast_utils
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.ast_utils import UnmodeledConstruct, check_modeled, is_aggregate
from kumo_sql_prover.smt_equivalence import TableConstraints

# each engine's documented aggregates that sqlglot 30.21.0 parses, in that engine's dialect, as an anonymous call
BY_DIALECT = {
    "mysql": ["STD", "JSON_ARRAYAGG"],
    "bigquery": ["ST_UNION_AGG", "ST_CENTROID_AGG", "ST_EXTENT", "JSON_ARRAY_AGG"],
    "postgres": ["EVERY", "JSONB_AGG", "RANGE_AGG", "RANGE_INTERSECT_AGG", "XMLAGG", "JSON_ARRAYAGG"],
    "sqlite": ["TOTAL", "JSONB_GROUP_ARRAY", "JSONB_GROUP_OBJECT", "PERCENTILE"],
    "duckdb": [
        "ARBITRARY", "BITSTRING_AGG", "FAVG", "FSUM", "GEOMEAN", "GEOMETRIC_MEAN", "HISTOGRAM", "PRODUCT", "SUMKAHAN", "KAHAN_SUM",
        "WEIGHTED_AVG", "ENTROPY", "KURTOSIS_POP", "MAD", "RESERVOIR_QUANTILE", "SEM", "BOOL_XOR", "MEAN", "SUM_NO_OVERFLOW",
    ],
}
TWO_ARGUMENTS = {"JSONB_GROUP_OBJECT", "PERCENTILE", "WEIGHTED_AVG", "RESERVOIR_QUANTILE"}
NAMED = [(dialect, name) for dialect, names in BY_DIALECT.items() for name in names]
DOTTED = ["HLL_COUNT.INIT", "HLL_COUNT.MERGE", "HLL_COUNT.MERGE_PARTIAL", "KLL_QUANTILES.INIT_INT64", "KLL_QUANTILES.MERGE_FLOAT64", "KLL_QUANTILES.MERGE_PARTIAL"]

SCHEMA = {"t": ["id", "grp", "x"]}
TYPES = {"t": {"id": "INT", "grp": "INT", "x": "INT"}}
KEYED = {"t": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),))}


def _call(name: str) -> str:
    return f"{name}(x, grp)" if name in TWO_ARGUMENTS else f"{name}(x)"


def test_the_list_is_the_names_the_tests_hold_against_the_parser():
    assert ast_utils.UNKNOWN_AGGREGATES == {name for names in BY_DIALECT.values() for name in names}
    assert {name.split(".")[0] for name in DOTTED} == set(ast_utils.DOTTED_AGGREGATES)


@pytest.mark.parametrize("dialect, name", NAMED)
def test_the_parser_still_has_no_class_for_the_name(dialect, name):
    # when a sqlglot version classes one of these, this fails: take the name off the list, the class is read as an aggregate
    call = sqlglot.parse_one(f"SELECT {_call(name)} AS s FROM t", read=dialect).expressions[0].this
    assert isinstance(call, exp.Anonymous) and not isinstance(call, exp.AggFunc)
    assert is_aggregate(call)


@pytest.mark.parametrize("dialect, name", NAMED)
def test_a_query_that_calls_one_is_refused_in_every_dialect(dialect, name):
    for read in {dialect, "mysql", "bigquery"}:
        tree = sqlglot.parse_one(f"SELECT {_call(name)} AS s FROM t", read=read)
        if not any(isinstance(node, exp.Anonymous) for node in tree.walk()):
            continue  # this dialect's parser has a class for the name
        with pytest.raises(UnmodeledConstruct, match=name):
            check_modeled(tree)


@pytest.mark.parametrize("name", DOTTED)
def test_a_dotted_sketch_aggregate_is_refused(name):
    tree = sqlglot.parse_one(f"SELECT {name}(x) AS s FROM t", read="bigquery")
    assert is_aggregate(tree.expressions[0].this)
    with pytest.raises(UnmodeledConstruct, match=name.replace(".", r"\.")):
        check_modeled(tree)


@pytest.mark.parametrize("sql", ["SELECT HLL_COUNT.EXTRACT(x) AS n FROM t", "SELECT KLL_QUANTILES.EXTRACT_INT64(x, 2) AS q FROM t", "SELECT my_function(x) AS y FROM t", "SELECT other.INIT(x) AS y FROM t"])
def test_a_call_that_reads_one_row_is_not_refused(sql):
    tree = sqlglot.parse_one(sql, read="bigquery")
    assert check_modeled(tree) is not None
    assert not any(is_aggregate(node) for node in tree.walk())


# (dialect, a query, a query that returns other rows): every pair was proved before the fix
PAIRS = [
    ("mysql", "SELECT COUNT(*) AS n FROM (SELECT STD(x) AS s FROM t) AS d", "SELECT COUNT(*) AS n FROM t"),
    ("sqlite", "SELECT COUNT(*) AS n FROM (SELECT TOTAL(x) AS s FROM t) AS d", "SELECT COUNT(*) AS n FROM t"),
    ("mysql", "SELECT STD(x) AS s FROM t GROUP BY id", "SELECT STD(x) AS s FROM t"),
    ("mysql", "SELECT JSON_ARRAYAGG(x) AS s FROM t GROUP BY id", "SELECT JSON_ARRAYAGG(x) AS s FROM t"),
    ("sqlite", "SELECT id, TOTAL(x) AS s FROM t GROUP BY id", "SELECT id, TOTAL(x) AS s FROM t"),
    ("bigquery", "SELECT HLL_COUNT.INIT(x) AS s FROM t GROUP BY id", "SELECT HLL_COUNT.INIT(x) AS s FROM t"),
    ("bigquery", "SELECT ST_UNION_AGG(x) AS s FROM t GROUP BY id", "SELECT ST_UNION_AGG(x) AS s FROM t"),
    ("duckdb", "SELECT PRODUCT(x) AS s FROM t GROUP BY id", "SELECT PRODUCT(x) AS s FROM t"),
    ("postgres", "SELECT EVERY(x > 1) AS s FROM t GROUP BY id", "SELECT EVERY(x > 1) AS s FROM t"),
]


@pytest.mark.parametrize("dialect, left, right", PAIRS)
def test_pairs_that_differ_are_not_proved(dialect, left, right):
    result = prove_equivalent_algebraic(left, right, schema=SCHEMA, constraints=KEYED, types=TYPES, compare_names=False, dialect=dialect)
    assert not result.proven
    result = prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect=dialect)
    assert not result.proven


def test_the_engine_returns_one_row_where_the_row_reading_saw_one_per_row():
    import sqlite3

    database = sqlite3.connect(":memory:")
    database.execute("CREATE TABLE t (id INT, grp INT, x INT)")
    database.executemany("INSERT INTO t VALUES (?, ?, ?)", [(1, 1, 5), (2, 1, 7), (3, 2, 9)])
    assert database.execute(PAIRS[1][1]).fetchall() == [(1,)] and database.execute(PAIRS[1][2]).fetchall() == [(3,)]
    assert len(database.execute(PAIRS[4][1]).fetchall()) == 3 and len(database.execute(PAIRS[4][2]).fetchall()) == 1
