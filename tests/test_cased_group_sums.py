"""A CASE that picks whole groups of a pre-aggregated table moves onto the summed value (``kumo_sql_prover.cased_group_sums``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.cased_group_sums import unnest_cased_sums

SCHEMA = {"l": ["pk", "price", "qty"], "p": ["pk", "kind"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}

GROUPED = "(SELECT l.pk, SUM(l.price) AS total, COUNT(*) AS n, MAX(l.qty) AS top FROM l GROUP BY l.pk) AS d"


def flat(sql, dialect="postgres"):
    out = unnest_cased_sums(sqlglot.parse_one(sql, read=dialect))
    return out.sql(dialect=dialect) if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(f'{c} BIGINT' for c in columns)})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def test_the_case_moves_onto_the_value():
    out = flat(f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.total ELSE 0 END), SUM(d.total) FROM {GROUPED} JOIN p ON d.pk = p.pk")
    assert out is not None and "GROUP BY" not in out and "kumosql_cased" not in out
    tree = sqlglot.parse_one(out, read="postgres")
    first, second = (item.sql(dialect="postgres") for item in tree.expressions)
    source = next(t.alias_or_name for t in tree.find_all(exp_table()) if t.name == "l")
    assert first == f"SUM(CASE WHEN p.kind = 1 THEN {source}.price ELSE 0 END)"
    assert second == f"SUM({source}.price)"
    # no ELSE stays no ELSE, and a count of the group becomes 1 per row
    out = flat(f"SELECT p.kind, SUM(CASE WHEN d.pk > 2 THEN d.total END), SUM(CASE WHEN p.kind = 1 THEN d.n ELSE 0 END) FROM {GROUPED}, p WHERE d.pk = p.pk GROUP BY p.kind")
    tree = sqlglot.parse_one(out, read="postgres")
    source = next(t.alias_or_name for t in tree.find_all(exp_table()) if t.name == "l")
    assert [item.sql(dialect="postgres") for item in tree.expressions[1:]] == [
        f"SUM(CASE WHEN {source}.pk > 2 THEN {source}.price END)",
        "SUM(CASE WHEN p.kind = 1 THEN 1 ELSE 0 END)",
    ]


def exp_table():
    from sqlglot import exp

    return exp.Table


@pytest.mark.parametrize("sql", [
    f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.total ELSE 1 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",  # 1 per group is not 1 per row
    f"SELECT SUM(CASE WHEN d.total > 5 THEN d.total ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",  # the test reads a total
    f"SELECT SUM(CASE WHEN d.n > 1 THEN d.total ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.top ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",  # a MAX is not summed row by row
    f"SELECT SUM(CASE WHEN p.kind = 1 THEN p.pk ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",  # not the grouped table's value
    f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.total WHEN p.kind = 2 THEN d.total ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    f"SELECT SUM(CASE p.kind WHEN 1 THEN d.total ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.total + 1 ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    f"SELECT SUM(DISTINCT CASE WHEN p.kind = 1 THEN d.total ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    f"SELECT AVG(CASE WHEN p.kind = 1 THEN d.total ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.total ELSE 0 END), COUNT(*) FROM {GROUPED} JOIN p ON d.pk = p.pk",  # a count of groups
    f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.total ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk WHERE d.total > 0",  # a filter on a total
    f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.total ELSE 0 END) FROM {GROUPED} LEFT JOIN p ON d.pk = p.pk",
    f"SELECT SUM(CASE WHEN RANDOM() < 0.5 THEN d.total ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    f"SELECT SUM(CASE WHEN p.kind IN (SELECT x.kind FROM p AS x) THEN d.total ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    "SELECT SUM(CASE WHEN p.kind = 1 THEN l.price ELSE 0 END) FROM l JOIN p ON l.pk = p.pk",  # nothing grouped
    f"SELECT d.pk, CASE WHEN p.kind = 1 THEN d.total ELSE 0 END FROM {GROUPED} JOIN p ON d.pk = p.pk",  # no sum
])
def test_shapes_left_alone(sql):
    assert flat(sql) is None


def test_proofs_and_near_misses():
    flat_sql = "SELECT SUM(CASE WHEN p.kind = 1 THEN l.price ELSE 0 END), SUM(l.price) FROM l, p WHERE l.pk = p.pk"
    grouped = f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.total ELSE 0 END), SUM(d.total) FROM {GROUPED} JOIN p ON d.pk = p.pk"
    assert proved(flat_sql, grouped) and proved(grouped, flat_sql)
    by_kind = "SELECT p.kind, SUM(CASE WHEN l.pk > 2 THEN l.price END) FROM l, p WHERE l.pk = p.pk GROUP BY p.kind"
    assert proved(by_kind, f"SELECT p.kind, SUM(CASE WHEN d.pk > 2 THEN d.total END) FROM {GROUPED}, p WHERE d.pk = p.pk GROUP BY p.kind")

    # ELSE 1 counts a group once on one side and each of its rows on the other
    one, two = (
        "SELECT SUM(CASE WHEN p.kind = 1 THEN l.price ELSE 1 END) FROM l, p WHERE l.pk = p.pk",
        f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.total ELSE 1 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    )
    first, second = rows_on_duckdb([one, two], {"l": [(1, 5, 0), (1, 6, 0)], "p": [(1, 2)]})
    assert first != second
    assert not proved(one, two) and not proved(two, one)
    # a test on the group's total is not a test on each row's value
    one, two = (
        "SELECT SUM(CASE WHEN l.price > 5 THEN l.price ELSE 0 END) FROM l, p WHERE l.pk = p.pk",
        f"SELECT SUM(CASE WHEN d.total > 5 THEN d.total ELSE 0 END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    )
    first, second = rows_on_duckdb([one, two], {"l": [(1, 3, 0), (1, 4, 0)], "p": [(1, 2)]})
    assert first != second
    assert not proved(one, two) and not proved(two, one)
    # another condition
    other = f"SELECT SUM(CASE WHEN p.kind = 2 THEN d.total ELSE 0 END), SUM(d.total) FROM {GROUPED} JOIN p ON d.pk = p.pk"
    first, second = rows_on_duckdb([flat_sql, other], {"l": [(1, 3, 0)], "p": [(1, 1)]})
    assert first != second
    assert not proved(flat_sql, other) and not proved(other, flat_sql)


@pytest.mark.parametrize("sql", [
    f"SELECT SUM(CASE WHEN p.kind = 1 THEN d.total ELSE 0 END), SUM(d.total), SUM(CASE WHEN p.kind = 2 THEN d.n END) FROM {GROUPED} JOIN p ON d.pk = p.pk",
    f"SELECT p.kind, SUM(CASE WHEN d.pk > 1 THEN d.total END), SUM(CASE WHEN p.kind = 1 THEN d.n ELSE 0 END), MAX(d.top) FROM {GROUPED}, p WHERE d.pk = p.pk GROUP BY p.kind",
])
def test_the_rewrite_returns_the_same_rows(sql):
    after = flat(sql)
    assert after is not None
    rng = random.Random(5)
    value = lambda: rng.choice([None, 0, 1, 2, 3])
    for _ in range(40):
        rows = {
            "l": [(value(), value(), value()) for _ in range(rng.randint(0, 6))],
            "p": [(value(), value()) for _ in range(rng.randint(0, 4))],
        }
        first, second = rows_on_duckdb([sql, after], rows)
        assert first == second, rows
