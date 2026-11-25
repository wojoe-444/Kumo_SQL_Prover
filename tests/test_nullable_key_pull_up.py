"""A keyed table joined to a pre-aggregated one is read as one flat grouping only under a key that cannot be NULL.

``eager_aggregation.pull_up_aggregate`` adds the table's key to the ``GROUP BY`` so that every row of the table is
its own group. A declared key says nothing about rows that hold NULL in it (a UNIQUE column admits several), so two
such rows that agree on the columns read fall into one group, while the join with the pre-aggregated table lists
both. The rule used any declared key and proved such a pair; it now needs every key column to be NOT NULL or to be
the column the join equates with a group key.
"""

from __future__ import annotations

import sqlite3

import pytest
import sqlglot

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.eager_aggregation import pull_up_aggregate
from kumo_sql_prover.smt_equivalence import SmtStatus, TableConstraints

SCHEMA = {"clients": ["id", "town"], "orders": ["town", "paid"]}
TYPES = {"clients": {"id": "INT", "town": "INT"}, "orders": {"town": "INT", "paid": "INT"}}
ROWS = {"clients": [(None, 5), (None, 5), (1, 6), (2, None)], "orders": [(5, 10), (6, 1), (6, 2), (None, 9)]}
NULLABLE = {"clients": TableConstraints(keys=(("id",),))}
NEVER_NULL = {"clients": TableConstraints(keys=(("id",),), not_null=frozenset({"id"}))}

# pre-aggregated orders joined on a column that is not the key, and the flat grouping the rule writes
JOINED = "SELECT c.town, g.total FROM clients AS c JOIN (SELECT o.town, SUM(o.paid) AS total FROM orders AS o GROUP BY o.town) AS g ON c.town = g.town"
FLAT = "SELECT c.town, SUM(o.paid) AS total FROM clients AS c JOIN orders AS o ON c.town = o.town GROUP BY c.id, c.town"
# the same pair joined on the key itself: a row with a NULL key meets nothing
JOINED_ON_KEY = JOINED.replace("ON c.town = g.town", "ON c.id = g.town")
FLAT_ON_KEY = FLAT.replace("ON c.town = o.town", "ON c.id = o.town")


def _rows(sql: str, dialect: str, rows=ROWS) -> list:
    database = pytest.importorskip("duckdb").connect() if dialect == "duckdb" else sqlite3.connect(":memory:")
    for name, columns in SCHEMA.items():
        database.execute(f"CREATE TABLE {name} ({', '.join(c + ' INT' for c in columns)})")
        database.executemany(f"INSERT INTO {name} VALUES (?, ?)", rows[name])
    return sorted(database.execute(sql).fetchall(), key=repr)


def _prove(left: str, right: str, constraints, **options):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=constraints, compare_names=False, **options)


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_two_rows_without_a_key_value_are_listed_twice_and_grouped_once(dialect):
    assert _rows(JOINED, dialect) == [(5, 10), (5, 10), (6, 3)]
    assert _rows(FLAT, dialect) == [(5, 20), (6, 3)]
    # joined on the key, the two spellings agree on the same rows
    assert _rows(JOINED_ON_KEY, dialect) == _rows(FLAT_ON_KEY, dialect)


@pytest.mark.parametrize("dialect", ["mysql", "sqlite", "duckdb", "bigquery", "postgres"])
def test_the_pair_is_not_proved_under_a_key_that_may_be_null(dialect):
    assert not _prove(JOINED, FLAT, NULLABLE, dialect=dialect).proven
    assert _prove(JOINED, FLAT, NEVER_NULL, dialect=dialect).proven


@pytest.mark.parametrize("constraints", [NULLABLE, NEVER_NULL])
def test_a_key_the_join_equates_is_enough(constraints):
    assert _prove(JOINED_ON_KEY, FLAT_ON_KEY, constraints, dialect="duckdb").proven


def test_half_of_a_two_column_key_is_not_enough():
    # (id, town) unique and nullable: two rows (1, NULL) are two rows, joined on id both meet the group
    two = {"clients": TableConstraints(keys=(("id", "town"),))}
    assert not _prove(JOINED_ON_KEY, FLAT_ON_KEY, two, dialect="duckdb").proven
    rows = {"clients": [(1, None), (1, None)], "orders": [(1, 4)]}
    assert _rows(JOINED_ON_KEY, "sqlite", rows) == [(None, 4), (None, 4)]
    assert _rows(FLAT_ON_KEY, "sqlite", rows) == [(None, 8)]


def test_the_conditions_it_is_equal_under_name_the_not_null_column():
    result = _prove(JOINED, FLAT, {}, dialect="duckdb", conditional=True)
    assert result.status is SmtStatus.PROVEN_CONDITIONALLY
    assert any(c.kind == "not_null" and c.columns == ("id",) for c in result.conditions)


def test_the_rule_reads_the_join_only_with_such_a_key():
    select = sqlglot.parse_one(JOINED, read="duckdb")
    keys = {"clients": [("id",)]}
    assert pull_up_aggregate(select, keys) is None
    assert pull_up_aggregate(select, keys, {"clients": frozenset()}) is None
    assert pull_up_aggregate(select, keys, {"clients": frozenset({"town"})}) is None
    assert pull_up_aggregate(select, keys, {"CLIENTS": frozenset({"ID"})}) is not None
    on_key = sqlglot.parse_one(JOINED_ON_KEY, read="duckdb")
    assert pull_up_aggregate(on_key, keys) is not None
    assert pull_up_aggregate(on_key, {"clients": [("id", "town")]}) is None
    assert pull_up_aggregate(on_key, {"clients": [("id", "town")]}, {"clients": frozenset({"town"})}) is not None
