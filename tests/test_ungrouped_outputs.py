"""A grouping with no aggregate that shows a column outside its ``GROUP BY`` is not the ``DISTINCT`` of its outputs.

MySQL and SQLite accept ``SELECT id, label FROM t GROUP BY id`` and return one row per ``id`` with the label of some
row of the group. ``SELECT DISTINCT id, label FROM t`` returns one row per pair. The solver read a grouping without
aggregates as ``DISTINCT`` as soon as its outputs determined its group keys (``_Prover.as_distinct_spj``), which is
half of what that needs: the keys must determine the outputs too. So it proved the two selects equal on any table,
and with them every query built on such a pair. The made-up rows below hold one ``id`` with two labels; each pair is
run on SQLite, where both sides are accepted, to show that the answers differ in their number of rows whichever
label the engine picks.
"""

from __future__ import annotations

import sqlite3

import pytest

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, TableConstraints, prove_equivalent_smt

SCHEMA = {"items": ["id", "label", "shelf"], "picks": ["item", "n"]}
TYPES = {"items": {"id": "INT", "label": "INT", "shelf": "INT"}, "picks": {"item": "INT", "n": "INT"}}
ROWS = {"items": [(1, 5, 1), (1, 6, 2), (2, 7, 1)], "picks": [(5, 10), (6, 1), (1, 2), (1, 3)]}
KEYED = {"items": TableConstraints(keys=(("id",),), not_null=frozenset({"id"}))}
NULLABLE_KEY = {"items": TableConstraints(keys=(("id",),))}

# (grouped, the DISTINCT it was read as, rows of each on the made-up tables)
PAIRS = [
    ("SELECT id, label FROM items GROUP BY id", "SELECT DISTINCT id, label FROM items", 2, 3),
    ("SELECT id, label FROM items GROUP BY id", "SELECT id, label FROM items GROUP BY id, label", 2, 3),
    ("SELECT DISTINCT id, label FROM items GROUP BY id", "SELECT DISTINCT id, label FROM items", 2, 3),
    ("SELECT i.id, i.label FROM items AS i WHERE i.shelf > 0 GROUP BY i.id", "SELECT DISTINCT i.id, i.label FROM items AS i WHERE i.shelf > 0", 2, 3),
    ("SELECT id, label + 1 AS next FROM items GROUP BY id", "SELECT DISTINCT id, label + 1 AS next FROM items", 2, 3),
    ("SELECT i.id, i.label FROM items AS i JOIN picks AS p ON p.item = i.id GROUP BY i.id", "SELECT DISTINCT i.id, i.label FROM items AS i JOIN picks AS p ON p.item = i.id", 1, 2),
    ("SELECT i.id, p.n FROM items AS i JOIN picks AS p ON p.item = i.id GROUP BY i.id", "SELECT DISTINCT i.id, p.n FROM items AS i JOIN picks AS p ON p.item = i.id", 1, 2),
    ("SELECT i.id, i.label FROM items AS i JOIN picks AS p ON p.item = i.label GROUP BY i.id", "SELECT DISTINCT i.id, i.label FROM items AS i JOIN picks AS p ON p.item = i.label", 1, 2),
    ("SELECT COUNT(*) AS n FROM (SELECT id, label FROM items GROUP BY id) AS t", "SELECT COUNT(*) AS n FROM (SELECT DISTINCT id, label FROM items) AS t", None, None),
    (
        "SELECT t.id, t.label FROM (SELECT id, label FROM items GROUP BY id) AS t JOIN picks AS p ON p.item = t.id",
        "SELECT t.id, t.label FROM (SELECT DISTINCT id, label FROM items) AS t JOIN picks AS p ON p.item = t.id",
        2,
        4,
    ),
    ("SELECT id, label FROM items GROUP BY id UNION SELECT item, n FROM picks", "SELECT id, label FROM items UNION SELECT item, n FROM picks", 6, 7),
]


def _rows(sql: str) -> list:
    database = sqlite3.connect(":memory:")
    for name, columns in SCHEMA.items():
        database.execute(f"CREATE TABLE {name} ({', '.join(c + ' INT' for c in columns)})")
        database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(columns))})", ROWS[name])
    return sorted(database.execute(sql).fetchall(), key=repr)


@pytest.mark.parametrize("grouped, distinct, few, many", PAIRS)
def test_the_two_selects_differ_on_the_made_up_rows(grouped, distinct, few, many):
    left, right = _rows(grouped), _rows(distinct)
    if few is None:
        assert (left, right) == ([(2,)], [(3,)])
    else:
        assert (len(left), len(right)) == (few, many)


@pytest.mark.parametrize("grouped, distinct, few, many", PAIRS)
@pytest.mark.parametrize("dialect", ["mysql", "sqlite"])
def test_the_pair_is_not_proved(grouped, distinct, few, many, dialect):
    options = dict(schema=SCHEMA, constraints={}, compare_names=False, dialect=dialect)
    assert not prove_equivalent_algebraic(grouped, distinct, types=TYPES, **options).proven
    assert prove_equivalent_smt(grouped, distinct, **options).status is not SmtStatus.PROVEN_EQUIVALENT
    # a key that may be NULL does not fix the label either: two rows without an id are one group
    assert not prove_equivalent_algebraic(grouped, distinct, types=TYPES, **{**options, "constraints": NULLABLE_KEY}).proven


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5])
def test_under_a_not_null_key_the_group_is_one_row_and_the_pair_is_proved(index):
    grouped, distinct, _, _ = PAIRS[index]
    assert prove_equivalent_algebraic(grouped, distinct, schema=SCHEMA, types=TYPES, constraints=KEYED, compare_names=False, dialect="mysql").proven


def test_outputs_that_are_the_group_keys_are_still_read_as_distinct():
    options = dict(schema=SCHEMA, types=TYPES, constraints={}, compare_names=False, dialect="mysql")
    assert prove_equivalent_algebraic("SELECT id, label FROM items GROUP BY id, label", "SELECT DISTINCT id, label FROM items", **options).proven
    assert prove_equivalent_algebraic("SELECT id, id + 1 AS n FROM items GROUP BY id", "SELECT DISTINCT id, id + 1 AS n FROM items", **options).proven
    assert prove_equivalent_algebraic("SELECT shelf FROM items WHERE id > 1 GROUP BY shelf", "SELECT DISTINCT shelf FROM items WHERE id > 1", **options).proven
