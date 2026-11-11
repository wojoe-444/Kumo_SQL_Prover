"""A ``UNION`` of tagged branches that each list a NOT NULL key is a ``UNION ALL`` (``keyed_union``)."""

import pytest
import sqlglot

from kumo_sql_prover.keyed_union import separate_keyed_unions

KEYS = {"products": [("product_id",)], "stock": [("shop", "item")]}
NOT_NULL = {"products": frozenset({"product_id"}), "stock": frozenset({"shop", "item"})}
BRANCH = "SELECT product_id, '{tag}' AS store, {column} AS price FROM products"
UNPIVOT = " UNION ".join(BRANCH.format(tag=f"store{n}", column=f"store{n}") for n in (1, 2, 3))
TWO = " UNION ".join(BRANCH.format(tag=f"store{n}", column=f"store{n}") for n in (1, 2))


def marked(sql: str, keys=KEYS, not_null=NOT_NULL) -> str:
    return separate_keyed_unions(sqlglot.parse_one(sql, read="mysql"), keys, not_null).sql(dialect="mysql")


def test_tagged_branches_that_list_a_key_need_no_duplicate_removal():
    assert marked(UNPIVOT) == UNPIVOT.replace("UNION", "UNION ALL")
    filtered = UNPIVOT.replace("FROM products", "FROM products WHERE price > 0")
    assert marked(filtered) == filtered.replace("UNION", "UNION ALL")
    # a composite key, one column of it fixed by the WHERE clause; a one-row branch from ``disjoint_union``
    composite = "SELECT item, 1 AS tag FROM stock WHERE shop = 'north' UNION SELECT shop, 2 FROM stock WHERE item = 'pen' UNION SELECT MAX(item), 3 FROM stock"
    assert marked(composite).count("UNION ALL") == 2
    # a union is read with the unions under it: the first two branches qualify on their own, the third does not
    mixed = TWO + " UNION SELECT store1, 'store3', store3 FROM products"
    assert marked(mixed) == TWO.replace("UNION", "UNION ALL") + " UNION SELECT store1, 'store3', store3 FROM products"


@pytest.mark.parametrize(
    "sql, keys, not_null",
    [
        (TWO, None, NOT_NULL),  # no key declared
        (TWO, KEYS, {}),  # the key may be NULL: two rows with a NULL id and one price are the same row
        (TWO.replace("'store2'", "'store1'"), KEYS, NOT_NULL),  # both branches carry the same tag
        (TWO.replace("'store2'", "'STORE 1'"), KEYS, NOT_NULL),  # tags a collation could call equal
        (TWO.replace("SELECT product_id, 'store2'", "SELECT store1, 'store2'"), KEYS, NOT_NULL),  # a branch without the key
        (TWO.replace("SELECT product_id, 'store2'", "SELECT product_id + 0, 'store2'"), KEYS, NOT_NULL),  # not the column itself
        (TWO.replace("FROM products", "FROM products JOIN stock ON TRUE", 1), KEYS, NOT_NULL),  # a join repeats rows
        (TWO.replace("SELECT product_id, 'store2'", "SELECT DISTINCT product_id, 'store2'"), KEYS, NOT_NULL),  # left to the set comparison
        ("SELECT item, 1 AS tag FROM stock UNION SELECT shop, 2 FROM stock", KEYS, NOT_NULL),  # half of a composite key
        (TWO.replace("UNION", "INTERSECT"), KEYS, NOT_NULL),
        (TWO + " LIMIT 2", KEYS, NOT_NULL),
    ],
)
def test_other_unions_keep_their_duplicate_removal(sql, keys, not_null):
    assert "UNION ALL" not in marked(sql, keys, not_null)


def test_on_duckdb_the_two_differ_exactly_when_the_key_fails():
    duckdb = pytest.importorskip("duckdb")
    union, union_all = UNPIVOT, UNPIVOT.replace("UNION", "UNION ALL")

    def rows(data):
        con = duckdb.connect()
        con.execute("CREATE TABLE products (product_id INT, store1 INT, store2 INT, store3 INT)")
        con.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", data)
        return [sorted(con.execute(sql).fetchall(), key=repr) for sql in (union, union_all)]

    a, b = rows([(1, 5, 5, None), (2, 5, None, 7), (3, None, None, None)])
    assert a == b
    a, b = rows([(1, 5, 6, 7), (1, 5, 8, 9)])  # the id repeats
    assert a != b
    a, b = rows([(None, 5, 6, 7), (None, 5, 8, 9)])  # unique as SQL counts it, but NULL
    assert a != b


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"products": ["product_id", "store1", "store2", "store3"]}
WRAPPED = f"SELECT * FROM ({UNPIVOT}) AS t WHERE price IS NOT NULL"
FLAT = " UNION ALL ".join(BRANCH.format(tag=f"store{n}", column=f"store{n}") + f" WHERE store{n} IS NOT NULL" for n in (1, 2, 3))


def prove(left, right, **kwargs):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, dialect="mysql", compare_names=False, **kwargs)


def test_the_pair_proves_only_with_the_key():
    key = {"products": TableConstraints(not_null=frozenset({"product_id"}), keys=(("product_id",),))}
    assert prove(WRAPPED, FLAT, constraints=key).proven
    assert not prove(WRAPPED, FLAT).proven
    assert not prove(WRAPPED, FLAT, constraints={"products": TableConstraints(not_null=frozenset(), keys=(("product_id",),))}).proven
    assert not prove(WRAPPED, FLAT, constraints={"products": TableConstraints(not_null=frozenset({"product_id"}), keys=())}).proven
    assert not prove(WRAPPED, FLAT.replace("'store3'", "'store2'"), constraints=key).proven


def test_the_conditional_verdict_names_the_key():
    result = prove(WRAPPED, FLAT, conditional=True)
    assert result.status.value == "proven_conditionally"
    assert sorted((c.kind, tuple(c.columns)) for c in result.conditions) == [("not_null", ("product_id",)), ("unique", ("product_id",))]
