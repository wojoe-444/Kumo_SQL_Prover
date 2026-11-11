"""Joins on a unique key in the bounded checker: one joined row per row of the other side, same answers."""

from __future__ import annotations

import random
from collections import Counter

import pytest
import sqlglot

pytest.importorskip("z3")
pytest.importorskip("duckdb")

from kumo_sql_prover import bounded_equivalence as be  # noqa: E402
from kumo_sql_prover import bounded_keyed_joins as kj  # noqa: E402
from kumo_sql_prover.bounded_equivalence import BColumn, BoundedSchema, BoundedStatus, BTable, check_bounded  # noqa: E402


def schema() -> BoundedSchema:
    customers = BTable("customers", [BColumn("id", "INT64"), BColumn("name", "STRING"), BColumn("region", "INT64")], keys=[("id",)])
    orders = BTable("orders", [BColumn("oid", "INT64"), BColumn("cid", "INT64"), BColumn("amount", "INT64")], keys=[("oid",)])
    items = BTable("items", [BColumn("oid", "INT64"), BColumn("line", "INT64"), BColumn("qty", "INT64")], keys=[("oid", "line")])
    notes = BTable("notes", [BColumn("cid", "INT64"), BColumn("body", "STRING")])
    return BoundedSchema({"customers": customers, "orders": orders, "items": items, "notes": notes})


def compiled(sql: str, rows: int = 3, keyed: bool = True) -> be.Rel:
    return be.Compiler(be.SymbolicDatabase(schema(), rows), "mysql", keyed_joins=keyed).compile(sql)


# --- how many joined rows ---------------------------------------------------------------------------

KEYED = [
    # (query, joined rows at 3 rows per table)
    ("select o.oid, c.name from orders o join customers c on o.cid = c.id", 3),
    ("select o.oid, c.name from orders o join customers c on c.id = o.cid and c.region > 1", 3),
    ("select o.oid, c.name from orders o left join customers c on o.cid = c.id", 3),
    ("select o.oid, c.name from customers c join orders o on o.cid = c.id", 3),  # the keyed table first
    ("select o.oid, c.name from customers c right join orders o on o.cid = c.id", 3),
    ("select o.oid, c.name from orders o, customers c where o.cid = c.id", 3),
    ("select o.oid, c.name from customers c, orders o where o.cid = c.id and o.amount > 5", 3),
    ("select i.qty, o.amount, c.name from items i join orders o on i.oid = o.oid join customers c on o.cid = c.id", 3),
    ("select i.qty, o.amount, c.name from customers c, orders o, items i where i.oid = o.oid and o.cid = c.id", 3),
    ("select o.oid, c.name from orders o join customers c on o.cid + 1 = c.id", 3),
    ("select o.oid, c.name from orders o join customers c on c.id = 7", 3),
    # a key that survives a filter, a rename and a projection
    ("select o.oid, d.name from orders o join (select id, name from customers where region = 2) d on o.cid = d.id", 3),
    ("select o.oid, d.n from orders o join (select name as n, id as k from customers) d on d.k = o.cid", 3),
    # GROUP BY and DISTINCT make a key
    ("select o.oid, g.total from orders o join (select cid, sum(amount) as total from orders group by cid) g on g.cid = o.cid", 3),
    ("select o.oid from orders o join (select distinct cid from notes) d on d.cid = o.cid", 3),
    # a self-join on the key, and a join with the same table grouped
    ("select o.oid, p.amount from orders o join orders p on o.cid = p.oid", 3),
    # ``o.oid = 5`` picks one row of ``o`` (the left side is the keyed one): a row per row of ``p``
    ("select o.oid, p.amount from orders o join orders p on o.oid = 5", 3),
    # both columns of a two-column key
    ("select o.oid, i.qty from orders o join items i on i.oid = o.oid and i.line = 1", 3),
]

NOT_KEYED = [
    ("select o.oid, c.name from orders o join customers c on o.amount = c.region", 9),  # not a key column
    ("select o.oid, c.name from orders o join customers c on o.cid <=> c.id", 9),  # NULL keys match each other
    ("select o.oid, c.name from orders o join customers c on o.cid = c.id or c.region = 1", 9),
    ("select o.oid, c.name from orders o join customers c on o.cid >= c.id", 9),
    ("select n.body, i.qty from notes n join items i on i.oid = n.cid", 9),  # half of a two-column key
    ("select o.oid, c.name from orders o join customers c on c.id = c.region", 9),  # the key against its own row
    ("select o.oid, c.name from orders o join customers c on o.cid = c.id + 1", 9),  # not the key column itself
    ("select o.oid, c.name from orders o join customers c on c.id * 1 = o.cid", 9),
    ("select n.body, m.body from notes n join notes m on n.cid = m.cid", 9),
    ("select o.oid, n.body from orders o join notes n on n.cid = o.cid", 9),  # no key declared
    ("select o.oid, d.k from orders o join (select id + 1 as k from customers) d on d.k = o.cid", 9),  # a computed column
    ("select o.oid, d.cid from orders o join (select cid from orders group by cid, amount) d on d.cid = o.cid", 9),  # part of the grouping
]


@pytest.mark.parametrize(("sql", "rows"), KEYED)
def test_a_join_on_a_key_takes_one_row_per_row_of_the_other_side(sql, rows):
    assert len(compiled(sql).rows) == rows
    assert len(compiled(sql, keyed=False).rows) > rows


@pytest.mark.parametrize(("sql", "rows"), NOT_KEYED)
def test_any_other_join_is_encoded_as_before(sql, rows):
    assert len(compiled(sql).rows) >= rows
    assert len(compiled(sql).rows) == len(compiled(sql, keyed=False).rows)


def test_where_the_order_of_rows_can_be_seen_only_the_order_keeping_form_is_used():
    # an inner join keyed on its right side keeps the rows in order, so a LIMIT does not stop it
    assert len(compiled("select o.oid from orders o join customers c on o.cid = c.id order by o.amount limit 2").rows) == 3
    # an outer join, or the keyed table on the left, would move rows: with a LIMIT or a window they stay as they were
    for sql in (
        "select o.oid from orders o left join customers c on o.cid = c.id order by o.amount limit 2",
        "select o.oid from customers c join orders o on o.cid = c.id order by o.amount limit 2",
        "select o.oid, row_number() over (order by o.amount) from customers c join orders o on o.cid = c.id",
        "select o.oid from customers c, orders o where o.cid = c.id order by o.amount limit 2",
        # a self-join holds the same cells on both sides: ``o.oid = 5`` is about ``o`` and says nothing about ``p``
        "select o.oid, p.amount from orders o join orders p on o.oid = 5 order by p.amount limit 2",
        "select o.oid, p.amount from orders o, orders p where o.oid = 5 order by p.amount limit 2",
    ):
        assert len(compiled(sql).rows) == len(compiled(sql, keyed=False).rows), sql


# --- the keys a relation carries ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "keys"),
    [
        ("select * from customers", ((0,),)),
        ("select name, id from customers where region = 1", ((1,),)),
        ("select name from customers", ()),
        ("select id + 0 as id from customers", ()),
        ("select line, qty, oid from items", ((0, 2),)),
        ("select qty, oid from items", ()),
        ("select cid, count(*) from orders group by cid", ((0,),)),
        ("select count(*), amount, cid from orders group by cid, amount", ((1, 2),)),
        ("select count(*) from orders group by cid", ()),
        ("select distinct cid, amount from orders", ((0, 1),)),
        ("select d.id from (select id from customers) d", ((0,),)),
        ("select id from customers union all select cid from orders", ()),
    ],
)
def test_keys_follow_a_relation(sql, keys):
    assert tuple(sorted(compiled(sql).keys)) == keys


def test_an_alias_that_shadows_a_column_is_not_mistaken_for_the_key():
    # ``id`` in the output is the region, not the key
    assert compiled("select region as id, name from customers").keys == ()
    assert len(compiled("select o.oid from orders o join (select region as id from customers) d on d.id = o.cid").rows) == 9


def test_position_free():
    def free(sql):
        return kj.position_free(sqlglot.parse_one(sql, read="mysql"))

    assert free("select a from t order by a")
    assert not free("select a from t order by a limit 1")
    assert not free("select a from (select a from t order by a limit 3) s")
    assert not free("select a, rank() over (order by a) from t")


# --- the two encodings give the same rows on concrete databases ------------------------------------------

QUERIES = [sql for sql, _ in KEYED + NOT_KEYED] + [
    "select c.region, sum(o.amount), count(*) from orders o join customers c on o.cid = c.id group by c.region",
    "select c.name, count(o.oid) from customers c left join orders o on o.cid = c.id group by c.name",
    "select o.oid, c.name from orders o left join customers c on o.cid = c.id and c.region = 1",
    "select o.oid from orders o left join customers c on o.cid = c.id where c.id is null",
    "select c.name, i.qty from customers c, orders o, items i where i.oid = o.oid and o.cid = c.id and i.qty > 1 and c.region < 3",
    "select c.name, i.qty from items i, orders o, customers c where i.oid = o.oid and o.cid = c.id",
    "select * from orders o, customers c where o.cid = c.id",
    "select * from customers c, orders o, items i where i.oid = o.oid and o.cid = c.id",
    "select o.oid from orders o join customers c on o.cid = c.id where exists (select 1 from items i where i.oid = o.oid)",
    "select o.oid, (select count(*) from items i join orders x on i.oid = x.oid where x.cid = o.cid) from orders o",
    "select o.oid from orders o join customers c on o.cid = c.id join customers d on c.region = d.id",
    "select a.oid, b.oid from orders a join orders b on a.oid = b.oid",
    "select o.oid, p.amount from orders o join orders p on o.oid = 2",
    "select o.oid, p.amount from orders o, orders p where p.oid = o.cid and o.oid = 2",
    "select o.oid, p.amount from orders o left join orders p on p.oid = o.cid + 1",
    "select o.oid, c.name from customers c right join orders o on o.cid = c.id and c.region = 2",
    "select o.oid, c.name from orders o join customers c on o.cid = c.id order by o.amount, o.oid limit 2",
]


def random_data(rng: random.Random) -> dict[str, list[tuple]]:
    def value(low=0, high=3):
        return None if rng.random() < 0.15 else rng.randint(low, high)

    def keyed(rows, key):
        seen, out = set(), []
        for row in rows:
            marker = tuple(row[k] for k in key)
            if None not in marker and marker in seen:
                continue
            seen.add(marker)
            out.append(row)
        return out

    return {
        "customers": keyed([(value(1, 4), rng.choice(["a", "b", None]), value()) for _ in range(rng.randint(0, 3))], (0,)),
        "orders": keyed([(value(1, 4), value(1, 4), value(0, 9)) for _ in range(rng.randint(0, 3))], (0,)),
        "items": keyed([(value(1, 4), value(1, 2), value()) for _ in range(rng.randint(0, 3))], (0, 1)),
        "notes": [(value(1, 4), rng.choice(["x", None])) for _ in range(rng.randint(0, 3))],
    }


@pytest.mark.parametrize("sql", QUERIES)
def test_both_encodings_return_the_same_bag(sql):
    rng = random.Random(sum(map(ord, sql)))
    for _ in range(12):
        data = random_data(rng)
        new = be.evaluate(sql, schema(), data, "mysql")
        old = be.evaluate(sql, schema(), data, "mysql", keyed_joins=False)
        assert Counter(new) == Counter(old), (sql, data)


def test_select_star_keeps_the_column_order_of_the_from_clause():
    data = {"customers": [(1, "a", 5)], "orders": [(7, 1, 30)], "items": [(7, 1, 2)], "notes": []}
    rows = be.evaluate("select * from customers c, orders o, items i where i.oid = o.oid and o.cid = c.id", schema(), data, "mysql")
    assert rows == [(1, "a", 5, 7, 1, 30, 7, 1, 2)]


def test_a_database_that_breaks_a_declared_key_is_declined():
    data = {"customers": [(1, "a", 5), (1, "b", 6)], "orders": [(7, 1, 30)], "items": [], "notes": []}
    with pytest.raises(be.Unsupported):
        be.evaluate("select o.oid, c.name from orders o join customers c on o.cid = c.id", schema(), data, "mysql")


# --- verdicts ---------------------------------------------------------------------------------------


def check(left, right, **kwargs):
    return check_bounded(left, right, schema(), rows=3, dialect="mysql", **kwargs)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            "select c.region, sum(o.amount) from orders o join customers c on o.cid = c.id group by c.region",
            "select c.region, sum(o.amount) from customers c, orders o where c.id = o.cid group by c.region",
        ),
        (
            "select o.oid from orders o join customers c on o.cid = c.id",
            "select o.oid from orders o where o.cid in (select id from customers)",
        ),
        (
            "select i.qty from items i join orders o on i.oid = o.oid join customers c on o.cid = c.id where c.region = 1",
            "select i.qty from customers c, orders o, items i where c.region = 1 and c.id = o.cid and o.oid = i.oid",
        ),
        (
            "select o.oid, c.name from orders o left join customers c on o.cid = c.id",
            "select o.oid, (select name from customers c where c.id = o.cid) from orders o",
        ),
    ],
)
def test_equal_pairs_stay_bounded_equivalent(left, right):
    for keyed in (True, False):
        assert check(left, right, keyed_joins=keyed).status is BoundedStatus.BOUNDED_EQUIVALENT


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # an inner join drops the orders without a customer
        ("select o.oid from orders o join customers c on o.cid = c.id", "select o.oid from orders o"),
        ("select o.oid from orders o left join customers c on o.cid = c.id", "select o.oid from orders o join customers c on o.cid = c.id"),
        # region is not a key: an order can meet two customers
        ("select o.oid from orders o join customers c on o.cid = c.region", "select o.oid from orders o where o.cid in (select region from customers)"),
        (
            "select c.region, sum(o.amount) from orders o join customers c on o.cid = c.id group by c.region",
            "select c.region, sum(o.amount) from orders o join customers c on o.oid = c.id group by c.region",
        ),
        ("select c.name from customers c, orders o where o.cid = c.id", "select c.name from customers c, orders o where o.oid = c.id"),
    ],
)
def test_different_pairs_are_still_refuted_with_a_replayed_database(left, right):
    for keyed in (True, False):
        result = check(left, right, keyed_joins=keyed)
        assert result.status is BoundedStatus.DIFFERENT and result.counterexample is not None


def test_a_chain_of_key_joins_no_longer_grows_with_every_table():
    sql = "select count(*) from items i, orders o, customers c, customers d where i.oid = o.oid and o.cid = c.id and c.region = d.id"
    assert len(compiled(sql, keyed=False).rows) == 1  # a global aggregate: one output row either way
    source = sqlglot.parse_one(sql, read="mysql")
    compiler = be.Compiler(be.SymbolicDatabase(schema(), 3), "mysql")
    compiler.reorder = True
    relation, remaining = kj.inner_chain(compiler, source, None)
    assert len(relation.rows) == 3 and remaining == []
