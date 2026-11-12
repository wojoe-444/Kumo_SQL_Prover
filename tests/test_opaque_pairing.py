"""The first round of derived-relation comparisons, across the two queries (``kumo_sql_prover.opaque_pairing``)."""

import pytest

pytest.importorskip("z3")

from kumo_sql_prover import opaque_pairing, shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.opaque_pairing import pair_across

SCHEMA = {"s": ["k", "q", "x"], "u": ["k", "q", "x"]}
TYPES = {table: {column: "BIGINT" for column in columns} for table, columns in SCHEMA.items()}


def body(table, quarter, spelling="BETWEEN 1 AND 5"):
    spelling = spelling.replace("T.", table + ".")
    return f"SELECT {table}.k, SUM({table}.x) AS t FROM {table} WHERE {table}.x {spelling} AND {table}.q = {quarter} GROUP BY {table}.k"


def bodies(*texts):
    return {f"({text})": (text, 2, text) for text in texts}


def test_each_body_meets_its_counterpart_first():
    left = [body("s", q) for q in (1, 2, 3)]
    right = [body("s", q, ">= 1 AND T.x <= 5") for q in (1, 2, 3)]  # other text, and so other keys
    asked = []

    def equal(rep, key):
        asked.append((rep, key))
        return rep[-20:] == key[-20:]  # the same quarter

    table = bodies(*left, *right)
    keys = list(table)
    mapping, tried = pair_across(table, set(keys[:3]), set(keys[3:]), equal)
    assert mapping == {keys[3]: keys[0], keys[4]: keys[1], keys[5]: keys[2]}
    assert asked == [(keys[0], keys[3]), (keys[1], keys[4]), (keys[2], keys[5])]  # the same literals first: no wasted proof
    assert tried == set(asked)


def test_without_matching_literals_an_unpaired_body_comes_first_in_reading_order():
    left = [body("s", q) for q in (1, 2, 3)]
    right = [body("s", f"{q} AND s.q IN (1, 2, 3)") for q in (1, 2, 3)]
    asked = []

    def equal(rep, key):
        asked.append((rep, key))
        return rep.split("s.q = ")[1][0] == key.split("s.q = ")[1][0]

    table = bodies(*left, *right)
    keys = list(table)
    mapping, _ = pair_across(table, set(keys[:3]), set(keys[3:]), equal)
    assert mapping == {keys[3]: keys[0], keys[4]: keys[1], keys[5]: keys[2]}
    assert len(asked) == 3


def test_only_across_the_two_queries_and_only_like_with_like():
    s1, s2, u1 = body("s", 1), body("s", 2, ">= 1 AND T.x <= 5"), body("u", 1)
    wide = "SELECT s.k, s.q, SUM(s.x) AS t FROM s GROUP BY s.k, s.q"
    table = {**bodies(s1, s2, u1), f"({wide})": (wide, 3, wide)}
    keys = list(table)
    asked = []

    def equal(rep, key):
        asked.append((rep, key))
        return False

    # a body both queries read is already one relation; another table or another width is never compared
    assert pair_across(table, {keys[0], keys[1]}, {keys[1], keys[2], keys[3]}, equal) == ({}, set())
    assert asked == []
    assert pair_across(table, {keys[0]}, {keys[1]}, equal) == ({}, {(keys[0], keys[1])})
    assert pair_across(table, set(keys), set(), equal)[0] == {}


def test_the_budget_ends_the_round():
    left = [body("s", q) for q in range(1, 6)]
    right = [body("s", q + 10, ">= 1 AND T.x <= 5") for q in range(1, 6)]
    table = bodies(*left, *right)
    keys = list(table)
    asked = []

    def equal(rep, key):
        asked.append((rep, key))
        return False

    mapping, tried = pair_across(table, set(keys[:5]), set(keys[5:]), equal, budget=4)
    assert mapping == {} and len(asked) == 4 and len(tried) == 4


def query(spelling, comma):
    reads = []
    for table in ("s", "u"):
        for quarter in (1, 2, 3):
            reads.append((f"{table}{quarter}", "(" + body(table, quarter, spelling) + ")"))
    first = reads[0][0]
    items = f"{first}.k, " + ", ".join(f"{alias}.t" for alias, _ in reads)
    if comma:
        sources = ", ".join(f"{text} AS {alias}" for alias, text in reads)
        return f"SELECT {items} FROM {sources} WHERE " + " AND ".join(f"{first}.k = {alias}.k" for alias, _ in reads[1:])
    joined = f"{reads[0][1]} AS {first} " + " ".join(f"JOIN {text} AS {alias} ON {first}.k = {alias}.k" for alias, text in reads[1:])
    return f"SELECT {items} FROM {joined}"


def prove(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="postgres")


def test_six_reads_of_two_grouped_tables_are_matched(monkeypatch):
    left, right = query("BETWEEN 1 AND 5", comma=True), query(">= 1 AND T.x <= 5", comma=False)
    assert prove(left, right).proven
    # a near miss: one read takes another quarter
    assert not prove(left, right.replace("u.q = 3", "u.q = 2")).proven
    assert not prove(left, right.replace("u.x <= 5", "u.x <= 4")).proven
    # what the round buys: without it the budget goes to the three reads of one table compared with each other
    monkeypatch.setattr(opaque_pairing, "pair_across", lambda *args, **kwargs: ({}, set()))
    assert not prove(left, right).proven


def test_off_on_the_second_attempt(monkeypatch):
    called = []
    real = opaque_pairing.pair_across

    def spy(*args, **kwargs):
        called.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(opaque_pairing, "pair_across", spy)
    left, right = query("BETWEEN 1 AND 5", comma=True), query(">= 1 AND T.x <= 5", comma=False)
    with shape_fallback.disabled():
        prove(left, right)
    assert called == []
