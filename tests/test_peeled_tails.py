"""A pair whose queries end in the same ORDER BY .. LIMIT is proved on the queries without it (``peeled_tails``)."""

import random

import duckdb
import pytest

from kumo_sql_prover import shape_fallback
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.peeled_tails import NAME, peel
from kumo_sql_prover.smt_equivalence import TIE_ASSUMPTION, SmtStatus, TableConstraints, _split_limit

SCHEMA = {"notes": ["id", "title", "made", "owner", "kind"], "replies": ["id", "note", "body"], "people": ["id", "name"]}
TYPES = {
    "notes": {"id": "INT", "title": "TEXT", "made": "INT", "owner": "INT", "kind": "INT"},
    "replies": {"id": "INT", "note": "INT", "body": "TEXT"},
    "people": {"id": "INT", "name": "TEXT"},
}
CONSTRAINTS = {
    "notes": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
    "replies": TableConstraints(not_null=frozenset({"id", "note"}), keys=(("id",),)),
    "people": TableConstraints(not_null=frozenset({"id"}), keys=(("id",),)),
}
# The same rows two ways: a count over an outer join, and a count subquery per row. Proved without a tail; the rules
# that bring them together decline a select that has an ORDER BY or a LIMIT of its own.
GROUPED = "SELECT n.title, n.made, COUNT(r.id) AS replies FROM notes AS n LEFT JOIN replies AS r ON n.id = r.note GROUP BY n.id, n.title, n.made"
COUNTED = "SELECT n.title, n.made, (SELECT COUNT(*) FROM replies AS r WHERE r.note = n.id) AS replies FROM notes AS n"


def prove(left: str, right: str, dialect: str = "postgres"):
    return prove_equivalent_algebraic(
        left, right, schema=SCHEMA, constraints=CONSTRAINTS, types=TYPES, compare_names=False, dialect=dialect, exact_arithmetic=True, timeout_ms=5000
    )


def test_the_pair_is_proved_without_a_tail():
    assert prove(GROUPED, COUNTED).proven


@pytest.mark.parametrize(
    "tail, ties",
    [
        (" ORDER BY n.made DESC LIMIT 10", True),
        (" ORDER BY 2 DESC LIMIT 10", True),
        (" ORDER BY made, title LIMIT 3 OFFSET 2", True),
        (" ORDER BY n.kind LIMIT 5", True),  # a key that is not an output: an extra column of both queries
        (" ORDER BY n.made DESC NULLS LAST, n.kind + 1 LIMIT 5", True),
        (" ORDER BY 1, 2, 3 LIMIT 10", False),  # the ordering covers every output: tied rows are equal rows
        (" ORDER BY n.made DESC OFFSET 4", True),
    ],
)
def test_a_shared_tail_is_proved_on_the_queries_without_it(tail, ties):
    result = prove(GROUPED + tail, COUNTED + tail)
    assert result.proven, result.reason
    assert (TIE_ASSUMPTION in result.assumptions) is ties
    with shape_fallback.without(NAME):
        assert not prove(GROUPED + tail, COUNTED + tail).proven  # the step is what proves it


@pytest.mark.parametrize(
    "left, right",
    [
        (GROUPED + " ORDER BY n.made DESC", COUNTED + " ORDER BY n.made DESC"),
        (GROUPED + " ORDER BY n.made DESC", COUNTED + " ORDER BY n.title"),  # neither changes the bag
        (GROUPED, COUNTED + " ORDER BY n.title"),
    ],
)
def test_an_order_by_that_cuts_no_rows_is_dropped_before_the_rules_run(left, right):
    result = prove(left, right)
    assert result.proven, (left, right, result.reason)
    assert TIE_ASSUMPTION not in result.assumptions


@pytest.mark.parametrize(
    "left_tail, right_tail",
    [
        (" ORDER BY n.made DESC LIMIT 10", " ORDER BY n.made DESC LIMIT 5"),
        (" ORDER BY n.made DESC LIMIT 10", " ORDER BY n.made LIMIT 10"),
        (" ORDER BY n.made DESC LIMIT 10", " ORDER BY n.title DESC LIMIT 10"),
        (" ORDER BY n.made DESC LIMIT 10", " ORDER BY n.made DESC NULLS LAST LIMIT 10"),
        (" ORDER BY n.made DESC LIMIT 10 OFFSET 1", " ORDER BY n.made DESC LIMIT 10"),
        (" ORDER BY n.made DESC LIMIT 10", ""),
        (" ORDER BY n.made DESC LIMIT 10", " ORDER BY n.made DESC"),
        (" ORDER BY n.made, n.title LIMIT 10", " ORDER BY n.title, n.made LIMIT 10"),
        (" ORDER BY n.kind LIMIT 10", " ORDER BY n.owner LIMIT 10"),  # two keys that are not outputs, and not the same
        (" ORDER BY n.kind LIMIT 10", " ORDER BY n.made LIMIT 10"),  # a key that is not an output against one that is
        (" LIMIT 10", " LIMIT 10"),  # no ORDER BY: any ten rows
    ],
)
def test_tails_that_differ_prove_nothing(left_tail, right_tail):
    result = prove(GROUPED + left_tail, COUNTED + right_tail)
    assert not result.proven and not result.conditionally_proven, result.reason


@pytest.mark.parametrize("other", [COUNTED.replace("COUNT(*)", "COUNT(r.body)"), COUNTED + " WHERE n.kind = 1"])
def test_other_rows_under_the_same_tail_prove_nothing(other):
    tail = " ORDER BY n.made DESC LIMIT 10"
    assert not prove(GROUPED + tail, other + tail).proven


def test_queries_that_differ_before_the_cut_are_not_called_different():
    # Without the tail the first query has fewer rows, yet both give the smallest kind: not a proof, and no counterexample.
    result = prove("SELECT DISTINCT n.kind FROM notes AS n ORDER BY n.kind LIMIT 1", "SELECT n.kind FROM notes AS n ORDER BY n.kind LIMIT 1")
    assert result.status is not SmtStatus.NOT_EQUIVALENT


@pytest.mark.parametrize(
    "left, right",
    [
        ("SELECT title FROM notes ORDER BY made LIMIT 3", "SELECT title FROM notes ORDER BY made LIMIT 4"),
        ("SELECT title FROM notes ORDER BY made LIMIT 3", "SELECT title FROM notes ORDER BY made DESC LIMIT 3"),
        ("SELECT title FROM notes ORDER BY made LIMIT 3", "SELECT title FROM notes"),
        ("SELECT title FROM notes LIMIT 3", "SELECT title FROM notes LIMIT 3"),  # LIMIT without ORDER BY is not read
        ("SELECT title FROM notes ORDER BY made LIMIT kind", "SELECT title FROM notes ORDER BY made LIMIT kind"),  # not a constant
        ("SELECT title FROM notes", "SELECT title FROM notes"),  # nothing to peel
        ("SELECT DISTINCT title FROM notes ORDER BY made LIMIT 3", "SELECT DISTINCT title FROM notes ORDER BY made LIMIT 3"),  # a key DISTINCT does not keep
        ("(SELECT title FROM notes ORDER BY made LIMIT 3) ORDER BY 1 DESC", "(SELECT title FROM notes ORDER BY made LIMIT 3) ORDER BY 1 DESC"),  # stacked tails
    ],
)
def test_peel_declines(left, right):
    assert peel(left, right, "postgres") is None


def test_peel_keeps_the_order_by_of_distinct_on():
    # The ORDER BY of DISTINCT ON picks the row each key keeps: it is not dropped, so there is nothing to peel.
    sql = "SELECT DISTINCT ON (owner) owner, title FROM notes ORDER BY owner, made DESC"
    assert peel(sql, sql, "postgres") is None
    peeled = peel(sql, "SELECT owner, title FROM notes ORDER BY owner", "postgres")
    assert peeled is not None and "ORDER BY" in peeled.left and "ORDER BY" not in peeled.right


def test_peel_gives_the_queries_without_their_tails():
    peeled = peel("SELECT title FROM notes ORDER BY made DESC LIMIT 3", "SELECT title FROM notes AS n ORDER BY n.made DESC LIMIT 3", "postgres")
    assert peeled is not None and peeled.ties
    assert peeled.left == "SELECT title, made AS kq_ord0 FROM notes"
    assert peeled.right == "SELECT title, n.made AS kq_ord0 FROM notes AS n"
    covered = peel("SELECT title, made FROM notes ORDER BY 2, 1 LIMIT 3", "SELECT title, made FROM notes ORDER BY made, title LIMIT 3", "postgres")
    assert covered is not None and not covered.ties and covered.left == covered.right == "SELECT title, made FROM notes"
    bare = peel("SELECT title FROM notes ORDER BY made", "SELECT title FROM notes", "postgres")
    assert bare is not None and not bare.ties and bare.left == bare.right == "SELECT title FROM notes"


TAILED = [
    "SELECT title, made FROM notes ORDER BY made DESC, id LIMIT 4",
    "SELECT title, made AS m FROM notes ORDER BY m, id DESC LIMIT 3 OFFSET 2",
    "SELECT id, title FROM notes ORDER BY 2 DESC, 1 LIMIT 5",
    "SELECT id FROM notes ORDER BY kind NULLS FIRST, made DESC NULLS LAST, id LIMIT 6",
    "SELECT owner, COUNT(*) AS n FROM notes GROUP BY owner ORDER BY n DESC, owner NULLS FIRST LIMIT 3",
    "SELECT owner, MAX(made) FROM notes GROUP BY owner ORDER BY COUNT(*) DESC, owner LIMIT 2 OFFSET 1",
    "SELECT n.id, r.body FROM notes AS n JOIN replies AS r ON r.note = n.id ORDER BY r.id DESC LIMIT 7",
    "SELECT id FROM notes UNION SELECT note FROM replies ORDER BY 1 DESC LIMIT 4",
    "SELECT id, kind + made AS score FROM notes ORDER BY kind + made DESC NULLS LAST, id OFFSET 3",
]


def test_the_peeled_query_and_its_tail_say_what_the_query_says():
    """On generated rows, a query equals the first columns of its peeled query under the tail that ``_split_limit``
    reports (positions, directions, NULL places, LIMIT and OFFSET). Every ordering here ends in a key, so no rows tie."""

    rng = random.Random(15)
    con = duckdb.connect()
    con.execute("CREATE TABLE notes (id INT, title TEXT, made INT, owner INT, kind INT)")
    con.execute("CREATE TABLE replies (id INT, note INT, body TEXT)")
    for _ in range(25):
        con.execute("DELETE FROM notes")
        con.execute("DELETE FROM replies")
        for i in range(rng.randrange(0, 12)):
            con.execute("INSERT INTO notes VALUES (?, ?, ?, ?, ?)", [i, rng.choice(["a", "b", "c", None]), rng.choice([1, 2, 3, None]), rng.choice([1, 2, None]), rng.choice([0, 1, None])])
        for i in range(rng.randrange(0, 12)):
            con.execute("INSERT INTO replies VALUES (?, ?, ?)", [i, rng.randrange(0, 12), rng.choice(["x", "y", None])])
        for sql in TAILED:
            core, spec = _split_limit(sql, "duckdb")
            assert core is not None and spec is not None, sql
            limit, offset, ordering, _covers, visible = spec
            keys = ", ".join(f"{position + 1} {'DESC' if desc else 'ASC'} NULLS {'FIRST' if nulls_first else 'LAST'}" for position, desc, nulls_first in ordering)
            again = f"SELECT * FROM ({core}) AS k ORDER BY {keys}" + (f" LIMIT {limit}" if limit is not None else "") + f" OFFSET {offset}"
            assert [row[:visible] for row in con.execute(again).fetchall()] == con.execute(sql).fetchall(), sql


def test_proved_pairs_agree_in_duckdb():
    """The pairs proved above, under an ordering that ends in a key, give the same rows in the same order."""

    rng = random.Random(151)
    con = duckdb.connect()
    con.execute("CREATE TABLE notes (id INT PRIMARY KEY, title TEXT, made INT, owner INT, kind INT)")
    con.execute("CREATE TABLE replies (id INT PRIMARY KEY, note INT NOT NULL, body TEXT)")
    grouped = GROUPED.replace("SELECT n.title", "SELECT n.id, n.title")
    counted = COUNTED.replace("SELECT n.title", "SELECT n.id, n.title")
    tails = [" ORDER BY n.made DESC NULLS LAST, n.id LIMIT 4", " ORDER BY 4 DESC, 1 LIMIT 3 OFFSET 1", " ORDER BY n.made NULLS FIRST, n.id DESC LIMIT 5"]
    for tail in tails:
        assert prove(grouped + tail, counted + tail).proven, tail
    for _ in range(25):
        con.execute("DELETE FROM notes")
        con.execute("DELETE FROM replies")
        for i in range(rng.randrange(0, 10)):
            con.execute("INSERT INTO notes VALUES (?, ?, ?, ?, ?)", [i, rng.choice(["a", "b", None]), rng.choice([1, 2, 3, None]), rng.choice([1, 2, None]), rng.choice([0, 1, None])])
        for i in range(rng.randrange(0, 14)):
            con.execute("INSERT INTO replies VALUES (?, ?, ?)", [i, rng.randrange(0, 12), rng.choice(["x", None])])
        for tail in tails:
            assert con.execute(grouped + tail).fetchall() == con.execute(counted + tail).fetchall(), tail


def test_a_pair_over_the_size_limit_gets_no_last_step(monkeypatch):
    # each last step is another whole attempt, so a long pair is proved as it was before the steps
    tail = " ORDER BY n.made DESC LIMIT 10"
    left, right = GROUPED + tail, COUNTED + tail
    assert shape_fallback.last_steps_worth_trying(left, right) and prove(left, right).proven
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(left) + len(right) - 1)
    assert not shape_fallback.last_steps_worth_trying(left, right)
    assert not prove(left, right).proven
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(left) + len(right))
    assert prove(left, right).proven


def test_a_long_pair_runs_once_and_the_proofs_inside_it_take_no_last_step(monkeypatch):
    from kumo_sql_prover import algebraic_equivalence

    tail = " ORDER BY n.made DESC LIMIT 10"
    left, right = GROUPED + tail, COUNTED + tail
    runs = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    def counted(*args, **kwargs):
        runs.append((shape_fallback.fifteenth_staged(), shape_fallback.fifteenth_watching(), shape_fallback.fifteenth_batch_enabled()))
        return whole(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_equivalent_algebraic_whole", counted)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", 10)
    assert not prove(left, right).proven
    assert runs and set(runs) == {(True, False, False)}  # staged, so a proof inside returns at once; nothing watches
    outer = len(runs)
    runs.clear()
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", 4000)
    assert prove(left, right).proven and len(runs) > outer
