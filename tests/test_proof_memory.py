"""A long proof frees what it discards while it runs (workstream issue 2, row 23).

A serialized call pauses automatic garbage collection in every thread, so that no other thread finalizes Z3 terms
while this one is inside Z3. Python frees a syntax tree only by collection (a node and its parent refer to each
other), so every copy a proof discarded stayed until the call returned: a 30 s normalization that needs 150 MB held
3 GB, and four benchmark workers doing that at once were killed for memory. The thread that holds the lock now
collects at ``solver_lock.sweep()``, which long loops call.
"""

import gc
import threading
import time

import pytest

pytest.importorskip("z3")

from kumo_sql_prover import solver_lock  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.bounded_equivalence import check_bounded, schema_from_prover  # noqa: E402
from kumo_sql_prover.smt_equivalence import prove_equivalent_smt  # noqa: E402
from kumo_sql_prover.solver_lock import SOLVER_LOCK, serialized, sweep  # noqa: E402


class _Probe:
    """Garbage that only a collection frees, and that records who freed it."""

    def __init__(self, seen):
        self.seen = seen
        self.cycle = self

    def __del__(self):
        self.seen.append((gc.isenabled(), SOLVER_LOCK._is_owned()))


@pytest.fixture
def eager(monkeypatch):
    """Every sweep is due."""

    monkeypatch.setattr(solver_lock, "SWEEP_AFTER", 0)


@pytest.fixture
def asked(monkeypatch):
    """The arguments of each collection the code asks for (automatic ones are not in it)."""

    calls = []
    collect = gc.collect

    def noted(*generation):
        calls.append(generation)
        return collect(*generation)

    monkeypatch.setattr(gc, "collect", noted)
    return calls


@pytest.fixture
def collections():
    """Every collection that runs, as ``(objects waiting when it began, this thread holds the lock)``."""

    seen = []

    def note(phase, info):
        if phase == "start":
            seen.append((gc.get_count()[0], SOLVER_LOCK._is_owned()))

    gc.callbacks.append(note)
    try:
        yield seen
    finally:
        gc.callbacks.remove(note)


def test_sweep_frees_what_a_proof_discarded_while_the_proof_runs(eager):
    seen = []

    @serialized
    def proof():
        _Probe(seen)
        assert sweep() is True
        return list(seen)

    # freed inside the call, by the thread that holds the lock, with automatic collection still paused
    assert proof() == [(False, True)]


def test_sweep_does_nothing_outside_a_proof(eager, asked):
    assert sweep() is False
    assert asked == []


def test_sweep_waits_until_enough_has_piled_up(monkeypatch, asked):
    monkeypatch.setattr(solver_lock, "SWEEP_AFTER", 10**12)

    @serialized
    def proof():
        return sweep(), list(asked)

    assert proof() == (False, [])


def test_a_thread_that_does_not_hold_the_lock_never_collects(eager, asked):
    answers = []

    @serialized
    def proof():
        other = threading.Thread(target=lambda: answers.append(sweep()))
        other.start()
        other.join()
        return list(asked)

    assert proof() == []
    assert answers == [False]


def test_the_end_of_a_nested_call_is_a_sweep(eager):
    seen = []

    @serialized
    def inner():
        return None

    @serialized
    def outer():
        _Probe(seen)
        inner()
        return list(seen)

    assert outer() == [(False, True)]


def test_a_caller_who_turned_collection_off_keeps_it_off(eager, asked):
    @serialized
    def proof():
        return sweep()

    original = gc.isenabled()
    gc.disable()
    try:
        assert proof() is False
        assert not gc.isenabled()
        assert asked == []
    finally:
        gc.enable() if original else gc.disable()


def test_a_failed_proof_hands_the_collector_back(eager):
    @serialized
    def proof():
        raise ValueError("proof failure")

    original = gc.isenabled()
    gc.enable()
    try:
        with pytest.raises(ValueError, match="proof failure"):
            proof()
        assert gc.isenabled()
        assert solver_lock._paused_by is None
        assert sweep() is False
    finally:
        gc.enable() if original else gc.disable()


def test_full_collections_take_a_bounded_share_of_the_time(eager, monkeypatch, asked):
    @serialized
    def proof():
        swept = sweep()
        return swept, list(asked)

    # the last full collection took long and ended just now: only what was made since is collected
    monkeypatch.setattr(solver_lock, "_full_took", 1000.0)
    monkeypatch.setattr(solver_lock, "_full_ended", time.monotonic())
    assert proof() == (True, [(1,)])
    # enough time has passed since: everything is collected, and the new duration is noted
    del asked[:]
    monkeypatch.setattr(solver_lock, "_full_ended", time.monotonic() - 20000.0)
    assert proof() == (True, [()])
    assert solver_lock._full_took < 1000.0


SCHEMA = {
    "orders": ["o_id", "o_cust", "o_total", "o_date"],
    "customer": ["c_id", "c_name", "c_nation"],
    "lineitem": ["l_order", "l_part", "l_qty", "l_price"],
}


def _wide_join(branches: int) -> str:
    parts = [
        f"(SELECT c_nation AS k{i}, SUM(l_price * l_qty) AS v{i} FROM customer JOIN orders ON o_cust = c_id "
        f"JOIN lineitem ON l_order = o_id WHERE o_total > {i} AND l_qty < {i + 5} GROUP BY c_nation) AS s{i}"
        for i in range(branches)
    ]
    joins = "".join(f" JOIN {part} ON s0.k0 = s{i}.k{i}" for i, part in enumerate(parts) if i)
    return "SELECT s0.k0, " + ", ".join(f"s{i}.v{i}" for i in range(branches)) + " FROM " + parts[0] + joins


def _pile(prove) -> tuple[str, int]:
    """The proof's status, and how many objects were waiting for a collection when it ended."""

    @serialized
    def proof():
        result = prove()
        return result.status.value, gc.get_count()[0]

    gc.collect()
    return proof()


def test_a_long_normalization_does_not_keep_what_it_discards(monkeypatch, collections):
    sql = _wide_join(5)

    def prove():
        return prove_equivalent_algebraic(sql, sql, schema=SCHEMA)

    # without sweeps everything the proof discarded is still there when it ends
    monkeypatch.setattr(solver_lock, "SWEEP_AFTER", 10**12)
    status, kept = _pile(prove)
    assert kept > 80_000, kept
    # with them the pile never grows far past the point where a sweep is due
    monkeypatch.setattr(solver_lock, "SWEEP_AFTER", 10_000)
    del collections[:]
    swept_status, left = _pile(prove)
    inside = [pending for pending, held in collections if held]
    assert swept_status == status == "proven_equivalent"
    assert len(inside) >= 3, collections
    assert max(inside) < 50_000 and left < 50_000, (inside, left)


def test_a_long_expression_does_not_keep_what_compiling_it_discards(monkeypatch, collections):
    columns = [f"c{i}" for i in range(40)]
    sql = "SELECT " + " + ".join(columns) + " AS s FROM t"
    monkeypatch.setattr(solver_lock, "SWEEP_AFTER", 2_000)
    status, left = _pile(lambda: prove_equivalent_smt(sql, sql, schema={"t": columns}, timeout_ms=1000))
    inside = [pending for pending, held in collections if held]
    assert status == "proven_equivalent"
    assert len(inside) >= 3, collections
    assert max(inside) < 12_000 and left < 12_000, (inside, left)


BOUNDED = schema_from_prover({"t": ["a", "b"]}, types={"t": {"a": "INT64", "b": "INT64"}})

PAIRS = [
    ("algebraic", "SELECT a FROM t WHERE a > 1 AND a IN (SELECT a FROM t WHERE a < 9)", "SELECT a FROM t WHERE a IN (SELECT a FROM t WHERE a < 9) AND a > 1"),
    ("algebraic", "SELECT a, COUNT(*) FROM t GROUP BY a", "SELECT a, COUNT(b) FROM t GROUP BY a"),
    ("algebraic", "SELECT DISTINCT a FROM (SELECT a, b FROM t) AS s", "SELECT a FROM t GROUP BY a"),
    ("smt", "SELECT a + b + 1 FROM t WHERE a > 0", "SELECT 1 + b + a FROM t WHERE 0 < a"),
    ("smt", "SELECT a FROM t WHERE a > 1 OR b > 1", "SELECT a FROM t WHERE a > 1"),
    ("smt", "SELECT SUM(a) FROM t GROUP BY b", "SELECT SUM(a + 0) FROM t GROUP BY b"),
    ("bounded", "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE 1 < a"),
    ("bounded", "SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE a >= 1"),
]


def _answer(prover: str, left: str, right: str) -> str:
    if prover == "algebraic":
        result = prove_equivalent_algebraic(left, right, schema={"t": ["a", "b"]})
    elif prover == "smt":
        result = prove_equivalent_smt(left, right, schema={"t": ["a", "b"]}, types={"t": {"a": "INT64", "b": "INT64"}})
    else:
        result = check_bounded(left, right, BOUNDED)
    return result.status.value


@pytest.mark.parametrize("prover, left, right", PAIRS)
def test_collecting_at_every_point_changes_no_answer(prover, left, right, monkeypatch, collections):
    usual = _answer(prover, left, right)
    monkeypatch.setattr(solver_lock, "SWEEP_AFTER", 0)
    monkeypatch.setattr(solver_lock, "_FULL_SHARE", 0)
    del collections[:]
    assert _answer(prover, left, right) == usual
    assert any(held for _, held in collections), "no sweep ran: the pair no longer tests anything"
