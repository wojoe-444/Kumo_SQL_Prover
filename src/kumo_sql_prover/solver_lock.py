"""Shared rules for using Z3: one lock around its default context, and a deterministic work cap per check.

Z3 terms, sorts and solvers built without an explicit context share one global context, and Z3 objects
must not be used from two threads at once: concurrent first calls fail with ``Context mismatch`` and can
leave the context unusable for the rest of the process. The UI serves requests on threads and analyses
pipelines in the background, so each public entry point that builds Z3 terms runs under this lock.
It is re-entrant, so a prover that calls another (or a rule that uses Z3) does not deadlock.
"""

from __future__ import annotations

import functools
import gc
import threading
import time

try:
    import z3
except ImportError:  # pragma: no cover - z3-solver is optional
    z3 = None

SOLVER_LOCK = threading.RLock()

# What every entry point answers without the optional solver (the extra is ``smt``; the package name differs between
# the product and the split-out prover, so the hint names the dependency itself).
Z3_MISSING = "z3-solver is not installed (pip install z3-solver)"


# While a serialized call runs, automatic collection is paused in every thread (see ``serialized``). Python frees a
# syntax tree only by collection (a node and its parent refer to each other), so everything a long call discards
# would stay until the call returns: a 30 s normalization that needs 150 MB held 3 GB. The thread that holds the
# lock may collect, since no other thread is inside Z3 then, and long loops ask it to with ``sweep``.
SWEEP_AFTER = 100_000  # container objects made since the last collection
_FULL_SHARE = 10  # a full collection runs once this many times its own duration has passed since the last one
_paused_by: int | None = None  # the thread whose outermost serialized call paused the collector
_full_ended = 0.0
_full_took = 0.0


def sweep() -> bool:
    """Free what the running proof has discarded, when enough has piled up. Call it from long loops.

    It does nothing unless this thread is inside a ``serialized`` call that paused the collector, and nothing
    until ``SWEEP_AFTER`` objects were made since the last collection, so a call costs a comparison. Objects made
    since the last sweep are collected each time; older ones only when a full collection is due, which keeps
    the cost of full collections under about a tenth of the running time however much the process holds.
    """

    global _full_ended, _full_took
    if _paused_by != threading.get_ident() or gc.get_count()[0] < SWEEP_AFTER:
        return False
    now = time.monotonic()
    if now - _full_ended >= _FULL_SHARE * _full_took:
        gc.collect()
        _full_ended = time.monotonic()
        _full_took = _full_ended - now
    else:
        gc.collect(1)
    return True


def serialized(function):
    """Run ``function`` while holding :data:`SOLVER_LOCK`."""

    @functools.wraps(function)
    def locked(*args, **kwargs):
        global _paused_by
        with SOLVER_LOCK:
            # Cyclic Z3 terms from a completed proof can otherwise be collected
            # by a different Python thread while this thread uses the context.
            restore_gc = gc.isenabled()
            if restore_gc:
                gc.disable()
                _paused_by = threading.get_ident()
            try:
                return function(*args, **kwargs)
            finally:
                if restore_gc:
                    _paused_by = None
                    try:
                        if gc.get_count()[0] >= gc.get_threshold()[0]:
                            gc.collect()
                    finally:
                        gc.enable()
                else:
                    sweep()  # the end of a nested call is a point where this thread may collect

    return locked


# Each solver check also gets a deterministic work cap (Z3's rlimit) of ``timeout_ms * WORK_PER_MS`` units.
# A check that runs out of it stops at the same point on any machine and under any load; the wall-clock
# ``timeout_ms`` still stops checks that do little counted work. Measured on the QED pairs, checks do 150 to
# 1,000,000 units per millisecond and the largest finished check used 4.3 million units, so the cap
# (100 million at the default 5000 ms) never binds before a check that finishes today would.
WORK_PER_MS = 20000


def bounded_solver(timeout_ms: int):
    """A ``z3.Solver`` whose checks stop at a deterministic work cap or at ``timeout_ms`` (see ``WORK_PER_MS``).

    Z3 reports ``reason_unknown()`` ``"canceled"`` for the cap and ``"timeout"`` for the wall clock.
    """

    sweep()  # a proof makes one solver per question, so this is a point every long proof passes often
    return bound(z3.Solver(), timeout_ms)


def bound(solver, timeout_ms: int):
    """Apply the work cap and the wall-clock limit to ``solver``; ``Solver.translate`` drops both."""

    solver.set("rlimit", timeout_ms * WORK_PER_MS)
    solver.set("timeout", timeout_ms)
    return solver
