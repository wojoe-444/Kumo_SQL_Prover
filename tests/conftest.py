import functools
import os
import sys
from pathlib import Path

import pytest



def _one_thread(config):
    """``config`` with ``threads`` set to 1, unless the caller chose a thread count; never mutates the caller's dict."""

    if config is None:
        return {"threads": 1}
    if not isinstance(config, dict) or any(str(key).lower() in ("threads", "worker_threads") for key in config):
        return config  # the caller's own choice, or a value DuckDB rejects exactly as before
    return {**config, "threads": 1}


def _single_threaded_duckdb():
    """Run DuckDB on one thread in tests unless a caller sets ``threads`` itself.

    Tests open thousands of tiny databases. Each ``duckdb.connect`` otherwise starts a thread per core, and under
    pytest-xdist those threads compete with the other workers for the same cores while doing nothing useful on a few
    rows. Every other argument and config key is passed on unchanged. Pool workers forked by a test inherit this.
    """

    try:
        import duckdb
    except ImportError:
        return
    connect = duckdb.connect
    if getattr(connect, "_kumosql_one_thread", False):
        return

    @functools.wraps(connect)
    def connect_one_thread(*args, **kwargs):
        if len(args) >= 3:  # connect(database, read_only, config)
            args = (*args[:2], _one_thread(args[2]), *args[3:])
        else:
            kwargs["config"] = _one_thread(kwargs.get("config"))
        return connect(*args, **kwargs)

    connect_one_thread._kumosql_one_thread = True
    duckdb.connect = connect_one_thread


def _no_pandas_probe():
    """Mark pandas as missing in ``sys.modules`` when it is not installed, so DuckDB stops searching for it.

    DuckDB tries ``import pandas`` for every bound parameter of ``execute`` and ``executemany``; without pandas each try
    is a full, failing search of ``sys.path``. A ``None`` entry fails the same import at once. ``import pandas`` still
    raises ``ModuleNotFoundError`` and ``importlib.util.find_spec("pandas")`` still returns None, and an installed pandas
    is never shadowed.
    """

    import importlib.util

    try:
        missing = importlib.util.find_spec("pandas") is None
    except (ImportError, ValueError):
        return
    if missing:
        sys.modules.setdefault("pandas", None)


# Test files that score the prover against a benchmark corpus and hold its floors. They are marked ``eval``, so
# ``pytest -m eval`` runs just the floors and ``pytest -m "not eval"`` everything else. The pull-request gate runs
# the second; the scheduled benchmark run (``.github/workflows/benchmarks.yml``) runs the first.
EVAL_FILES = {
    "test_calcite_mined_benchmarks.py",
    "test_conditional_benchmark.py",
    "test_conditional_candidates.py",
    "test_constraint_dependence.py",
    "test_cosette_benchmarks.py",
    "test_dbgpt_rules_bench.py",
    "test_dlbench_bench.py",
    "test_engine_pairs_bench.py",
    "test_eval_execution_cache.py",
    "test_join_rewrite_bench.py",
    "test_keyed_set_join.py",
    "test_llm_sql_solver_bench.py",
    "test_logos_bench.py",
    "test_numeric_traps_bench.py",
    "test_optimizer_bugs_bench.py",
    "test_output_properties.py",
    "test_qed_benchmarks.py",
    "test_quite_bench.py",
    "test_rbot_benchmarks.py",
    "test_rbot_normalise.py",
    "test_singh_bedathur_benchmark.py",
    "test_smt_counterexample_determinism.py",
    "test_soundness_fuzz.py",
    "test_sqliq_bench.py",
    "test_sqlsolver_benchmarks.py",
    "test_uexpr_tpch.py",
    "test_verieql_benchmarks.py",
}


@pytest.fixture(autouse=True)
def isolated_state(tmp_path_factory, monkeypatch):
    """Keep any saved state out of the real user directory."""

    monkeypatch.setenv("KUMOSQL_HOME", str(tmp_path_factory.mktemp("home") / "kumosql-home"))


def pytest_configure(config):
    """Run DuckDB single-threaded without the pandas probe (see the helpers above)."""

    _single_threaded_duckdb()
    _no_pandas_probe()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.name in EVAL_FILES:
            item.add_marker(pytest.mark.eval)
