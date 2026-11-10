import functools
import json
import os
import re
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


# A test file that scores the prover against a benchmark corpus and holds its floors declares it with
# ``pytestmark = pytest.mark.eval``: ``pytest -m eval`` runs just the floors and ``pytest -m "not eval"`` everything
# else. Nothing registers the file here; ``eval_files`` finds the declaration.
_EVAL_DECLARATION = re.compile(r"^pytestmark\s*=.*\bmark\.eval\b", re.M)


@functools.lru_cache(maxsize=1)
def eval_files() -> frozenset[str]:
    """Names of the test files that declare themselves benchmark floors."""

    found = set()
    for path in Path(__file__).resolve().parent.glob("test_*.py"):
        try:
            if _EVAL_DECLARATION.search(path.read_text(encoding="utf-8")):
                found.add(path.name)
        except OSError:
            continue
    return frozenset(found)


# The pull-request gate runs ``pytest -m "not eval" --quick``; the scheduled benchmark run
# (``.github/workflows/benchmarks.yml``) runs ``pytest -m eval``, and a full run is plain ``pytest``.
ORDER_FILE = Path(__file__).resolve().parent / "order.json"


def pytest_addoption(parser):
    parser.addoption("--quick", action="store_true", default=False, help="skip the slow tier (the tests listed as slow in tests/order.json)")


@functools.lru_cache(maxsize=1)
def _slow_tests() -> frozenset[str]:
    """Node ids of the tests that take about two seconds or more, measured over many runs (tests/order.json)."""

    try:
        return frozenset(json.loads(ORDER_FILE.read_text(encoding="utf-8")).get("slow", {}))
    except (OSError, ValueError):
        return frozenset()


def pytest_collection_modifyitems(config, items):
    """``--quick`` drops the slow tier. A canary stands for its benchmark family in the gate, so it is never dropped."""

    if not config.getoption("--quick"):
        return
    slow = _slow_tests()
    dropped = [item for item in items if item.nodeid in slow and not item.get_closest_marker("canary")]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        gone = {id(item) for item in dropped}
        items[:] = [item for item in items if id(item) not in gone]


@pytest.fixture(autouse=True)
def isolated_state(tmp_path_factory, monkeypatch):
    """A test never finds a SQLSolver installation (Java, jar) of whoever runs the suite: the home it looks in is empty."""

    monkeypatch.setenv("KUMOSQL_SQLSOLVER_HOME", str(tmp_path_factory.mktemp("sqlsolver-home")))


def pytest_configure(config):
    """Run DuckDB single-threaded without the pandas probe (see the helpers above)."""

    _single_threaded_duckdb()
    _no_pandas_probe()
