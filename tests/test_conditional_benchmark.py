"""The conditional verdict on Singh & Bedathur's and VeriEQL's LeetCode pairs: floors, and zero wrong or crashed.

``tools/conditional_bench.py`` re-runs every conditional proof on random DuckDB databases repaired to meet its conditions and
checks that dropping any one condition loses the proof. The Singh data is downloaded on first use (see
tools/singh_bedathur_bench.py); the tests skip when it cannot be fetched. ``FLOORS`` only ever goes up.
"""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.eval

pytest.importorskip("z3")
pytest.importorskip("duckdb")

_tools = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_tools))
_spec = importlib.util.spec_from_file_location("conditional_bench", _tools / "conditional_bench.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["conditional_bench"] = bench
_spec.loader.exec_module(bench)

# measured 20 of 120 Singh pairs, 13 of 160 VeriEQL cases and 590 of 2,800 Singh pairs (610 before 28 more pairs were proved
# with no condition, 9 of them conditional until then); a little room for solver timeouts under load
FLOORS = {"singh-sample": 17, "verieql-sample": 11, "singh-all": 570}


@pytest.fixture(autouse=True)
def _own_singh_module(monkeypatch):
    """The Singh eval's own test loads ``singh_bedathur_bench`` under the same name; pickling a ``Pair`` for the worker
    processes finds the class by that name, so point it at the module these pairs come from while a test runs."""
    monkeypatch.setitem(sys.modules, "singh_bedathur_bench", bench.singh)


def _singh_pairs():
    try:
        return bench.singh.load_pairs()
    except OSError as error:
        pytest.skip(f"benchmark data not available: {error}")


def _check(report, floor):
    counts = report.counts
    assert counts["wrong"] == 0, [o.detail for o in report.outcomes if o.kind == "wrong"]
    assert counts["crash"] == 0, [o.detail for o in report.outcomes if o.kind == "crash"]
    assert counts["conditional"] >= floor, report.line()
    conditional = [o for o in report.outcomes if o.kind == "conditional"]
    assert all(o.minimal for o in conditional), "a named condition could be dropped"
    assert all(o.conditions for o in conditional)


# A made-up pair in the corpus's shape (names only, no types): one side compares the column with a date, the other with the
# string. For a column that holds text MySQL reads the two differently, so the prover declines unless the column is declared
# the way the re-check fills it.
_DATED = (
    "SELECT s.product_id FROM sales AS s GROUP BY s.product_id HAVING MIN(s.sale_date) >= '2019-01-01' AND MAX(s.sale_date) <= '2019-03-31'",
    "SELECT s.product_id FROM sales AS s GROUP BY s.product_id "
    "HAVING MIN(s.sale_date) >= CAST('2019-01-01' AS DATE) AND MAX(s.sale_date) <= CAST('2019-03-31' AS DATE)",
)


def test_columns_used_as_dates_are_declared_to_the_prover(monkeypatch):
    pair = bench.singh.Pair(0, *_DATED, {"sales": ["product_id", "sale_date"]}, "equivalent")
    assert bench._prove_options(pair)["types"] == {"sales": {"sale_date": "DATE"}}
    assert bench.decide_singh(pair).kind == "equivalent"
    monkeypatch.setattr(bench, "_date_types", lambda pair: {})
    assert "types" not in bench._prove_options(pair)
    # the second attempt declares the columns used as text or as dates, and proves it too
    retried = bench.decide_singh(pair)
    assert retried.kind == "equivalent" and retried.detail.endswith(bench.TEXT_NOTE)
    monkeypatch.setattr(bench.singh, "_text_types", lambda pair: {})
    assert bench.decide_singh(pair).kind == "unknown"


def test_a_pair_with_no_date_column_declares_nothing():
    pair = bench.singh.Pair(0, "SELECT t.a FROM t WHERE t.b = 'x'", "SELECT t.a FROM t WHERE 'x' = t.b", {"t": ["a", "b"]}, "equivalent")
    assert "types" not in bench._prove_options(pair)


# A made-up pair whose proof needs two different text literals to differ on the column: 'S' and 'I' are the same value to
# MySQL when the column holds numbers (then the second query returns nothing), so no condition helps and the prover
# declines until the column is declared as text. The first query's NOT NULL test is what the condition is for.
_LABELLED = (
    "SELECT DISTINCT sale.buyer FROM sale JOIN item ON sale.item_id = item.item_id WHERE label = 'S' AND sale.buyer IS NOT NULL",
    "SELECT DISTINCT sale.buyer FROM item JOIN sale ON sale.item_id = item.item_id WHERE label = 'S' AND label <> 'I'",
)


def test_a_pair_still_unknown_is_tried_again_with_its_text_columns_declared(monkeypatch):
    pair = bench.singh.Pair(0, *_LABELLED, {"item": ["item_id", "label"], "sale": ["buyer", "item_id"]}, "equivalent")
    assert "types" not in bench._prove_options(pair)
    assert bench._prove_options(pair, text=True)["types"] == {"item": {"label": "STRING"}}
    outcome = bench.decide_singh(pair)
    assert outcome.kind == "conditional" and outcome.detail.endswith(bench.TEXT_NOTE)
    assert outcome.validated > 0 and outcome.minimal
    monkeypatch.setattr(bench.singh, "_text_types", lambda pair: {})
    assert bench.decide_singh(pair).kind == "unknown"


def test_a_pair_proved_without_text_types_is_not_tried_again(monkeypatch):
    pair = bench.singh.Pair(0, *_DATED, {"sales": ["product_id", "sale_date"]}, "equivalent")
    seen = []
    original = bench._singh_prove
    monkeypatch.setattr(bench, "_singh_prove", lambda pair, options=None: seen.append(options) or original(pair, options))
    outcome = bench.decide_singh(pair)
    assert outcome.kind == "equivalent" and outcome.detail == "proved outright" and seen == [None]


@pytest.mark.parametrize("rewritten_proves", [True, False])
def test_a_proof_with_no_condition_is_preferred(monkeypatch, rewritten_proves):
    """The pair as written is proved under conditions; its rewritten form is then tried for a proof that needs none."""

    from kumo_sql_prover import algebraic_equivalence, canonical_rules
    from kumo_sql_prover.smt_equivalence import SmtEquivalenceResult, SmtStatus

    calls = []

    def prove(left, right, conditional=False, **options):
        calls.append((left, conditional))
        if left == "written":
            return SmtEquivalenceResult(SmtStatus.PROVEN_CONDITIONALLY if conditional else SmtStatus.NOT_PROVEN, "as written")
        return SmtEquivalenceResult(SmtStatus.PROVEN_EQUIVALENT if rewritten_proves else SmtStatus.NOT_PROVEN, "rewritten")

    monkeypatch.setattr(algebraic_equivalence, "prove_equivalent_algebraic", prove)
    monkeypatch.setattr(canonical_rules, "canonicalize", lambda sql, dialect, tables: "rewritten " + sql)
    pair = bench.singh.Pair(0, "written", "written too", {"t": ["a"]}, "equivalent")
    result, texts = bench._singh_prove(pair)
    assert calls == [("written", True), ("rewritten written", False)]
    if rewritten_proves:
        assert result.status is SmtStatus.PROVEN_EQUIVALENT and texts == ("rewritten written", "rewritten written too")
    else:
        assert result.status is SmtStatus.PROVEN_CONDITIONALLY and texts == ("written", "written too")


def test_singh_sample():
    pairs = random.Random(2024).sample(_singh_pairs(), 120)
    _check(bench.run_singh(pairs), FLOORS["singh-sample"])


def test_verieql_sample():
    veri = pytest.importorskip("verieql_bench")
    try:
        cases = veri.load_cases("leetcode")[::150]
    except OSError as error:
        pytest.skip(f"benchmark data not available: {error}")
    _check(bench.run_verieql(cases), FLOORS["verieql-sample"])


@pytest.mark.slow
def test_all_singh_pairs():
    _check(bench.run_singh(_singh_pairs()), FLOORS["singh-all"])
