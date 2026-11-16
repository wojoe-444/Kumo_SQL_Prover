"""Singh & Bedathur's LeetCode equivalence pairs: decided-count floors and zero wrong verdicts.

The data is downloaded on first use (see tools/singh_bedathur_bench.py); the tests skip when
it cannot be fetched. ``FLOORS`` only ever goes up.
"""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.eval

pytest.importorskip("z3")
pytest.importorskip("duckdb")

_path = Path(__file__).resolve().parent.parent / "tools" / "singh_bedathur_bench.py"
_spec = importlib.util.spec_from_file_location("singh_bedathur_bench", _path)
bench = importlib.util.module_from_spec(_spec)
sys.modules["singh_bedathur_bench"] = bench
_spec.loader.exec_module(bench)

SAMPLE = 120
FLOORS = {"sample": 116, "all": 2747}  # measured 118 and 2,763; a little room for solver timeouts under load


def _pairs():
    try:
        return bench.load_pairs()
    except OSError as error:
        pytest.skip(f"benchmark data not available: {error}")


def test_sample_of_pairs():
    pairs = random.Random(2024).sample(_pairs(), SAMPLE)
    for number, pair in enumerate(pairs):
        pair.index = number
    report = bench.run(pairs)
    assert report.counts["wrong"] == 0, [i for i, v in report.verdicts.items() if v.kind == "wrong"]
    decided = report.counts["equivalent"] + report.counts["different"]
    assert decided >= FLOORS["sample"], report.line()


@pytest.mark.slow
def test_all_pairs():
    report = bench.run(_pairs())
    assert report.counts["wrong"] == 0, [i for i, v in report.verdicts.items() if v.kind == "wrong"]
    decided = report.counts["equivalent"] + report.counts["different"]
    assert decided >= FLOORS["all"], report.line()


# A made-up pair (no benchmark data): equal when the column holds dates, different when it holds numbers, because MySQL
# reads both strings as 2019 against a number. The harness declares the column the way its search reads it.
_DATED_LEFT = (
    "SELECT s.product_id FROM sales AS s WHERE s.name <> 'S8' GROUP BY s.product_id "
    "HAVING MIN(s.sale_date) >= '2019-01-01' AND MIN(s.sale_date) <> '2019-03-31'"
)
_DATED_RIGHT = (
    "SELECT s.product_id FROM sales AS s WHERE s.name <> 'S8' GROUP BY s.product_id "
    "HAVING (MIN(s.sale_date) > '2019-01-01' AND MIN(s.sale_date) <> '2019-03-31') OR MIN(s.sale_date) = '2019-01-01'"
)


def test_columns_used_as_dates_are_declared_to_the_prover(monkeypatch):
    pair = bench.Pair(0, _DATED_LEFT, _DATED_RIGHT, {"sales": ["product_id", "name", "sale_date"]}, "equivalent")
    assert bench._text_types(pair) == {"sales": {"name": "STRING", "sale_date": "DATE"}}
    assert bench.prove(pair)[0]
    assert bench.decide(pair).kind == "equivalent"
    monkeypatch.setattr(bench, "_text_types", lambda pair: {"sales": {"name": "STRING"}})
    proved, _, reason = bench.prove(pair)
    assert not proved and "same number" in reason
