"""The admission ledgers (benchmarks/admitted/, tools/proof_ledger.py).

The first test is the gate: a results file counts exactly the proofs its ledger admits, and every ledger has
a results file to be compared with.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import proof_ledger as L  # noqa: E402


def test_every_results_file_counts_exactly_the_proofs_its_ledger_admits():
    assert L.check() == []


def test_every_ledger_is_as_the_admission_tool_writes_it():
    assert L.ledgers(), "benchmarks/admitted holds no ledger"
    for name in L.ledgers():
        lines = (L.LEDGERS / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith(f"# Proofs of {name} admitted to its count.") and "do not edit by hand" in lines[0], name
        entries = L.read_ledger(name)  # raises on a line that is not a ledger line, or a pair listed twice
        assert len(entries) == len(lines) - 1, name  # one header, then nothing but proofs
        assert list(entries) == sorted(entries), name


def _results(folder: Path, name: str, **fields) -> None:
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(fields), encoding="utf-8")


def _ledger(folder: Path, name: str, *lines: str) -> None:
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.txt").write_text("# header\n" + "".join(line + "\n" for line in lines), encoding="utf-8")


def test_check_compares_the_count_with_the_ledger(tmp_path):
    ledgers, results = tmp_path / "admitted", tmp_path / "results"
    _ledger(ledgers, "demo", "000000000000 checked p1", "111111111111 baseline p2")
    _results(results, "demo", coverage={"proven": 2}, score="2/3, 0 wrong")
    assert L.check(ledgers, results) == []
    _results(results, "demo", coverage={"proven": 3})
    (problem,) = L.check(ledgers, results)
    assert "counts 3 proofs and 2 are admitted" in problem and "admit demo" in problem
    _results(results, "demo", coverage={"proven": 1})  # a lost proof still listed leaves room for an unchecked one
    assert len(L.check(ledgers, results)) == 1
    _results(results, "demo", score="1,003/1,100 proved, 0 wrong")  # no coverage block: the number that opens the score
    assert L.counted("demo", results) == 1003 and len(L.check(ledgers, results)) == 1


def test_a_ledger_with_nothing_to_be_compared_with_fails_the_check(tmp_path):
    ledgers, results = tmp_path / "admitted", tmp_path / "results"
    _ledger(ledgers, "moved", "000000000000 baseline p1")
    results.mkdir()
    (problem,) = L.check(ledgers, results)  # the results file is gone: say so, never skip
    assert "no results file" in problem and "moved" in problem
    _results(results, "moved", score="all proved", coverage={})
    (problem,) = L.check(ledgers, results)
    assert "gives no proven count" in problem
    _results(results, "moved", coverage={"proven": True})  # a flag is not a count
    assert "gives no proven count" in L.check(ledgers, results)[0]


def test_a_damaged_ledger_is_an_error(tmp_path):
    _ledger(tmp_path, "demo", "000000000000 checked p1", "not-a-digest checked p2")
    with pytest.raises(ValueError, match="demo.txt:3"):
        L.read_ledger("demo", tmp_path)
    _ledger(tmp_path, "demo", "000000000000 checked p1", "111111111111 checked p1")
    with pytest.raises(ValueError):  # one pair twice
        L.read_ledger("demo", tmp_path)
    _ledger(tmp_path, "demo", "000000000000 trusted p1")
    with pytest.raises(ValueError):  # not one of the ways a proof is admitted
        L.read_ledger("demo", tmp_path)
    with pytest.raises(ValueError):
        L.check(tmp_path, tmp_path)  # the gate does not pass over a ledger it cannot read


def test_a_pair_name_may_hold_spaces(tmp_path):
    _ledger(tmp_path, "demo", "aaaaaaaaaaaa searched case 7 (b)")
    assert L.read_ledger("demo", tmp_path) == {"case 7 (b)": ("a" * 12, "searched")}


def test_the_command_line_reports_and_exits_nonzero_on_a_mismatch(tmp_path, monkeypatch, capsys):
    ledgers, results = tmp_path / "admitted", tmp_path / "results"
    _ledger(ledgers, "demo", "000000000000 checked p1")
    _results(results, "demo", coverage={"proven": 2})
    monkeypatch.setattr(L, "LEDGERS", ledgers)
    monkeypatch.setattr(L, "RESULTS", results)
    assert L.main(["check"]) == 1 and "counts 2 proofs and 1 are admitted" in capsys.readouterr().out
