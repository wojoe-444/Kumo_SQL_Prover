"""Admission ledgers: the proofs each eval's count is made of, and the check that the two agree.

An eval's results file (``benchmarks/results/<eval>.json``) holds a count of proven pairs. The ledger
``benchmarks/admitted/<eval>.txt`` lists which pairs: one line ``<digest> <how> <pair>`` per proof that was
admitted to the count. ``how`` is ``baseline`` (counted before the ledger existed), ``checked`` (a heavy
executed search and a near-miss probe found nothing), ``searched`` (only the executed search: a held-out
pair, or an eval whose pairs cannot be mutated) or ``unrunnable`` (DuckDB cannot run the pair, so the proof
stands on the prover alone).

    python tools/proof_ledger.py status   # ledger size against each results file
    python tools/proof_ledger.py check    # exit 1 when a count and its ledger disagree

``check`` reads two small files per eval and runs in the pull-request gate (``tests/test_proof_ledger.py``).
A ledger whose results file is missing, or gives no count, is a failure too: a check that silently compares
against nothing is no check.

This tool only reads. Ledgers are written by KumoSQL's ``tools/proof_admission.py admit <eval> --prover
<this checkout>``, which proves every pair with the prover installed in its environment and runs the checks
on each new proof (docs/proof-admission.md). Never add a line by hand.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
LEDGERS = ROOT / "benchmarks" / "admitted"
RESULTS = ROOT / "benchmarks" / "results"
HOWS = ("baseline", "checked", "searched", "unrunnable")
ADMIT = "python tools/proof_admission.py admit {name} --prover <this checkout>"  # run in a KumoSQL checkout


def read_ledger(name: str, folder: Path = LEDGERS) -> dict[str, tuple[str, str]]:
    """``pair -> (digest, how)``; empty when the eval has no ledger. A line that is not a ledger line is an error."""

    path = folder / f"{name}.txt"
    entries: dict[str, tuple[str, str]] = {}
    if not path.is_file():
        return entries
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(" ", 2)
        if len(parts) != 3 or not re.fullmatch(r"[0-9a-f]{12}", parts[0]) or parts[1] not in HOWS or parts[2] in entries:
            raise ValueError(f"{path.name}:{number}: not a ledger line: {line[:80]!r}")
        entries[parts[2]] = (parts[0], parts[1])
    return entries


def ledgers(folder: Path = LEDGERS) -> list[str]:
    return sorted(path.stem for path in folder.glob("*.txt")) if folder.is_dir() else []


def counted(name: str, results: Path = RESULTS) -> int | None:
    """The number of proofs the eval's results file counts: ``coverage.proven``, else the number that opens
    ``score``. ``None`` when the file is missing or gives neither."""

    path = results / f"{name}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    proven = (data.get("coverage") or {}).get("proven")
    if isinstance(proven, int) and not isinstance(proven, bool):
        return proven
    match = re.match(r"\s*([\d,]+)", str(data.get("score", "")))
    return int(match.group(1).replace(",", "")) if match else None


def check(folder: Path = LEDGERS, results: Path = RESULTS) -> list[str]:
    """One line per ledger that does not match its results file.

    More counted than admitted: a proof was counted without the checks. Fewer: the ledger still lists a proof
    the eval lost, which would let a new, unchecked proof take its place without the count moving. A ledger
    with no results file, or with one that gives no count, has nothing to be compared with and is reported."""

    problems = []
    for name in ledgers(folder):
        admitted = len(read_ledger(name, folder))
        if not (results / f"{name}.json").is_file():
            problems.append(f"{name}: benchmarks/admitted/{name}.txt has no results file benchmarks/results/{name}.json to be compared with")
            continue
        count = counted(name, results)
        if count is None:
            problems.append(f"{name}: benchmarks/results/{name}.json gives no proven count (coverage.proven, or a number that opens score)")
        elif count != admitted:
            problems.append(
                f"{name}: benchmarks/results/{name}.json counts {count} proofs and {admitted} are admitted; "
                f"run `{ADMIT.format(name=name)}` in a KumoSQL checkout and commit benchmarks/admitted/{name}.txt (docs/proof-admission.md)"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("status", "check"))
    args = parser.parse_args(argv)
    if args.command == "status":
        for name in ledgers(LEDGERS):
            hows = Counter(how for _, how in read_ledger(name, LEDGERS).values())
            print(f"{name}: {sum(hows.values())} admitted ({', '.join(f'{n} {how}' for how, n in sorted(hows.items()))}); results file counts {counted(name, RESULTS)}")
        return 0
    problems = check(LEDGERS, RESULTS)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
