# Proof admission

[Plain-language version](../docs_simple/proof-admission.md)

An eval's results file (`benchmarks/results/<eval>.json`) holds a count of proven pairs, and the README scoreboard is generated from it. The count alone does not say *which* pairs are counted, so a rule that newly proves a pair could raise it with no more evidence than the eval's own short random run. The admission ledger closes that gap: every counted proof is listed, a new proof gets onto the list only after heavy checks, and the gate fails when a count and its list disagree.

```sh
python tools/proof_ledger.py status   # ledger size against each results file
python tools/proof_ledger.py check    # exit 1 when a count and its ledger disagree
```

## The ledger

`benchmarks/admitted/<eval>.txt` has one line per admitted proof: `<digest> <how> <pair>`. The digest is twelve hex digits over the eval's name and the pair exactly as the admission tool reads it, so a pair whose text changes is a new pair. `how` says what stands behind the line:

| `how` | Meaning |
| --- | --- |
| `baseline` | Counted before the ledger existed (2026-11-10). No evidence from the admission checks. |
| `checked` | An executed search (3,000 databases or 120 seconds by default: exhaustive tiny databases, edge databases and random ones, with DuckDB's optimizer on and, for a sample, off) and a near-miss probe (single-site mutants of the pair that the prover also accepts, each sent to the same search) both found nothing. |
| `searched` | Only the executed search ran: a held-out pair (never mutated), or an eval whose pairs cannot be mutated. |
| `unrunnable` | DuckDB cannot run the pair, so no executed check exists; the proof stands on the prover alone. |

Eleven evals have a ledger: the SQLSolver suites (Calcite, Spark, TPC-H, TPC-C), QED, mined Calcite, R-Bot, Cosette, SPES, and VeriEQL's Calcite-397 and Literature suites. The other evals are not covered.

## The gate

`tests/test_proof_ledger.py` runs `tools/proof_ledger.py`'s check in the pull-request gate. It reads two small files per eval and fails when:

- the results file counts more proofs than the ledger admits (a proof was counted without the checks);
- it counts fewer (the ledger still lists a proof the eval lost, which would let a new, unchecked proof take its place without the count moving);
- a ledger has no results file, or its results file gives no proven count (`coverage.proven`, or the number that opens `score`). A ledger with nothing to be compared with is reported, never skipped.

So a pull request that changes a covered eval's proven count has to bring the updated ledger with it.

## Admitting a proof

The tool that writes ledgers lives in KumoSQL, with the executed-search engine it uses ([its page](https://github.com/wojoe-444/KumoSQL/blob/master/docs/proof-admission.md)). With this checkout installed in a KumoSQL environment (`pip install -e <this checkout>`), run in the KumoSQL checkout:

```sh
python tools/proof_admission.py admit qed-calcite --prover <this checkout>             # prove every pair, check the new proofs, update the ledger
python tools/proof_admission.py admit qed-calcite --prover <this checkout> --dry-run   # report unadmitted and lost proofs, write nothing
```

`admit` proves every pair as the eval does, keeps the ledger lines that are still proven, drops the ones that are not, and puts each new proof through the checks above. A new proof that a database separates is a **false proof**: it is printed with its witness, it is not added, and the command exits 1. That is a bug to fix, never a line to add by hand. Commit the updated ledger with the results file. The command stops without writing anything when no checkout is named, or when the prover installed in the environment is not the checkout it was given, so a ledger is never written for other code than it is committed with.

## Limits

- Surviving the search and the probe is evidence, not a proof. Four families of false proofs were found by hand on 2026-11-10 that the fuzzers, the search and the probe had all passed; two of them rest on BigQuery reading a query differently from DuckDB, which is the oracle of every executed check.
- The gate compares two files. A prover change that loses one proof and gains another leaves the count unchanged, and a change that loses a proof is not noticed until the eval is measured again. `admit --dry-run` proves every pair and is the exact comparison; it belongs with the scheduled eval runs and is not scheduled yet.
- `baseline` lines carry no evidence from the admission checks.

## Tests

`tests/test_proof_ledger.py`: the gate on this repository's ledgers, the ledger format, the count comparison in both directions, a ledger with no results file or no count, a damaged ledger, and the command line.
