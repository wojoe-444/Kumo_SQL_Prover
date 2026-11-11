# Optimizer wrong-result bugs

[Plain-language version](../../docs_simple/evals/optimizer-bugs.md)

24 query pairs taken from public bug reports in which a database's optimizer returned wrong rows: Apache Calcite, Apache Spark, CockroachDB, DuckDB, MySQL and ClickHouse. Each pair is the query as written and the rewrite the optimizer made of it, spelled out as SQL, with the report's tables and rows. The two return different rows, so this is a **must-not-prove** set: a proof is a soundness bug. The pairs are in `tests/fixtures/optimizer_bugs` ([sources, adaptations and overlap](../../tests/fixtures/optimizer_bugs/README.md)). Results file: `optimizer-bugs`.

```
python tools/optimizer_bugs_bench.py                    # a few seconds
python tools/optimizer_bugs_bench.py --show unknown
python tools/optimizer_bugs_bench.py --write-results
```

## How a pair is decided

1. **proven**: `prove_equivalent_algebraic` (DuckDB dialect, output names ignored) proves the pair, using the primary keys, UNIQUE and NOT NULL columns the setup declares (CHECK constraints are not used). Any proof is **wrong**.
2. **refuted**: the prover itself finds the two queries differ. Either its solver does, or its executed search does (`search_counterexample=True` with the setup's column types): it builds databases around the two queries and returns the first on which the queries, run on DuckDB as written, return different rows ([the executed search for DuckDB pairs](../provers.md#executed-counterexamples-for-duckdb-queries)). The prover is not given the case's own data.
3. **unknown**: anything else.

Every run also checks that each pair really differs on its own data: DuckDB with its optimizer off (`kumo_sql_prover.duckdb_load.run_unoptimized`), or SQLite for the one pair DuckDB cannot run. Results are compared as lists when both queries end in `ORDER BY`, and as bags otherwise.

**Held out.** One case in five, by a hash of its id (5 of 24), is held out. The first run showed every case.

## Scores

2026-11-11: **19/24 refuted, 0 proved, 0 wrong**; every pair confirmed to differ. Held out: 4/5 refuted, 0 proved. Five are refuted by the solver and fourteen by the executed search.

| Outcome | Pairs |
| --- | --- |
| refuted by the solver | bug-001 (null-safe equality under `NOT`), bug-013 (a dropped filter), bug-015 (a lost `GROUP BY`), bug-019 (`= NULL` read as `IS NULL`), bug-021 (an empty-input aggregate) |
| refuted by the executed search | 14 pairs, among the development ones: bug-002 (`EXCEPT` read as a negated filter), bug-004 (a global aggregate over an empty input), bug-009 and bug-010 (a correlated `COUNT` decorrelated into an inner join), bug-011 (an anti-join against an empty table), bug-014 (a decimal constant truncated to an integer), bug-017 and bug-018 (rows duplicated by an inferred cross join or branch), bug-020 (an INT64 compared with a FLOAT64 past 2**53, found by the z3 candidates), bug-022 (a correlated `NOT IN` turned into a join), bug-023 (`LEFT JOIN ... ON FALSE` read as empty), bug-024 (a scalar subquery `IS NULL` folded to `FALSE`) |
| unknown | 5: bug-005 runs only on SQLite (DuckDB rejects a subquery in an outer join's `ON`); bug-006 has a `ROWS` window frame, bug-012 an `OFFSET` and bug-025 a `DISTINCT ON`, where the engine may pick the rows, so the search does not run them; one held-out pair |

The counterexamples are small and are not the reports' rows: for bug-020 the search asks z3 for a database and gets the single row `9007199254740993`, which equals the FLOAT64 constant once converted and does not equal it as an integer.

Before the executed search read DuckDB queries (2026-10-03): 5/24 refuted, 0 proved, held out 2/5.

**Baseline.** On master before 2026-10-03 01:51 UTC, bug-004 was **proved**: a false proof. `SELECT 'US' FROM events` was proved equal to `SELECT 'US' FROM (SELECT COUNT(*) FROM events) g`, because a rule folded the outer select into a global aggregate and dropped its one-row result. The global-aggregate cardinality fix (#417) landed before this eval; bug-004 is kept as a regression in the eval's test. No prover change was made for the first version of this eval. The executed search for DuckDB queries (2026-11-11) was written with the development pairs in view, so the score is tuned on test; the held-out pairs were scored by verdict only, and the held-out split had already been shown by the first run.

## Limits

* Most rewrites are SQL readings of the plan a report prints, not SQL the reporter wrote. Where a report shows only a plan, the pair encodes its reported effect on a minimal fixture.
* The rows come from the reports, so many pairs are on tiny or empty tables. That is the point: most of these bugs appear only on empty inputs, NULLs or duplicates.
