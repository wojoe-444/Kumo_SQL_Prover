# Evals

[Plain-language version](../../docs_simple/evals/README.md)

KumoSQL is scored on public benchmarks and on suites generated for its own features. None of them calls a language model at run time. This folder has one page per eval family: where the data comes from, how a case is scored, how to rerun it and what the limits are. The headline numbers are in the [README scoreboard](../../README.md#benchmark-scoreboard).

## How the evals are organised

- **Results.** Every eval owns one file, `benchmarks/results/<name>.json` ([format](../../benchmarks/README.md)). `python tools/scoreboard.py` turns those files into the README scoreboard; a score is never edited by hand. The tables below name the file for each eval.
- **Evidence level.** A score says how strong its positive answers are: an unbounded proof, bounded verification (no difference on any database up to a row limit) or agreement on executed datasets. A suite that mixes levels has one results file per level.
- **Zero wrong.** An eval reports `X/Y, 0 wrong`. A false proof, an incorrect counterexample or a behaviour-changing rewrite is a bug; an unknown is allowed.
- **Held-out cases.** Where a held-out split exists, the page and the results file say so. Cases a rule was developed against are marked `tuned on test`.
- **Rerun.** `command` in each results file reruns the eval. `python -m pytest -n auto -m eval` runs every benchmark floor, and `python tools/eval_diff.py` compares every eval on `origin/main` with your checkout. It masks timing before comparing (including fields ending in `_seconds`, numbers with a unit, `HH:MM:SS`, and the bare elapsed-seconds column that ends a SQLSolver summary row), so only answers and counts count as a difference. See [eval diff](eval-diff.md) for stable samples, per-eval time caps, process cleanup, and timing records.

## Source inventory

[Public SQL evaluation sources](https://github.com/wojoe-444/KumoSQL/blob/master/docs/evals/public-sources.md) lists every public suite, database and project considered, whether an eval below already scores it, and what is being added.

## Equivalence and proofs

| Page | What it scores | Results files |
| --- | --- | --- |
| [Algebraic prover on SQLSolver, R-Bot, QED, Cosette, SPES and mined Calcite tests](sqlsolver.md) | Equivalent query pairs proved, with refutations by a replayed counterexample | `sqlsolver-calcite`, `sqlsolver-spark`, `sqlsolver-tpch`, `sqlsolver-tpcc`, `rbot-calcite`, `qed-calcite`, `cosette`, `cosette-adapted`, `spes-only`, `calcite-mined` |
| [VeriEQL](verieql.md) | LeetCode, Literature and Calcite suites, proofs and counterexamples scored separately | `verieql-leetcode-proof`, `verieql-leetcode-executed`, `verieql-literature-proof`, `verieql-literature-executed`, `verieql-calcite-proof`, `verieql-calcite-executed` |
| [Logos' TPC-H, DSB and TPC-DS pairs](logos.md) | R-Bot's Calcite rewrites of TPC-H and DSB queries and TPC-DS query variants: proved, or shown different on generated data | `logos-core-proof`, `logos-core-executed` |
| [Singh and Bedathur](singh-bedathur.md) | 2,800 LeetCode equivalence pairs | `singh-bedathur-leetcode` |
| [Equivalent under conditions](conditional-equivalence.md) | The fourth verdict (proved equal under minimal NOT NULL, unique-key and foreign-key conditions) on the Singh and VeriEQL LeetCode pairs, plus three hand-checked suites | `conditional-equivalence-singh`, `conditional-equivalence-verieql` |
| [Bounded verification](bounded-verification.md) | The same suites under the z3 bounded checker (at most 3 rows per table) | `bounded-sqlsolver-calcite`, `bounded-sqlsolver-spark`, `bounded-sqlsolver-tpch`, `bounded-sqlsolver-tpcc`, `bounded-qed`, `bounded-rbot`, `bounded-cosette`, `bounded-spes`, `bounded-singh`, `bounded-literature`, `bounded-calcite`, `bounded-leetcode` |
| [SQL-IQ](sql-iq.md) | Equivalence judge, SQL judge and error classification | `sql-iq-equivalence`, `sql-iq-judge`, `sql-iq-errors` |
| [LLM-SQL-Solver](llm-sql-solver.md) | 180 Spider pairs that must never be proved, 70 pairs with expert labels | `llm-sql-solver-negatives`, `llm-sql-solver-relaxed` |
| [QUITE LLM rewrites](quite.md) | The 4,160 rewrites that 13 LLM and learned rewriters published for TPC-H, DSB, Calcite and SQLStorm queries, each flagged equal or not on the authors' instance: flagged-equal pairs proved, flagged-unequal pairs refuted by a replayed database and never proved on grounds a replay contradicts | `quite-rewrites`, `quite-negatives` |
| [DLBench](dlbench.md) | Cross-dialect translations from SQLite, MySQL and PostgreSQL into six databases: parsed, and proved equal to the source | `dlbench` |
| [DB-GPT rewrite examples](dbgpt-rules.md) | DB-GPT's 36 PostgreSQL before/after rewrites, labelled by hand and checked on DuckDB | `dbgpt-rules` |
| [Optimizer wrong-result bugs](optimizer-bugs.md) | Query pairs from public optimizer bug reports (Calcite, Spark, CockroachDB, DuckDB, MySQL, ClickHouse): none may be proved | `optimizer-bugs` |
| [Paired engine tests](engine-paired-tests.md) | Trino's two-query join assertions, Spark's predicate-subquery tests, a PostgreSQL join-removal regression and a DuckDB collation fixture, with authored JoinEquiv and jOOQ guards: proved, refuted or unknown against hand labels | `engine-paired-tests` |
| [Numeric traps](numeric-traps.md) | BigQuery number and error rules around the SMT prover (2**53, INT64 overflow, NaN, `-0.0`, NUMERIC, `/` against `DIV`, an operation moved ahead of its guard): trap pairs never proved, and whether a rewrite can raise an error the original cannot | `numeric-traps` |
| [Join rewrites to LEFT JOIN](join-rewrites.md) | Hand-checked rewrites between CROSS, INNER, RIGHT, FULL, semi and anti joins and LEFT JOIN, proved or refuted | `join-rewrites` |
| [Metamorphic fuzzing](fuzzing.md) | TLP/NoREC fuzzing, unsafe-rewrite detection, rewrite composition and the typed soundness fuzzer | `sqlancer-tlp-norec`, `unsafe-rewrite-detection`, `rewrite-composition`, `soundness-fuzz` |

## Evals documented with their feature

These features keep their eval results on the feature's own page.

| Page | What it scores | Results files |
| --- | --- | --- |
| [Output properties](../output-properties.md) | Never-NULL columns, unique keys and row bounds | `output-properties`, `output-properties-adapted` |
| [Constraint-dependent rewrites](../constraint-rewrites.md) | Rewrites valid only under declared keys and constraints | `constraint-rewrites` |

## Adding an eval

1. Write the harness under `tools/` and a test under `tests/`; put `pytestmark = pytest.mark.eval` in the test file (no list to edit). Use [eval diff](eval-diff.md) to compare slow evals without running their full corpora.
2. Pin the source version, keep original and adapted cases apart, and reserve held-out cases.
3. Add `benchmarks/results/<name>.json` and run `python tools/scoreboard.py`.
4. Add or extend a page in this folder, list it in the table above and in [docs/README.md](../README.md), or leave that to `python tools/docs_index.py --write`, which adds any page or results file no index names to a generated block at the end of each index. `tests/test_docs.py` checks the links and that every page has a plain-language companion.
