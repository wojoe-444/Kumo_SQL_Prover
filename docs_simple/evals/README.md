# Understanding the evaluations

[All simple guides](../README.md) · [Full reference and results-file index](../../docs/evals/README.md)

An evaluation, or eval, is a repeatable test suite measuring a feature. Some use public SQL test cases; others generate projects whose expected behavior is known in advance. None calls a language model while deciding the results.

## Read a score carefully

“X/Y, 0 wrong” needs two questions: what counts as success, and how was correctness checked? An equivalence proof, a bounded check, and agreement on executed data are different evidence levels.

An unknown answer is allowed. A false proof or a rewrite accepted despite changing results is a bug. A suite can avoid wrong answers by declining everything, so coverage and usefulness matter too.

**Held-out** cases were kept out of development. **Tuned on test** means developers used those cases while building the feature. **Adapted** SQL was changed to fit a dialect or test harness; it is reported separately from original SQL.

For comparing slow evals between code versions, see [Eval diff](eval-diff.md).

## Query equivalence and safety

| Guide | Question it tests |
| --- | --- |
| [SQLSolver and related corpora](sqlsolver.md) | Which optimizer query pairs can be proved? |
| [VeriEQL](verieql.md) | Can queries be proved or separated on constraint-respecting data? |
| [Singh and Bedathur](singh-bedathur.md) | Do alternative LeetCode solutions agree? |
| [Logos' TPC-H, DSB and TPC-DS pairs](logos.md) | Are Calcite's rewrites of benchmark queries really equivalent? |
| [Equivalent under conditions](conditional-equivalence.md) | How often is a pair equal under a short list of facts, and is any answer wrong? |
| [Bounded verification](bounded-verification.md) | Do queries agree on every modeled small database? |
| [SQL-IQ](sql-iq.md) | Equivalence, candidate choice, and error classification |
| [LLM-SQL-Solver](llm-sql-solver.md) | Does the checker reject wrong query pairs and handle expert labels? |
| [QUITE LLM rewrites](quite.md) | When language models rewrite a query, which rewrites can be proved right, and are any wrongly proved? |
| [Join rewrites](join-rewrites.md) | When does changing a join type preserve results? |
| [Optimizer wrong-result bugs](optimizer-bugs.md) | Does the prover avoid accepting known faulty rewrites? |
| [Numeric traps](numeric-traps.md) | Does the prover handle BigQuery's number and error rules, and tell when a rewrite could start failing? |
| [DB-GPT examples](dbgpt-rules.md) | Which demonstration rewrites preserve results under reviewed schemas? |
| [Paired engine tests](engine-paired-tests.md) | Which query pairs from Trino, Spark, PostgreSQL and DuckDB tests can be proved equal or shown different? |
| [DLBench](dlbench.md) | Are translations across SQL dialects faithful? |
| [Fuzzing](fuzzing.md) | Can generated queries expose unsafe proofs or rewrites? |

Feature guides also explain evaluations for [model reuse](https://github.com/wojoe-444/KumoSQL/blob/master/docs_simple/model-reuse.md), [join ordering](https://github.com/wojoe-444/KumoSQL/blob/master/docs_simple/joinorder.md), [output properties](../output-properties.md), [constraint-dependent rewrites](../constraint-rewrites.md), [incremental models](https://github.com/wojoe-444/KumoSQL/blob/master/docs_simple/incremental.md), and [scripts](https://github.com/wojoe-444/KumoSQL/blob/master/docs_simple/scripts.md).

## Find numbers or rerun a suite

Use the [full eval index](../../docs/evals/README.md) to find each `benchmarks/results/*.json` file. Its `command` field is the rerun command, and its date and caveats describe the measurement. The [project scoreboard](../../README.md#benchmark-scoreboard) displays those recorded results.

From a development checkout, `python -m pytest -n auto -m eval` runs the benchmark floor tests. Full corpus runs can require downloads, databases, or hours of execution; read the matching full guide first. When adding a docs page, add its simple counterpart and index entry too.
