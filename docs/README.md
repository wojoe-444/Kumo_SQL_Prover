# Kumo SQL Prover documentation

[Plain-language version](../docs_simple/README.md)

Start with the [README](../README.md): what the prover answers, how to install it and call it, and the benchmark scoreboard. This package is the equivalence prover of [KumoSQL](https://github.com/wojoe-444/KumoSQL); the documentation of the rest of KumoSQL (pipeline analysis, the browser UI, rewrite rules, cost and change reports) is in [its repository](https://github.com/wojoe-444/KumoSQL/blob/master/docs/README.md).

## Pages

| Page | What it covers |
| --- | --- |
| [Singleton aggregation and set identity](singleton-and-set-identity.md) | Key-fixed aggregation and scoped set-tree identity, with explicit collation conditions |
| [Equivalence provers](provers.md) | Structural prover, synthetic-data comparison, Z3, the algebraic prover and SQLSolver |
| [Proof safeguards](proof-safeguards.md) | The independent predicate, CTE, parenthesis, DISTINCT, qualification, layout, subquery-lift and prover column-resolution checkers, the checker registry, Dataform expressions in proofs, and what is not covered yet |
| [Proof admission](proof-admission.md) | The ledger of proofs each eval's count is made of, the gate that compares the two, and how a new proof is admitted |
| [Parser checks](parser-checks.md) | A second reading of every query a proof depends on, checked against MySQL, DuckDB and BigQuery, and what sqlglot gets wrong |
| [Equivalent under conditions](conditional-equivalence.md) | The fourth verdict: a pair that is equal when stated NOT NULL, unique or foreign-key facts hold, with a SQL check for each |
| [Difference explanations](difference-explanations.md) | "Equivalent except when P": a verified predicate for the rows two different queries disagree on, in the API, CLI, Compare queries and change reports |
| [Running BigQuery SQL on DuckDB](bigquery-on-duckdb.md) | How executed counterexamples stay BigQuery refutations: settings, translation fixes and guards |
| [Output properties](output-properties.md) | Never-NULL columns, unique keys and row bounds, inferred without running a query |
| [Ties and nondeterministic results](ties.md) | Windows, LIMITs and aggregates whose result can depend on how tied rows are ordered, what would pin them down, and small databases that show the difference (tie witnesses) |
| [Constraint-dependent rewrites](constraint-rewrites.md) | Rewrites that hold only under declared NOT NULL columns, keys and foreign keys |

## Evals

The [evals folder](evals/README.md) has one page per eval family (SQLSolver, VeriEQL, Logos, Singh and Bedathur, SQL-IQ, bounded verification, numeric traps and the others) and a table naming the `benchmarks/results/*.json` file behind every eval. Each eval's numbers are in the README scoreboard ([format](../benchmarks/README.md)).
