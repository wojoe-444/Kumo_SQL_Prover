# Kumo SQL Prover, explained simply

Kumo SQL Prover answers one question: do these two BigQuery queries return the same results? It says yes with a proof, no with a small database where they differ, "yes, if these facts hold", or "I cannot tell". It never guesses: when it cannot be sure it says unknown.

It is the prover of [KumoSQL](https://github.com/wojoe-444/KumoSQL), which uses it to check its rewrites. This folder is the plain-language companion to [the full documentation](../docs/README.md); each guide links to its full reference for API details and recorded scores.

## Choose a guide

| I want to… | Read this |
| --- | --- |
| Understand whether two queries match | [Provers](provers.md) |
| Understand singleton joins and nested UNION identities in a proof | [Singleton aggregation and set identity](singleton-and-set-identity.md) |
| See how rewrites are double-checked | [Proof safeguards](proof-safeguards.md) |
| See how KumoSQL checks it read a query correctly | [Parser checks](parser-checks.md) |
| See when two queries match only if some facts hold | [Equivalent under conditions](conditional-equivalence.md) |
| See why two queries differ, as one condition | [Difference explanations](difference-explanations.md) |
| Learn what a query guarantees about its rows | [Output properties](output-properties.md) |
| Find results that change when rows tie | [Ties](ties.md) |
| Use keys and other data guarantees in a proof | [Constraint-dependent rewrites](constraint-rewrites.md) |
| Understand the test suites and their scores | [Evals](evals/README.md) |
| Record a benchmark result | [Benchmark results format](benchmarks/README.md) |
| Understand local BigQuery-to-DuckDB execution | [BigQuery on DuckDB](bigquery-on-duckdb.md) |
