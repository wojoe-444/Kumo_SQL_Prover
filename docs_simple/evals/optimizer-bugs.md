# Query pairs that must never be proved equivalent

[Simple eval index](README.md) · [Full reference](../../docs/evals/optimizer-bugs.md)

This suite turns public database optimizer bug reports into SQL pairs with witness data. The original query and faulty rewrite return different results. Any equivalence proof is therefore a soundness bug.

## Two independent checks

The prover tries to decide each pair using the supplied supported constraints. Separately, every run confirms the difference on the stored witness, usually with DuckDB's optimizer disabled so another optimizer does not hide the bug.

A prover refutation is useful. The prover can refute a pair in two ways: its solver finds a difference, or it builds small databases of its own and runs both queries on DuckDB exactly as written until the results differ. For example, for a rewrite that turns `NOT EXISTS` against an empty table into "no rows", it finds a one-row table on which the original returns the row and the rewrite returns nothing. It never sees the report's own data. An unknown is allowed: failing to prove a known-different pair is safe, but it is not the same as finding the difference.

```sh
python tools/optimizer_bugs_bench.py
```

Run from a development checkout. The full guide explains SQL reconstructions from optimizer plans, adaptations, constraints used, row-order comparisons, and recorded outcomes.

The search does not run queries where the engine may choose which rows come back (`LIMIT`, `DISTINCT ON`, a `ROWS` window frame), so those pairs stay unknown. It was written while looking at the development pairs, so its score is not an independent measurement.

The split marked held out was introduced after the first run showed all cases. It should not be read as a fully untouched test. A bug witness also depends on its engine and setup; preserve those details when adapting it.
