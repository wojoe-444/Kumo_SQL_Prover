# Checking the VeriEQL benchmark pairs

[Simple eval index](README.md) · [Full reference](../../docs/evals/verieql.md)

This evaluation reads VeriEQL's benchmark query pairs, schemas, and constraints. KumoSQL uses its own deterministic prover and database generator; it does not call a language model or reuse VeriEQL's implementation.

The corpora include LeetCode solutions, pairs from equivalence research, and Calcite optimizer tests. They contain both equivalent and different queries, without gold labels in the input files.

## Read the verdict by its evidence

- **Equivalent** is an unbounded proof under the stated supported constraints and assumptions. Additional random databases cross-check it.
- **Different** includes a database satisfying the applicable constraints on which the queries return different results.
- **Unknown** means neither answer was established.

Candidate counterexamples are replayed and shuffled to avoid differences caused only by arbitrary row order or tied LIMIT choices. Some engine-sensitive cases receive additional checks with the optimizer disabled. Picks the shuffle cannot move (`ANY_VALUE` over several values, a `LIMIT` that cuts through ties) get a further check, so a difference through such a pick stays unknown. The full reference has the details.

## Why the test run is cheaper

Almost all of a run is DuckDB answering both queries on many tiny databases. The search now uses a single DuckDB thread, rewrites only the tables whose rows changed, and does not rerun a database it has already seen both queries agree on. The databases it tries are the same, so the answers are the same, with one known exception: when a query picks an arbitrary value from a group, DuckDB's pick can depend on how the table was filled, so one LeetCode pair whose refutation rested on a lucky pick is no longer refuted. Timings and the pair are in the [full reference](../../docs/evals/verieql.md); they come from one container and a sample of the pairs, not the whole corpus.

## Latest scores

On 2026-11-12 the whole LeetCode set (23,994 pairs) was rerun: 9,022 pairs are proved equivalent (4,787 before), none wrongly. Calcite-397 went from 337 to 342 proved. The gain comes from new rules for families of spellings of one LeetCode answer (matching with `COUNT(DISTINCT ..)`, values that occur twice, `NOT IN` against the table's own key) and from other rules merged since October. Every fifth pair was never read while writing them; it is proved about as often as the others (37.3% against 37.7%). Many unproved pairs agree on every generated database but differ on a NULL the databases never contain; they stay unknown. Details and the per-problem counts are in the [full reference](../../docs/evals/verieql.md).

## Constraints matter

The input can declare keys, non-NULL columns, and other restrictions. When a table declares two primary keys, each one is kept as a key of its own. A database violating those restrictions is not a valid counterexample to a conditional claim. Read which constraints the prover models and which the execution generator enforces.

The full guide documents translation, constraint handling, known non-equivalent pairs, commands, and separate proof/executed score rows. [Bounded verification](bounded-verification.md) covers KumoSQL's separate row-limited checker, inspired by the bounded approach used in VeriEQL's research.
