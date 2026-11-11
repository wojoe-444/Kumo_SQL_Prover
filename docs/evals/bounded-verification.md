# Bounded verification (z3, at most N rows per table)

[Plain-language version](../../docs_simple/evals/bounded-verification.md)

`kumo_sql_prover.bounded_equivalence` checks that two queries return the same bag of rows on **every database with at most N rows per table**, with symbolic values, so the solver covers every combination of values and NULLs up to the bound. It is a third evidence level, next to an unbounded proof and agreement on executed random databases:

| Evidence | What it says | Where it comes from |
| --- | --- | --- |
| Unbounded proof | equal on every database that meets the declared facts | `algebraic_equivalence`, `smt_equivalence` ([sqlsolver.md](sqlsolver.md)) |
| **Bounded, N rows** | equal on every database with at most N rows per table; says nothing about larger ones | `bounded_equivalence` (this page) |
| Executed datasets | equal on the random databases that were run | `counterexample`, `result_equivalence` |
| Counterexample | a database, replayed on both queries, on which they differ | any of the above, after a replay |

Bounded is not a proof and is never reported as one. The approach is the one VeriEQL describes ("VeriEQL: Bounded Equivalence Verification for Complex SQL Queries with Integrity Constraints", OOPSLA 2024, https://github.com/VeriEQL/VeriEQL): tables of N symbolic tuples, queries as symbolic relations, constraints as assertions. The code here is written from the paper and shares none of VeriEQL's source, which is CC BY-NC-SA 4.0 (the same reason `docs/verieql.md` reads only its benchmark files).

## How it works

Each table has N slots. A slot has a presence flag and, per column, a symbolic value and a NULL flag. Slots fill from the front (a present slot implies the one before it is present), which removes symmetric models. A query becomes a *relation*: a list of rows, each with a presence flag and symbolic cells.

- `WHERE`, `ON`, `HAVING`: conjoin the row's presence with the condition (three-valued logic: NULL does not pass).
- Joins (inner, left, right, full, cross, `USING`, `NATURAL`): every pair of rows, with an outer row for each unmatched side.
- `GROUP BY`: slot *i* is a group representative when it is present and no earlier present slot has the same key (NULLs equal each other); aggregates run over the members. Without `GROUP BY`, an aggregate query always returns one row. A column that is neither grouped nor aggregated is an *arbitrary pick* from the group: a fresh choice constrained to a member, so two queries are equivalent only if they agree whatever the pick.
- `DISTINCT`, `UNION [ALL]`, `INTERSECT [ALL]`, `EXCEPT [ALL]` (branches matched by position; `BY NAME` and `CORRESPONDING` are first rewritten to the positional form by `kumo_sql_prover.set_operations`, and the answer is unknown when a branch's columns are not known), `IN`, `NOT IN`, `EXISTS`, correlated and scalar subqueries, CTEs, `CASE`, `COALESCE`, `NULLIF`, `LIKE` (literal patterns), arithmetic, `ROUND`, date arithmetic in days, `ORDER BY ... LIMIT ... OFFSET`, `FETCH FIRST n ROWS ONLY`, window functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, `FIRST_VALUE`, aggregates over the default frame or the whole partition), also over the rows of a grouped query.
- `ROLLUP`, `CUBE`, `GROUPING SETS` and `GROUPING()`: one grouping per set, with the keys a set leaves out NULL. Aggregate `FILTER (WHERE ...)`, `BOOL_AND` and `BOOL_OR`. Date and timestamp literals, a date moved by a constant number of days (or by months and years when the date is a literal), numbers written as strings where a number is compared, and column lists on an alias (`AS t (a, b)`).
- Uninterpreted predicates such as `B(X)` (VeriEQL's symbolic predicates) become z3 functions, so equivalence has to hold for every meaning of `B`.
- Constraints: NOT NULL, keys, foreign keys, enum values, consecutive-id columns and cross-row predicates become assertions.

**Joins on a unique key** (`src/kumo_sql_prover/bounded_keyed_joins.py`). "Every pair of rows" makes three tables of three rows joined in a chain 27 symbolic rows, and every aggregate, grouping and bag comparison above them grows with that number; an eight-table TPC-H join did not fit at all. Most joins follow a key, though: `orders JOIN customers ON orders.customer_id = customers.id` gives each order at most one customer. Such a join gets one joined row per row of the *other* side: it is present when some row of the keyed side matches, and its keyed-side cells are an if-then-else over that side's rows (a single choice, because two present rows cannot both match). A `LEFT JOIN` row with no match takes NULLs in the same row. The chain of three tables is then three rows.

- A relation carries its *keys*: sets of columns whose non-NULL combinations are unique among the rows present. They start from the keys the schema declares (which the symbolic database asserts) and survive a filter, a rename, a projection that keeps the columns as they are, and a `LIMIT`; `DISTINCT` makes all columns one, and a `GROUP BY` whose keys are all in the output makes those one. A projected column is recognised by the identity of its symbolic cells, not by its name, so an alias that shadows a column is not mistaken for the key.
- A join is read this way only when its condition is a conjunction that holds, for every column of one key, an `=` between that column and plain arithmetic over the other side (columns, literals, `+ - *`, casts), compared without converting the key column (INT64 against a decimal is fine, that conversion is one-to-one). Which side a column belongs to is read from its position in the joined row, never from its cells: in a self-join both sides hold the same cells. `<=>` does not qualify (NULL keys match each other), nor a condition under `OR`, part of a two-column key, or a computed column.
- A chain of plain inner joins (written with `JOIN ... ON`, with commas and a `WHERE`, or as a `CROSS JOIN`) is encoded one way however it is written: every `ON` and `WHERE` conjunct that reads only the joined tables is the condition of the first join after which all its columns are in scope, which for inner joins selects the same rows, and the conjuncts of one join are put in one order. Where no order of rows can be seen, a tie between join orders goes to the table whose cells were made first, not to the one written first. `a JOIN b ON c WHERE d` and `b, a WHERE d AND c` then give the same rows in the same order with the same presence terms, so the solver has nothing to work out about the join.
- An inner join keyed on its right side keeps the joined rows in the order the row-per-pair encoding gave them, so it is always used. The other forms change that order (an outer join's unmatched rows move, the keyed side on the left turns the order around, and a comma chain is joined in the order that keeps it smallest), so they are used only in a query with no `LIMIT`, `OFFSET` or window function anywhere, where no order of rows can be seen.
- Any other join is encoded as before, and `check_bounded(..., keyed_joins=False)` gives the old encoding for comparison. `tests/test_bounded_keyed_joins.py` checks, on random databases that respect the keys, that both encodings return the same bag for 47 query shapes, that pairs that differ are still refuted with a replayed database, and that the near misses above are not compacted. `evaluate` now asserts the schema's constraints, so a concrete database that breaks a declared key is declined instead of evaluated.

**A looser question first** (`src/kumo_sql_prover/bounded_abstraction.py`). Two answers to one LeetCode problem usually compute the same thing by another route, such as `ROUND(SUM(price * units) / SUM(units), 2)` over a join written two ways. The product of two unknowns, a division by an unknown and the rounding inside `ROUND` are what the solver spends its time limit on, although both queries apply them to the same values. So each bound is first asked with those operations *uninterpreted*: a product of two or more non-constant terms, `/`, integer division and modulo by a non-constant, and real-to-integer conversion become unknown functions of their operands (`x * 3` and `x / 2` stay as they are). The real operations are one meaning those functions can take, so if the looser formula has no model the exact one has none either, and that answer usually comes in a fraction of the time, because equal operands give equal results without any arithmetic. A model of the looser formula proves nothing (the solver may have picked a meaning under which 2 * 3 is 7), so it is used only as a database to try: the replay runs both queries on it, and a difference there is a counterexample like any other. A model the replay does not confirm, or a timeout (the looser question gets a quarter of the time limit), leads to the exact question as before. `check_bounded(..., abstract_arithmetic=False)` skips the looser question. `tests/test_bounded_abstraction.py` checks on random formulas that "no model" carries over, that pairs equal only by a fact of arithmetic (`price * (units + 1)` against `price * units + price`) are still decided by the exact question, and that differing pairs are reported only with a replayed database.

**Division by a row count, case by case** (`src/kumo_sql_prover/bounded_count_division.py`). `AVG(x)` is a sum divided by a count, and answers divide by `COUNT(*)` everywhere (`ROUND(100 * SUM(...) / COUNT(*), 2)`). A division by a term that is not a constant is nonlinear to the solver, although under the bound a count is one of a handful of numbers. A division (`/`, integer division, modulo) whose divisor can only take a few known values is therefore written as a choice of divisions by constants: `x / n` becomes `x / 1` when `n = 1`, `x / 2` when `n = 2`, and so on. Each branch is the original division under the condition that selects it, so the formula has exactly the models it had; nothing is abstracted. The values a divisor can take are read from its shape alone (a numeral, an if-then-else of such terms, a sum of them, a numeral times one, at most 16 values); a column or a `SUM` of a column has none and is left alone. `tests/test_bounded_count_division.py` checks with the solver that each rewritten formula is equivalent to the original, that integer division stays integer division, and that a different count (`COUNT(*)` against `COUNT(x)`) is still refuted with a replayed database.

**Simplify, then search.** Each bounded question is one large formula asked once, so it goes through Z3's standard simplifications (constants propagated, solved equalities substituted, unconstrained terms dropped) before the SMT core, under the same deterministic work cap and time limit as before (`prepared_solver`). Most of what two encodings of one problem share, the same table cells joined another way, disappears in that step. On the LeetCode sample this alone answers several times faster and decides pairs the plain solver timed out on. A model is still believed only after the replay.

Two relations are compared as bags: a difference exists when some row's multiplicity differs. `sat` gives a model; `unsat` is "equivalent within the bound".

Anything the encoding does not model (`GROUP_CONCAT`, regular expressions, `UPPER`, date parts, recursive CTEs, a window frame counted in rows up to the current row or with `EXCLUDE` (it depends on how ties are placed), MySQL's `date + 1` and `WITH ROLLUP`, a month or year interval on a date that is not a literal, a cast between numbers and strings or from a real to an integer, an unknown function outside `WHERE` and `ON`, a `LIMIT` without `ORDER BY`, a `LIMIT` or `OFFSET` on a set operation itself, `SELECT * EXCEPT/REPLACE/RENAME/ILIKE`, a table missing from the schema) raises `Unsupported` and the answer is **unknown**, never a verdict.

## Replay, assumptions, and what a result means

A counterexample is returned only after both queries were executed on DuckDB over the model's database, the two result bags differ, and the difference survives three shuffles of every table's rows (so a `LIMIT` tie, or MySQL's arbitrary pick of an ungrouped column, never produces one). A model the replay does not confirm gives `unknown`, after two retries with printable strings and quarter-step reals.

Column domains: a `NUMERIC(p, s)` column holds only values with `s` decimals and `|value| < 10**(p - s)`, and a `DATE` or `DATETIME` / `TIMESTAMP` column only values from year 1 to year 9999, so the solver cannot pick a value the replayed database cannot hold. (Until 2026-10 a model date below year 1 was clamped to `0001-01-01` when decoded, so two rows that differ only in a DATE key column came out equal and the counterexample repeated the primary key; the range is now a constraint instead and nothing is clamped.)

Assumptions reported with every result: bounded databases only; exact arithmetic (no `FLOAT64` rounding or integer overflow); runtime errors are not modeled (division by zero gives NULL, a scalar subquery with several rows takes the first); strings compare case-sensitively; results are compared as bags; ties in `ORDER BY` (`LIMIT`, `ROW_NUMBER`, `LAG`, `LEAD`, `FIRST_VALUE`) are broken by row position, so a difference that depends on the tie-break is reported only if it survives the replay's shuffles, otherwise the answer is unknown.

`check_bounded` tries bounds 1, 2, ..., N in turn, so a counterexample is reported at its smallest size and a timeout still reports the largest bound that finished (`bounded, 2 rows`).

## Testing the encoding

A wrong encoding could hide a difference, so the encoding itself is tested against DuckDB:

- `tests/test_bounded_equivalence.py` compiles 30 query shapes (joins, groups, windows, set operations, subqueries, NULL cases), pins the symbolic database to random concrete ones and compares the result with DuckDB's.
- `python tools/bounded_bench.py differential SUITE` does the same on every query of a VeriEQL suite. Any mismatch is a bug and must stay 0 before a bounded number is quoted.
- `run` cross-checks every bounded verdict: a counterexample against a pair the unbounded prover proved, VeriEQL's published counterexample (replayed on DuckDB) that fits inside the bound, or a random-search counterexample that fits inside the bound each count as **wrong**.

## In the app

Settings → Solver has a "rows per table, bounded check" setting (0 turns it off; the default is 3). **Compare queries** and **Compare tables** show the result as its own label, "bounded, 3 rows", or "different results" with the database, next to the unbounded answer. The check runs when nothing was proven, on the columns, types, NOT NULLs and keys of the saved BigQuery catalog, so queries over tables without a saved schema stay unknown.

## Running it on the evals

```
python tools/bounded_bench.py run literature --rows 3
python tools/bounded_bench.py run calcite --rows 3
python tools/bounded_bench.py run leetcode --rows 3 --every 24
python tools/bounded_bench.py run sqlsolver-calcite --rows 3     # also sqlsolver-spark, -tpch, -tpcc, qed, rbot, cosette, spes
python tools/bounded_bench.py run singh --rows 3
python tools/bounded_bench.py differential calcite
```

Each run first takes the eval's own baseline verdict (unbounded prover and executed search), then the bounded check, and prints the two side by side. The scoreboard rows are in `benchmarks/results/bounded-*.json`.

## Results

Measured 2026-10-02 at 3 rows per table.

| Suite | Pairs | Bounded, 3 rows | Different (replayed) | Timeout | Unsupported | Unknown |
| --- | --- | --- | --- | --- | --- | --- |
| SQLSolver Calcite | 232 | 212 | 0 | 1 | 18 | 1 |
| SQLSolver Spark SQL | 127 | 106 | 0 | 0 | 20 | 1 |
| SQLSolver TPC-H | 22 | 7 | 0 | 4 | 11 | 0 |
| SQLSolver TPC-C | 19 | 19 | 0 | 0 | 0 | 0 |
| QED Calcite | 375 | 363 | 0 | 3 | 9 | 0 |
| R-Bot Calcite | 45 | 23 | 0 | 0 | 22 | 0 |
| Cosette examples | 60 | 52 | 6 | 0 | 2 | 0 |
| SPES Calcite | 34 | 26 | 3 | 0 | 5 | 0 |
| Singh & Bedathur LeetCode pairs | 1006 | 824 | 71 | 97 | 13 | 1 |
| VeriEQL Literature | 64 | 28 | 24 | 4 | 4 | 4 |
| VeriEQL Calcite-397 | 397 | 276 | 2 | 4 | 112 | 3 |
| VeriEQL LeetCode (1,000-case sample) | 1000 | 452 | 231 | 123 | 179 | 15 |

Every row has 0 wrong. A *timeout* is a pair whose 3-row check timed out, some after finishing a smaller bound (reported as e.g. "bounded, 2 rows"); *unknown* is a model the replay did not confirm or a parse failure. For pairs the unbounded prover already proves, the bounded verdict is a cross-check of the encoder. On pairs the proofs leave unknown, the bounded check refutes pairs the 150-database random search missed (LeetCode sample: 90 refutations on pairs the executed search called "agrees"; Literature: 6 of 25), each replayed on DuckDB. Singh & Bedathur's 1,794 pairs the prover already refutes are not rerun.

Disputed labels: Cosette's `testDecorrelateTwoIn` and SPES's three semi-join cases (`testSemiJoinRule`, `testSemiJoinRuleExists`, `testSemiJoinTrim`) are labelled equivalent, but a replayed bounded counterexample exists; they are counted as label disputes, not as wrong.

Checks behind "0 wrong": the encoding agrees with DuckDB on 92 Literature, 511 Calcite and 1,504 LeetCode-sample queries (`differential`, 0 mismatches; queries with an ungrouped column are excluded because their answer is arbitrary). The first LeetCode run found the encoder assumed ORDER BY ties away, which hid real differences in 18 pairs; ties are now broken by row position and the same run is 0 wrong. A doubled primary key in one family of pairs is read as the shared harness reads it (the later one wins).

The bounded check is developed with these suites in view (tuned on test), and held-out cases are not reserved yet. Caveats per suite are in `benchmarks/results/bounded-*.json`.

## SQLite (SQL-IQ)

`SQLiteReplay` replays on the standard-library `sqlite3` module, for queries in SQLite's dialect. For SQLite only counterexamples are offered: the encoding's `LIKE` (case-sensitive) and `/` (exact) differ from SQLite's, so "no counterexample" is reported as unknown. `tools/sqliq_bench.py` uses it as a last step of the Equivalence Judge, after the random and targeted databases agree: a database of at most 3 rows per table on which SQLite itself returns different results answers "no". On the 1,390 pairs it settles 16 more (15 match the labels, 1 "equivalent" label is a label dispute), moving the score from 1165 to 1179 ([sql-iq.md](sql-iq.md#scores)).
