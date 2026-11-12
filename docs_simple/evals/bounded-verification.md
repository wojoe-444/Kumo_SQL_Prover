# Checking every small database

[Simple eval index](README.md) · [Full reference](../../docs/evals/bounded-verification.md)

Bounded verification asks: do these queries agree on every database with at most N rows per table, within the modeled SQL and assumptions?

At a bound of 3, each table can have zero, one, two, or three rows. Z3 reasons about symbolic cell values and NULLs, rather than trying only a few randomly chosen values.

Columns keep to the values they can really hold: a `DATE` stays between year 1 and year 9999 and a `NUMERIC(10, 2)` keeps two decimals, so a counterexample never needs a date or number the database could not store. See the [full reference](../../docs/evals/bounded-verification.md) for the details.

Joins used to be the expensive part: with three rows per table, joining three tables meant reasoning about 27 possible combined rows, and eight tables did not fit. Most joins follow a key, for example each order has at most one customer. The checker now notices that (from the keys the schema declares) and keeps one combined row per order instead of one per pair, so the same three-table join is three rows. The answers are the same; they arrive much sooner, and some checks that used to time out now finish.

The checker also asks an easier question first. Two queries often do the same multiplication, division and rounding on the same numbers, just reached by different routes. Instead of working through that arithmetic, the checker first treats "multiply", "divide" and "round" as black boxes: the same inputs give the same output, and nothing else is known. If the queries cannot differ even then, they cannot differ at all, and that is found quickly. If they might differ, the checker tries the suggested data for real, and falls back to the full arithmetic when the data shows no difference.

Three smaller changes help in the same way. A join is now encoded identically however it is written: `a JOIN b ON ...` and `b, a WHERE ...` become the same rows in the same order, so the checker does not have to rediscover that they are the same join. A division by a row count, as in an average, is written out case by case (divide by 1, by 2, by 3), because a count of at most three rows can only be one of those; that is exactly the same question in a form the solver handles easily. And the solver now simplifies the whole question before it starts searching. For example, two answers to one LeetCode problem that differ only in writing `100 * SUM(...) / COUNT(*)` or `SUM(...) / COUNT(*) * 100` inside a `ROUND` used to run into the time limit at three rows; they are now compared in about a second.

More SQL is understood than before, including `ROLLUP`, `CUBE` and `GROUPING SETS`, aggregate `FILTER`, `FETCH FIRST`, window functions over grouped rows, and date literals. SQL the checker still does not model gives **unknown**, never a verdict.

## Read the result

- **Bounded, 3 rows** means no difference exists within that model and row limit. It says nothing about larger databases.
- **Counterexample** means the solver found data and execution confirmed different results.
- **Unknown** means unsupported SQL, a time limit, or an unconfirmed model prevented an answer.

Bounded evidence is separate from an unbounded equivalence proof. It is also stronger within its row limit than random execution of just a few databases.

## In the app

**Settings → Solver** controls the row bound; zero turns it off. Compare queries and Compare tables display the bounded result alongside other evidence. Saved schemas and constraints are needed to describe the source tables.

The checker replays candidate counterexamples in DuckDB and shuffles table rows to avoid treating arbitrary tie choices as a real difference. Its arithmetic and runtime-error assumptions still matter.

For SQLite, only replayed counterexamples are offered: differences between the symbolic model and SQLite mean a successful bounded search cannot be reported as bounded equality there. The full guide covers encoding tests, assumptions, rerun commands, and scores.
