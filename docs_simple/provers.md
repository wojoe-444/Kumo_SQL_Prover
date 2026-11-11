# How KumoSQL checks whether queries match

[All simple guides](README.md) · [Full reference](../docs/provers.md)

None of these checks uses the network. Calling a prover or a rewrite from Python makes no request; the one optional lookup (the columns of tables the project does not define) is off unless you turn it on. See [pipeline analysis](https://github.com/wojoe-444/KumoSQL/blob/master/docs_simple/pipeline-analysis.md).

Two queries are equivalent when they return the same results under the comparison's rules. Usually that means the same rows with the same duplicate counts, ignoring unspecified row order. Column names, types, ordering, and declared data guarantees can also matter; read the check's assumptions.

Suppose you replace `WHERE 1 = 1` with no WHERE clause. A checker can establish that the removed condition never filtered anything. More complicated changes need stronger reasoning.

## The different kinds of checks

| Check | What a successful result establishes |
| --- | --- |
| Structural proof | After safe simplifications, both queries have the same supported structure |
| SMT / algebraic proof | Mathematical reasoning establishes equivalence for the supported SQL under listed assumptions |
| Bounded verification | No difference exists in the model for any database up to a stated row limit |
| Executed comparison | Both queries matched on the particular datasets tested |
| BigQuery dry run | The statements plan and, for a rewrite check, their output schemas match |

The first two can give an unbounded proof: the database's row count has no fixed test limit. The bounded check covers every modeled value combination, including NULLs, within its row bound. Random-data execution only covers the databases actually tried. A dry run does not compare returned values.

## Which prover is which?

The structural prover compares normalized query trees. The SMT prover uses Z3, a solver that reasons about constraints. The algebraic prover treats duplicate row counts as arithmetic, helping it reason about joins, unions, and aggregates. The optional Java SQLSolver backend is a separate setup; ordinary Python proofs do not require Java.

From a checkout, install solver and execution support with:

```sh
python -m pip install ".[smt,execution]"
```

A small Python example:

```python
from kumo_sql_prover import prove_equivalent

result = prove_equivalent(
    "SELECT id FROM orders WHERE 1 = 1",
    "SELECT id FROM orders",
)
print(result.status.value)  # proven_equivalent
```

## Comments

A comment such as `-- note` does not change what a query returns, so it does not stop two queries from being proven the same. The one exception is a comment holding a Dataform `${...}` expression: Dataform fills those in even inside comments, and the result can turn into real SQL, so such a comment is compared like code. For example `SELECT 1 AS a -- note` and `SELECT 1 AS a` are proven the same. See the [full reference](../docs/provers.md) for details.

## Moving a subquery into a WITH

Before comparing two queries KumoSQL moves each subquery in a `FROM` into a `WITH` so both sides are in the same shape. A subquery that reads a column of the query around it, or a table a nested `WITH` defines, would mean something else once moved, so it stays where it is and the comparison is made with it in place. Without this, a lifted form that no longer ran could be called equal to the query it came from. A column with no table in front of it, read over a real table, still counts as that subquery's own; KumoSQL cannot tell without the table's columns. See the [full reference](../docs/provers.md).

## Which table does a bare column belong to?

In `SELECT id, name FROM orders JOIN customers ON ...`, the provers have to work out for themselves which table `id` and `name` come from, and inside a subquery whether a name means the subquery's own table or the query around it. Two queries that differ only in such a pick can look the same to a prover that picks wrong. So a separate reader goes through the query text, works out the owner of each bare column on its own, and compares it with what the prover chose. If they disagree, the pair is reported as unknown. If the reader cannot tell (it does not know a table's columns, or the name could mean several things), the proof stands, because refusing every such case would throw away correct proofs. This reader does not check every place the algebraic prover rewrites columns; the [full reference](../docs/proof-safeguards.md#the-provers-own-column-resolution) lists what it covers, how often it could decide on the benchmark sets, and what it cannot do.

## What if the answer is unknown?

It may mean the SQL uses an unsupported feature, a table's columns are missing, or the solver reached its time or work limit. It does not establish that the queries differ. Text that cannot be read at all (an unclosed quote, very deep nesting) and queries that would take too long to even set up (a chain of CTEs that each read the previous one twice) also come back unknown rather than as an error.

Some text is refused on purpose, before it is read, because reading it would end the program or never finish. Three shapes do this: more than 100 brackets (or `CASE`s) inside each other, more than 10 calls such as `DATE(DATE(DATE(..)))` inside each other, and more than 10 joins in a row with no `ON`. For example, twenty-four `JOIN t` in a row with no condition would use more memory than most computers have, so the answer is "cannot be read" at once. Ordinary long queries are not affected: these limits do not refuse a sum of 3,000 columns or a list of 5,000 values. The limits were tested on 113 kinds of repeated SQL in four dialects; they are not a proof that no other shape exists. See [the full reference](../docs/provers.md#algebraic-prover-and-sqlsolver-no-admin-rights) for the measurements.

Each solver check has a fixed work limit as well as a time limit. When the work limit stops a check, the same query pair gets the same answer on a fast or a busy machine; the result says "timed out" only when the clock stopped it, which a faster machine might not. With a very small time limit the clock usually stops a check first, so whether a pair is proven or unknown can still vary with machine load; it never flips between proven and different. The bounded check follows the same work limit, and a number such as `1e100000000` is declined everywhere rather than read. `GROUP BY ALL` is read as the columns it actually groups by, so an aggregate-only query still returns its one row on an empty table.

`SELECT * EXCEPT (b)`, `* REPLACE (...)` and `* RENAME (...)` change which columns a query returns, so the provers either apply them or say unknown. Example: `SELECT * EXCEPT (b) FROM t` is no longer treated as the same query as `SELECT * FROM t`. The cases the SMT prover cannot apply (a name the star does not have, `* ILIKE`) come back unknown, and the bounded checker always declines a modified star. See the full reference for details.

A replayed counterexample does establish a difference: the report includes a database where the results disagree. Matching a few random databases does not establish that no counterexample exists. That agreement is also weaker than BigQuery agreement: the queries run on DuckDB, decimals are compared to 12 significant digits, and column types are not compared.

Two details of the structural check: it drops a result ordering only when no sort key could raise an error (a `LIMIT` that cannot cut any row does not change that), and it treats a function that might be a user-defined one as a different function when its spelling differs, because BigQuery reads those names case-sensitively.

## Comparing scripts that change a table

An `UPDATE` or `DELETE` only reports how many rows it touched, and two updates can touch the same number of rows while writing different values: `UPDATE t SET a = 1 WHERE TRUE` and `UPDATE t SET a = 2 WHERE TRUE` both touch every row. The executed comparison therefore works on a private copy of the table and compares the rows that are left afterwards, not the count. This also covers `TRUNCATE`.

Some writes cannot be compared faithfully, so the check answers `error` instead of guessing: `MERGE` and `UPDATE ... FROM` (BigQuery fails when a target row matches several source rows, while DuckDB quietly picks one), an `UPDATE` or `DELETE` with no `WHERE` clause (BigQuery rejects it), and a statement whose target table cannot be identified. The limit is that a script writing several tables is still compared on the last table it wrote. See the [full reference](../docs/provers.md) for details.

Repeating the same comparison should find the same counterexample regardless of earlier comparisons or unrelated imports. For example, removing a lookup join can lose its treatment of a user whose plan is NULL: the join drops that user, while reading the users table keeps them. The solver isolates its candidate search to avoid changing this witness with process history. The search can still miss a difference or run out of time; returned examples must respect the declared data guarantees and make the query results differ.

An example database also has to be one that could exist. Three things used to go wrong. A text column could be shown holding the number 1, or a whole-number column holding `'A'`. Text could be put in an order it does not have: the check reported that `WHERE s < ''` and a filter that keeps nothing are different, with the example `s = 'A'`, although nothing sorts before empty text. And with about ninety text values in the queries, the search for a readable example never came back, whatever time limit was set. Now each value matches the declared type of its column, text is ordered the way BigQuery and DuckDB order it (an example that breaks this is thrown away, and the answer is "unknown" if no better one is found), and every step of the search has a time limit.

The limits: this needs declared column types, so a column without one can still be shown with a number where the queries compare it with text. Text order is plain character order, so an example may not hold in a database set up to ignore upper and lower case. The [full reference](../docs/provers.md#smt-equivalence-prover) has the details and the tests.

## Strings compared with numbers

`WHERE '2' <> 2` looks like a condition that is never true, but engines disagree: MySQL turns the string into a number and finds them equal, DuckDB and PostgreSQL cast it the same way, and BigQuery refuses the query. Treating the two as always different once led the checker to say this query matches one with no filter, when real engines return different rows.

The checkers no longer guess. When the string is a small whole number (`'2'`), MySQL, DuckDB and PostgreSQL all read it as that number, so `'2' = 2` is simply true, and a column declared as an integer compared with `'2'` is compared with `2`. BigQuery refuses such queries, so they stay "not proven" there. Any other string-versus-number comparison is treated as unknown, so `WHERE '2.5' <> 2` is "not proven" equal to the query without the filter, while two queries that make the same such comparison in the same way can still match. A few other forms (for example `IN` lists, or one column compared with both a string and a number) are simply declined. Comparing two numbers or two strings is unaffected.

The checkers also work out what kind of value a column holds from the query itself. If one table's column is matched against a string (`a.x = 'abc'`) and joined to another table's column that is matched against a number (`b.y = 0`), MySQL still joins them, because it reads `'abc'` as 0, so the checker no longer calls that query empty. The same goes for `COALESCE(n, 'abc') = 0` or a `CASE` that returns a string in one branch and a number in another, and for values that pass through a derived table or a `WITH` name. The same holds when the column passes through `SELECT *`, is written without its table name, or comes out of a `UNION`. The same holds when the column passes through `SELECT *`, is written without its table name, or comes out of a `UNION`. Columns that are only ever compared with their own kind are unaffected. One more MySQL quirk is handled: if a column has no declared type and is compared with two different strings such as `'x'` and `'y'`, an integer column holding 0 matches both, because MySQL reads any non-numeric string as 0. A claim that such a query is empty is now accepted only if it also holds when the strings are read as numbers. The same table read under two names (`t AS a JOIN t AS b`) counts as one column here, since a key can make `a` and `b` the same row. Declaring the column as text brings the old behavior back. A declared type is tied to its own table: when a query writes `s` without a table name, another table's text column called `s` does not make it text, unless the list of each table's columns shows that only the declared table can hold it. For example, with `u.s` declared as text and `t.s` not declared, `SELECT k FROM t WHERE s = 'x' AND s = 'y'` is not called empty. See the [full reference](../docs/provers.md) for the exact list.

When results are compared by running both queries, a value keeps its kind: `TRUE` is not `1`, a NaN is not the text `NaN`, and a struct is not a list of pairs. Floats are compared rounded to 12 significant digits unless you ask for an exact comparison, and each result records which one it used.

The structural checker also refuses a query BigQuery would plainly reject (for example `HAVING` with no grouping, or a column that the subquery it reads does not have), because two such queries matching proves nothing about results. It also refuses a type name BigQuery does not have, such as `FLOAT`, `INT32` or `VARCHAR`, in a cast, an `ARRAY<...>`, a typed value or a script's column list (every prover and the bounded check do, not only the structural one) (BigQuery has `FLOAT64`, `INT64` and `STRING`), because KumoSQL would otherwise read the invalid name as the valid one and call the two casts equal. For example, `CAST(x AS FLOAT)` against `CAST(x AS FLOAT64)` is not proven. It only recognizes these cases. See the [full reference](../docs/provers.md).

## A limit that moves into a projection

KumoSQL knows that `SELECT d.b + 1 FROM (SELECT a AS b FROM t ORDER BY a LIMIT 1) AS d` returns the same row as `SELECT a + 1 FROM t ORDER BY a LIMIT 1`: it takes the first row, then computes the value. That rewrite had three holes, all now closed. When the outer query contained its own subquery, such as `(SELECT MAX(b) FROM u)`, renaming `d.b` to `a` also changed which column that subquery read, so two different queries were called equal. A value like `RAND()` that the outer query used twice would be drawn twice after the rewrite. And a `GROUP BY 1` could point at a different item once the select list changed. In each case the checker now declines, so the pair is "not proven" rather than wrongly "proven", and the common rewrites still go through. The evidence is a handful of hand-made witnesses plus fuzzing, not a proof that no such hole is left. The [full reference](../docs/provers.md) has the details.

Proofs may depend on declared keys, non-NULL columns, arithmetic assumptions, or restrictions on runtime errors. Check those before applying a change to real data. [Constraint-dependent rewrites](constraint-rewrites.md) explains data guarantees, and [bounded verification](evals/bounded-verification.md) explains the row limit.

## A subquery compared on its own is read in your dialect

Some subqueries in `FROM` are too involved for the prover to read row by row, for example one that groups and sums. It treats such a subquery as a black box, and when the two queries hold differently written black boxes it runs a small proof of its own to see whether they are the same table. That small proof used to read both subqueries as BigQuery, even when you said the queries were MySQL, PostgreSQL or DuckDB.

The dialects disagree about decimals. `1.00` is an exact decimal in MySQL and a floating-point number in BigQuery. So under MySQL the prover could not see that `SUM(price * (1 - discount))` and `SUM(price * (1.00 - discount))` are the same sum, and it called two filters equal that are not: `x = 0.1` and `x = 0.10000000000000000001` are the same floating-point number in BigQuery but two different decimals in MySQL, where a row holding exactly 0.1 passes only the first. Now the small proof reads the subqueries in the dialect you named. When a subquery cannot be written back in that dialect without changing it, it is not compared at all, which can only cost a proof.

The evidence is the two examples above as tests in three dialects and a rerun of the benchmarks that read other dialects; it is not a proof that no other step still assumes BigQuery. The [full reference](../docs/provers.md) has the details.

## Broken input gets an answer, not a crash

Sometimes the text handed to a checker is not a query it can read: a quote that never closes, sixty parentheses inside each other, a table written `t..u`, or a table description with no columns. These used to stop the program with an internal error, which is hard to tell apart from a real bug and takes down whatever tool was calling the checker. Now each checker gives its ordinary "I cannot tell" answer: the provers say "not proven", the run-both-queries check says "error" with the reason, and the bounded check says "unknown".

For example, comparing `SELECT '` with `SELECT a FROM t` now returns "not proven" with a parse error as the reason. The cases come from an automated search over the entry points and are kept as tests; other odd inputs nobody has tried may still fail. The [full reference](../docs/provers.md) lists them.

## Numbers and errors

BigQuery numbers have traps. A whole number past about nine quadrillion loses its last digits once it meets a decimal, `0.1 + 0.2` is not `0.3`, and dividing by zero is an error. Worse, BigQuery does not promise to filter rows before it computes the select list, so `SELECT x / y FROM t WHERE y <> 0` can still divide by zero.

When you tell the prover the column types, it reads numbers the way BigQuery does and keeps track of every operation that could fail (a division, an integer overflow, a `CAST`). It then says what a rewrite does to those failures: nothing changes, it removes one that the original had, or it adds one the original did not have. In the last case it refuses to call the pair equal and shows why. For example, replacing `IF(y = 0, 0, x / y)` with `x / y` is flagged, because the `IF` was what kept the division away from zero.

A sum can fail too: adding up the numbers in a group can leave the range of a 64-bit integer. The prover compares each `SUM` group by group. If a rewrite adds up exactly the same rows, in the same groups, as the original, the two fail together. If it moves a `HAVING` on the grouping column into `WHERE`, the original also adds up the groups `HAVING` throws away, so the rewrite is safer; moving it the other way makes the rewrite add up groups the original never did, and the prover flags that and shows a small table where the sum overflows. For example, `SELECT y, SUM(x) FROM t WHERE y > 0 GROUP BY y` rewritten as `... GROUP BY y HAVING y > 0` can fail on two rows with `y = 0` and the largest integer in `x`. Anything the prover cannot match this way stays unknown, in particular a sum that is split into partial sums and added up again: whether BigQuery fails on a partial sum that overflows when the total does not is something the documentation does not settle, so the prover does not guess. Two more unverified points: that BigQuery really adds up a group that `HAVING` then drops (an optimizer may skip it), and that `SUM(DISTINCT)` and window sums fail on overflow like `SUM`.

The limits: without declared types the older, looser reading applies and the result lists that runtime errors were not modelled. Exact decimal rounding is reasoned about only for `NUMERIC` and `BIGNUMERIC` values whose number of decimal places is known (a column declared with that type, a plain decimal in quotes cast to it, and `+ - * /`, `CAST` and `ROUND` over those): `n * m` is rounded to nine places, so `ROUND(n * m, 9)` is the same number and `n / 3 * 3` is not `n`. Where the number of places is not known (a sum, a `CASE`, a floating-point value) the pair comes back unknown; the rule that products and quotients round half away from zero is not confirmed in the documentation the project has. Floating-point columns are still assumed never to hold `NaN`, and a sum of floating-point values is assumed not to depend on row order unless the two queries add the same values in the same way (an identical `SUM` or `AVG` on both sides, which the result then lists as a narrower claim) or the values are whole numbers or exact decimals (nothing is assumed); the result lists what it assumed. Adding the same decimals in a different order, or splitting one sum into two, can change the last digits, so those pairs are either listed with the order assumption or not proved. The recorded scores and the case list are in the [numeric traps eval](evals/numeric-traps.md); the details are in the [full reference](../docs/provers.md#numbers-and-runtime-errors).

## A decimal written next to an exact-decimal column

BigQuery has two kinds of decimal numbers: floating-point ones, which are approximate, and `NUMERIC` and `BIGNUMERIC`, which are exact. When you write `0.1` next to an exact column, as in `WHERE price = 0.1`, BigQuery reads the `0.1` as an exact decimal too, so the filter finds the rows where the price is 0.1. The prover used to read every written decimal as a floating-point number, which is very slightly different from 0.1, and so it "proved" that this filter never matches anything. It also treated `0.1` and `0.10000000000000001` as the same number everywhere, although a `BIGNUMERIC` column can tell them apart. Both are fixed: the prover now reads the decimal the way the column next to it requires, and when it does not know the column's type it only proves what holds for every type. The behaviour was checked on BigQuery itself with small inline queries; the [full reference](../docs/provers.md#numbers-and-runtime-errors) lists them. The limit is that only the operators and functions listed there were checked; anything else is treated as unknown.

The same question comes up when the value next to the decimal is computed by a subquery. In `WHERE price > (SELECT 1.3 * AVG(amount) FROM sales WHERE ...)` the `1.3` is exact if the average is an exact decimal and approximate if it is floating-point. The prover keeps such a grouped subquery as one block and did not know which kind of number came out of it, so it could not prove two versions of a query like this equal unless they were written identically. It now works the type out when the subquery takes a plain `SUM`, `MIN`, `MAX`, `AVG` or `COUNT` of a table column whose type was declared, which is the common case in TPC-H and DSB style reports. The limits: an aggregate of an expression (`AVG(a * b)`) is still not typed, and for engines other than BigQuery nothing is read this way, so some of those pairs stay "not proven". The rules are in the [full reference](../docs/provers.md).

## A filter before or after grouping

`SELECT ... FROM (SELECT store, reason, SUM(amount) AS total FROM returns GROUP BY store, reason) AS r JOIN stores ON ... WHERE r.reason BETWEEN 72 AND 75` filters the groups after they are built. A database optimizer prefers to filter the rows first, inside the subquery. Both give the same answer, because the filter only looks at a column the subquery groups by, so it keeps or drops whole groups. The prover treats a grouped subquery as one block and could not match a block with the filter inside against a block with the filter outside. It now moves such a filter inside in both queries. It also removes a `store IS NOT NULL` that an optimizer adds inside the subquery when the join outside already requires the store to match something. A filter that the subquery already has is simply removed outside, not added inside a second time; two rules used to hand the same filter back and forth and leave one query with repeated conditions. The limits: this is done only for inner joins, never across a `LIMIT` or a window function, and only for simple conditions on the grouped columns. The check behind it compares original and rewritten queries on small DuckDB tables full of NULLs. The rules are in the [full reference](../docs/provers.md).

## Stacking columns into rows

A common way to turn columns into rows is to stack one query per column: `SELECT product_id, 'store1' AS store, store1 AS price FROM products UNION SELECT product_id, 'store2', store2 FROM products`. Some people write `UNION`, which removes duplicate rows, and some write `UNION ALL`, which does not. The two give the same answer when every product has its own id, because then no row can appear twice: rows from one query differ in their id, and rows from different queries differ in the store label. The prover now sees this when it is told that `product_id` is a key and never empty, and when it is asked for conditions it answers "equal whenever `product_id` is a primary key". The limits: each stacked query must read one table and list the key itself, and the labels must be clearly different words or numbers. If ids can repeat or be empty, the two queries really do differ, and the prover does not call them equal.

## Looking a row up through its own id

People often ask "which students have no department?" in a roundabout way: `SELECT id, name FROM students WHERE id NOT IN (SELECT s.id FROM students AS s JOIN departments AS d ON s.department_id = d.id)`. The inner query reads the students table a second time, finds the students that do have a department, and the outer query keeps the rest. When every student has their own id, the "second" student found by id is the same student, so the question is simply "is there no department for this student?". The prover now reads it that way when it is told that `id` is a key and never empty, and it can then match the roundabout query with the direct one. When asked for conditions it answers "equal whenever `students(id)` is a primary key". The limits: the id must be a single column, the inner query must be a plain join with no totals or limits, and if two students could share an id the two queries can differ, so without the key the prover does not call them equal.

## Counting distinct values in two steps, with a top-N on the end

"How many different suppliers per brand?" can be written in one step with `COUNT(DISTINCT supplier)`, or in two: first list each brand and supplier once, then count the rows per brand. Query optimizers often produce the two-step form. The prover already knew the two are the same, but only when the query ended there. Real reports usually end with "sort by the count and keep the top 10", and with that ending the prover gave up. It now sets the ending aside, recognises the two-step count, and puts the ending back, so the two reports are proved equal. The limits: both queries must sort by the same things in the same direction and keep the same number of rows, and the two-step query must be one the prover already understood without the ending.

## Grouping with nothing to count is "remove duplicates"

`SELECT customer, address FROM (online sales stacked on catalog sales), customers WHERE .. GROUP BY customer, address` computes no total. It only lists each customer and address once, which is exactly what `SELECT DISTINCT customer, address ..` does. Optimizers write the same list as "customers with an online sale `UNION` customers with a catalog sale". The prover already matched the `DISTINCT` spelling with that union; it now reads the `GROUP BY` spelling the same way, also when the stacked sales are joined to other tables. The limits: the query must compute no aggregate, have no `HAVING`, sorting or row limit of its own, use only inner joins, and list nothing but its grouping columns.

## Looking a row up in its own table's groups

`SELECT s.employee, m.top FROM salaries AS s LEFT JOIN (SELECT company, MAX(salary) AS top FROM salaries GROUP BY company) AS m ON s.company = m.company` attaches each company's top salary to every employee. `LEFT JOIN` means "keep the employee even if no group matches", but every employee's company has a group, because the groups were made from the same table. So a plain `JOIN` gives the same rows, and people write either. The prover now treats the two as one query when it is told that `company` is never empty (an employee with no company is kept by `LEFT JOIN` and dropped by `JOIN`). The limits: the grouped side must read the same table with no filter of its own, and the join must compare exactly the grouping columns. When asked for conditions, the prover answers "equal whenever `company` is never empty".

## Two ways to find duplicates

"Which emails are used by more than one person?" has two common spellings. One counts: `SELECT email FROM person GROUP BY email HAVING COUNT(email) > 1`. The other pairs every person with every other person: `SELECT DISTINCT a.email FROM person a JOIN person b ON a.email = b.email AND a.id <> b.id`. The second one relies on `id` to tell two people apart. If two rows could have the same id, or no id, it would miss them or count a person against themselves. So the two are the same exactly when `id` is a key and never empty, and the prover now proves them equal when it is told so. Asked for conditions, it answers "equal whenever `id` is a primary key". The limits: the pairing query must compare nothing else, read the table exactly twice, and list only the columns it matches on.

## "Bought this and never that", counted or filtered

"Which buyers bought an S8 and never an iPhone?" is often written by counting inside each buyer's group: `GROUP BY buyer_id HAVING SUM(CASE WHEN name = 'S8' THEN 1 ELSE 0 END) > 0 AND SUM(CASE WHEN name = 'iPhone' THEN 1 ELSE 0 END) = 0`. It is as often written with a filter and a list to stay out of: `SELECT DISTINCT buyer_id ... WHERE name = 'S8' AND buyer_id NOT IN (the buyers of an iPhone)`. A count above zero only asks "is there such a row", and a count of zero asks "is there none", so the prover reads the counting form as the filtering form and can then compare the two. They are the same only when `buyer_id` is never empty: one sale with no buyer makes the `NOT IN` list reject everybody, while the counting form does not care. So the prover proves them equal when it is told the column is NOT NULL, and asked for conditions it answers "equal whenever `buyer_id` is NOT NULL". The limits: the `HAVING` must hold only such "at least one" and "none" counts, the query must list exactly the columns it groups by, and those columns must be declared NOT NULL.

## Text columns when the query is read as BigQuery

Other databases have a fixed-width text type, `CHAR(2)`, that pads values with spaces. BigQuery has only one text type. So when a query written for another database is translated to BigQuery, `CAST(state AS STRING)` does nothing to a text column, whatever the original schema called its type. The prover now knows this for the BigQuery reading, which matters because translated queries are full of such casts. For other databases the cast is kept, because there it can change how trailing spaces compare.

## A name that means two things

In `SELECT b AS a FROM t ORDER BY a + 1`, is `a` the new output (really `b`) or the table's own column `a`? BigQuery says the output; DuckDB says the table column. The same question comes up in `QUALIFY`. The prover used to pick the table column in these places for every engine, and so it called some pairs equal that BigQuery answers differently. Now it follows BigQuery when the query is BigQuery SQL, and for other engines it answers "not proven" whenever the name could be read both ways. A name in `WHERE` or in the select list is the table column on both engines and was never affected. A related slip was fixed the same day: with `SELECT id, CAST(b AS STRING) FROM t ORDER BY b`, the prover took `b` to mean the converted text and not the column, and text sorts differently from numbers. The evidence is a set of small queries run on BigQuery and DuckDB, listed in the [full reference](../docs/provers.md#a-name-that-is-both-a-select-alias-and-a-column); `GROUP BY` and `HAVING` were tried with the same examples and showed no wrong answer, but are not covered by the new check.

## Functions that give a different answer each time

Most functions always give the same answer for the same input: `UPPER('a')` is always `'A'`. A few do not. `RAND()` gives a new number every time, `nextval('q')` hands out the next number of a counter, and `NOW()` depends on when you ask. The prover treats a function it has no special rule for as the first kind, so it will say that `F() = F()` is always true when `F()` is not NULL.

That is wrong for the second kind. In PostgreSQL, `WHERE nextval('q') = nextval('q')` never matches a row, because the counter moves on between the two calls. The prover used to keep a short list of such functions that was written for BigQuery, so for other engines it called this filter the same as `WHERE nextval('q') IS NOT NULL`, which matches every row.

Now there is one list for all engines: random numbers, fresh identifiers, counters, the clock, and who or where the session is. When either query uses one of them, the solver-based provers never answer "proven", even when the two queries are word for word the same, because the answer depends on something outside the tables. The simpler structural check still accepts such a call when the rewrite leaves it exactly where it was.

There is one useful middle case. `NOW()`, `CURRENT_DATE` and the name of the current user do not change while one query runs. So if you rewrite `SELECT a FROM t WHERE d >= CURRENT_DATE AND a > 1` to filter on `a` first, the two queries do return the same rows whenever both are run at the same moment. For that the provers answer "proven under conditions" and name the condition: both queries read the same statement time (or the same session). It is never counted as a plain proof, and no benchmark score includes it. It applies only to engines that promise one value per query (BigQuery, PostgreSQL, MySQL, DuckDB, Spark, Databricks, Trino, Presto). `RAND()`, `nextval` and `clock_timestamp()` can change within one query, so they stay "not proven". The prover assumes nothing about the value itself: it will not agree that `CURRENT_DATE > DATE '2020-01-01'` is always true. And if the two queries really run at different times, as two scheduled jobs do, the condition does not hold.

The limits: the list was written from each engine's documentation, so a function that is missing from it is still trusted. To make that visible, a proof that relies on a function the prover does not know (your own `my_udf(x)`, for example) now says so in its list of assumptions, by name. If that function is not repeatable, the proof does not hold. See the [full reference](../docs/provers.md#volatile-calls).

## The parser is checked too

The provers also refuse to trust the parser blindly. Before a proof counts, the text is read a second time by a small separate reader that knows each engine's operator order, and if the two readings group the operators differently (for example `a | b & c` in BigQuery, or `a = b < c` in MySQL) the answer is "not proven" with the reason "parser disagreement". The limit is that this can only take proofs away, and a construct the second reader does not know is left to sqlglot's reading. See [parser checks](parser-checks.md).

## Example: grouping with a grand total

`GROUP BY ROLLUP (x)`, `CUBE` and `GROUPING SETS` can add a grand-total row, even when no input row exists, and a list that repeats a grouping set returns each group twice. A rule that assumes one row per group (summing per-group counts into one count, say) would then give a different number than the real query. KumoSQL's rules now recognise these groupings, `GROUP BY ()` and `DISTINCT ON` everywhere and decline to rewrite them, so such pairs come back unproven instead of proven. The evidence is regression pairs checked on DuckDB, so a pair that is still unproven may well be equivalent. The exact conditions are in the [full reference](../docs/provers.md).

## Example: whole numbers that turn into decimals

A database that compares a whole number with a decimal column first converts the whole number to a decimal, and a very large whole number loses its last digits in that conversion. So `a = b AND b = c` does not always mean `a = c`: 9007199254740992 and 9007199254740993 both equal the decimal 9007199254740992.0. Likewise `1e-324 < 2e-324` is false, because both literals round to zero. KumoSQL's SMT prover now models the conversion when the column types are declared, and outside BigQuery it does not treat tiny, huge or long decimal literals as exact numbers (BigQuery decimal literals are read as the nearest floating-point value, as BigQuery does). When column types are not declared, a proof that compares columns states the assumption that they have the same type. The evidence is regression pairs; the exact rules are in the [full reference](../docs/provers.md).

## Example: a clause the prover does not understand

`CAST(x AS NUMBER DEFAULT 0 ON CONVERSION ERROR)` returns 0 for a value that cannot be converted, where a plain `CAST` fails. If the prover quietly ignored the extra clause, it would call the two queries equal. Now the SMT prover checks every query for clauses it does not model (this one, `FOR UPDATE`, `CLUSTER BY`, `SETTINGS`, `SELECT` modifiers, `* EXCEPT`) and gives up on the pair instead. Some of those clauses cannot change a result, so a few pairs that are really equal now come back unproven. The evidence is a small set of regression pairs; the exact list is in the [full reference](../docs/provers.md).

## Example: a DISTINCT that can move outward

A query that removes duplicates inside a subquery, then joins it to a table on whole-number key columns, can have its duplicate removal moved to the outside when the join already makes every output row unique. KumoSQL's prover applies that move only when the columns are declared whole numbers and the joined tables' keys are fully pinned down; any grouping, limit, outer join or other twist makes it decline and leave the pair unproven. The evidence is a handful of textbook query pairs, so treat it as a narrow rule. The reference page has the exact conditions and the recorded scores: [Full reference](../docs/provers.md).

## Trying again without the newest rules

The prover rewrites both queries into a standard form and then compares the two. Every rewrite keeps the query's meaning, so a rewrite can never make the prover call two different queries equal. But a rewrite can get in the way. If it changes one query and has nothing to change in the other, two forms that used to match can stop matching, and a proof that worked before is lost.

Example: one query filters the rows of a `FULL JOIN`. The other builds the same rows from two one-sided joins glued with `UNION ALL` and filters those. A new rule moved the filter inside on the second query only, and the prover stopped recognizing the pair.

So the newest group of rules is now tried first, and if the prover reaches no answer, it tries the pair once more with that whole group switched off, exactly as it worked before those rules existed. Whichever attempt finds an answer wins. An answer of "not equal" from the first attempt is never overridden.

The limit: this protects only against the rules in that group. A rule added later needs to be put in the group to get the same protection. The second attempt also takes extra time on pairs the prover cannot decide, so it is only made when the two queries together are at most 4,000 characters long; a very large pair gets the first attempt only. The evidence is the comparison of every benchmark before and after the group was added (four proofs had been lost, all four are back) and a small set of made-up test pairs. Details: [Full reference](../docs/provers.md#a-second-attempt-without-the-shape-rules).

## Avoiding repeated work

When a proof tries several fallback rules, it keeps the earlier normalization result for that same proof. It also keeps the assumptions attached to it. A different proof, such as one with different declared keys, starts fresh.

A long calculation used to cost far more than its length. Take `SELECT c1 + c2 + .. + c40 FROM t` compared with itself: every `+` might overflow, and for each one the prover asked the solver two questions that carried all the columns before it. That took one to two minutes, however short the time limit was, because the limit applies to each question and not to their number. The questions are now put so that the earlier columns drop out of them where they cannot change the answer, and the same forty-column comparison takes two to three seconds. The answers are the same; only the work is less. The limit is still per question, so a very long expression can still take many times the limit. The [full reference](../docs/provers.md#numbers-and-runtime-errors) has the details and the measurements.

If a query returns two columns and a view returns twenty, returning the entire view cannot answer that query. KumoSQL skips that impossible candidate and still proves any proposed two-column projection. Stars whose width is not settled keep the existing checks. These changes reduce work; they do not relax proof acceptance.

Python can clean up old objects on a different thread from the one using the solver. KumoSQL now keeps that automatic cleanup inside the solver lock as well, and restores the caller's cleanup setting afterwards. This addresses concurrent proof hangs without changing proof acceptance or increasing test timeouts.

Holding that cleanup back had a cost: a long proof kept every intermediate copy of the query until it finished. One 30-second proof that needs about 200 MB held more than 3 GB, and a benchmark with four workers ran out of memory on a 16 GB machine. The thread that is using the solver now does the cleanup itself at regular points during a long proof, which is safe because no other thread is using the solver at that moment. The same proof now stays near 200 MB and takes the same time. No answer changes. The measurements and the list of points are in [the full reference](../docs/provers.md#proof-search-performance).
