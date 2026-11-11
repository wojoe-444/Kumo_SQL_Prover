# DLBench

[Plain-language version](../../docs_simple/evals/dlbench.md)

[DLBench](https://github.com/DLBenchll/DLBench) (Apache-2.0, ASE 2025) pairs a query with its translation into another database's dialect: BIRDTrans translates 3,206 of BIRD's SQLite queries and BUTTERTrans 3,196 queries from MySQL's and PostgreSQL's own test suites, into MySQL, MariaDB, PostgreSQL, ClickHouse, MonetDB and DuckDB. Its labels (6,199 "exact", 203 "approximate" equivalence) come from language models and human review. KumoSQL uses it for cross-dialect coverage: does each side parse in its own dialect, and can the translation be proved equal to its source? The repository pins 807 pairs: every approximate pair and one exact pair in ten ([source, subset rule and overlap](../../tests/fixtures/dlbench/README.md)). Results file: `dlbench`.

```
python tools/dlbench_bench.py                     # the pinned subset, about a minute on 4 cores
python tools/dlbench_bench.py --data DLBench      # all 6,402 pairs from a checkout
python tools/dlbench_bench.py --write-results
DLBENCH_NATIVE='{...}' python tools/dlbench_bench.py   # also replay every proof on the databases themselves (below)
```

## How a pair is decided

1. **parsed**: sqlglot reads the source in its dialect and the translation in the target's. sqlglot has no MariaDB or MonetDB dialect, so they are read as MySQL and PostgreSQL.
2. **proven**: the translation's respelled columns and tables are mapped back to the source's names (see below), identifiers are lower-cased, sqlglot writes the translation in the source's dialect, and `prove_equivalent_algebraic` proves it equal to the source (no keys, output names ignored). A source that ends in `ORDER BY` is compared as a list, so the translation must keep the order.
3. **proven, same statement**: a MariaDB translation that is its MySQL source token for token (see below).
4. **proven, a statement that writes rows**: an `INSERT` or `CREATE TABLE ... AS` that reads back as its source, or whose inner queries are proved equal (see below).
5. **unknown** otherwise, including every **dialect gap** (the two databases read the same text differently and sqlglot does not translate the difference, so no proof is taken) and every pair with a **volatile call** (below).

| Dialect gap | Rule |
| --- | --- |
| LIKE and case | `LIKE` ignores case in SQLite and MySQL, not in the others |
| MySQL string comparison | MySQL and MariaDB compare strings without case (default collation). Between a MySQL-family query and another dialect a pair is proved only when no string can be compared, grouped, ordered or converted (see below). With MonetDB, which was not replayed, the first rule stays: no string literal and no column at all |
| MySQL and MariaDB collations | MySQL 8's default collation (`utf8mb4_0900_ai_ci`, NO PAD) tells `'a'` from `'a '`; MariaDB's (PAD SPACE) does not. The same places as above, except that two plain ASCII literals without a space at either end may meet |
| row comparison | a compared row value (`ROW(a, b) IN (...)`, `(a, b) = (1, 2)`) with a field that is not provably a number; any compared row value when DuckDB, ClickHouse or MonetDB is involved |
| timestamp range | `UNIX_TIMESTAMP`, `FROM_UNIXTIME` and their names elsewhere, between two different engines, unless the argument is a literal from 1971 to 2037 |
| ClickHouse date range | `MAKEDATE`, `FROM_DAYS` and ClickHouse's date builders: out of range is 1970-01-01 there |
| PostgreSQL time zone | `TO_TIMESTAMP` gives a timestamp with a time zone, whose text ends in `+00` |
| population variance | `VARIANCE`, `STD` and `STDDEV` are the population's in MySQL and MariaDB and the sample's elsewhere; sqlglot reads them as one function |
| LEAST and GREATEST with NULL | NULL in MySQL when any argument is; PostgreSQL, DuckDB and ClickHouse skip a NULL argument. Literal arguments pass |
| integer division | `/` on integers truncates in PostgreSQL and MonetDB, not in MySQL |
| division by zero | NULL in SQLite and MySQL, infinity in DuckDB and ClickHouse (an error elsewhere, which proofs exclude) |
| 32-bit float | `REAL` (and `FLOAT` in most targets) is 32-bit; sqlglot writes it as SQLite's 64-bit `REAL` |
| decimal arithmetic | `1.0` is an exact decimal in PostgreSQL, MonetDB and MySQL, a float in SQLite; `/` and `AVG` give a decimal with four more digits in MySQL and MariaDB (`1 / 3` is `0.3333`) |
| ClickHouse defaults | ClickHouse returns 0 for `SUM` of no rows and fills outer-join gaps with default values, not NULL |
| SQLite type conversion | a text column compared with a number (see [LLM-SQL-Solver](llm-sql-solver.md)) |
| SQLite text against a number | the text a function returns compared with a bare number: `STRFTIME('%Y', d) < 1997` is false for every row in SQLite, where any text sorts above any number; the other databases convert one side |
| date format | a DuckDB or MonetDB translation of a SQLite source that formats a date with a specifier other than `%Y %m %d %H %M %S` |
| rounding | `ROUND` between two different engines (below) |
| logarithm base, ClickHouse NaN | `LOG(x)` between two different engines; a logarithm, root or power of anything but numeric literals when one side is ClickHouse |
| substring position | `SUBSTR` from position 0 or below, with a negative length, or with a position or length that is not a literal, except where the engines were seen to agree (below) |
| CHAR without a length, bitwise operators, integer division, HEX of a number | `CAST(x AS CHAR)`; `\|`, `&`, `~`, `<<`, `>>`, XOR; `DIV` and `//`; `HEX` of anything but a string literal or a column declared as text; each between two different engines |
| set-returning function | `generate_series`, `unnest`, `range` and the like between two different engines |
| SQLite month arithmetic | `date(x, '+1 month')` in a SQLite source |
| deviation precision | `STD`, `STDDEV`, `VARIANCE` and their `_POP` and `_SAMP` forms between MySQL and MariaDB |

**Where a string may be compared.** The MySQL rule looks for a place where a string could take part in a comparison or a conversion: an operand of `=`, `<`, `IN`, `BETWEEN` or a simple `CASE`; a key of `GROUP BY`, `ORDER BY`, `DISTINCT`, a window's partition, `UNION`, `INTERSECT`, `EXCEPT` or `USING`; the argument of `MIN` or `MAX`; `LIKE` and the functions that compare strings (`STRCMP`, `INSTR`, `LEAST`, `NULLIF`, ...); and an operand of arithmetic, `SUM`, `AVG`, a numeric `CAST` or a condition, because the databases convert a string to a number differently. A place is safe only when every operand is provably a number: a numeric literal, a column that every table of the source's schema declares numeric, `COUNT`, `LENGTH`, arithmetic and `SUM` over such. A column no table declares is a string. BIRDTrans translations into MySQL and MariaDB all compare a string somewhere, so the rule still blocks every one of them; replayed with the rule off, four such development proofs differ natively (a join on `'a'` and `'A'`, a `GROUP BY` over `'a'` and `'a '`).

**Volatile calls.** A pair is not proved when either query calls something whose answer is not a function of the tables: the clock (`NOW()`, `CURRENT_DATE`, `date('now')`, `clock_timestamp()`), random values (`RAND()`, `random()`, `randomblob()`, `gen_random_uuid()`), sequences (`nextval`), the session and server (`CONNECTION_ID()`, `USER()`, `VERSION()`, MySQL variables). The provers read a function they do not know as a function of its arguments, and sqlglot keeps most of these as plain calls outside BigQuery, so `nextval('q') = nextval('q')` would be proved always true where it is never true. The harness reads the names from each query's tokens in its own dialect. On this subset the refusal removes no proof (12 pairs that were unknown already now say why).

**One name, two functions.** sqlglot reads `ROUND`, `LOG`, `SUBSTR` or `HEX` as the same function in every dialect, and the provers read a function they do not model as a function of its arguments, so a translation that keeps such a call was proved on the name alone. Running one made-up statement on every engine shows where that is wrong:

| Statement | Answers |
| --- | --- |
| `SELECT ROUND(c)` with `c` a double holding 0.5, 2.5, -2.5 | 1, 3, -3 in SQLite and DuckDB; 0, 2, -2 in MySQL, MariaDB, PostgreSQL and ClickHouse |
| `SELECT ROUND(c, 2)` with `c = 2.675` | 2.67 in SQLite, 2.68 in the others |
| `SELECT ROUND(2.5)` | 3 everywhere but ClickHouse, which answers 2 |
| `SELECT LOG(a)` with `a = 2` | 0.693147 in MySQL and ClickHouse, 0.30103 in PostgreSQL and DuckDB |
| `SELECT SQRT(a)` with `a = -1` | NULL in MySQL, NaN in ClickHouse |
| `SELECT SUBSTR('abc', -2)` | `bc` in SQLite, DuckDB, ClickHouse and MySQL, `abc` in PostgreSQL |
| `SELECT SUBSTR('a', -8)` | `a` in SQLite, DuckDB and ClickHouse, empty in MySQL |
| `SELECT SUBSTR('a', 0, 2)` | `a` in SQLite, DuckDB and PostgreSQL, empty in MySQL, empty or an error in ClickHouse |
| `SELECT CAST(a AS CHAR)` with `a = -1` | `-1` in MySQL, `-` in PostgreSQL |
| `SELECT ~a` with `a = 3` | 18446744073709551612 in MySQL, -4 in PostgreSQL and DuckDB |
| `SELECT HEX(a)` with `a = 3` | `3` in MySQL and DuckDB, `03` in ClickHouse, `33` in SQLite |
| `SELECT generate_series(1, 2) AS id` | two rows in PostgreSQL, one row holding a list in DuckDB |
| `SELECT date('2020-01-31', '+1 month')` | 2020-03-02 in SQLite; the other engines' month arithmetic gives 2020-02-29 |
| `SELECT MD5('test')` | 32 hex digits in MySQL, PostgreSQL and DuckDB, 16 raw bytes in ClickHouse |
| `SELECT STD(a)` over 2, 0, 0, -1, 3 | 1.469694 in MySQL, 1.4697 in MariaDB |

Each is a gap between engines of different families (MySQL and MariaDB count as one, except for the last row; the `MD5` row is a gap named ClickHouse digest). The rules are as narrow as the comparison supports: `SUBSTR(s, i)` and `SUBSTR(s, i, n)` were run on every engine for four short strings, every `i` from -8 to 8 and every `n` from -3 to 8. All engines agree for a position of 1 or more and a length of 0 or more; SQLite and DuckDB agree everywhere; from a negative position ClickHouse agrees with them; from position 0 PostgreSQL does. `DIV` is a gap for another reason: sqlglot writes MySQL's `a DIV 2` back as `CAST(a / 2 AS SIGNED)`, which rounds. None of these rules removed a proof on the pinned subset (they name 42 pairs that were unknown already), so they guard the rest of DLBench and whatever is added to it. The list is what a sweep of about a hundred made-up statements found; a same-named function that differs on inputs the sweep did not try is still read as one function.

**Date formatting read back.** Two date-formatting calls reach the harness untranslated when a SQLite source is translated: DuckDB's `strftime('%Y', x)` with the format first (DuckDB takes both orders; sqlglot reads the first argument as the value), and MonetDB's `sys.timestamp_to_str(x, '%Y')`, which sqlglot does not know. Both are read as SQLite's `STRFTIME('%Y', x)`, only for a format made of `%Y %m %d %H %M %S` and plain text (SQLite and DuckDB were compared on each, with years 0099, 0999 and 9999), and only when exactly one of the two arguments is a string literal holding a `%` specifier, so the order is never a guess. MonetDB formats through C's `strftime`, which writes the year 99 as `99` where SQLite writes `0099` (seen in this container's C library; MonetDB itself was not available). So a MonetDB call with `%Y` is read back only where that cannot show: the text is cast to a whole number at once (`CAST(sys.timestamp_to_str(d, '%Y') AS INT)`), or it is tested with `=`, `<>` or `IN` against literals that are a possible output with a four-digit year (`= '2016'`, `= '2014-02'`). An ordering comparison (`< '1960'`) stays unknown, because `'0099' < '1960'` and `'99' > '1960'`. The DuckDB proofs are replayed natively; **the MonetDB proofs are not replayed** and rest on that reading of its documentation.

**Statements that write rows.** The provers read queries. An `INSERT` or a `CREATE TABLE ... AS` is decided in two cases, after the dialect-gap rules and the volatile-call refusal have passed, between any two engines except MonetDB (which is read as PostgreSQL and was never run):

* *Same statement, read back.* The translation, written in the source's dialect with its names read back, is the source statement token for token: `INSERT INTO t1 (data) VALUES (repeat('a',1*1024))` against `INSERT INTO t1 (data) VALUES (REPEAT('a', 1 * 1024))`, or `CREATE TABLE tail AS SELECT concat(b1.b, b2.b) ...` against PostgreSQL's `b1.b || b2.b`, which sqlglot reads as the same `CONCAT`.
* *Inner query proved.* Everything outside the inner query is the same text (the table, the column list, `IGNORE`, every table property such as `ENGINE` or `ORDER BY`), and the provers prove the two inner queries equal, with `VALUES` rows read as `SELECT ... UNION ALL SELECT ...`. For `CREATE TABLE ... AS` the two inner queries must also name their outputs alike, because those names are the new table's columns.

Both refuse what the same-statement rule refuses, on either side (the clock, random values, the session, variables, `GROUP_CONCAT`, window functions, `LIMIT` without `ORDER BY`), and neither applies to a table definition, an `UPDATE`, a `DELETE`, `EXPLAIN`, `SHOW` or `DESCRIBE` between two different engines. A decision is about the rows the statement writes: what the table's column types then do to them (rounding, truncation, an error) is the engine's own and is not modelled, and **these proofs are not replayed natively** (the replay runs queries). The calls in the development pairs they decide (`REPEAT`, `MD5`, `GREATEST` and `LN` of literals, `CONCAT`) were run on the engines of those pairs and agree.

**Names read back.** BIRDTrans's translated schemas respell names the target would not take as written: `Type` becomes `_Type`, `Date received` becomes `Date_received`, the table `Sales Orders` becomes `Sales_Orders`. The fixture lists the column renames it could derive (`renames.json`, keys as the translated schema quoted them). A name the list misses is read back by its spelling: lower-cased, with every character that is not a letter or digit written as an underscore, and optionally one more underscore in front, it must match exactly one column (or table) of the source's schema, and must not be a source name itself. An unqualified name that is also an output name of the translation (`SELECT x AS _year ... ORDER BY _year`) and a table name the translation defines itself (a CTE, an alias) are left alone.

**The same statement.** MySQL and MariaDB are read as one dialect and no gap rule tells them apart, and many BUTTERTrans translations between them change only letter case and spacing. Where the provers have nothing to read (an `INSERT`, a `CREATE TABLE ... AS`, a column of a table whose definition was not published), such a pair is decided by comparing the two statements' tokens: comments and the closing semicolon dropped, everything but a literal lower-cased. Nothing rests on how sqlglot parses the statement. Refused: anything that is not a query or a data change (`EXPLAIN` prints the engine's own plan); a statement whose answer can change between two runs (the clock, random values, the session, variables, locks, `GROUP_CONCAT`, window functions, `LIMIT` without `ORDER BY`); and a string literal that ends in a space, because MySQL 8's default collation compares trailing spaces and MariaDB's ignores them. The rule never applies between two other databases: the same text is not the same statement there (`generate_series(1, 2)` as a column is two rows in PostgreSQL and one list in DuckDB).

A proof therefore says the two queries mean the same under sqlglot's reading of each dialect, outside these gaps. MariaDB-specific and MonetDB-specific behaviour is not modelled, and PostgreSQL is assumed to sort text bytewise (the C collation). Every proof of a BIRDTrans pair is re-run: source and translation on 300 random SQLite databases (lists when ordered, with a reverse-order reload so ties never count), and a DuckDB translation also natively in DuckDB on the same data (a difference confirmed with the optimizer off). A difference makes the proof **wrong**. The random data follows the declared types: an `INTEGER` column holds whole numbers only (SQLite would keep a stored `0.1` there, DuckDB rounds it, and the translation's table could not hold it).

## Native replay

Set `DLBENCH_NATIVE` and every proof is also run where it belongs: the source on its own database, the translation on its own, on the same random typed data (40 trials; `tools/dlbench_native.py`). Without the variable the harness behaves exactly as before.

```
export DLBENCH_NATIVE='{
  "mysql":      {"unix_socket": "/path/mysql.sock", "user": "root"},
  "mariadb":    {"unix_socket": "/path/mariadb.sock", "user": "root"},
  "postgresql": {"host": "/path/pg-socket-dir", "port": 5432, "user": "postgres", "dbname": "postgres"},
  "clickhouse": "chdb"
}'
```

The MySQL and MariaDB entries are arguments of `pymysql.connect`, the PostgreSQL entry of `psycopg.connect`, and `"chdb"` is ClickHouse embedded in the `chdb` package. SQLite and DuckDB run in process. Use throwaway servers: each replay creates a scratch database and drops it. There is no MonetDB entry, so MonetDB proofs are not replayed and its gap rules were not narrowed.

The random data holds what the engines are known to disagree about: NULL, `'a'` beside `'A'` and `'a '`, a six-letter string, halves and 2.675 in fractional columns. A difference marks the proof **wrong**. What the replay does not cover: a statement that is not a query, a table with no published definition, a column of a type it does not load (binary, `ENUM`, `YEAR`, `TIME`, JSON), and `LIMIT` without `ORDER BY`. An ordered source is compared as a list, on data where its `ORDER BY` keys tell every row apart. Tables are built from the source's schema text, so a translation is run against the source's column types. All engines ran in UTC.

`--record-contradictions` writes every pair whose proof the replay contradicted to `tests/fixtures/dlbench/native_contradictions.json` (id, route and engine versions only), and a listed pair is never counted as proved again. That is how a held-out contradiction is kept out without anyone reading the pair. The file is empty: the replay of 2026-11-11 found none on the held-out pairs.

**What the first replay found** (2026-11-11; MySQL 8.0.46, MariaDB 10.11.14, PostgreSQL 16.15, ClickHouse 26.9.2.1 through chdb 4.4.0, DuckDB 1.5.6, SQLite 3.53.1). Of 130 development proofs, 106 could be replayed and six differed. None was a mistake in a prover's reasoning: each is a place where two engines answer one statement differently and the harness had assumed they agree. Made-up examples of each:

| Engines | Statement | Answers |
| --- | --- | --- |
| MySQL, MariaDB | `SELECT UNIX_TIMESTAMP('2039-01-20 01:00:00')` | 2179098000, NULL |
| MySQL, MariaDB | `SELECT UNIX_TIMESTAMP('1969-12-31 16:00:00')` | 0, NULL |
| MySQL, MariaDB | `SELECT ROW('b', 1.5, 3) IN (ROW('b', NULL, 3), ROW('a', 1.5, 3), ROW(1, 3, 3))` | NULL, 1 |
| MySQL, MariaDB | `SELECT STRCMP(a, 'a ') FROM t` with `a = 'a'` | -1, 0 |
| MySQL, ClickHouse | `SELECT MAKEDATE(9999, 365)` | 9999-12-31, 1970-01-01 |
| MySQL, ClickHouse | `SELECT FROM_UNIXTIME(a) FROM t` with `a = -1` | NULL, 1970-01-01 00:00:00 |
| MySQL, DuckDB | `SELECT ROW(1, 2, ROW(3, NULL)) IN (ROW(3, 2, ROW(3, 4)), ROW(1, 2, ROW(3, 5)))` | NULL, false |
| MySQL, PostgreSQL | `SELECT a FROM t GROUP BY a HAVING VARIANCE(b) IS NOT NULL`, one row per group | the row, nothing |
| MySQL, PostgreSQL | `CONCAT('', STR_TO_DATE(...))` against `'' \|\| TO_TIMESTAMP(...)` | `2003-01-02 08:11:02`, `2003-01-02 08:11:02+00` |
| MySQL, PostgreSQL | `SELECT SUM(LEAST(a, b)) FROM t` with a NULL in `a` | the row is left out, the row counts |

Each became a gap rule above. The table has more rows than six because the `STRCMP` row appeared once the schema reader kept column types with parentheses, and the `FROM_UNIXTIME`, `VARIANCE`, `TO_TIMESTAMP` and `LEAST` rows appeared while the string rule was being narrowed, among pairs it had blocked before. The same replay is the reason the MySQL string rule could be narrowed from "no string literal and no column" to "no place where a string is compared": 23 pairs it blocked are now proved, and every one the replay could run showed no difference.

**An ordered PostgreSQL source** was compared as a bag until 2026-11-11: the check for a closing `ORDER BY` looked the dialect up under sqlglot's name for it and found nothing, so `SELECT a FROM t ORDER BY a` against the same query with `ORDER BY a DESC` was proved for a PostgreSQL source (never for a SQLite or MySQL one). The made-up sweep found it; no proof on the pinned subset depended on it. It is compared as a list now.

## Scores

Pinned subset, 2026-11-11, on 0ec0ce4: **190/604 exact pairs proved, 0 wrong; 799/807 parsed**. Held out (103 exact pairs): 38 proved. Of the 190, 10 are same-statement decisions (MySQL to MariaDB), 8 are an `INSERT` or `CREATE TABLE ... AS` that reads back as its source and 4 one whose inner queries are proved. Approximate pairs proved: 2/203, both BIRDTrans PostgreSQL and both replayed natively without a difference (108: `STRFTIME('%Y', date)` against `TO_CHAR(date, 'YYYY')` on a column that is text in SQLite and a date in PostgreSQL; 393: a difference of two such years, with a `NULLIF(COUNT(DISTINCT ...), 0)` divisor where SQLite's division by zero is NULL already).

Native replay of that run: 141 of the 192 proofs ran on their own engines with **0 differences** (MySQL to MariaDB 29, to PostgreSQL 14, to ClickHouse 11, to DuckDB 10; SQLite to DuckDB 29, to PostgreSQL 29, to ClickHouse 19). **Not replayed: 51.** 24 are MonetDB proofs (no engine; 14 of them rest on the date-formatting read-back), 21 are statements that are not queries (9 MariaDB same-statement decisions and the 12 `INSERT` and `CREATE TABLE ... AS` decisions above), and 6 the replay cannot load or run.

| Dataset | Target | Pairs | Parsed | Proved | Dialect gap |
| --- | --- | ---: | ---: | ---: | ---: |
| BIRDTrans | ClickHouse | 57 | 57 | 19 | 33 |
| BIRDTrans | DuckDB | 77 | 77 | 29 | 42 |
| BIRDTrans | MariaDB | 91 | 91 | 0 | 91 |
| BIRDTrans | MonetDB | 83 | 83 | 24 | 39 |
| BIRDTrans | MySQL | 117 | 117 | 0 | 117 |
| BIRDTrans | PostgreSQL | 68 | 66 | 29 | 22 |
| BUTTERTrans | ClickHouse | 60 | 59 | 16 | 29 |
| BUTTERTrans | DuckDB | 62 | 59 | 14 | 27 |
| BUTTERTrans | MariaDB | 67 | 67 | 42 | 20 |
| BUTTERTrans | MonetDB | 56 | 55 | 0 | 48 |
| BUTTERTrans | MySQL | 3 | 3 | 1 | 2 |
| BUTTERTrans | PostgreSQL | 66 | 65 | 18 | 37 |

**History.** 2026-10-03: 109/604 (held out 19). The same harness on f1d36cc (2026-11-11, after the prover's shape rules) measured 110/604. The 46 more come from two harness changes made on the development split (501 exact pairs: 91 to 129 proved; the 103 held-out pairs were scored once, at the end): 24 from reading translated spellings back and 22 same-statement decisions. No proof was lost.

Later on 2026-11-11 (156 to 161 on f6bef8f): the gap rules that the native replay led to removed 21 proofs (19 development, 2 held out; 16 of them MySQL to MariaDB: collations 10, timestamp range 5, row values 1) and narrowing the MySQL string rule added 23 (18 development, 5 held out); 3 more development pairs come from the prover's new SQLite date-text rule. The rules are by class, so not every removed proof differs on the replay's data: of the nine development collation proofs, three differ, three do not and three could not be replayed. The volatile-call refusal removed none. The floor went up, but the 21 are removals of proofs the harness had no right to and would have moved it down on their own. The held-out pairs were replayed for their verdicts only. Of those 161, 3 development pairs are due to the prover rule `sqlite_date_text.py` (SQLite's `STRFTIME` digits read as text or as a number, checked in SQLite itself).

Later again on 2026-11-11 (161 to 190 on 0ec0ce4; development 131 to 152 of 501, held out 30 to 38): 17 from the date-formatting read-back (MonetDB 14, DuckDB 3) and 12 from the two rules for statements that write rows. The gaps for same-named calls and the fix for ordered PostgreSQL sources removed none. A change to drop the division-by-zero gap when every divisor is a literal other than zero was tried and left out: one development pair has only such divisors, and it is unknown for another reason (sqlglot reads DuckDB's `/` back as a division of reals, which the provers do not equate with SQLite's).

Unparsed: 2 BIRDTrans PostgreSQL translations and 6 BUTTERTrans queries (one MySQL source, five translations).

**Baseline.** The first run on master, with only the LIKE, MySQL, integer-division and SQLite type rules, proved 112 pairs, and the native DuckDB re-run showed two of them (both "approximate") return different results: `CAST(.. AS REAL)` is 32-bit in DuckDB, and dividing by zero gives infinity there. The 32-bit float, division-by-zero, decimal and ClickHouse rules were added after that run, with the whole subset (held-out pairs included) in view, so they count as **tuned on test**.

## Limits

* Most BUTTERTrans queries test one function on constants (`SELECT hex(concat(regexp_instr('a', 'a')))` against `SELECT HEX(CONCAT(POSITION('a' IN 'a')))`); the prover treats such functions as opaque, so they stay unknown unless the two sides are the same function.
* The MySQL rule still blocks every BIRDTrans translation into MySQL and MariaDB: each compares a string somewhere, and SQLite compares strings exactly.
* A same-statement decision says MySQL and MariaDB were given one statement; it rests on the reading of MariaDB as MySQL, like every other proof for that target. The gap rules cover the differences the replay showed (collations, Unix time, row values); a function only one of them has, or one they compute differently on data the replay did not try, the harness does not know.
* The native replay is evidence, not proof: 40 random trials on small tables, the engine versions named above, every server in UTC with its default collation. The gap rules it led to were written from the development pairs; a difference of another kind among the pairs it could not replay (MonetDB, statements that are not queries) would go unseen.
* 51 of the 192 proofs were not replayed natively: MonetDB proofs, statements that write rows, and a few queries the replay cannot load. For those the harness has the provers, the gap rules and nothing else.
* A decision about an `INSERT` says the same rows are written. It does not say the two tables then hold the same values: a column type may round, truncate or reject them, differently in each engine.
* The gap rules are conservative by name: a query that only mentions `FROM_UNIXTIME` over a column, or `LEAST` over columns that are never NULL, stays unknown.
* A spelling is matched against the tables the source's schema text describes. A translation that reads a renamed column the fixture's schema does not list stays unknown.
* Labels are not proofs: "exact" pairs can still differ on edge cases (time zones in `UNIX_TIMESTAMP`, the `SUBSTR('Last Updated', -4)` string literal in BIRD's own gold query). Only a proof contradicted by running both queries counts as wrong.
