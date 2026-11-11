# Testing SQL translations between dialects

[Simple eval index](README.md) · [Full reference](../../docs/evals/dlbench.md)

DLBench pairs a query with its translation into another database's dialect. KumoSQL uses it to check parsing and whether supported translations can be proved equivalent to their sources.

The pinned subset includes approximate-label pairs and a sample of exact-label pairs. Labels came from language models and human review upstream; KumoSQL does not call a model while checking them.

## Why translation is hard

The same text can have different behavior across engines: LIKE can have different case rules, integer division can truncate, and division by zero can produce NULL, infinity, or an error.

The harness reads each side in its dialect, reads respelled column and table names back to the source's (`_date` for `date`, `Sales_Orders` for `Sales Orders`, only when one source name fits), and tries proof. Known dialect gaps stay unknown rather than being accepted as equivalent.

Many translations from MySQL to MariaDB change nothing but letter case and spacing. When the two statements are the same token for token, the pair is decided without a prover, for example an `INSERT` that was only re-capitalised. A statement that can answer differently on a second run (one that reads the clock, a random value or a variable) is not decided this way, and neither is the same text sent to two unrelated databases.

A query that reads the clock, a random value, a sequence or the session is never proved: its answer does not depend on the tables alone, and the provers would treat `nextval('q') = nextval('q')` as always true when it is never true.

The harness can also run every proof on the real databases. Set `DLBENCH_NATIVE` to name test servers (MySQL, MariaDB, PostgreSQL, ClickHouse; SQLite and DuckDB need none) and each proved pair runs on both of its databases with the same random rows. For example, MySQL 8 answers `SELECT UNIX_TIMESTAMP('2039-01-20 01:00:00')` with a number and MariaDB 10.11 answers NULL, so that statement is now a known gap instead of a proof. Without the setting nothing changes.

The same function name can mean different things in two databases. `LOG(2)` is 0.69 in MySQL and 0.30 in PostgreSQL, and `ROUND(2.5)` on a floating-point value is 3 in SQLite and 2 in PostgreSQL. The harness lists the calls that were seen to differ when one statement was run on every engine, and leaves a pair that uses one of them unknown.

Two date-formatting calls are read back as SQLite's `STRFTIME`: DuckDB's `strftime` with the format written first, and MonetDB's `sys.timestamp_to_str`. Only the common pieces (`%Y %m %d %H %M %S`) are accepted. The MonetDB case rests on its documentation, because no MonetDB server was available to run it.

An `INSERT` or a `CREATE TABLE ... AS` is decided when the translation reads back as the very same statement, or when everything but the inner query is the same text and the inner queries are proved equal. For example `INSERT INTO t1 VALUES (ln(1))` against the same statement with different letter case. This says the same rows are written; it does not say what each database's column types then do to them, and these statements are not re-run on the real databases.

Three of the development pairs are proved thanks to a prover rule (`sqlite_date_text.py`) that reads the digits SQLite's `STRFTIME` returns as text or as a number, as SQLite itself does.

ORDER BY on the source can require an ordered comparison. Output names are otherwise ignored under this evaluation's policy.

```sh
python tools/dlbench_bench.py
```

The full guide describes the subset, dialect approximations, proof assumptions, gaps, source provenance, and scores. Parsing both statements is coverage evidence; it does not establish faithful translation. A same-statement decision assumes MariaDB answers a statement as MySQL does; the native replay checks that only for queries it can run, on small random tables, for the engine versions that were installed. No MonetDB engine was available, so MonetDB proofs are not replayed.
