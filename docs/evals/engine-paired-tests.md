# Paired engine tests

157 query pairs taken from other engines' own tests and bug fixes, plus authored guards, each with a hand-checked label. The provers must prove the equivalent pairs and may refute the others; a proof of a pair that is not equivalent, or any claim that an equivalent pair differs, is wrong. Results file: `engine-paired-tests`. The pairs are in `tests/fixtures/engine_pairs` ([sources, pins, adaptations and overlap](../../tests/fixtures/engine_pairs/README.md)).

```
python tools/engine_pairs_bench.py                    # about two minutes
python tools/engine_pairs_bench.py --show unknown
python tools/engine_pairs_bench.py --write-results
python tools/engine_pairs_bench.py --check-sources    # download the pinned files, check SHA-256, re-extract
```

## Sources

| Source (inventory row) | Pairs | What a pair is |
| --- | --- | --- |
| Trino `AbstractTestJoinQueries` (A-R01, Apache-2.0) | 137 extracted, 134 scored | Every `assertQuery(sql, expectedSql)` whose two SQL arguments are string literals (four more pass a session first; it only fixes the join order). Trino runs the first query and H2 the second over TPC-H tiny, and the test asserts the same rows. 27 pairs read tables on both sides (alternative SQL); 107 compare a query with its expected result written as SQL (`VALUES`, `SELECT 0`). Calls built with `format()` or a query template are not extracted |
| Spark `SubquerySuite` (A-R02, Apache-2.0) | 9 extracted, 7 scored | The EXISTS / IN / NOT EXISTS / NOT IN tests over the `l` and `r` tables (NULLs and duplicates). Queries that expect the same rows are paired; NOT EXISTS against NOT IN is the NULL-sensitive negative. Two pairs use multi-column `NOT IN`, which has no GoogleSQL form, and are kept unscored |
| PostgreSQL commit f4c00d1, bug 17976 (A-R04, PostgreSQL licence) | 2 | The regression query the fix added to `join.sql`, against the join-removed form the expected plan in `join.out` shows (equivalent: `t.a` is UNIQUE) and against the dropped-qual reading of the fault (not equivalent: one row instead of none) |
| DuckDB `test_window_order_collate.test`, issue 20608 (A-R08, MIT) | 4 extracted, 3 scored | The file's two window queries over a `COLLATE NOCASE` column (kept unscored: `row_number()` without `ORDER BY` is not determined), the same pair with `ORDER BY name` (equivalent), an authored `NOACCENT.NOCASE` sibling (not equivalent) and the 20608 `EXCEPT ALL` fixture against the join it subtracts from (same rows on the fixture, `cOB4` kept as stored). These stay in DuckDB SQL |
| JoinEquiv projection boundary (A-R06, authored) | 3 | Our counterexample: an always-false join under `SELECT DISTINCT 1` is empty, but the INTERSECT of the projected LEFT and RIGHT joins is `{1}`. The full-row form is equivalent with NOT NULL columns and not equivalent without them |
| jOOQ documented transforms (A-R07, authored) | 8 | `COUNT(*) > 0` to `EXISTS`, `COUNT(x) > 0` to `EXISTS ... x IS NOT NULL`, unnecessary `DISTINCT` in `EXISTS`, and negative siblings: the dropped `IS NOT NULL`, `HAVING`, an aggregate inside `EXISTS`, `LIMIT 0`. A `GROUP BY` on the correlated key stays equivalent. Written by us from the [jOOQ manual's transformation pages](https://www.jooq.org/doc/latest/manual/sql-building/queryparts/sql-transformation/transform-patterns/); no text is copied |

Labels are **equivalent** (133: same rows on every database the declared keys and NOT NULL columns allow), **fixture** (15: same rows on the source's own data only, for example "no part is named 'a'") and **not_equivalent** (9). Each is written in `why`. TPC-H pairs get the specification's primary keys and NOT NULL columns.

## How a pair is decided

1. **proven**: `kumo_sql_prover.prove_equivalent` (structural) or `prove_equivalent_algebraic` (with the fixture's types, keys and NOT NULL columns) proves it.
2. **refuted**: the algebraic prover's counterexample (SMT model or executed search) or the targeted refuter (`kumo_sql_prover.refute.find_targeted_difference`) gives a database, and the database keeps the declared keys and NOT NULL columns and separates the two queries in DuckDB, confirmed with the optimizer off (`run_unoptimized`). An SMT model can use placeholder values (a string in an `INT64` column); such a claim does not count as a refutation, and it was the case for one TPC-H pair, which the targeted refuter then refuted with a typed database.
3. **unknown**: anything else.

Every pair also runs on its source's data: TPC-H at scale 0.01 (Trino's `tiny` schema, made by `tools/benchmark_corpora.py`; it reproduces Trino's expected rows for order 31718), the inline Spark, PostgreSQL, DuckDB and authored tables otherwise. All 157 agree with their label there. Trino's inline `VALUES` become `UNNEST([STRUCT(...)])` in GoogleSQL, which KumoSQL's DuckDB reading refuses; for execution only, the harness rewrites those to the same rows as `SELECT ... UNION ALL SELECT ...`. The provers get the pairs as written (and since 2026-11-11 read the inline rows themselves, `inline_rows.py`).

**Held out.** One case in five, by a hash of its id (36 of 157). Every case, held-out ones included, was seen while building the harness and its Trino translations, so the held-out score is not a clean estimate (tuned on test). The prover rules of 2026-11-11 were developed on the other four fifths; the held-out pairs were scored, not read.

## Scores

2026-11-11, on 10f3be2 (sqlglot 30.21 compiled, z3-solver 5.4.1.0, a 4-core machine): **145/157 decided correctly, 0 wrong**: 124 proved, 21 refuted, 12 unknown. Held out: 33/36, 0 wrong.

| Source | Scored | Proved | Refuted | Unknown |
| --- | --- | --- | --- | --- |
| Trino | 134 | 114 | 12 | 8 |
| Spark | 7 | 5 | 2 | 0 |
| PostgreSQL | 2 | 1 | 1 | 0 |
| DuckDB | 3 | 0 | 0 | 3 |
| JoinEquiv (authored) | 3 | 0 | 2 | 1 |
| jOOQ (authored) | 8 | 4 | 4 | 0 |

By origin: original pairs (upstream pairs, translated) 114/122, adapted pairs (paired, completed or changed by us) 21/23, authored guards 10/12. By label: 124/133 equivalent pairs proved, 13/15 fixture-only pairs refuted, 8/9 not-equivalent pairs refuted. 0 prover claims were left unknown for failing to replay.

The baseline, 2026-10-04 on KumoSQL master 0f13b9af with no prover change made for this eval, was 41/157 (20 proved, 21 refuted, 116 unknown; held out 11/36). The refutations are the same 21 pairs; the 104 new proofs come from the rules below, written for this eval on its development pairs (the held-out fifth was scored only as a count, and every case had already been seen while the harness was built, so it is tuned on test either way).

| Module | Reads | Pairs it opened |
| --- | --- | --- |
| `inline_rows.py` | `UNNEST([STRUCT(1 AS a, 2 AS b), ...]) AS t`, how sqlglot writes another engine's `VALUES`, as the derived table `(SELECT 1 AS a, 2 AS b UNION ALL ...) AS t` | with the next one, Trino's joins over inline rows: 81 of the 104 new proofs |
| `closed_queries.py` | a query that reads no table has one result: a small interpreter works out its rows and the query becomes that list, sorted. Integers, Booleans, string equality, NULL, three-valued logic, `IN`, `EXISTS`, scalar subqueries, every join kind, `GROUP BY` with `COUNT` / `SUM` / `MIN` / `MAX`, `DISTINCT`, set operations, an element of an array written in place, and decimals of up to 15 digits in comparisons only; anything else is declined | the same, plus `1.0 IN (..)` and an array subscript |
| `counted_joins.py` | `SELECT COUNT(*)` over a `FULL JOIN` whose `ON` compares a left column, as the count of the `LEFT JOIN` plus the `RIGHT JOIN` rows whose left column is NULL; a `FULL JOIN` under an `IS NOT NULL` test keeps the tested side; `COUNT(*)` does not read the select lists of the `UNION ALL` it counts | Trino's FULL JOIN counts against their `LEFT JOIN UNION ALL RIGHT JOIN` rewrites |
| `count_exists_rules.py` | `(SELECT COUNT(*) ..) > 0` and its spellings as `EXISTS`, the tests for no row as `NOT EXISTS`; `COUNT(x)` moves `x IS NOT NULL` into the subquery; a count grouped on its correlated key only as a `WHERE` conjunct, where its NULL does what FALSE does | the jOOQ transforms (their negatives stay refuted) |
| `subquery_absorption.py` | `x IN (..) OR x IN (the same with more filters)` is the first test, likewise `EXISTS`, under `NOT`, and for `AND`: by the order FALSE < NULL < TRUE, so NULLs included | Spark's implied ORs |
| `typed_comparisons.py` | with declared types: a DATE plus a whole-day interval, compared with a date, is the `DATE_ADD` spelling; `x + n OP y + n` over integers is `x OP y` | two date-bounded joins, `orderkey + 1 = orderkey + 1` |

All are shape rules (off on the prover's second attempt, see [provers.md](../provers.md#a-second-attempt-without-the-shape-rules)) and read GoogleSQL only. Each has made-up pairs it must prove and near misses it must not, in its own test file. The interpreter of `closed_queries.py` is compared with DuckDB on seeded random queries of its fragment (400 in the gate; a longer run of 14,340 found no difference), and the other rules run their rewrite against the original in DuckDB on random rows with NULLs.

What is refuted: twelve Trino pairs that agree only on the tiny data (the four empty-build-side tests, true only while no part is named 'a'; six single-nation-row and single-order tests), Spark's NOT IN against NOT EXISTS and its fixture-only pair, the PostgreSQL dropped-qual reading, the JoinEquiv projection counterexample and the unguarded full-row form, and four jOOQ negatives. Every refutation database respects the fixture's declared keys and NOT NULL columns (the harness checks this before it counts one).

What stays unknown (12):

* Nine equivalent pairs: `COUNT(*)` over `orders UNION ALL orders` joined on its key against `2 * COUNT(*)`; a FULL JOIN count whose `ON` compares no column of the left side (`counted_joins.py` needs one to tell the padded rows); closed queries that read a struct field through a path, unnest an array column or call `SAFE_DIVIDE`; a DuckDB collation pair (the provers do not model collations); the JoinEquiv `DISTINCT` join against an `INTERSECT` of outer joins; and two held-out pairs.
* Two fixture-only pairs and one not-equivalent pair that no refuter separates (none is proved, including the `NOACCENT.NOCASE` collation trap).

## Limits

* The GoogleSQL translations were run on DuckDB through KumoSQL's BigQuery reading, not on BigQuery. Two readings are worth knowing: a Trino `a.*` after `USING` leaves out the join column, so the translation says `a.* EXCEPT (orderkey)`; and `date + INTERVAL n DAY` is kept as written on the Trino side while H2's `DATEADD` becomes `DATE_ADD`.
* The Spark pairing, the PostgreSQL and DuckDB counterparts and the JoinEquiv and jOOQ guards are ours. The PostgreSQL dropped-qual query is our reading of the commit message ("we may drop essential join quals"); the bug report itself is on a blocked host, and so are PG17982, PG17985 and DuckDB issues 20483 and 20486.
* Trino's labels assume no arithmetic overflow and the TPC-H keys; Trino's own test compares results on the tiny data only.
* Overlap: none of these pairs is in the [optimizer-bug pairs](optimizer-bugs.md) (R019); the closest are its NOT IN decorrelation cases (bug-003, bug-022), which are different queries. The DuckDB test file's queries are also run one at a time, through the rewrites, by the [DuckDB SQLLogicTest eval](https://github.com/wojoe-444/KumoSQL/blob/master/docs/evals/engine-suites.md); here they are pairs.
