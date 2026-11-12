# DB-GPT rewrite examples

[Plain-language version](../../docs_simple/evals/dbgpt-rules.md)

[DB-GPT](https://github.com/TsinghuaDatabaseGroup/DB-GPT) (Apache-2.0) ships 36 before/after PostgreSQL rewrites as demonstrations for its LLM query rewriter. They come without labels or schemas, so each one was given a schema and a label by hand, and the test suite checks every label on DuckDB. Data, labels and adaptations: [tests/fixtures/dbgpt_rules](../../tests/fixtures/dbgpt_rules/README.md). Results file: `dbgpt-rules`.

```
python tools/dbgpt_rules_bench.py                  # score, and check every label on DuckDB
python tools/dbgpt_rules_bench.py --write-results
```

## Labels

| Label | Cases | Checked by |
| --- | ---: | --- |
| equivalent | 30 | 400 random DuckDB databases that respect the case's keys and NOT NULL columns, filled from the literals the queries mention: no difference |
| not equivalent | 3 | a stored counterexample database (14: `'abc'` matches `LIKE 'abc%'` but is not `> 'abc'`; 30: a row matching both sides of an `OR` appears twice after the split into `UNION ALL`; 34: the rewrite joins `t1.c1` to `t2.c2`) |
| invalid | 3 | DuckDB rejects a query (10, 33, 36); left out of the score |

Every DuckDB difference is confirmed with the optimizer off (`kumo_sql_prover.duckdb_load.run_unoptimized`). Case 35 is equal as bags but its `ORDER BY` differs (a missing count sorts as 0 on one side and as NULL on the other); KumoSQL compares bags, so it is labelled equivalent.

## How a case is decided

**proven**: `prove_equivalent_algebraic` in the PostgreSQL dialect with the case's keys and NOT NULL columns, output names ignored, the column types of the case's schema declared, and exact arithmetic (every number column is an INTEGER). **refuted**: a random DuckDB database on which the bags differ, confirmed with the optimizer off. **unknown**: neither. The label is read only to score. Wrong is a proof against a not-equivalent label or a refutation against an equivalent one.

## Scores

| Date | Equivalent proved | Not equivalent refuted | Wrong | What moved it |
| --- | ---: | ---: | ---: | --- |
| 2026-10-03 | 24/30 | 3/3 | 0 | baseline on master, default arithmetic |
| 2026-10-03 | 27/30 | 3/3 | 0 | exact arithmetic for these integer schemas (`a+1+2` against `a+3`, `a+1=2` against `a=1`, `-a=3` against `a=-3`) |
| 2026-11-11 | 29/30 | 3/3 | 0 | `LIKE` with no wildcard as `=` on a column declared VARCHAR or TEXT (15, `literal_like.py`), and an `OR` of `EXISTS` over one `FROM` as one `EXISTS` (29, `exists_disjunction.py`); the eval now passes the declared column types |
| 2026-11-12 | 30/30 | 3/3 | 0 | case 35 (a correlated `COUNT(*)` in the select list against a grouped `LEFT JOIN` with `CASE WHEN cnt IS NULL THEN 0`) is proved by the counted-subquery and outer-count rules written for QUITE's TPC-H, Calcite and SQLStorm pairs; no rule was written for this eval |

Nothing is left unknown: case 35, unknown on 2026-11-11, is proved since 2026-11-12.

**Tuned on test.** The 36 cases are all there is: the labels, schemas and the exact-arithmetic setting were chosen with every case in view, the two rules of 2026-11-11 were written for cases 15 and 29 after reading them, and there is no held-out split. The gain of 2026-11-12 came from rules written for other benchmarks, but the case was read before, so it counts as tuned on test too.
