# QED Calcite equivalence pairs

SQL pairs converted from the Calcite test corpus of the QED prover.

* Source: https://github.com/qed-solver/prover, `tests/calcite/*.json`
* Commit: `31f4b6c271440942ecaca1e1111d4beeabf1f14c`
* Licence: MIT, (c) 2021 The QED Team (see `LICENSE`)
* Converter: `tools/qed_to_sql.py` (regenerate with
  `python tools/qed_to_sql.py --src <prover>/tests/calcite`)
* Checker: `python tools/validate_qed_pairs.py` (needs `duckdb`)

## Files

| file | content |
| --- | --- |
| `qed_calcite_pairs.jsonl` | one case per line: `name`, `schema_id`, `sql_a`, `sql_b`, `ddl` (CREATE TABLE text), `schemas` (structured: table, columns with type/nullable, keys) |
| `qed_calcite_schemas.json` | each distinct DDL set by `schema_id` |
| `qed_calcite_skipped.jsonl` | skipped cases with the reason |
| `summary.json` | counts |

Every QED test is a Calcite rewrite rule, so the two sides of a pair are meant
to be equivalent (QED proves them); the corpus has no deliberately
inequivalent pairs.

## Result

444 files: **407 converted** (32 of them from the plan text, see the conversion
notes), 37 skipped.

| skipped | reason |
| ---: | --- |
| 8 | aggregate FILTER clause (not represented in QED's IR) |
| 7 | unsupported operator ST_POINT |
| 4 | integer-typed STDDEV/VAR aggregate (Calcite integer arithmetic not modelled) |
| 3 | projection with no columns |
| 2 | table is empty only by test-harness convention (not expressible in the schema) |
| 2 | nondeterministic aggregate ANY_VALUE |
| 2 | aggregate WITHIN DISTINCT (not represented in QED's IR) |
| 1 | unsupported operator CURRENT_TIMESTAMP |
| 1 | unsupported aggregate GROUPING |
| 1 | unsupported aggregate LITERAL_AGG |
| 1 | implicit cross-type comparison |
| 1 | window function |
| 1 | column type ANY |
| 1 | unsupported operator ROW |
| 1 | unsupported operator USER |
| 1 | LIMIT/OFFSET without ORDER BY (nondeterministic) |

(The first matching reason is recorded, so a case that has several is counted once.
The 8 FILTER cases left are `WITHIN DISTINCT` tests, whose plans raise an error when
the values of a group differ; the plan-text route declines the GROUPING, LITERAL_AGG
and window cases left as well.)

## Conversion notes

* MySQL-flavoured SQL, parseable by sqlglot (`read="mysql"`). Tables are named by
  the last component of the schema name; scans select the schema's own field
  names and alias them `c0..cN`. Every other operator is a derived table whose
  columns are `c0..cN`, so QED's ordinal column references become `alias.c<i>`.
  Calcite flattens struct columns into names like `"F1"."A0"`; the quotes are
  dropped and the name is back-quoted.
* Subqueries (EXISTS, IN, scalar, ANY/ALL) and the right side of `correlate`
  follow QED's convention: ordinals count the enclosing relation's columns first,
  then the subquery's own input. These become correlated references
  (`LATERAL` for INNER/LEFT correlate, `[NOT] EXISTS` for SEMI/ANTI).
* `union` is UNION ALL, `distinct` is SELECT DISTINCT, `intersect` and `except`
  are the set forms (the Calcite plans say `all=[false]`). SEMI/ANTI joins are
  `[NOT] EXISTS`.
* `SEARCH`/Sarg is expanded to comparisons; `IS NOT DISTINCT FROM` is `<=>`;
  `||` is `CONCAT`; integer `/` is `DIV`; AVG over an integer type is
  `SUM DIV COUNT` (Calcite types it as an integer); `+ - *` on integers cast
  operands to the result type (Calcite promotes, DuckDB would overflow).
* Sort: QED's collation has no NULLS FIRST/LAST, so plain `ORDER BY` is emitted
  (MySQL: NULLs lowest). DECIMAL columns are `DECIMAL(19, 2)` and VARCHAR is
  `VARCHAR(255)` because the IR carries no precision or length.
* Keys: the first key whose columns are all NOT NULL is `PRIMARY KEY`, the rest
  are `UNIQUE`.
* **Plan-text route.** QED's JSON form drops grouping sets, `GROUPING()`, aggregate
  `FILTER`, multi-column `COUNT(DISTINCT)`, `SINGLE_VALUE` and window functions, but
  each file also records the two Calcite plans as text (its `help` field). When the
  JSON conversion skips a case for one of those reasons (`PLAN_TEXT_REASONS` in the
  converter), the plan text is translated with `tools/calcite_plan_to_sql.py`'s
  converter, the one behind `tests/fixtures/calcite_mined/`, and its notes apply
  (explicit null-ordering keys, Calcite's types). The result is accepted only when the
  plans' tables match the tables the QED file declares (names, base types,
  nullability) and the translated root column types equal the types of the JSON
  form's root; otherwise the case stays skipped with its original reason. Such a pair
  carries `"translated_from": "plan text"`, and `summary.json` counts them
  (`converted_from_plan_text`, 32). The 375 pairs converted from the JSON keep their
  SQL byte for byte.
* Three of the 32 are real differences, found on random data: `testAggregateCaseToFilter`,
  `testDistinctCountGroupingSets1` and `testDistinctCountGroupingSets2`. QED recorded
  the plans of an older Calcite whose rewrite was later corrected (Calcite's current
  plans for the same tests agree on DuckDB, see the mined fixture). They are kept as
  pairs that must never prove, like `testRightEmptyFullJoin` and
  `testStrengthenJoinType`.
* Skipped, never guessed: aggregate FILTER / WITHIN DISTINCT (the IR drops them;
  detected from the `help` plan text), window functions, GROUPING, LITERAL_AGG,
  SINGLE_VALUE, ANY_VALUE, geospatial and dynamic functions, struct (`ANY`)
  columns, implicit cross-type comparisons, tables that are empty only by
  Calcite test-harness convention, LIMIT without ORDER BY, and anything else
  the converter does not model exactly.

## Validation

`tools/validate_qed_pairs.py` creates each case's schema in DuckDB (sqlglot
transpiles MySQL to DuckDB), fills it with random small databases that respect
NOT NULL and keys (NULLs sort first), runs both sides and compares the result
multisets. Latest run (8 databases per case, 407 cases): 403 agree on every database,
2 differ (`testAggregateCaseToFilter` and `testDistinctCountGroupingSets2`, two of the
real differences above; the benchmark's 60 databases per pair also separate the other
three), and 2 DuckDB cannot execute (`testExpandJoinExists` and
`testExpandJoinScalar`: "Cannot perform non-inner join on subquery"). A database on
which a scalar subquery meets several rows is an error in Calcite too and is skipped;
such a pair draws more databases until 8 run. This is a consistency check of the
conversion, not a proof of equivalence.
