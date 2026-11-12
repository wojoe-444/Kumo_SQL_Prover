# QUITE's LLM query rewrites

[QUITE](https://github.com/Yuyang-Song/QUITE) (Song et al., "QUITE: A Query Rewrite System Beyond Rules with LLM Agents") compares LLM-based query rewriters on TPC-H, DSB, Calcite and SQLStorm (StackOverflow) queries. Its `experiments_results/` folder publishes every rewrite the systems produced, each with an `equivalence` flag from running the original and the rewrite on the authors' PostgreSQL instance. KumoSQL reads those pairs with no model at run time: the algebraic prover, then databases on which both queries are replayed in DuckDB.

```
python tools/quite_bench.py                    # every distinct pair (about 40 minutes on 4 cores)
python tools/quite_bench.py --split dev        # the pairs used while developing
python tools/quite_bench.py --sample 48        # a pinned stratified sample (the test draws it from the development pairs)
python tools/quite_bench.py --workers 2        # fewer processes on a shared machine
python tools/quite_bench.py --log run.jsonl    # append verdicts as they land; a rerun resumes
python tools/quite_bench.py --write-results    # quite-rewrites and quite-negatives (whole corpus only)
python tools/quite_bench.py --contains "fetch first" --no-fetch-as-limit   # the baseline reading of FETCH, on the pairs that use it
python tools/quite_bench.py --overlap          # which originals other evals hold (fetch them first, below)
```

## Source

The repository has no licence, so nothing from it is copied into KumoSQL. `tools/quite_bench.py` downloads the 52 flagged result files and the four schema files from commit `0cffd7ce412dcd5c46bf272cd6464a00120db91c` into `~/.cache/kumosql-bench/quite` (or `$KUMOSQL_BENCH_DATA`) and checks each file's SHA-256. Credit for the data belongs to the authors.

| Benchmark | Queries | Systems | Rewrites | Distinct pairs | Flagged equal | Flagged unequal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TPC-H | 65 | 13 | 819 | 591 | 764 | 55 |
| DSB | 174 | 13 | 2,028 | 1,545 | 1,764 | 264 |
| Calcite | 59 | 13 | 754 | 583 | 618 | 136 |
| SQLStorm | 41 | 13 | 559 | 455 | 427 | 132 |
| Total | 339 | 13 | 4,160 | 3,174 | 3,573 | 587 |

The 13 systems are QUITE and QUITE with hints, LLM-R2, R-Bot and a plain LLM agent on several models (Claude 3.7, DeepSeek R1 and V3, GPT-4o), and LearnedRewrite. A fourteenth file per benchmark (`EXP_QUITE_*`) holds QUITE's raw output without a flag and is not used. "Queries" counts distinct originals: the files spell some queries differently (`interval '30 days'` against `interval '30' day`, quoted and unquoted names), number them differently, and a few systems store an original with a renamed alias (`total_value` against `value`), so a query is identified by its parsed, case-folded text. That is why the counts exceed the 63, 156, 58 and 43 queries named in the file names. Identical (original, rewrite) texts are scored once, with the number of systems that produced them kept; 324 of the 3,174 distinct pairs have a rewrite identical to the original up to spaces (the system found nothing to change).

## What a flag means

The flags are execution results on one database, not semantic labels. The authors document one rewrite (TPC-H Q2, `documents/equivalence_definition_and_known_cases.md`) that returns the same rows on their instance but drops a `region` filter, and they set its flag to false by hand. Their evaluation script also flags a pair unequal when either query hits the 300-second timeout or the rewrite fails to run. Each flagged-unequal rewrite is classified from its recorded times:

| Why it was flagged unequal | How it is recognised | Distinct pairs |
| --- | --- | ---: |
| `rows`: both queries ran and the results differed | the rewrite's recorded time differs from the original's | 214 |
| `error`: the rewrite failed to run on the authors' instance | its recorded time equals the original's | 197 |
| `timeout`: either query hit the 300-second limit | a recorded time of exactly 300 s | 85 |
| `documented`: the TPC-H Q2 case above | named in the authors' document (QUITE and QUITE with hints) | 2 |

## How a pair is decided

Nothing reads the flag while deciding.

1. **Proven.** `prove_equivalent_algebraic` with the published schema (types, primary keys, NOT NULL, foreign keys), PostgreSQL dialect, output names ignored, results compared as bags. One proof attempt is capped at 60 seconds.
2. **Refuted.** A database on which DuckDB returns different bags for the two queries. Candidates are the prover's counterexample, KumoSQL's targeted databases (`kumo_sql_prover.refute`, 8 seconds per pair), the database the authors describe for their Q2 case (for TPC-H pairs), and a TPC-H instance at scale factor 0.01 generated with `tpchgen-cli` when it is installed. Every candidate is repaired to respect the keys and foreign keys (composite ones included), rejected if it still breaks one, and replayed: the difference counts only when DuckDB's unoptimized plan agrees (`run_unoptimized`). On the TPC-H instance that unoptimized run is capped at 30 seconds, and one that does not finish is not a refutation.
   When none of these separates the queries, a **directed search** (`tools/witness_search.py`) draws small databases from the two queries' own text for 8 seconds (1 second for a proved pair): every constant a column is compared with, its neighbours and NULL; one small shared set of values for columns compared with each other, so joins match and groups repeat; groups larger than any number a count is compared with; values within the factor when two columns are compared as `a BETWEEN 0.9 * b AND 1.1 * b`; a single row holding the constant that a subquery read as one value filters on; and, in each table, rows that copy another row but for one or two columns. A database that separates the queries is reduced row by row and then goes through the same replay as every other candidate. The search adds three conditions of its own, each of which can only reject a database. A divisor that is zero raises an error in the replayed text, as it does in PostgreSQL (DuckDB alone would return NULL), and a column read inside a divisor only holds powers of two, so the floating-point quotient DuckDB computes compares with a constant as PostgreSQL's exact one does. A pair whose rows can depend on input order is searched only when the one cause is a `LIMIT` on the outermost query, and a database counts only if neither query returns more rows than that `LIMIT` when run without it. The unoptimized replay of the reduced database runs under the 10-second query limit.
3. **Unknown**, **unsupported** (sqlglot cannot parse a query), **timeout** or **error** otherwise.

Every proof goes through step 2 as well; a proof with a replayed difference is **wrong**.

DuckDB runs with PostgreSQL's integer division and NULL ordering, `now()` and `current_date` read one fixed instant on both sides, and `char(n)` padding is ignored. Queries with `random()` are never refuted. When a query's rows can depend on input order (a `LIMIT` whose `ORDER BY` does not cover every output column, `ROW_NUMBER`, `LAG`, `STRING_AGG`, `DISTINCT ON`, a `ROWS` frame), a difference counts only if both queries return the same bags with every table loaded in three row orders, and the generated instance (whose row order cannot be changed) is not used.

Two harness-side readings, no KumoSQL change: identifiers are folded to lower case, as PostgreSQL folds unquoted ones (the Calcite queries quote `"EMP"` against tables created as `emp`; 330 pairs needed a quoted name folded and are counted apart below), and `FETCH FIRST n ROWS ONLY` is read as `LIMIT n`, which the prover handles.

**Held out.** One query in five, by SHA-1 of `quite/<benchmark>/<query hash>`, is held out with all its rewrites; `--split dev` runs the rest. Rules have been written against the development pairs since 2026-11-10 ([Before and after](#before-and-after)); the held-out queries are run for scoring only.

## Scores

Measured on 2026-11-12 over the whole corpus (3,174 distinct pairs from 4,160 rewrites) on commit b01aa90 (1,827 flagged-equal pairs proved on 2026-11-11, on ff364a7; 1,659 on 2026-11-10, on a861879), with `tpchgen-cli` installed so the TPC-H pairs also run on the generated instance. **0 wrong** in both parts. One run with four worker processes took 43 minutes on 4 cores.

### Flagged equal (`quite-rewrites`)

A pair scores proved when the prover shows it equivalent; it leaves the denominator when a replayed database separates the queries (the flag is then an instance-label disagreement, not an error of either side).

| Benchmark | Pairs | Proved | Refuted | Unknown | Timeout or error | Score |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TPC-H | 543 | 496 | 8 | 39 | 0 | 496/535 |
| DSB | 1,319 | 967 | 102 | 239 | 11 | 967/1,217 |
| Calcite | 477 | 405 | 40 | 32 | 0 | 405/437 |
| SQLStorm | 335 | 174 | 37 | 124 | 0 | 174/298 |
| **All** | **2,674** | **2,042** | **187** | **434** | **11** | **2,042/2,487, 0 wrong** |

**Which pairs leave the denominator, and why.** On 2026-11-10 the total was 2,609: 2,674 pairs are flagged equal and 65 were refuted. It is 2,487 in the run of 2026-11-12 because 122 more pairs are refuted (2,479 on 2026-11-11, with 130). Six of the eight pairs that the run of 2026-11-11 refutes and this one does not (DSB, all found by the `wide` directed search) are refuted again when decided alone on this tree, 12 to 18 seconds each: the search runs against a time budget and does not find every refutation under whole-run load. Counting them, the total would be 2,481 and the score 2,042/2,481 (82.3%). Of the 195 refuted pairs of 2026-11-11, 130 were found by the directed search and 65 by the earlier sources of databases; the earlier sources alone give the 65 of 2026-11-10 (64 on a rerun on main with the same harness, one SQLStorm pair that the 8-second search does not find every time). A pair leaves the total for one reason only: a legal database on which the two queries return different rows, replayed with DuckDB's optimizer off. The authors' single PostgreSQL instance did not separate them, so their equal flag is a label disagreement. No pair is dropped for being hard. Read against the old total the score is 2,042/2,609 (78.3%), against all flagged pairs 2,042/2,674 (76.4%), against the new total 2,042/2,487 (82.1%). The pairs that left, by benchmark, as recorded on 2026-11-11: DSB 109 (1 to 110), SQLStorm 5, Calcite 11, TPC-H 5; this run has DSB 101 (1 to 102). Their causes, read for the development pairs, are a grouping key dropped, `COALESCE` dropped, `x / y BETWEEN a AND b` rewritten as `x BETWEEN a * y AND b * y` (wrong for a negative divisor), an `IN` list member or a `BETWEEN` condition dropped, a `ROLLUP` with swapped columns, a filter moved into a CTE that also feeds an average, and a cast to `DECIMAL(19,0)` that loses a half. None of them is proved by any version of the prover.

* **Held out** (one query in five, 564 pairs, 70 queries): 409/529 proved, 0 wrong, 35 refuted (376/528 on 2026-11-11, 346/555 on 2026-11-10; 409/564 over all flagged pairs). The development pairs score 1,633/1,958, 0 wrong, 152 refuted (1,451/1,951 on 2026-11-11, 1,313/2,054 on 2026-11-10). The rules of 2026-11-12 were written on the TPC-H, Calcite and SQLStorm development pairs: those went from 1,451 to 1,633 proved (+182 of the +215), and the held-out fifth, never read one by one, from 376 to 409 (+33, +9%), so these rules carry over less than the earlier ones did. The rules of 2026-11-10 were written on the development pairs, which went from 1,075 to 1,313 proved (+22%); the held-out fifth, never read, went from 279 to 346 (+24%), so the rules carry over to queries they were not written for. The rules of 2026-11-11 were written on the DSB development pairs: those went from 663 proved (of 1,075) to 785 proved (of 985, after 90 more refutations), and the held-out DSB pairs, never read, from 146 to 176 proved (+21%, of 224). Two more pairs, flagged equal by one system and unequal by another, are not scored.
* **Original and adapted apart.** 271 of the pairs needed a quoted name folded to lower case (the Calcite queries): 210 proved, 28 refuted, 33 unknown. The other 2,403 are scored as published: 1,832 proved, 159 refuted, 401 unknown, 10 timeouts (DSB pairs of 8,000 to 38,000 characters that pass the harness's 60-second limit) and 1 error.
* **Errors.** One DSB development pair raises a z3 `ArgumentError` about a deadline argument inside the harness's alarm; it is counted as neither proved nor unknown, and main has one such pair too (a different one: the error follows the 60-second alarm, not the pair). The three TPC-H pairs that made the prover raise a z3 exception in October no longer do.
* **Why unknown.** Across all 654 unknown pairs (both flags) the commonest reasons are that no row-preserving mapping between the queries was found (434), a `UNION` whose branches differ in shape (43), a `LIMIT` on one side only (21), a derived table that is not joined on all of its columns (16) and a `LIMIT`, `OFFSET` or `ORDER BY` that differs (12). (On 2026-11-11 there were 1,241 unknown pairs; the other reasons were not recounted.) The reasons of October (an aggregate over an outer join, 148; a query that could not be printed back faithfully, 111) are gone. The unknowns were not classified further.

### Flagged unequal (`quite-negatives`)

| Why it was flagged | Pairs | Refuted | Proved | Unknown | Unsupported or timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| rows differed | 214 | 153 | 14 | 47 | 0 |
| rewrite failed to run | 197 | 14 | 5 | 140 | 38 |
| a query timed out | 85 | 35 | 12 | 32 | 6 |
| documented TPC-H Q2 | 2 | 2 | 0 | 0 | 0 |
| **All** | **498** | **204** | **31** | **219** | **44** |

204 of 498 are refuted by a replayed database (134 on 2026-11-10; 70 are found by the directed search); held out, 43 of 106 (8 proved, 0 wrong). **No pair is proved on grounds that a replayed database contradicts.** The 31 proofs are not errors, and each is accounted for:

* 14 pairs flagged for different rows are `ORDER BY ... LIMIT` queries whose sort key has ties (Calcite's `ORDER BY SAL`), so the authors' PostgreSQL happened to cut the tied rows differently for the two queries. The prover proves them equal on the stated assumption that tied rows are cut alike ([ORDER BY, LIMIT and OFFSET](sqlsolver.md#order-by-limit-and-offset)); no replayed database separates them.
* 17 pairs were flagged only because a query timed out (12) or the rewrite failed to run (5): a rewrite that fails on PostgreSQL (one has an output alias spelled `0`) can still be equivalent as written. Five of the ten are development pairs the rules of 2026-11-10 newly prove (rewrites of two DSB queries). Each was read by hand and is equivalent as written: a correlated `AVG` turned into a join with the grouped average, comma joins written as `JOIN`, `75 * 0.01` written as `0.75` (exact decimals in PostgreSQL), and a date plus 90 days written as the date it is. The rules of 2026-11-12 prove six more development pairs of this kind: two TPC-H pairs flagged for a timeout and one for a failed run (rewrites of TPC-H Q20 and Q11 with the same rows as written) and three Calcite window-function rewrites whose output alias is spelled with a digit; each was read by hand and is equivalent as written.
* The database search has a budget of 8 seconds per pair and does not find the same refutations in every run: a rerun of main with the same harness refutes one flagged-equal SQLStorm pair fewer than the recorded run (64, not 65), and the two SQLStorm pairs this branch's run does not refute are ones main refutes. Run alone, neither tree refutes them: whether the search finds them seems to depend on what the worker process ran before (not tracked down). Flagged unequal, main refutes 134 in the recorded run and 114 in a rerun in this container (cause not tracked down: other machine, z3 5.4.1 instead of 5.4.0, or the time budgets); this branch refutes 204 here, so against the same container it is +90 and no pair that main refutes is lost except the SQLStorm pair named above.

The two documented TPC-H Q2 rewrites are refuted on the database the authors describe for the case (a supplier outside EUROPE that ties the EUROPE minimum cost). The same rewrite from a third system (DeepSeek R1), flagged equal, is refuted there too: one of the 65 refuted flagged-equal pairs.

### Before and after

The first run of this harness was the baseline; nothing was tuned on it. One harness reading was then changed, on development pairs only: `FETCH FIRST n ROWS ONLY` is read as `LIMIT n`, because DuckDB cannot replay `FETCH` and the prover's tie check cannot see it. On the 100 development pairs that use `FETCH` (81 flagged equal) the baseline proved 16/67 and the final harness proves 7/67, both 0 wrong: 13 identical-text pairs are no longer proved because the prover declines an aggregate over an outer join under a row cap, and 4 pairs are gained. The lower number is the honest one. A second change bounds the unoptimized replay of a TPC-H instance to 30 seconds, because a six-table comma join becomes a cross product with DuckDB's optimizer off; a difference there is counted only when that run finishes and agrees, so a timeout is an unknown, never a refutation.

**Rules for the unknown pairs (2026-11-10).** Until then no prover rule had been written for this eval, and the harness on main (c384968) measured 1,354/2,608 proved, 0 wrong (held out 279/555). The largest groups of unknown development pairs were then read and eight modules written, each an exact rewrite with its own test file of proofs, near misses that DuckDB separates and random replays:

| Module | What was declined before | What it does |
| --- | --- | --- |
| `tailed_outer_aggregates.py` | an aggregate over an outer join with `ORDER BY` / `LIMIT` ("aggregate over an outer join") | reads the join as one derived relation and keeps the tail on the select |
| `tailed_grouping_sets.py` | `GROUP BY ROLLUP / CUBE / GROUPING SETS` with a row tail | moves the tail onto a select around the grouped one |
| `printed_spelling.py` | a query whose normal form did not print back to itself (`a \|\| b \|\| c`, a grouping set of one column) | spells both the way they are printed |
| `closed_sources.py` | a correlated scalar aggregate over a `WITH` table | treats a derived table that reads nothing from outside as a relation, so the aggregate becomes a grouped join |
| `wrapped_join_filters.py` | a filter above an outer join against the same filter pushed below it | evaluates the filter inside the derived table that wraps the join |
| `listed_output_names.py` | a column list on a derived table with an unnamed aggregate (TPC-H Q13) | names the output before the list is expanded |
| `one_row_tails.py` | a one-row report with a pointless `ORDER BY` / `LIMIT` (TPC-DS style) against the same without | drops the tail of a global aggregate |
| `string_spelling_rules.py` | `CAST(c AS CHAR(n))` of a column declared `CHAR(n)`, `LEFT(x, n)` against `SUBSTRING(x, 1, n)` | reads both as the value itself |

`string_number_compare.py` also stopped reading a date plus an interval as a number. Together: 1,354 of 2,608 to 1,659 of 2,609 proved (+305), by benchmark TPC-H 378 to 403, DSB 592 to 809, Calcite 345 to 349, SQLStorm 39 to 98. No pair proved on main is lost, development or held out. Other evals did lose four proofs to these rules firing on one query of a pair only, so a pair of up to 4,000 characters that they leave unproven is tried again without them ([a second attempt without the shape rules](../provers.md#a-second-attempt-without-the-shape-rules)); the numbers here are measured with that in place.

**Rules for the unknown TPC-H, Calcite and SQLStorm pairs (2026-11-11/12).** The 329 development pairs left unknown in those three parts (the DSB part has its own batch) were read in groups and about forty modules written, each an exact rewrite with its own test file of proofs, shapes it must leave alone, near misses that DuckDB separates and random replays ([the list, by benchmark](../provers.md#rules-for-quites-tpc-h-calcite-and-sqlstorm-pairs-2026-11-11)). Together with the DSB batch of 2026-11-11 and the Calcite batch of the same day, which were merged first: 1,827 of 2,479 to 2,042 of 2,487 proved, by benchmark TPC-H 417 to 496, DSB 961 to 967, Calcite 349 to 405, SQLStorm 100 to 174. The total moved by 8 because the directed search refuted 8 fewer DSB pairs in this run than in the recorded one (its budget is 8 seconds per pair).

Three safeguards came out of the whole-corpus runs, because the gate does not see these losses. A rule that fires on one query of a pair only moves that side away from the form the two share: twelve DSB pairs of 6,000 to 14,000 characters lost their proofs to `keyed_join_groups.py` and `membership_joins.py` until the first fired only when something is carried over and the second left `EXISTS` inside `EXISTS` alone. Larger pairs have a [narrower second attempt](../provers.md#a-narrower-second-attempt) that switches off only the rules that fired (`nested_membership`, `window_filters`, `one_row_groups`, `keyed_join_groups`); it recovered a held-out LOGOS pair and a SQLStorm pair. And any rule that reads a declared unique key requires NOT NULL on every key column, which [conditional equivalence](conditional-equivalence.md) on the Singh pairs showed. Against main (ff364a7, run the same way in the same container), no flagged-equal pair that main proves is lost on QUITE; the other evals moved only where recorded on their pages (conditional Singh 590 to 783 conditional, DB-GPT 29 to 30 of 30, LOGOS 42 to 43 of 69, 7 more proofs and no false one in the soundness fuzz).

**Exposure.** The development pairs are tuned on: the rules were written while reading them. The held-out queries were run for scoring only and never read one by one; their share of the gain is given above. The two harness changes of October came from development pairs.

Results files: `benchmarks/results/quite-rewrites.json` and `benchmarks/results/quite-negatives.json`. The pinned-sample test (`tests/test_quite_bench.py`) runs 51 stratified development pairs without the generated instance: 32 proved, 2 refuted, 0 wrong, with floors of 30 and 2.

## Overlap with other evals

`python tools/benchmark_corpora.py fetch llm-r2 sqlstorm dsb`, then `python tools/quite_bench.py --overlap`, matches each QUITE original against the other evals' queries, as the same text (case, quotes and layout ignored) and as the same shape (every literal masked, so another instance of the same template):

| Other eval's queries | QUITE queries | Same text | Same shape |
| --- | ---: | ---: | ---: |
| R-Bot's Calcite pairs | 59 (Calcite) | 12 | 12 |
| SQLSolver's Calcite pairs | 59 (Calcite) | 1 | 1 |
| SQLSolver's TPC-H pairs | 65 (TPC-H) | 0 | 29 |
| LLM-R2 train, TPC-H (3,772 queries) | 65 (TPC-H) | 29 | 42 |
| LLM-R2 test, TPC-H (500 queries) | 65 (TPC-H) | 10 | 42 |
| LLM-R2 train, DSB (1,635 queries) | 174 (DSB) | 3 | 108 |
| LLM-R2 test, DSB (500 queries) | 174 (DSB) | 1 | 93 |
| Analytical coverage, DSB (156 queries) | 174 (DSB) | 0 | 120 |
| Analytical coverage, SQLStorm v1.0 StackOverflow (18,251 queries) | 41 (SQLStorm) | 5 | 5 |
| Analytical coverage, SQLStorm v1.0 TPC-H (17,036 queries) | 65 (TPC-H) | 0 | 0 |
| Analytical coverage, SQLStorm v0.0 TPC-H (44,000 queries) | 65 (TPC-H) | 47 | 62 |

The TPC-H and DSB originals are template instances of the kind [LLM-R2](https://github.com/wojoe-444/KumoSQL/blob/master/docs/evals/llmr2-bench.md) and [analytical coverage](https://github.com/wojoe-444/KumoSQL/blob/master/docs/evals/analytical-sql-coverage.md) already run: 62 of the 65 TPC-H originals have the shape of a SQLStorm v0.0 TPC-H query (47 verbatim), and 120 of the 174 DSB originals have the shape of a DSB query already scored. What is new here is the rewrites and their flags, which no other eval has. The Calcite originals overlap [R-Bot's Calcite pairs](sqlsolver.md#r-bots-calcite-pairs) for 12 of 59 queries, but the rewrites differ (R-Bot's are Calcite's rule outputs). Five of the 41 SQLStorm queries are verbatim SQLStorm v1.0 StackOverflow queries; the others are not in SQLStorm v1.0 as published. So source queries recur across evals, but no pair does.

## Limits

* The authors' database instances are not public, so a refutation comes from a database KumoSQL builds; a flagged-equal pair that is refuted is an instance-label disagreement, not an error of either side.
* Results are compared as bags. The authors compare as lists when the original has an outer `ORDER BY`; 37 distinct pairs drop the original's top-level `ORDER BY`.
* PostgreSQL and DuckDB can still differ in places the harness does not cover (string collation, numeric precision of `AVG` beyond six decimal places); a replayed difference is shown with its database, so each can be checked by hand.
