# Singh & Bedathur LeetCode equivalence pairs

[Plain-language version](../../docs_simple/evals/singh-bedathur.md)

Rajat Singh and Srikanta Bedathur (IIT Delhi), "Can the Rookies Cut the Tough Cookie? Exploring the Use of LLMs for SQL Equivalence Checking" ([arXiv 2412.05561](https://arxiv.org/abs/2412.05561), [repository](https://github.com/rajatb115/LLMs-for-SQL-Equivalence-Checking)), studied language models as SQL equivalence judges. KumoSQL answers the same question with no model at run time: the algebraic prover, a few extra rewrites, and counterexample databases checked in DuckDB.

## Which data is public

| Set in the paper | Pairs | Available? |
| --- | ---: | --- |
| LeetCode fine-tuning pairs (`finetune/dataset`, 2,000 train + 800 eval) | 2,800 | Yes, in the repository. **Scored here.** |
| SQLEquiQuest (student submissions) | 404 | Only by request through a Google Form |
| Spider + DIN-SQL | 1,034 | Only by request through a Google Form |
| Calcite | 232 | The same pairs as SQLSolver's Calcite suite, already scored in [sqlsolver.md](sqlsolver.md#benchmark-coverage) |

The repository has no licence file, so the pairs are not copied into KumoSQL. `tools/singh_bedathur_bench.py` downloads the two files from a pinned commit (`930569d894c950c0b64f2150214394b52a40e11b`) into `~/.cache/kumosql-bench` (or `$KUMOSQL_BENCH_DATA`) and checks their SHA-256 digests. Credit for the data belongs to the authors.

**Overlap.** Every one of the 2,800 pairs (both queries, same text) is also in [VeriEQL](https://github.com/VeriEQL/VeriEQL)'s LeetCode benchmark. The difference is what comes with them: these files list only table and column names (their "Primary Keys" and "Foreign Keys" lines are a fixed placeholder naming tables the pairs never use), while VeriEQL keeps each problem's types, keys and foreign keys. This eval is therefore the *original*, constraint-free reading of the pairs; nothing is adapted.

## How a pair is decided

Nothing reads the published label while deciding; labels are only compared afterwards.

1. **Proved equivalent.** `prove_equivalent_algebraic` runs on the pair as written (MySQL dialect, output names ignored, no keys). When it finds no proof, both queries go through `kumo_sql_prover.canonical_rules` and the prover runs again. Every proof is then re-run on 900 random DuckDB databases; a database that separates the queries would make the pair **wrong**.
2. **Proved different.** A counterexample is a database on which DuckDB returns different result bags. Candidates are the prover's own counterexample, when it proposes one, and 300 random databases (half plain, half skewed towards one or two values per column so groups, duplicates and ties appear, with exact duplicate rows in some of them; table sizes reach past any `COUNT(..) >= n` threshold in the queries; values include every literal and its neighbours, the day before and after each date literal, a literal times each constant the queries divide or multiply by (`duration / 60 >= 5` gets 299, 300 and 301), and decimals with three places so `ROUND(x, 2)` differs from `x`). When those find nothing, 300 more databases come from a stream seeded by the pair itself, with tables of up to 10 rows, so pairs of one shape do not all miss on the same databases. Each candidate is checked by running both queries; a database already checked in the same search (small tables repeat) is not run again, since its answer would repeat. MySQL's `ORDER BY NULL` (no sorting) is dropped before DuckDB runs the query.
3. **Unknown** otherwise. Unknown is always preferred to wrong.

DuckDB runs with MySQL's NULL ordering and case-insensitive string comparison. A pair never gets "different" where DuckDB and MySQL could disagree: a `LIMIT` whose `ORDER BY` does not cover every output column, `LOWER()` (`LIKE` ignores case only in MySQL), or a non-deterministic function. Column types are not given, so they are inferred from use: compared with a string, a string; used as a date, a date; summed, averaged or rounded, a decimal that may hold fractions; otherwise an integer.

### Canonical rewrites

`kumo_sql_prover.canonical_rules.canonicalize` keeps every result bag on every database, NULLs included, and uses no constraints:

- `ORDER BY` without `LIMIT` is dropped.
- `SELECT DISTINCT` over a `GROUP BY` that selects every group key is the plain grouped select.
- `HAVING COUNT(*) > 0` (or `>= 1`) under a `GROUP BY` is always true.
- A select that only projects or filters one derived table is merged into it; the filter joins the inner `WHERE`, or its `HAVING` when the inner select is grouped (one row per group).
- `(a, b) IN (SELECT a2, agg FROM t GROUP BY a2)` in a top-level `WHERE` conjunct is an inner join with that grouped table, which holds at most one row per `a2`; correlated subqueries are left alone.

`tests/test_canonical_rules.py` checks each rule on random DuckDB databases and pins the near misses it must leave alone.

## Scores

Measured 2026-11-24 over all 2,800 pairs (`python tools/singh_bedathur_bench.py`, about 45 minutes on 4 cores, commit 8a0a786, hardest-cases batches 25 and 26; 2,769 from 2026-11-20 to the batch 25 head e5a5f33, 2,768 on 2026-11-19 on c7d7021 and on 6fedbbc; 2,767 on 2026-11-18 on 873f667, 2,766 earlier that day on e203738 and on main c97feeb, 2,765 on 2026-11-12 on 951be35, 2,763 on 2026-11-10 on a861879 and again on b01aa90):

**2770/2800, 0 wrong**: 896 proved equivalent, 1,874 proved different, 30 unknown. Held-out fifth: 570/580 (569 before). The newest proof (2026-11-24, hardest-cases batch 26) is a pair of the held-out fifth labelled equivalent that was declined as unsupported before. It was not read: one of its queries has a join written with `USING` beside a star, which the batch reads before its other steps, and the harness's 900 generated databases show no difference. Every other pair has the answer it had on the batch 25 head e5a5f33. The proof before it (2026-11-20, the twelfth hardest-cases batch, the date windows written for the [conditional](conditional-equivalence.md) VeriEQL sample) is a development pair labelled equivalent that was declined as unsupported before: one query tests a DATE column against `'2019-07-04'` and the other against `'2019-7-4'`, which MySQL reads as the same day. Every other pair has the answer it had on main 2a98b73 (measured on 90461c1, whose prover source is the merged one). The proof before it (2026-11-19, the tenth hardest-cases batch, written for the same sample) is a development pair labelled equivalent: the fraction of players who came back the day after their first login, written as a sum of a 1-or-0 flag over the joined rows and as a count of the joined rows that meet the test. Every other pair had the answer it had on 873f667. The two extra proofs between 2026-11-12 and 2026-11-18 are two pairs of the held-out fifth, both labelled equivalent. One is proved outright by the rules written for the [conditional eval](conditional-equivalence.md) on its development pairs; it was not read, and it agrees on 1,800 generated databases. The other is proved by the worker rules written for the [VeriEQL](verieql.md) LeetCode cases; it was not read either. Each change was measured alone at 2,766 (e203738, and main c97feeb on the same machine), and the merge of the two (873f667) has every answer of both runs pair by pair. The two proofs before it came from rules written for other evals ([QUITE](quite.md), VeriEQL); none was written for this one.

| Outcome | Pairs |
| --- | ---: |
| Proven equivalent | 896 |
| Refuted (counterexample) | 1,874 |
| Unknown | 27 |
| Unsupported (the prover cannot read a query, and no counterexample) | 3 |
| Timeout | 0 |
| Error | 0 |
| Wrong | 0 |

Supported subset: 2770/2797 (three pairs that compare a string with a number are now declined as unsupported, #542, #613 and the SELECT * and set-operation tracking that followed; the second is a dev pair whose `COALESCE(REFEREE_ID, 'SOME BOGUS VALUE') != 2` was proved equal to `REFEREE_ID <> 2 OR REFEREE_ID IS NULL`: right on MySQL only because the string reads as 0, DuckDB rejects the mixed `COALESCE`). Held-out fifth (pairs whose text hash is divisible by 5): **570/580, 0 wrong**. The split was made partway through, after the first rewrite rules, and later full-corpus runs were still read while tuning the counterexample search, so this is a weak check. From now on development runs use `--split dev`.

Column types: the pairs carry only column names, so `_text_types` in `tools/singh_bedathur_bench.py` declares every column the queries compare with a string as text, the reading the counterexample search already used. Without it the prover's check on untyped string columns (a column compared with two different strings that MySQL reads as the same number could be an integer column; see `docs/provers.md`) declined 17 proofs, 4 of them held out, all variants of one `NOT IN` query on product names. With the declaration the score and the 861 proofs of that time were unchanged. The held-out pairs were not read; the loss was found from the proof counts.

Since 2026-11-16 it declares the columns the search reads as dates (`DATE` in `column_kinds`) the same way. The string check now follows a column through an aggregate or a derived column (see `docs/provers.md`), so `MIN(d) >= '2019-01-01'` beside `'2019-03-31'` asks, like the bare column already did, whether the two strings are one number: both are 2019 to MySQL when `d` holds numbers. One held-out proof was no longer accepted for that reason. With the declaration the outcome of each of the 837 development pairs that have such a column is unchanged, and of the 240 held-out ones only that pair's changes, back to proven. The held-out pair was not read; the cause was found from its verdict, the reason the prover gave and counts of its literals.

The last step was developed on dev pairs only (the held-out fifth went from 509 to 524 without being looked at). It also fixed two ways the harness could see a difference that is not one: DuckDB returns `DECIMAL` results as Python `Decimal` and `DOUBLE` ones as `float`, and `0.33` never equals `Decimal("0.33")`, so numbers are now compared as floats rounded to six places; and every fraction the generator draws is exact in binary (eighths), because DuckDB averages decimals in floating point and a value like `1.005` lands on the other side of a `ROUND(.., 2)` midpoint from MySQL's exact result. Neither had produced a published refutation. Every dev pair labelled equivalent that this step newly refutes was checked by hand: they hinge on a NULL inside `NOT IN`, duplicate rows, or an inclusive `BETWEEN` against a half-open range.

What moved the score:

| Step | Decided |
| --- | ---: |
| Prover alone, random search on small tables | 2,211 |
| + prover counterexamples, canonical rewrites, bigger tables for `COUNT` thresholds | 2,317 |
| + skewed databases, fractional decimals, `LIMIT` over a full ordering, MySQL NULL order and case rules | 2,434 |
| + a second search stream seeded per pair with tables up to 10 rows, duplicate rows, three-decimal values, date and divisor neighbours, `ORDER BY NULL` | 2,513 |
| + `LEFT JOIN` read as inner under a NULL-rejecting `WHERE`, `WHERE` pushed into `UNION` branches (2,533 on the same master without them) | 2,581 |
| + identity casts, `x * 1.0`, `ROUND(x)` as `ROUND(x, 0)`, `NATURAL JOIN` (`cast_rules.py`; held-out fifth unchanged at 534) | 2,586 |
| + aggregate facts in the SMT model and the set-of-values reduction ([provers.md](../provers.md)), developed on dev pairs only (held-out fifth 534 to 538) | 2,614 |
| + derived tables holding an outer join read through (`outer_join_flatten.py`), developed on dev pairs only (held-out fifth 539) | 2,626 |
| + structural aggregate rewrites (`aggregate_rules.py`: shared filters into `WHERE`, filters into `HAVING`, expressions lifted out of grouped derived tables), developed on dev pairs only (held-out fifth 542), master 80ac740 | 2,650 |
| + `DISTINCT` queries split into unions (`set_split_rules.py`: `CASE` join keys, derived unions, `IN`/`EXISTS` over unions), `OR` splits in set containment, self-join symmetry; developed on dev pairs only (held-out fifth 557), master 696c6c8 | 2,728 |
| + `ABS(x - y)` facts and `x <> y` split into `x < y` and `y < x` in set containment ([provers.md](../provers.md)), developed on dev pairs only; also includes prover PRs merged since 696c6c8 (held-out fifth 560), master 30eb0cb | 2,736 |
| the same harness on main c384968 on 2026-11-10, before the next step (three pairs declined since as string-number comparisons, one proof not traced; held-out fifth 560) | 2,734 |
| + six rule modules for the undecided development pairs ([below](#the-undecided-development-pairs)), developed on dev pairs only (held-out fifth 565), commit a861879 | 2,763 |

### The undecided development pairs

On 2026-11-10 main left 46 development pairs undecided (43 unknown, 3 unsupported). Six rule modules, each an exact rewrite with its own test file of proofs, near misses that DuckDB separates and random replays, prove 24 of them. The held-out fifth gained 5 proofs and lost none.

| Module | What it reads | Example |
| --- | --- | --- |
| `case_arm_rules.py` | arithmetic moved into the literal arms of a `CASE`, and an arm that is never taken (a comparison with `NULL`, returning `NULL`) dropped | `salary * CASE WHEN c THEN 0.5 ELSE 1 END` against `CASE WHEN c THEN salary * 0.5 ELSE salary END` |
| `unmatched_outer_join.py` | a `LEFT JOIN` whose every match the `WHERE` rules out (`NOT IN` or `NOT EXISTS` on the join key) is dropped | `FROM c LEFT JOIN o ON c.id = o.cid WHERE c.id NOT IN (SELECT cid FROM o)` |
| `having_count_rules.py` | `COUNT(*) >= n` implied by a bound on another `COUNT` of the same group is dropped; a counted group key is not NULL | `GROUP BY email HAVING COUNT(email) > 1` against `WHERE email IS NOT NULL GROUP BY email HAVING COUNT(*) > 1` |
| `grouped_self_membership.py` | the distinct values of a column that are among its own table's groups are those groups | `SELECT DISTINCT x FROM t WHERE x IN (SELECT x FROM t GROUP BY x HAVING ..)` |
| `disjoint_union.py` | a `UNION` of one-row branches tagged with different literals is a `UNION ALL` | `SELECT 'low', COUNT(*) .. UNION SELECT 'high', COUNT(*) ..` |
| `semijoin_shapes.py` | a chain of `IN` tests restructured through a CTE, a derived table, a join or `EXISTS` | `x IN (SELECT d.c FROM (SELECT c FROM t WHERE p) AS d WHERE w)` |

Three things went wrong on the way and are recorded here because each was found by a measurement, not by review:

- Reading a correlated `EXISTS` as `IN` on every query lost a SQLSolver TPC-H proof and three soundness-fuzzer proofs: one side was rewritten and the other was not. The step is now taken only when the subquery holds a membership test for the other rules to work on.
- The counted-key rule adds `k IS NOT NULL` to one query at a time, and a query that spelled `COUNT(k) > 1` another way was left without it. One held-out proof was lost. It was found from the count, by switching the rule off on that pair (verdict only) and trying made-up spellings; the pair was not read. The rule now reads the bound through `NOT`, `BETWEEN`, a select alias and a qualified key.
- A stricter reading of bare column names in the string-type check ([provers.md](../provers.md)) declined 49 development proofs, because this harness declares only the text columns. The check now uses the schema's column lists to see that no other table can hold the name.

**Exposure of the held-out fifth.** The rules were developed on development pairs. During the work a subagent ran `tests/test_singh_bedathur_benchmark.py`, which runs every pair, for about 30 minutes; nothing of its output was read. The 19 development pairs still unknown are mostly pairs with no row-preserving mapping (10), a `LIMIT` on one side only (2) and one the prover's own counterexample does not replay.

### Against the published labels

| Verdict | Label "Equivalent" | Label "Non Equivalent" |
| --- | ---: | ---: |
| Proved equivalent | 895 | 0 |
| Proved different | 476 | 1,398 |
| Unknown | 29 | 2 |

No proof contradicts a label. The 476 pairs labelled equivalent that get a counterexample are equivalent only under LeetCode's constraints, which the files drop: most rely on a key (`UNION` versus `UNION ALL`), a NOT NULL column (`NOT IN` versus an anti-join) or a foreign key. A few differ outright, for example a typo inside a string literal (`'15 OR MORE AS BIN'`). Each comes with its counterexample: `--show-disagreements` prints them.

## Running it

```bash
pip install --user -e ".[dev]"
python tools/singh_bedathur_bench.py                          # all pairs
python tools/singh_bedathur_bench.py --split dev --sample 200 # quick development run
python tools/singh_bedathur_bench.py --show-unknown --show-disagreements --json verdicts.json
```

`tests/test_singh_bedathur_benchmark.py` runs a fixed sample of 120 pairs (floor 116 decided, 0 wrong) and, marked `slow`, all 2,800 (floor 2,747). Both skip when the data cannot be downloaded. A pair that ever comes out wrong is a soundness bug: fix the rule or prover and add the shape to `tests/test_canonical_rules.py` or the prover's tests as a regression case.


Bounded verification (3 rows per table, [bounded-verification.md](bounded-verification.md#results)) on the 1,006 pairs the prover does not already refute: 824 bounded-equivalent, 71 refuted with a replayed database, 0 wrong. Not a proof.
