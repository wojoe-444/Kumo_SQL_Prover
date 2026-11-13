# Equivalent under conditions

The [fourth verdict](../conditional-equivalence.md) is `proven_conditionally`: the pair is equal on every database where a listed, minimal set of NOT NULL, unique-key and foreign-key facts holds. This page scores it on two corpora whose "equivalent" labels assume keys and NOT NULL rules their files drop or keep only in part, and on three hand-checked suites. The verdict is off by default, so no existing score moves (the Singh and VeriEQL pages are unchanged).

## What is counted

`tools/conditional_bench.py` runs `prove_equivalent_algebraic(..., conditional=True)` on each pair and reports four counts, kept apart:

| Count | Meaning |
| --- | --- |
| Proved outright | The prover needs no condition. Scored in the [Singh](singh-bedathur.md) and [VeriEQL](verieql.md) pages; left out of the scoreboard row here. |
| Proved under named conditions | `proven_conditionally` with a minimal set. |
| Refuted with every candidate holding | A database that meets **every** candidate condition still separates the queries, so no set from the catalog can make the pair equal. |
| Unknown | Anything else. |

Nothing reads the published labels while deciding; they are compared afterwards. A label is not ground truth for a conditional pair: the Singh files list no keys and no NOT NULL facts, so a pair labelled equivalent can differ on a NULL or a duplicate row, and a pair labelled different can be equal once NULLs are excluded.

Every conditional proof is checked three ways, and a failure of the first or the second makes the pair **wrong** (the count must stay 0):

1. The pair is re-run on 900 random DuckDB databases repaired to meet the conditions (NULLs filled in, duplicate keys dropped, orphaned foreign keys pointed at a parent or dropped). Both queries must return the same bag on all of them.
2. For VeriEQL, the executed counterexample search of `kumo_sql_prover.counterexample` runs with the conditions added to the spec; it must find nothing.
3. The prover loses the proof when any one condition is dropped (reported as the minimal count). The share of pairs for which a random database that breaks the conditions separates the queries is reported too ("separated without the conditions"); agreement on sampled databases is evidence, never the ground truth.

## Singh and Bedathur LeetCode pairs

All 2,800 pairs, MySQL dialect, table and column names only (the [Singh page](singh-bedathur.md) has the data and its licence note). Measured 2026-11-13 on commit b389fe4 (the hardest-cases branch merged with main a23e844), about 85 minutes on 4 cores. Main a23e844 on the same corpus, measured by the validation thread: 878 outright, 793 conditional (2026-11-12 on 951be35 the same; 2026-11-11 on c52406f 590).

| Count | Pairs |
| --- | ---: |
| Proved outright | 889 |
| **Proved under named conditions** | **973** |
| Refuted with every candidate holding | 357 |
| Unknown | 581 |
| Wrong | 0 |
| Harness crashes | 0 |

- The 973 conditional proofs use 2,227 conditions: 1,705 NOT NULL, 471 unique keys, 51 single-column foreign keys. One condition alone suffices for 310 pairs, two for 240, three for 288, and four to seven for 135.
- All 973 were re-checked on 900 databases each (875,700 in total) with 0 differences, and each is minimal for the prover. For 935 of them a random database that breaks the conditions separates the queries; for the other 38 no separating database turned up in 300 tries, which does not mean there is none.
- Against the files' labels: 282 conditional pairs are labelled equivalent (the published answer assumed a key or NOT NULL rule the file does not state), 691 are labelled different (the labelled difference is a NULL or a duplicate; the pair is equal once it is excluded).
- Held-out fifth (hash of the pair divisible by 5, 580 pairs): **196 conditional, 0 wrong** (164 on main a23e844); the development split has 777 of 2,220 conditional.
- **The count rose from 793 to 973 on 2026-11-13.** The hardest-cases branch of the prover (a filter pushed into a grouped source, regroupings under a tail, joins to a table's own groups, duplicate-finding self joins, a `HAVING` that counts rows meeting a condition, a key looked up through its own table, `BETWEEN` bounds with interval dates, `ENUM` columns as strings) moved 180 pairs from unknown to conditional (32 of them held out) and 11 from unknown to proved outright (2 held out), compared pair by pair with main a23e844; no pair that main proves, outright or under conditions, is lost. The rules were written on the development split of this eval, of the LOGOS eval and of the Singh eval; the held-out fifth was run for scoring only and never read. One pair that was conditional on main ff364a7 is not conditional any more (a2826e16ae71, customers who bought A and B but not C: it needed different strings compared with an untyped column to differ, which MySQL reads as the same number for an integer column; it is not conditional on a23e844 either).
- **The count rose from 590 to 793 on 2026-11-12 (main).** The rule batch of 2026-11-11/12 (window filters and pruning, nested membership, keyed join groups, scalar and one-row joins; see the [QUITE page](quite.md)) and the VeriEQL batch of main were written for other pairs, not for this eval. Together they moved 203 pairs from unknown to conditional and 8 more to proved outright (870 to 878); an earlier measurement of the QUITE batch alone on b01aa90 gave 783 conditional and 870 outright. This eval was run as a soundness guard: it showed that a rule reading declared unique keys must require NOT NULL on every key column, which was fixed before merging. The held-out fifth was never read; it went from 126 to 164 conditional.
- **The count fell on 2026-11-11 because pairs left for the better.** Main without the rule modules of 2026-11-10 (9231bd9), measured the same way on the same machine, gives 842 proved outright, 597 conditional, 357 refuted and 1,004 unknown (held-out fifth 128 conditional). Those rules, written for the [Singh eval](singh-bedathur.md), prove 9 of the conditional pairs with no condition at all (2 of them held out). They also prove 19 unknown pairs outright (3 held out), make 2 unknown pairs conditional and let 4 conditional proofs drop one condition. No pair moved the other way, and no rule was written for this eval. The first measurement (2026-10-04) had 854 outright, 610 conditional, 357 refuted and 979 unknown; main's distance from it predates these rules and was not traced.

```bash
python tools/conditional_bench.py singh                  # all 2,800 pairs, about 40 minutes on 4 cores
python tools/conditional_bench.py singh --split dev      # development
python tools/conditional_bench.py singh --sample 120     # the test sample
```

## VeriEQL LeetCode

The same problems as VeriEQL ships them, with types, primary keys, NOT NULL columns and foreign keys already declared; a condition here is a fact **beyond** what the spec states. The run takes every eighth of the 23,994 cases (3,000), `--every 8`.

| Count | Cases |
| --- | ---: |
| Proved outright | 866 |
| **Proved under named conditions** | **297** |
| Refuted with every candidate holding | 0 |
| Unknown | 1,832 |
| Conditional but not re-checkable | 5 |
| Wrong | 0 |
| Harness crashes | 0 |

The 297 use 454 conditions (225 NOT NULL, 169 unique keys, 60 foreign keys). All 297 pass the 900-database re-check and the executed counterexample search with the conditions in the spec; 70 are separated by a database that breaks them. Against master on the same sample this is 23 fewer conditional proofs (320 there): the constant-result check could not run on VeriEQL type names before, and now rejects 24 proofs that hold only because the returned conditions make both queries return one result on every test database (a unique key on a self join that asks for two different rows); one proof was gained. Five more conditional proofs rest on a composite foreign key that the executed check cannot impose, so they are not re-checked and not scored. Fewer conditionals per case than Singh is expected: the declared keys already carry what the prover can use. No held-out split is reserved for this sample. The harness was developed against Singh dev pairs and an early 80-case VeriEQL sample (tuned on test); the full sample was run afterwards.

```bash
python tools/conditional_bench.py verieql --every 8 --workers 4   # about 15 minutes on 4 cores
```

## Constraint-touched candidate pairs

A list of 655 benchmark pairs whose queries read a declared key, NOT NULL column or foreign key in a WHERE, join, GROUP BY, DISTINCT, IN or COUNT (VeriEQL Calcite and Literature, SQLSolver Calcite, WeTune Calcite, Cosette examples) was mined syntactically; no prover had run on it. `tools/conditional_candidates.py <candidates.json>` runs the verdict on the 508 pairs of VeriEQL Calcite (321), VeriEQL Literature (5 of 6) and SQLSolver Calcite (182), with the schema's tables, columns and types kept and **every declared constraint removed**, so a proof has to name the facts it needs. It compares those with what the schema declares (a declared key covers every superset of its columns) and re-checks each conditional proof on databases that meet only the returned conditions. WeTune Calcite (142 pairs, a catalog format with no loader here), the four Cosette pairs and one Literature pair whose text does not match the pinned file are not run. None of the three corpora has a held-out split of its own, and mined Calcite pairs and adapted Cosette pairs, which do, are not in the list; the harness was not tuned on these pairs.

| Corpus | Pairs | Proved with no condition | Proved under conditions | Unknown | Of the unknown, proved when the schema's constraints are given |
| --- | ---: | ---: | ---: | ---: | ---: |
| VeriEQL Calcite | 321 | 233 | 30 | 58 | 2 |
| VeriEQL Literature | 5 | 0 | 3 | 2 | 1 |
| SQLSolver Calcite | 182 | 153 | 16 | 13 | 2 |

0 wrong and 0 crashes. All 49 conditional proofs were re-checked on databases that meet only their conditions, and each is minimal for the prover. What the run shows:

- **Touching a constraint is not depending on it.** 386 of the 508 pairs are proved with no constraint at all.
- **49 pairs need conditions** (85 conditions: 46 unique keys, 39 NOT NULL, no foreign key; one condition for 25 pairs, two for 14, three for 8, four for 2). 35 use only facts the schema declares (a primary key gives unique and NOT NULL, and a declared key makes every superset of its columns unique). 14 use a fact the schema does not state. Three read by hand: `WHERE DEPTNO IN (SELECT DEPTNO FROM DEPT)` against an inner join needs `DEPT.DEPTNO` unique, and VeriEQL's schema gives DEPT no key; a self join of `R2` on its key with `Y.B = Z.B` dropped needs `R2.b` NOT NULL (the two joined rows are one row, and `b = b` still rejects it when `b` is NULL); `(empno, deptno) IN (...)` against a LEFT JOIN needs NOT NULL on both columns and a unique pair, which the declared key on `empno` provides.
- **11 of the 49 are not shown to be semantically needed.** A database without the conditions separated the queries for 38; for these 11 none turned up in 300 tries, so the prover needs the conditions and the queries may not. Examples: the Calcite `INTERSECT` of `DEPTNO = 10`, `20` and `30` (a row cannot have two `DEPTNO` values, so the left side is empty on every database), and one pair (`MGR IS NULL` under a returned `NOT NULL mgr`) whose conditions empty both sides; its query already compares `HIREDATE = CURRENT_TIMESTAMP`, which the constant check treats as a query written to be empty, so the verdict stands.
- **5 pairs are proved with the schema's constraints but not conditionally** (2 VeriEQL Calcite, 1 Literature, 2 SQLSolver Calcite): the catalog, at most 40 single-column candidates, leaves out what they need.

```bash
python tools/conditional_candidates.py <candidates.json> --json results.json   # about 2 minutes on 4 cores
```

## Hand-checked suites

| Suite | Cases | What it checks |
| --- | ---: | --- |
| `tests/test_conditional_suite.py` | 17 cases, 5 controls, 4 decoys | Each case has an answer worked out by hand with and without its conditions: DISTINCT on a unique id, LEFT JOIN removal on a unique right key, INNER JOIN removal with a non-null foreign key to a unique parent, composite keys, NULLs, bag semantics. A database that breaks a condition separates the queries on DuckDB; random databases that meet the conditions never do; dropping any condition loses the proof. Controls stay unproven under every candidate, a set no database satisfies is refused, and a set that empties the query is passed over. |
| `tests/test_conditional_s001.py` | 29 pairs from an outside source | Every counterexample, deletion witness and sufficiency example is replayed on DuckDB; the verdict of both provers is pinned. 13 pairs get a conditional verdict from the algebraic prover, 11 from the SMT prover (the SMT prover has no use for foreign keys and no outer-join aggregates). Pairs whose expected conditions are outside the catalog (filtered keys, CHECK, FD, EXISTS) must stay unproven or refuted. |
| `tests/test_conditional_vendor.py` | 2 pairs from Databricks' documentation on `RELY` constraints | DISTINCT dropped on a key (needs unique **and** NOT NULL: repeated NULLs collapse under DISTINCT) and an unused LEFT JOIN dropped on a unique key (no NOT NULL needed). |

The outside cases are data in `tests/fixtures/conditional/` with a source note per case ([README](../../tests/fixtures/conditional/README.md)). Where a pinned set is stronger than the minimal set the file expects (a plain unique key where the file asks for a key that holds only on a filtered subset), the prover is sound but incomplete, never wrong.

## Limits

- Minimal means minimal for the prover: every single condition, the last one included, was tried against the final set and dropping it loses the proof. That is not a proof the condition is necessary (a "not proven" after a deletion is the prover giving up), and the search returns one sufficient set, not every alternative. Two different minimal sets can exist; the search keeps NOT NULL conditions longest and leaves unique keys out first.
- The condition search has a wall clock (30 seconds by default, `conditional_seconds`); when a loaded machine cuts it short the verdict still holds but the set is not minimal, and its reason says the search ran out of time. The scoring harness gives the search 600 seconds, so a busy machine cannot turn a minimal proof into a failing count.
- The catalog is NOT NULL, unique keys and single-column foreign keys read off the queries, at most 40 candidates. Non-empty tables, CHECK and value ranges, filtered keys and functional dependencies are outside it; pairs that need them stay unproven or refuted.
- A condition is data the owner must make true: the verdict carries a SQL check for each, and BigQuery does not enforce keys.
- Results: [`conditional-equivalence-singh`](../../benchmarks/results/conditional-equivalence-singh.json), [`conditional-equivalence-verieql`](../../benchmarks/results/conditional-equivalence-verieql.json).
