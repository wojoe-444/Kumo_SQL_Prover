# Repository instructions for Claude

This repository is the equivalence prover of [KumoSQL](https://github.com/wojoe-444/KumoSQL), split out as step 8 of its [roadmap](https://github.com/wojoe-444/KumoSQL/blob/master/docs/roadmap.md). KumoSQL depends on a pinned release of this package (`kumo-sql-prover @ git+https://github.com/wojoe-444/Kumo_SQL_Prover@vX.Y.Z`), so a change here reaches KumoSQL only as a version bump in its `pyproject.toml`.

Principles that carry over from KumoSQL:

- **Unknown is better than wrong.** Every scoreboard row says "0 wrong" only when it is true. A false proof, an incorrect counterexample or a rewrite that changes results is a bug.
- **Nothing in this package imports KumoSQL.** The import graph runs one way. `tests/test_package_boundary.py` fails the gate on a `kumosql` import. Scripting, the rule registry, Dataform project loading and the UI live in KumoSQL.
- **`sqlglot==30.21.0` is the only supported version.** Routine CI uses the matching compiled `sqlglotc==30.21.0` build. Pure Python is for focused diagnostics. A version change needs explicit validation and an intentional update to the pin, CI and docs.
- **Benchmarks report, they do not gate.** The pull-request gate runs `python -m pytest -m "not eval" --quick`. The scheduled workflow runs the eval floors (`-m eval`), regenerates the scoreboard and opens an issue when a floor drops.

Working rules:

- Run tests with `python -m pytest -n auto -m "not eval" --quick` for the gate (it skips the slow tier listed in `tests/order.json`; a test that grows past about two seconds belongs there), `-m "not eval"` for everything but the floors, `-m eval` for the benchmark floors. Tests must be safe to run in parallel: no shared files or module-level state between tests.
- The README scoreboard is generated from `benchmarks/results/*.json`, one file per eval. A change that moves a score, or adds an eval, updates that results file (score, size, date, caveats; keep "0 wrong" honest) and runs `python tools/scoreboard.py` in the same pull request. Never edit the table by hand. On a merge conflict keep either side and rerun the script. Mention the same numbers in the eval's page under `docs/evals/`.
- Documentation: every Markdown page in `docs/` has a plain-language companion at the same path in `docs_simple/`; `docs/README.md` and `docs/evals/README.md` (and the simple copies) index every page; `tests/test_docs.py` checks both. Update both sets when behavior changes.
- A shared prover change (parser, normalization, execution) needs a base/candidate comparison of the affected evals (`python tools/eval_diff.py`) and the soundness guards before merging.
- Releases are tags `vX.Y.Z` on `main`. Say in the release notes which evals moved, if any, and give the commit tested.
- This is a public repository: no secrets, real project identifiers or machine-specific paths in files, issues or pull requests.
