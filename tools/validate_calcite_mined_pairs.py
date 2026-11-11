"""Execute every mined Calcite pair in DuckDB over small random databases.

For each case in tests/fixtures/calcite_mined/pairs.jsonl this creates the
schema (schemas.json), fills it with a few small random databases that respect
column types, NOT NULL and key constraints, runs both SQL strings (transpiled
MySQL -> DuckDB by sqlglot) and compares the result multisets.

    python tools/validate_calcite_mined_pairs.py [--dbs 8] [--seed 1] [--write] [-v]

With ``--write`` it records the outcome in the fixture: every pair gets
``differs_in_duckdb`` (true when some database separates the two sides), the
first counterexample of each differing pair goes to
``duckdb_counterexamples.jsonl``, and ``summary.json`` gets a
``duckdb_validation`` block. A difference is either a translation bug (which
must be fixed or skipped in tools/calcite_plan_to_sql.py) or a genuine Calcite
non-equivalence (Calcite rule tests are not proofs; some rules are only
equivalent under Calcite's typing, or the test documents a known bug).

Two DuckDB/MySQL gaps are bridged when transpiling: MySQL ``DIV`` truncates
toward zero (DuckDB ``//`` floors), and sqlglot reads MySQL ``TIMESTAMP`` (columns
and literals) as TIMESTAMPTZ, while Calcite's TIMESTAMP has no time zone.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import duckdb
import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kumo_sql_prover.duckdb_load import small_database  # noqa: E402

logging.getLogger("sqlglot").setLevel(logging.ERROR)
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "calcite_mined"

STRINGS = ["A", "B", "", "abc", "CLERK", "dept1", "foo", "Charlie", "1", "2", "x", "MANAGER", "a", "bb", "cc"]
INTS = [0, 1, 2, 3, 7, 10, 11, 20, 30, 40, 50, 100, -1, 120, 5]  # all fit TINYINT
WIDE_INTS = [1000, 1500, 2000, 3000, 7369]


def gen_value(col: dict, rng: random.Random):
    t = col["type"]
    if col["nullable"] and rng.random() < 0.2:
        return None
    if t in ("INTEGER", "BIGINT", "SMALLINT", "TINYINT"):
        return rng.choice(INTS if t == "TINYINT" else INTS + WIDE_INTS)
    if t in ("VARCHAR", "CHAR"):
        n = int(col["ddl_type"].split("(")[1].rstrip(")"))
        return rng.choice([s for s in STRINGS if len(s) <= n])
    if t == "BOOLEAN":
        return rng.random() < 0.5
    if t == "DATE":
        return f"2014-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}"
    if t == "TIMESTAMP":
        return f"2014-0{rng.randint(1, 9)}-1{rng.randint(0, 9)} {rng.choice(['00', '13'])}:00:00"
    if t == "DECIMAL":
        return rng.choice([0, 1, 1.5, 10, 20.25, -1, 3000, 1250.5])
    if t in ("DOUBLE", "FLOAT", "REAL"):
        return rng.choice([0, 1, 2, 10, 20, 5.5])
    raise ValueError(t)


def make_db(schema: dict, rng: random.Random) -> tuple[duckdb.DuckDBPyConnection, dict]:
    con = small_database()
    for stmt in sqlglot.parse(schema["ddl"], read="mysql") if schema["ddl"] else []:
        con.execute(stmt.transform(_fix).sql(dialect="duckdb"))
    data = {}
    for t in schema["tables"]:
        cols = t["columns"]
        ph = ", ".join("?" for _ in cols)
        rows = []
        for _ in range(rng.randint(0, 6)):
            row = [gen_value(c, rng) for c in cols]
            try:  # key violations are simply skipped
                con.execute(f'INSERT INTO "{t["table"]}" VALUES ({ph})', row)
                rows.append(row)
            except duckdb.ConstraintException:
                pass
        data[t["table"]] = rows
    return con, data


def _fix(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.IntDiv):
        num = exp.cast(node.this, "DOUBLE")
        return exp.cast(exp.func("TRUNC", exp.Div(this=num, expression=node.expression)), "BIGINT")
    if isinstance(node, exp.DataType) and node.this == exp.DataType.Type.TIMESTAMPTZ:
        return exp.DataType.build("TIMESTAMP")
    return node


def to_duckdb(sql: str) -> str:
    tree = sqlglot.parse_one(sql, read="mysql")
    return tree.transform(_fix).sql(dialect="duckdb")


def norm(v):
    if isinstance(v, float):
        return round(v, 6)
    if hasattr(v, "is_finite"):  # Decimal
        return round(float(v), 6)
    return v


def run(con, sql: str):
    rows = [tuple(norm(x) for x in r) for r in con.execute(to_duckdb(sql)).fetchall()]
    return Counter(rows), rows


def jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--write", action="store_true", help="record results in the fixture")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--dir", type=Path, default=FIXTURES, help="fixture directory")
    args = ap.parse_args()
    fx = args.dir
    schemas = json.loads((fx / "schemas.json").read_text())
    cases = [json.loads(l) for l in (fx / "pairs.jsonl").read_text().splitlines() if l.strip()]
    errors, unsupported, differ, agree, nonempty, rejected, scalar_rejected = [], [], [], 0, 0, 0, 0
    examples = []
    status = {}
    for case in cases:
        rng = random.Random(f"{args.seed}:{case['name']}")
        several_rows = 0
        st = "agree"
        ran = attempts = 0
        # a pair with a scalar subquery draws further databases (up to 25 times as many) until enough of them run
        while attempts < args.dbs or (several_rows and ran < args.dbs and attempts < 25 * args.dbs):
            attempts += 1
            try:
                con, data = make_db(schemas[case["schema_id"]], rng)
                (ca, ra), (cb, rb) = run(con, case["sql_a"]), run(con, case["sql_b"])
            except duckdb.ConversionException:
                # a CAST in the plan overflows on this random database: an error in Calcite too
                rejected += 1
                continue
            except duckdb.InvalidInputException as e:
                if "More than one row returned by a subquery" not in str(e):
                    errors.append((case["name"], f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"))
                    st = "error"
                    break
                # a scalar subquery (Calcite's $SCALAR_QUERY or keyless SINGLE_VALUE) met several rows: an error
                # in Calcite too. The pair is compared on the databases where both queries run.
                several_rows += 1
                continue
            except duckdb.NotImplementedException as e:
                unsupported.append((case["name"], str(e).splitlines()[0][:160]))
                st = "duckdb_unsupported"
                break
            except Exception as e:  # noqa: BLE001
                errors.append((case["name"], f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"))
                st = "error"
                break
            ran += 1
            nonempty += bool(ra)
            if ca != cb:
                st = "differs"
                examples.append({
                    "name": case["name"],
                    "database": {t: [[jsonable(v) for v in r] for r in rows] for t, rows in data.items()},
                    "result_a": sorted(([jsonable(v) for v in r] for r in ra), key=repr),
                    "result_b": sorted(([jsonable(v) for v in r] for r in rb), key=repr),
                })
                break
        if st == "agree" and not ran and several_rows:
            errors.append((case["name"], "no database on which both queries run"))
            st = "error"
        scalar_rejected += several_rows
        status[case["name"]] = st
        if st == "differs":
            differ.append(case["name"])
        elif st == "agree":
            agree += 1
    print(f"cases {len(cases)}  agree {agree}  differ {len(differ)}  duckdb-unsupported {len(unsupported)}"
          f"  errors {len(errors)}  (non-empty sql_a results: {nonempty}; databases rejected by a CAST overflow: {rejected}, "
          f"by a scalar subquery with several rows: {scalar_rejected})")
    for n, e in unsupported:
        print("DUCKDB-UNSUPPORTED", n, e)
    for n, e in errors:
        print("ERROR", n, e)
    if args.verbose:
        for n in differ:
            print("DIFFER", n)
    if args.write:
        with (fx / "pairs.jsonl").open("w") as fh:
            for c in cases:
                c["differs_in_duckdb"] = status[c["name"]] == "differs"
                c["duckdb_status"] = status[c["name"]]
                fh.write(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n")
        with (fx / "duckdb_counterexamples.jsonl").open("w") as fh:
            for e in examples:
                fh.write(json.dumps(e, ensure_ascii=False, separators=(",", ":")) + "\n")
        summ = json.loads((fx / "summary.json").read_text())
        summ["duckdb_validation"] = {
            "databases_per_pair": args.dbs, "seed": args.seed, "agree": agree, "differs": len(differ),
            "duckdb_unsupported": len(unsupported), "errors": len(errors), "differing": differ,
        }
        (fx / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
