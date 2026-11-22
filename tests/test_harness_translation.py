"""Benchmark harness translation: Calcite-printed SQL run on DuckDB, and the SQL repairs in tools/bench_sql_repairs.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from bench_sql_repairs import (  # noqa: E402
    barewords_as_strings,
    expand_row_predicates,
    fold_table_names,
    pipes_as_concat,
    rebind_correlation_variables,
    uniquify_star_columns,
)

from kumo_sql_prover import counterexample as cx  # noqa: E402

EMP = ["EMPNO", "DEPTNO", "SAL"]


def emp_spec(*, foreign_key: bool = False) -> cx.Spec:
    spec = cx.Spec({
        "EMP": cx.Table("EMP", [cx.Column("EMPNO", "INT", not_null=True), cx.Column("DEPTNO", "INT"), cx.Column("SAL", "INT")], primary_key=("EMPNO",)),
        "DEPT": cx.Table("DEPT", [cx.Column("DEPTNO", "INT", not_null=True), cx.Column("NAME", "VARCHAR")], primary_key=("DEPTNO",)),
    })
    if foreign_key:
        spec.foreign_keys.append(("EMP", "DEPTNO", "DEPT", "DEPTNO"))
    return spec


def runs(sql: str) -> list:
    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE EMP (EMPNO BIGINT, DEPTNO BIGINT, SAL BIGINT)")
    db.execute("INSERT INTO EMP VALUES (1, 10, 5), (2, 10, NULL), (3, 20, 7)")
    return db.execute(cx.to_duckdb(sql)).fetchall()


def test_dollar_names_are_quoted():
    sql = "SELECT t.$f1, t.EXPR$0 FROM (SELECT DEPTNO AS EXPR$0, COUNT(*) AS $f1 FROM EMP GROUP BY DEPTNO) AS t"
    assert sorted(runs(sql)) == [(1, 20), (2, 10)]


def test_multi_argument_count_counts_rows_with_every_argument():
    assert sorted(runs("SELECT DEPTNO, COUNT(EMPNO, SAL) FROM EMP GROUP BY DEPTNO")) == [(10, 1), (20, 1)]


def test_filter_columns_and_grouping_set_keys_stay_unwrapped():
    sql = (
        "SELECT DEPTNO, COUNT(SAL) FILTER (WHERE $g) FROM (SELECT DEPTNO, SAL, GROUPING(DEPTNO, SAL) = 0 AS $g "
        "FROM EMP GROUP BY GROUPING SETS ((DEPTNO, SAL), DEPTNO)) AS t GROUP BY DEPTNO"
    )
    translated = cx.to_duckdb(sql)
    assert "ANY_VALUE" not in translated
    assert sorted(runs(sql)) == [(10, 1), (20, 1)]


def test_calcite_aggregates_and_order_by_null():
    assert runs("SELECT FIRST_VALUE(DEPTNO) FROM EMP WHERE EMPNO = 3") == [(20,)]
    assert runs("SELECT SINGLE_VALUE(SAL) FROM EMP WHERE EMPNO = 3") == [(7,)]
    assert runs("SELECT SINGLE_VALUE(SAL) FROM EMP WHERE EMPNO = 9") == [(None,)]
    with pytest.raises(duckdb.Error):
        runs("SELECT SINGLE_VALUE(SAL) FROM EMP")
    assert sorted(runs("SELECT EMPNO FROM EMP ORDER BY NULL")) == [(1,), (2,), (3,)]


def test_empty_select_list_keeps_the_row_count():
    assert runs("SELECT FROM EMP WHERE DEPTNO = 10") == [(1,), (1,)]
    assert runs("SELECT * FROM (SELECT DEPTNO FROM EMP WHERE EMPNO = 1) AS a, LATERAL (SELECT FROM EMP) AS b") == [(10,)] * 3


def test_child_tables_get_rows_although_the_parent_is_not_read():
    generator = cx._Generator(emp_spec(foreign_key=True), cx.Constants(), __import__("random").Random(0))
    sizes = [len(generator.database({"EMP"}, 3)["EMP"]) for _ in range(30)]
    assert max(sizes) > 0


def test_search_skips_databases_that_raise():
    spec = emp_spec()
    left = "SELECT SINGLE_VALUE(SAL) FROM EMP"
    right = "SELECT MAX(SAL) + 1 FROM EMP"
    for seed in range(3):
        found = cx.find_counterexample(spec, left, right, seed=seed)
        assert found and len(found.tables["EMP"]) == 1, "only one-row tables separate them; larger ones raise and are skipped"


def test_uninterpreted_predicates_run_as_macros():
    tables = {"r": ["X", "Y"]}
    left, arities = expand_row_predicates("SELECT * FROM R AS a WHERE B1(a)", tables)
    right, _ = expand_row_predicates("SELECT * FROM (SELECT * FROM R AS b WHERE B1(b)) AS c", tables)
    assert left == "SELECT * FROM R AS a WHERE B1(a.X, a.Y)" and arities == {"B1": 2}
    spec = cx.Spec({"R": cx.Table("R", [cx.Column("X", "INT"), cx.Column("Y", "INT")])})
    searcher = cx.Searcher(spec, left, right, predicates=arities)
    assert searcher.runs() and searcher.search(60) is None
    other = cx.Searcher(spec, left, "SELECT * FROM R", predicates=arities)
    assert other.search(60) is not None


def test_correlation_variables_rebind_to_the_lateral_source():
    sql = "SELECT $cor0.EMPNO FROM EMP AS $cor0, LATERAL (SELECT MAX(SAL) AS EXPR$0 FROM EMP) AS t6 WHERE $cor0.SAL = $cor0.EXPR$0"
    assert rebind_correlation_variables(sql, {"emp": EMP}).endswith("WHERE $cor0.SAL = t6.EXPR$0")
    ambiguous = "SELECT $cor0.EMPNO FROM EMP AS $cor0, LATERAL (SELECT 1 AS X) AS a, LATERAL (SELECT 2 AS X) AS b WHERE $cor0.X = 1"
    assert rebind_correlation_variables(ambiguous, {"emp": EMP}) == ambiguous


def test_repeated_star_columns_take_duckdb_names():
    sql = "SELECT t.SAL FROM (SELECT * FROM EMP AS a, EMP AS b) AS t"
    repaired = uniquify_star_columns(sql, {"emp": EMP})
    assert "b.SAL AS SAL_1" in repaired and "a.SAL AS SAL" in repaired
    assert uniquify_star_columns("SELECT t.SAL FROM (SELECT * FROM EMP AS a) AS t", {"emp": EMP}).endswith("FROM EMP AS a) AS t")
    full = "SELECT t.SAL FROM (SELECT * FROM EMP AS a FULL JOIN EMP AS b ON a.EMPNO = b.EMPNO) AS t"
    with pytest.raises(ValueError):
        uniquify_star_columns(full, {"emp": EMP})


def test_table_names_fold_only_when_a_pair_spells_them_two_ways():
    left, right = fold_table_names("SELECT * FROM a x", "SELECT * FROM A y")
    assert right == "SELECT * FROM a AS y" and left == "SELECT * FROM a x"
    assert fold_table_names("SELECT * FROM EMP", "SELECT * FROM EMP") == ["SELECT * FROM EMP", "SELECT * FROM EMP"]


def test_calcite_pipes_concatenate():
    sql = pipes_as_concat("SELECT DEPTNO || 'x' || SAL AS c FROM EMP WHERE DEPTNO || '' = '10' AND 'a||b' <> ''")
    assert sql == "SELECT CONCAT(CONCAT(DEPTNO, 'x'), SAL) AS c FROM EMP WHERE CONCAT(DEPTNO, '') = '10' AND 'a||b' <> ''"
    assert sorted(runs(sql), key=repr) == [("10x5",), (None,)]
    assert pipes_as_concat("SELECT a || b | c FROM t") == "SELECT a || b | c FROM t", "a query that also uses | is left alone"


SIDES = {"side_a", "side_b", "side_c", "colour"}

# (as VeriEQL's files hold the answer, with the quotes back): made-up tables, the shapes the files have
LOST_QUOTES = [
    (
        "SELECT *, CASE WHEN side_a + side_b > side_c THEN YES ELSE NO END AS fits FROM shapes",
        "SELECT *, CASE WHEN side_a + side_b > side_c THEN 'YES' ELSE 'NO' END AS fits FROM shapes",
    ),
    # nothing but the words changes: DIV, the spacing and the letter case stay as written
    (
        "select  side_a DIV 2 , IF( colour = Red , Warm , COLD ) from shapes  where  side_b<>3",
        "select  side_a DIV 2 , IF( colour = 'Red' , 'Warm' , 'COLD' ) from shapes  where  side_b<>3",
    ),
    (
        "SELECT s.side_a FROM shapes AS s WHERE s.colour IN (RED, 'BLUE') AND EXISTS (SELECT X FROM shapes AS o WHERE o.side_a = s.side_b)",
        "SELECT s.side_a FROM shapes AS s WHERE s.colour IN ('RED', 'BLUE') AND EXISTS (SELECT 'X' FROM shapes AS o WHERE o.side_a = s.side_b)",
    ),
    (
        "SELECT IF(colour LIKE RED, (WARM), COLD) AS feel, CASE colour WHEN BLUE THEN 1 ELSE 0 END AS chilly FROM shapes WHERE colour <> GREY UNION ALL SELECT NEITHER AS feel, 2",
        "SELECT IF(colour LIKE 'RED', ('WARM'), 'COLD') AS feel, CASE colour WHEN 'BLUE' THEN 1 ELSE 0 END AS chilly FROM shapes WHERE colour <> 'GREY' UNION ALL SELECT 'NEITHER' AS feel, 2",
    ),
    (
        "WITH sized (kind, n) AS (SELECT CASE WHEN side_a > 9 THEN BIG ELSE SMALL END, side_a FROM shapes) SELECT kind, COUNT(n) AS total FROM sized GROUP BY kind HAVING total > 1",
        "WITH sized (kind, n) AS (SELECT CASE WHEN side_a > 9 THEN 'BIG' ELSE 'SMALL' END, side_a FROM shapes) SELECT kind, COUNT(n) AS total FROM sized GROUP BY kind HAVING total > 1",
    ),
]

# every name here is a column of the schema, or an alias, a table or a WITH column of the query: nothing is a lost string
KNOWN_NAMES = [
    "SELECT side_a, side_b AS width FROM shapes WHERE colour = 'RED' ORDER BY width",
    "SELECT s.side_a, o.total FROM shapes AS s JOIN (SELECT side_b, COUNT(*) AS total FROM shapes GROUP BY side_b) AS o ON o.side_b = s.side_a",
    "WITH sized (kind, n) AS (SELECT colour, side_a FROM shapes) SELECT kind, SUM(n) FROM sized GROUP BY kind",
    "SELECT side_a, ROW_NUMBER() OVER (PARTITION BY colour ORDER BY side_b) AS place FROM shapes HAVING place = 1",
    "SELECT side_a FROM shapes WHERE side_b > CURRENT_DATE AND side_c IS NOT NULL AND colour LIKE 'R%'",
    "SELECT shapes.side_a, missing.thing FROM shapes",  # a qualified name is never a lost string
]


@pytest.mark.parametrize("held,repaired", LOST_QUOTES)
def test_lost_quotes_are_put_back_and_the_rest_is_kept_as_written(held, repaired):
    assert barewords_as_strings(held, SIDES) == repaired
    assert barewords_as_strings(repaired, SIDES) == repaired
    # the reading is the search's own: it turns both texts into the same DuckDB query
    assert cx.to_duckdb(repaired, "mysql", SIDES) == cx.to_duckdb(held, "mysql", SIDES)
    assert cx.to_duckdb(repaired, "mysql", SIDES) == cx.to_duckdb(repaired, "mysql", None), "no word is left for the search to read"


@pytest.mark.parametrize("sql", KNOWN_NAMES)
def test_a_name_the_schema_or_the_query_knows_is_never_quoted(sql):
    assert barewords_as_strings(sql, SIDES) == sql


# a word that is no known name, standing where a lost string is not the plain reading: the query stays as written
NOT_PLAINLY_STRINGS = [
    "SELECT colour, COUNT(GAME) AS games FROM shapes GROUP BY colour",  # an aggregate's argument: a lost column or alias
    "SELECT side_a FROM shapes ORDER BY WIDTH",
    "SELECT colour, SUM(side_a) FROM shapes GROUP BY colour, KIND",
    "SELECT side_a + EXTRA AS more FROM shapes",
    "SELECT side_a FROM shapes WHERE side_b > LIMIT_B",
    "SELECT ROW_NUMBER() OVER (PARTITION BY KIND ORDER BY side_a) AS place FROM shapes",
    "SELECT ROUND(side_a / (TABLE totals), 2) FROM shapes",  # MySQL's TABLE statement, which the parser reads as a name
    "SELECT CASE WHEN side_a > 1 THEN YES ELSE NO END AS fits, COUNT(GAME) FROM shapes",  # one such word keeps the whole query
]


@pytest.mark.parametrize("sql", NOT_PLAINLY_STRINGS)
def test_a_word_where_a_lost_string_is_not_the_plain_reading_keeps_the_query_as_written(sql):
    assert barewords_as_strings(sql, SIDES) == sql
    assert cx.to_duckdb(sql, "mysql", SIDES) != cx.to_duckdb(sql, "mysql", None), "the search does read a word here: the prover's reading is the narrower one"


def test_a_word_in_backticks_or_one_the_text_does_not_hold_is_left_alone():
    backticks = "SELECT CASE WHEN side_a > 1 THEN `YES` ELSE NO END AS fits FROM shapes"
    assert barewords_as_strings(backticks, SIDES) == backticks, "a name written as a name: the whole query stays as written"
    # the schema's column names decide: with colour unknown it would be a string, and the test of the function
    # is that it asks the search's own reader
    assert barewords_as_strings("SELECT colour FROM shapes", {"side_a"}) == "SELECT 'colour' FROM shapes"
    assert barewords_as_strings("SELECT COLOUR FROM shapes", {"Colour"}) == "SELECT COLOUR FROM shapes", "names compare without case"


@pytest.mark.canary  # one hand-written case through the VeriEQL harness; the family's floors stay in tests/test_verieql_benchmarks.py
def test_the_leetcode_harness_proves_a_pair_whose_strings_lost_their_quotes():
    import verieql_bench

    schema = {"SHAPES": {"SIDE_A": "INT", "SIDE_B": "INT", "SIDE_C": "INT"}}
    constraint = [{"primary": [{"value": "SHAPES__SIDE_A"}, {"value": "SHAPES__SIDE_B"}, {"value": "SHAPES__SIDE_C"}]}]
    quoted = "SELECT SIDE_A, IF(SIDE_A + SIDE_B > SIDE_C, 'YES', 'NO') AS FITS FROM SHAPES"
    lost = "SELECT SIDE_A, CASE WHEN SIDE_A + SIDE_B > SIDE_C THEN YES ELSE NO END AS FITS FROM SHAPES"
    other = "SELECT SIDE_A, CASE WHEN SIDE_A + SIDE_B > SIDE_C THEN YES ELSE NOPE END AS FITS FROM SHAPES"

    def case(right: str, suite: str) -> dict:
        return {"index": 7, "suite": suite, "schema": schema, "constraint": constraint, "pair": [quoted, right]}

    spec = verieql_bench.build_spec(case(lost, "leetcode"))
    left, right, _ = verieql_bench.repaired_pair(case(lost, "leetcode"), spec)
    assert left == quoted
    assert right == "SELECT SIDE_A, CASE WHEN SIDE_A + SIDE_B > SIDE_C THEN 'YES' ELSE 'NO' END AS FITS FROM SHAPES"
    assert verieql_bench.prove(case(lost, "leetcode"), spec, 3000, (left, right)).proven
    assert not verieql_bench.prove(case(lost, "leetcode"), spec, 3000, (quoted, lost)).proven, "as the file holds it, the pair is not read"
    assert verieql_bench.decide(case(lost, "leetcode")).status == verieql_bench.EQUIVALENT
    assert verieql_bench.decide(case(other, "leetcode")).status == verieql_bench.DIFFERENT, "another word is another string"
    # the other suites' queries are printed by Calcite or written in a paper's notation: no quotes were lost there
    assert verieql_bench.repaired_pair(case(lost, "literature"), spec)[1] == lost


EXISTS_SQL = "SELECT t.a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.d <> t.a AND t.b > u.c)"
COUNT_SQL = "SELECT t.a FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.d <> t.a AND t.b > u.c) > 0"


def test_optimizer_bugs_are_not_counterexamples():
    """DuckDB 1.5's optimizer drops rows of this correlated EXISTS when the tables hold NULLs."""

    from kumo_sql_prover.duckdb_load import run_unoptimized

    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE t (a BIGINT, b BIGINT)")
    db.execute("CREATE TABLE u (c BIGINT, d BIGINT)")
    db.execute("INSERT INTO t VALUES (1, 2), (NULL, 3), (2, 1), (3, 0)")
    db.execute("INSERT INTO u VALUES (0, 0), (0, NULL)")
    exists, count = run_unoptimized(db, EXISTS_SQL, COUNT_SQL)
    assert sorted(exists) == sorted(count) == [(1,), (2,)]
    assert sorted(db.execute(COUNT_SQL).fetchall()) == [(1,), (2,)]

    spec = cx.Spec({
        "t": cx.Table("t", [cx.Column("a", "INT"), cx.Column("b", "INT")]),
        "u": cx.Table("u", [cx.Column("c", "INT"), cx.Column("d", "INT")]),
    })
    for seed in range(2):  # without the recheck, both seeds report a false counterexample
        assert cx.find_counterexample(spec, EXISTS_SQL, COUNT_SQL, seed=seed) is None


@pytest.mark.canary  # builds one hand-written case with the harness's schema reader; the family's floors stay in tests/test_verieql_benchmarks.py
def test_each_primary_entry_of_a_verieql_schema_is_its_own_key():
    """A table that declares two primary keys keeps both: the earlier one as a unique key over NOT NULL columns."""

    import verieql_bench

    case = {
        "schema": {"SHOP": {"SHOP_ID": "INT", "TITLE": "VARCHAR", "CITY": "VARCHAR"}, "SALE": {"SALE_ID": "INT", "SHOP_ID": "INT"}},
        "constraint": [
            {"primary": [{"value": "SHOP__SHOP_ID"}]},
            {"primary": [{"value": "SHOP__TITLE"}]},
            {"primary": [{"value": "SALE__SALE_ID"}]},
            {"foreign": [{"value": "SALE__SHOP_ID"}, {"value": "SHOP__SHOP_ID"}]},
        ],
    }
    spec = verieql_bench.build_spec(case)
    shop, sale = spec.tables["SHOP"], spec.tables["SALE"]
    assert sorted([shop.primary_key, *shop.unique]) == [("SHOP_ID",), ("TITLE",)]
    assert [c.name for c in shop.columns if c.not_null] == ["SHOP_ID", "TITLE"]
    assert (sale.primary_key, sale.unique) == (("SALE_ID",), [])
    rng = __import__("random").Random(0)
    generator = cx._Generator(spec, cx.Constants(), rng)
    seen_rows = 0
    for _ in range(60):
        rows = generator.database({"SHOP", "SALE"}, 5)["SHOP"]
        seen_rows += len(rows)
        for position in (0, 1):  # both declared keys hold in every generated table
            values = [row[position] for row in rows]
            assert None not in values and len(set(values)) == len(values)
    assert seen_rows > 60


def test_every_small_database_keeps_consecutive_ids():
    # the "all small databases" step used to try seat ids 0 and 2 for a column declared consecutive
    seats = cx.Table("SEATS", [cx.Column("ID", "INT", not_null=True), cx.Column("FREE", "INT")], primary_key=("ID",), sequential=("ID",))
    spec = cx.Spec({"SEATS": seats})
    # equal when the ids are 1..n (every seat but the last has a successor), different on ids 1 and 3
    successor = "SELECT a.ID FROM SEATS AS a JOIN SEATS AS b ON b.ID = a.ID + 1"
    not_last = "SELECT a.ID FROM SEATS AS a WHERE a.ID < (SELECT MAX(ID) FROM SEATS)"
    outcome, found = cx.Searcher(spec, successor, not_last).exhaustive(2, 3000)
    assert (outcome, found) == ("verified", None)
    # a real difference is still found, on a database with ids 1..n
    outcome, found = cx.Searcher(spec, "SELECT ID FROM SEATS", "SELECT ID FROM SEATS WHERE ID > 1").exhaustive(2, 3000)
    assert outcome == "found"


TIMED_CASE = {
    "index": 7,
    "schema": {"SHOP": {"SHOP_ID": "INT", "CITY": "VARCHAR"}},
    "constraint": [{"primary": [{"value": "SHOP__SHOP_ID"}]}],
    "pair": ["SELECT CITY FROM SHOP WHERE SHOP_ID > 3", "SELECT CITY FROM SHOP WHERE NOT SHOP_ID <= 3"],
}


def test_a_search_the_time_budget_interrupts_is_a_timeout_not_a_crash(monkeypatch):
    """DuckDB reports the alarm that lands inside a query as its own RuntimeError caused by the harness's timeout. The
    case ran out of time: before any search finished it is unknown, after one it "agrees" (no difference was seen)."""

    import verieql_bench

    def interrupted_after(calls_allowed):
        calls = []

        def search(self, *args, **kwargs):
            calls.append(1)
            if len(calls) <= calls_allowed:
                return None
            try:
                raise verieql_bench._Timeout
            except verieql_bench._Timeout as alarm:
                raise RuntimeError("Query interrupted") from alarm

        return search

    monkeypatch.setattr(cx.Searcher, "search", interrupted_after(0))
    verdict = verieql_bench.decide(dict(TIMED_CASE))
    assert (verdict.status, verdict.detail) == (verieql_bench.UNKNOWN, "time budget")
    monkeypatch.setattr(cx.Searcher, "search", interrupted_after(1))  # the recheck of the proof is interrupted
    verdict = verieql_bench.decide(dict(TIMED_CASE))
    assert (verdict.status, verdict.detail) == (verieql_bench.AGREES, "time budget")


def test_another_runtime_error_in_a_search_is_still_a_crash(monkeypatch):
    import verieql_bench

    def broken(self, *args, **kwargs):
        raise RuntimeError("something else")

    monkeypatch.setattr(cx.Searcher, "search", broken)
    with pytest.raises(RuntimeError, match="something else"):
        verieql_bench.decide(dict(TIMED_CASE))
    verdict = verieql_bench._work((dict(TIMED_CASE), {}))
    assert (verdict.status, verdict.detail) == (verieql_bench.UNKNOWN, "crash: RuntimeError")
