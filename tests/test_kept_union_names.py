"""The output name a select keeps when the ``UNION ALL`` it reads is renamed (``kept_union_names``).

The prover's own text of each query is run on DuckDB over made-up rows beside the query as written. Without the
repair that text named a column that was gone.
"""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from kumo_sql_prover import kept_union_names, shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic

TABLES = {
    "p": ("pid INT, x INT", [(1, 10), (2, 20), (3, 30), (None, 40)]),
    "t": ("a INT, b INT", [(1, 2), (1, 3), (None, 3), (4, 4)]),
    "t2": ("a INT, b INT", [(3, 1), (2, 2)]),
}
SCHEMA = {name: [part.split()[0] for part in columns.split(", ")] for name, (columns, _) in TABLES.items()}
TYPES = {name: dict(part.split() for part in columns.split(", ")) for name, (columns, _) in TABLES.items()}
UNION = "(SELECT a AS id, 1 AS n FROM t UNION ALL SELECT b AS id, 1 AS n FROM t2)"
GROUPED = f"(SELECT id, SUM(n) AS n FROM {UNION} AS u GROUP BY id)"

# an outer query reads the name the renaming took
READ = [
    f"SELECT p.x, d.n FROM {GROUPED} AS d LEFT JOIN p ON p.pid = d.id",
    f"SELECT x, n FROM {GROUPED} AS d JOIN p ON pid = id",
    f"SELECT p.x, d.n FROM p LEFT JOIN {GROUPED} AS d ON p.pid = d.id AND d.n > 1",
    f"WITH d AS (SELECT id, COUNT(*) AS n FROM {UNION} AS u GROUP BY id) SELECT a.n, b.n AS m FROM d AS a JOIN d AS b ON a.id = b.id + 1",
    f"SELECT p.x, d.id, d.n FROM (SELECT u.id, MAX(u.n) AS n FROM {UNION} AS u GROUP BY u.id) AS d JOIN p ON p.pid < d.id",
    # an alias an earlier step has moved into the union is lost the same way
    f"SELECT p.x, d.n FROM (SELECT id AS k, SUM(n) AS n FROM {UNION} AS u GROUP BY id) AS d LEFT JOIN p ON p.pid = d.k",
]

# nothing outside reads the name, or another rule has merged the select away: the text is as it was
UNCHANGED = [
    f"SELECT d.n FROM {GROUPED} AS d",
    f"SELECT d.n FROM {GROUPED} AS d WHERE d.id = 3",
    f"SELECT id, SUM(n) AS n FROM {UNION} AS u GROUP BY id",
    f"SELECT p.x, d.n, q.id FROM (SELECT id, SUM(n) AS n FROM {UNION} AS u GROUP BY id) AS d CROSS JOIN (SELECT pid AS id FROM p) AS q JOIN p ON p.pid = q.id",
]


def _database():
    database = pytest.importorskip("duckdb").connect()
    for name, (columns, rows) in TABLES.items():
        database.execute(f"CREATE TABLE {name} ({columns})")
        database.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(rows[0]))})", rows)
    return database


def _rows(database, sql: str) -> list:
    return sorted(database.execute(sql).fetchall(), key=repr)


@pytest.mark.parametrize("written", READ)
def test_the_text_the_prover_reads_returns_the_rows_of_the_query(written):
    database = _database()
    text = normalize(written, schema=SCHEMA, dialect="duckdb")
    assert (" AS k" if "d.k" in written else " AS id") in text and _rows(database, text) == _rows(database, written)
    with shape_fallback.without("kept_union_names"):
        before = normalize(written, schema=SCHEMA, dialect="duckdb")
    assert before != text
    with pytest.raises(Exception, match="(?i)column"):  # the name the outer query reads is gone
        database.execute(before)


@pytest.mark.parametrize("written", UNCHANGED)
def test_a_name_nothing_outside_reads_is_not_added(written):
    database = _database()
    text = normalize(written, schema=SCHEMA, dialect="duckdb")
    with shape_fallback.without("kept_union_names"):
        assert normalize(written, schema=SCHEMA, dialect="duckdb") == text
    assert _rows(database, text) == _rows(database, written)


def _select(sql: str) -> exp.Select:
    return sqlglot.parse_one(sql).find(exp.Subquery).this


def test_keep_names_only_an_item_that_is_the_read_itself():
    outer = _select("SELECT d.id, d.w FROM (SELECT c1, c1 + 1 AS w, c0 AS n FROM u GROUP BY c1) AS d")
    item, inside, aliased = outer.expressions[0], outer.expressions[1].this.this, outer.expressions[2].this
    kept_union_names.keep(outer, [(item, exp.to_identifier("id")), (inside, exp.to_identifier("id")), (aliased, exp.to_identifier("v"))])
    assert outer.sql() == "SELECT c1 AS id, c1 + 1 AS w, c0 AS n FROM u GROUP BY c1"


def test_keep_leaves_a_name_no_outer_column_reads():
    for sql in (
        "SELECT d.w FROM (SELECT c1, c0 AS w FROM u) AS d",  # nothing reads id
        "SELECT q.id FROM (SELECT c1, c0 AS w FROM u) AS d, q",  # another source's id
        "SELECT d.id FROM (SELECT c1, c0 AS id FROM u) AS d",  # the name is another item's
    ):
        outer = _select(sql)
        before = outer.sql()
        kept_union_names.keep(outer, [(outer.expressions[0], exp.to_identifier("id"))])
        assert outer.sql() == before, sql
    top = sqlglot.parse_one("SELECT c1 FROM u")
    kept_union_names.keep(top, [(top.expressions[0], exp.to_identifier("id"))])
    assert top.sql() == "SELECT c1 FROM u"


def test_keep_keeps_a_quoted_name_quoted():
    outer = _select('SELECT d."Id" FROM (SELECT c1 FROM u) AS d')
    kept_union_names.keep(outer, [(outer.expressions[0], exp.to_identifier("Id", quoted=True))])
    assert outer.sql() == 'SELECT c1 AS "Id" FROM u'


def _prove(left: str, right: str):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql")


def test_a_pair_is_proved_and_was_declined_without_the_repair():
    left, right = READ[0], READ[0].replace("p.pid = d.id", "d.id = p.pid")
    assert _prove(left, right).proven
    with shape_fallback.without("kept_union_names"):
        result = _prove(left, right)
    assert not result.proven and "is not a column of" in (result.reason or "")


# the two unions call different columns id: the outer query reads a on one side and b on the other
SWAPPED = "(SELECT a AS n, b AS id FROM t UNION ALL SELECT a AS n, b AS id FROM t2)"
PLAIN = "(SELECT a AS id, b AS n FROM t UNION ALL SELECT a AS id, b AS n FROM t2)"


@pytest.mark.parametrize(
    "left, right",
    [
        (f"SELECT p.x, d.c FROM (SELECT id, n, COUNT(*) AS c FROM {PLAIN} AS u GROUP BY id, n) AS d LEFT JOIN p ON p.pid = d.id", f"SELECT p.x, d.c FROM (SELECT n, id, COUNT(*) AS c FROM {SWAPPED} AS u GROUP BY n, id) AS d LEFT JOIN p ON p.pid = d.id"),
        (f"SELECT p.x, d.c FROM p JOIN (SELECT id, n, COUNT(*) AS c FROM {PLAIN} AS u GROUP BY id, n) AS d ON p.pid = d.id", f"SELECT p.x, d.c FROM p JOIN (SELECT n, id, COUNT(*) AS c FROM {SWAPPED} AS u GROUP BY n, id) AS d ON p.pid = d.id"),
        (READ[0], READ[0].replace("p.pid = d.id", "p.pid = d.n")),
        (READ[0], READ[0].replace("SELECT b AS id", "SELECT a AS id")),
    ],
)
def test_pairs_that_differ_stay_unproved(left, right):
    assert not _prove(left, right).proven
