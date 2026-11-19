"""A bare ``GROUP BY`` name that two joined tables share, written as the select-list column it means (``shared_group_names``).

Checked on a MySQL 8.0.46 server on 2026-11-19, with ``p (pid, price, k)`` and ``u (pid, units, k)``, both holding
NULL and unmatched ``pid`` values:

* ``SELECT p.pid, SUM(units) FROM p JOIN u ON p.pid = u.pid GROUP BY pid`` runs, and with a ``LEFT JOIN`` it returns
  the groups of ``p.pid`` (the unmatched ``p`` rows stay apart), while ``SELECT u.pid .. GROUP BY pid`` returns the
  groups of ``u.pid`` (they fall into one NULL group). ``SELECT p.pid AS pid ..`` reads the same.
* ``HAVING pid IS NULL`` after ``GROUP BY u.pid`` or ``GROUP BY pid`` follows the output of that name.
* ``SELECT p.pid AS x .. GROUP BY pid``, ``SELECT SUM(units) .. GROUP BY pid`` and ``SELECT p.pid, u.pid .. GROUP BY pid``
  are error 1052 (ambiguous), and so are ``HAVING COUNT(pid) > 0``, ``HAVING MAX(pid) > 0`` and ``ORDER BY COUNT(pid)``.
* ``SELECT u.k AS pid .. GROUP BY pid`` groups by ``u.k`` (two tables have a ``pid``, so the output is meant), and
  ``SELECT u.units AS price .. GROUP BY price`` groups by ``p.price`` (one table has a ``price``, so the column is
  meant; the statement is then error 1055). Neither is read by the module: the output is not a column of its own name.
"""

import pytest
import sqlglot

from kumo_sql_prover.shared_group_names import qualify_shared_group_names

JOINED = "FROM prices AS p JOIN sold AS u ON p.product_id = u.product_id"


def reading(sql: str, dialect: str = "mysql") -> str | None:
    tree, changed = qualify_shared_group_names(sqlglot.parse_one(sql, read=dialect), dialect)
    return tree.sql(dialect=dialect) if changed else None


def test_the_bare_name_is_the_select_list_column():
    assert reading(f"SELECT p.product_id, SUM(u.units) {JOINED} GROUP BY product_id") == f"SELECT p.product_id, SUM(u.units) {JOINED} GROUP BY p.product_id"
    assert reading(f"SELECT u.product_id, SUM(u.units) {JOINED} GROUP BY product_id") == f"SELECT u.product_id, SUM(u.units) {JOINED} GROUP BY u.product_id"
    # with AS of the same name, in any letter case, beside another grouping column, and in a derived table
    assert reading(f"SELECT P.PRODUCT_ID AS product_id, u.day, SUM(u.units) {JOINED} GROUP BY Product_Id, u.day") == f"SELECT P.PRODUCT_ID AS product_id, u.day, SUM(u.units) {JOINED} GROUP BY P.PRODUCT_ID, u.day"
    assert reading(f"SELECT n FROM (SELECT p.product_id, COUNT(*) AS n {JOINED} GROUP BY product_id) AS t") == f"SELECT n FROM (SELECT p.product_id, COUNT(*) AS n {JOINED} GROUP BY p.product_id) AS t"
    # a derived table as the source the column names
    assert reading("SELECT t.k, COUNT(*) FROM (SELECT k FROM a) AS t JOIN b ON t.k = b.k GROUP BY k") == "SELECT t.k, COUNT(*) FROM (SELECT k FROM a) AS t JOIN b ON t.k = b.k GROUP BY t.k"


def test_a_bare_name_in_having_outside_an_aggregate_follows():
    assert reading(f"SELECT p.product_id {JOINED} GROUP BY product_id HAVING product_id > 3 AND COUNT(*) > 1") == f"SELECT p.product_id {JOINED} GROUP BY p.product_id HAVING p.product_id > 3 AND COUNT(*) > 1"
    assert reading(f"SELECT p.product_id {JOINED} GROUP BY p.product_id HAVING product_id NOT IN (SELECT product_id FROM returned)") == (
        f"SELECT p.product_id {JOINED} GROUP BY p.product_id HAVING NOT p.product_id IN (SELECT product_id FROM returned)"
    )
    # inside an aggregate the statement is an error in MySQL, and inside a subquery the name is that query's: both stay
    assert reading(f"SELECT p.product_id {JOINED} GROUP BY p.product_id HAVING COUNT(product_id) > 1") is None
    assert reading(f"SELECT p.product_id {JOINED} GROUP BY p.product_id HAVING EXISTS (SELECT 1 FROM returned WHERE product_id = 3)") is None


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT p.product_id AS x, SUM(u.units) {JOINED} GROUP BY product_id",  # no output of the name: an error in MySQL
        f"SELECT SUM(u.units) {JOINED} GROUP BY product_id",
        f"SELECT p.product_id, u.product_id {JOINED} GROUP BY product_id",  # two outputs of the name: an error in MySQL
        f"SELECT u.day AS product_id, SUM(u.units) {JOINED} GROUP BY product_id",  # an output that is another column
        f"SELECT p.product_id + 0 AS product_id, SUM(u.units) {JOINED} GROUP BY product_id",
        f"SELECT product_id, SUM(u.units) {JOINED} GROUP BY product_id",  # the output is bare too
        f"SELECT p.product_id, SUM(u.units) {JOINED} GROUP BY p.product_id",  # nothing bare
        f"SELECT p.product_id, SUM(u.units) {JOINED} GROUP BY 1",
        f"SELECT *, p.product_id {JOINED} GROUP BY product_id",  # a star may list another output of the name
        f"SELECT p.product_id, SUM(u.units) {JOINED} GROUP BY product_id WITH ROLLUP",
        "SELECT (SELECT o.k FROM b GROUP BY k) FROM a AS o",  # the qualifier is not a source of this select
        f"SELECT p.product_id, SUM(u.units) {JOINED} ORDER BY product_id",
    ],
)
def test_other_selects_are_left_alone(sql):
    assert reading(sql) is None


def test_other_dialects_are_left_alone():
    for dialect in ("sqlite", "postgres", "bigquery", "duckdb"):
        assert reading(f"SELECT p.product_id, SUM(u.units) {JOINED} GROUP BY product_id", dialect) is None


z3 = pytest.importorskip("z3")

from kumo_sql_prover import shape_fallback  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic  # noqa: E402

SCHEMA = {"prices": ["product_id", "price"], "sold": ["product_id", "units", "day"]}
TYPES = {"prices": {"product_id": "INT", "price": "INT"}, "sold": {"product_id": "INT", "units": "INT", "day": "INT"}}
OPTIONS = dict(schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql")
LEFT = "FROM prices AS p LEFT JOIN sold AS u ON p.product_id = u.product_id"


def test_the_pair_with_the_qualified_name_is_proved_and_the_other_table_is_not():
    bare = f"SELECT p.product_id, SUM(u.units) AS n {LEFT} GROUP BY product_id"
    assert prove_equivalent_algebraic(bare, f"SELECT p.product_id, SUM(u.units) AS n {LEFT} GROUP BY p.product_id", **OPTIONS).proven
    # grouping by the padded side's column puts the unmatched rows into one group: another query
    assert not prove_equivalent_algebraic(bare, f"SELECT p.product_id, SUM(u.units) AS n {LEFT} GROUP BY u.product_id", **OPTIONS).proven
    other = f"SELECT u.product_id, SUM(u.units) AS n {LEFT} GROUP BY product_id"
    assert prove_equivalent_algebraic(other, f"SELECT u.product_id, SUM(u.units) AS n {LEFT} GROUP BY u.product_id", **OPTIONS).proven
    assert not prove_equivalent_algebraic(other, bare, **OPTIONS).proven


def test_without_the_rule_the_query_is_declined():
    bare = f"SELECT p.product_id, SUM(u.units) AS n {LEFT} GROUP BY product_id"
    with shape_fallback.without_all(shape_fallback.ELEVENTH_BATCH):
        result = prove_equivalent_algebraic(bare, f"SELECT p.product_id, SUM(u.units) AS n {LEFT} GROUP BY p.product_id", **OPTIONS)
    assert not result.proven


def test_it_is_a_rule_of_the_eleventh_batch():
    assert "shared_group_names" in shape_fallback.ELEVENTH_BATCH
    sql = f"SELECT p.product_id, SUM(u.units) AS n {JOINED} GROUP BY product_id"
    with shape_fallback.tracking() as fired:
        normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
    assert "shared_group_names" in fired
    with shape_fallback.without("shared_group_names"), shape_fallback.tracking() as fired:
        try:
            normalize(sql, schema=SCHEMA, dialect="mysql", types=TYPES)
        except Exception:  # the query is declined without the rule
            pass
    assert "shared_group_names" not in fired
