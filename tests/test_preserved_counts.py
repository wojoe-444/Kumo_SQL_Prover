"""COUNT of a NOT NULL column of a table no outer join pads is COUNT(*) (``kumo_sql_prover.preserved_counts``)."""

import pytest
import sqlglot

pytest.importorskip("z3")

from _made_up_posts import SCHEMA, TYPES, differ_on_duckdb, random_databases
from kumo_sql_prover.algebraic_equivalence import _null_extended, normalize, prove_equivalent_algebraic
from kumo_sql_prover.preserved_counts import count_preserved_not_null
from kumo_sql_prover.smt_equivalence import TableConstraints

NOT_NULL = {"p": frozenset({"id", "kind"}), "u": frozenset({"id"}), "c": frozenset({"id"})}
CONSTRAINTS = {name: TableConstraints(not_null=columns) for name, columns in NOT_NULL.items()}
GROUPED = "SELECT u.name, COUNT(p.id) AS n FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post GROUP BY u.name"


def counted(sql, not_null=NOT_NULL):
    out = count_preserved_not_null(sqlglot.parse_one(sql, read="postgres"), not_null, _null_extended)
    return out.sql(dialect="postgres") if out is not None else None


def proved(left, right, constraints=CONSTRAINTS):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, constraints=constraints, dialect="postgres", compare_names=False).proven


def test_the_count_of_a_preserved_not_null_column_counts_rows():
    assert counted(GROUPED) == GROUPED.replace("COUNT(p.id)", "COUNT(*)")
    # every such count in the select list, HAVING and ORDER BY; the padded side's count stays
    assert (
        counted("SELECT COUNT(p.kind) AS a, COUNT(c.id) AS b, COUNT(u.id) AS d FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post HAVING COUNT(p.id) > 1 ORDER BY COUNT(p.id)")
        == "SELECT COUNT(*) AS a, COUNT(c.id) AS b, COUNT(*) AS d FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post HAVING COUNT(*) > 1 ORDER BY COUNT(*)"
    )
    assert counted("SELECT COUNT(c.id) AS n FROM p RIGHT JOIN c ON p.id = c.post") == "SELECT COUNT(*) AS n FROM p RIGHT JOIN c ON p.id = c.post"


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(p.id) AS n FROM p JOIN u ON p.owner = u.id",  # no outer join: read already
    "SELECT COUNT(c.id) AS n FROM p LEFT JOIN c ON p.id = c.post",  # the padded side
    "SELECT COUNT(p.id) AS n FROM p RIGHT JOIN c ON p.id = c.post",
    "SELECT COUNT(p.id) AS n FROM p FULL JOIN c ON p.id = c.post",
    "SELECT COUNT(c.id) AS n FROM p FULL JOIN c ON p.id = c.post",
    "SELECT COUNT(u.id) AS n FROM p LEFT JOIN c ON p.id = c.post LEFT JOIN u ON u.id = p.owner",
    "SELECT COUNT(p.id) AS n FROM p JOIN u ON p.owner = u.id RIGHT JOIN c ON p.id = c.post",  # a later RIGHT JOIN pads p
    "SELECT COUNT(p.owner) AS n FROM p LEFT JOIN c ON p.id = c.post",  # a nullable column
    "SELECT COUNT(DISTINCT p.id) AS n FROM p LEFT JOIN c ON p.id = c.post",
    "SELECT COUNT(p.id + 0) AS n FROM p LEFT JOIN c ON p.id = c.post",
    "SELECT COUNT(id) AS n FROM p LEFT JOIN c ON p.id = c.post",  # which id is not said
    "SELECT COUNT(p.id) OVER (PARTITION BY p.kind) AS n FROM p LEFT JOIN c ON p.id = c.post",
    "SELECT COUNT(p.id) FILTER (WHERE c.score > 0) AS n FROM p LEFT JOIN c ON p.id = c.post",
    "SELECT COUNT(d.id) AS n FROM (SELECT p.id FROM p) AS d LEFT JOIN c ON d.id = c.post",  # not a base table
    "SELECT COUNT(p.id0) AS n FROM p AS p(id0, o, k, d) LEFT JOIN c ON p.id0 = c.post",  # renamed columns
    "SELECT COUNT(p.id) AS n FROM s.p AS p LEFT JOIN c ON p.id = c.post",  # another schema's table
    "SELECT (SELECT COUNT(p.id) FROM u WHERE u.id = p.owner) AS n FROM p LEFT JOIN c ON p.id = c.post",  # the subquery's own aggregate
])
def test_other_counts_are_left_alone(sql):
    assert counted(sql) is None


def test_no_declaration_no_rewrite():
    assert counted(GROUPED, {}) is None
    assert counted(GROUPED, {"p": frozenset({"kind"})}) is None


TARGETS = [
    (GROUPED, GROUPED.replace("COUNT(p.id)", "COUNT(*)")),
    (GROUPED + " ORDER BY COUNT(p.id) DESC, u.name LIMIT 3", GROUPED.replace("COUNT(p.id)", "COUNT(*)") + " ORDER BY COUNT(*) DESC, u.name LIMIT 3"),
    ("SELECT u.name, COUNT(p.kind) AS n, SUM(c.score) AS s FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post GROUP BY u.name", "SELECT u.name, COUNT(*) AS n, SUM(c.score) AS s FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post GROUP BY u.name"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_preserved_counts_prove(left, right):
    assert proved(left, right)


# databases that respect the NOT NULL declarations
ROWS = {
    "p": [(1, 1, 1, 10), (2, 1, 1, 20), (3, None, 1, 30), (4, 2, 2, 40), (5, 9, 1, 50), (4, 2, 2, 40)],
    "u": [(1, 7, 5), (2, 8, 6), (2, 8, 6), (3, None, 0)],
    "c": [(1, 1, 3), (2, 1, 0), (3, 4, 5), (4, None, 1), (5, 8, 2)],
}
NEAR_MISSES = [
    # the padded side: COUNT(c.id) is 0 for a post without comments, COUNT(*) is 1
    ("SELECT p.id, COUNT(c.id) AS n FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.id", "SELECT p.id, COUNT(*) AS n FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.id"),
    # a nullable column of the preserved side
    ("SELECT p.kind, COUNT(p.owner) AS n FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.kind", "SELECT p.kind, COUNT(*) AS n FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.kind"),
    # a later RIGHT JOIN pads the table
    ("SELECT COUNT(p.id) AS n FROM p JOIN u ON p.owner = u.id RIGHT JOIN c ON p.id = c.post", "SELECT COUNT(*) AS n FROM p JOIN u ON p.owner = u.id RIGHT JOIN c ON p.id = c.post"),
    # DISTINCT counts values, not rows
    ("SELECT COUNT(DISTINCT p.id) AS n FROM p LEFT JOIN c ON p.id = c.post", "SELECT COUNT(*) AS n FROM p LEFT JOIN c ON p.id = c.post"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


def test_without_the_declaration_the_count_is_not_the_row_count():
    left, right = "SELECT p.kind, COUNT(p.id) AS n FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.kind", "SELECT p.kind, COUNT(*) AS n FROM p LEFT JOIN c ON p.id = c.post GROUP BY p.kind"
    assert proved(left, right)
    assert not proved(left, right, constraints={})
    assert differ_on_duckdb(left, right, {"p": [(None, 1, 1, 1), (2, 1, 1, 1)], "c": [(1, 2, 1)]})


COUNTED_QUERIES = [
    GROUPED,
    "SELECT COUNT(p.kind) AS a, COUNT(c.id) AS b, COUNT(u.id) AS d FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post HAVING COUNT(p.id) > 1",
    "SELECT c.score, COUNT(c.id) AS n FROM p RIGHT JOIN c ON p.id = c.post GROUP BY c.score",
    "SELECT p.owner, COUNT(p.id) AS n, COUNT(c.id) AS m FROM p LEFT JOIN c ON p.id = c.post AND c.score > 1 LEFT JOIN u ON u.id = p.owner GROUP BY p.owner",
]


@pytest.mark.parametrize("sql", COUNTED_QUERIES)
def test_rewritten_counts_give_the_same_rows(sql):
    out = counted(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, not_null=NOT_NULL, dialect="postgres")
    assert out is not None and "COUNT(*)" in out
    for rows in random_databases(len(sql)):
        legal = {  # keep the rows that respect the declarations
            "p": [r for r in rows.get("p", []) if r[0] is not None and r[2] is not None],
            "u": [r for r in rows.get("u", []) if r[0] is not None],
            "c": [r for r in rows.get("c", []) if r[0] is not None],
        }
        assert not differ_on_duckdb(sql, out, legal)
        assert not differ_on_duckdb(sql, normalized, legal)
