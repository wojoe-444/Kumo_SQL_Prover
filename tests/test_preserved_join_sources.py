"""An inner join written as a derived table on the preserved side of LEFT JOINs is read as the join (``kumo_sql_prover.preserved_join_sources``)."""

import random

import pytest
import sqlglot

pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.preserved_join_sources import merge_preserved_join_source

SCHEMA = {"p": ["id", "owner", "kind", "title"], "u": ["id", "name", "rep"], "c": ["id", "post", "score"]}
TYPES = {name: {column: "INT" for column in columns} for name, columns in SCHEMA.items()}
DERIVED = "(SELECT p.id, p.title, p.kind, u.name FROM p, u WHERE p.owner = u.id AND p.kind = 1) AS d"
FLAT = "p CROSS JOIN u LEFT JOIN c ON p.id = c.post WHERE p.owner = u.id AND p.kind = 1"


def merged(sql, dialect="postgres"):
    out = merge_preserved_join_source(sqlglot.parse_one(sql, read=dialect))
    return out.sql(dialect=dialect) if out is not None else None


def proved(left, right):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="postgres", compare_names=False).proven


def rows_on_duckdb(queries, rows):
    duckdb = pytest.importorskip("duckdb")
    from kumo_sql_prover.duckdb_load import insert_rows, run_unoptimized

    db = duckdb.connect()
    for name, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {name} ({', '.join(f'{column} BIGINT' for column in columns)})")
        insert_rows(db, name, rows.get(name, []))
    return [sorted(map(repr, result)) for result in run_unoptimized(db, *(sqlglot.transpile(q, read="postgres", write="duckdb")[0] for q in queries))]


def differ_on_duckdb(left, right, rows):
    a, b = rows_on_duckdb([left, right], rows)
    return a != b


def test_the_derived_join_is_lifted():
    assert merged(f"SELECT d.title, c.id FROM {DERIVED} LEFT JOIN c ON d.id = c.post") == f"SELECT p.title, c.id FROM {FLAT}"
    assert (
        merged(f"SELECT d.title AS t, COUNT(c.id) AS n FROM {DERIVED} LEFT JOIN c ON d.id = c.post WHERE d.name > 3 GROUP BY d.id, d.title ORDER BY d.title LIMIT 3")
        == "SELECT p.title AS t, COUNT(c.id) AS n FROM p CROSS JOIN u LEFT JOIN c ON p.id = c.post WHERE p.owner = u.id AND p.kind = 1 AND d.name > 3 GROUP BY p.id, p.title ORDER BY p.title LIMIT 3".replace("d.name", "u.name")
    )
    # an output that is an expression keeps its name in the select list and is parenthesized elsewhere
    assert (
        merged("SELECT d.s, d.s * 2 AS twice FROM (SELECT p.id, p.kind + u.rep AS s FROM p JOIN u ON p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post")
        == "SELECT p.kind + u.rep AS s, (p.kind + u.rep) * 2 AS twice FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT d.title FROM {DERIVED}",  # no join
    f"SELECT d.title, c.id FROM {DERIVED} JOIN c ON d.id = c.post",  # inner joins only: the prover flattens these itself
    f"SELECT d.title, c.id FROM {DERIVED} RIGHT JOIN c ON d.id = c.post",  # the derived table is padded
    f"SELECT d.title, c.id FROM {DERIVED} FULL JOIN c ON d.id = c.post",
    f"SELECT d.title, c.id FROM c LEFT JOIN {DERIVED} ON d.id = c.post",  # not the FROM item: it is the padded side
    f"SELECT d.title, c.id FROM {DERIVED} LEFT JOIN c USING (id)",
    f"SELECT * FROM {DERIVED} LEFT JOIN c ON d.id = c.post",
    f"SELECT title, c.id FROM {DERIVED} LEFT JOIN c ON d.id = c.post",  # a column without its source
    f"SELECT d.title, (SELECT MAX(u.rep) FROM u) AS m FROM {DERIVED} LEFT JOIN c ON d.id = c.post",  # a subquery outside the sources
    "SELECT d.title, c.id FROM (SELECT p.id, p.title FROM p WHERE p.kind = 1) AS d LEFT JOIN c ON d.id = c.post",  # one source: a filter
    "SELECT d.title, c.id FROM (SELECT DISTINCT p.id, p.title FROM p, u WHERE p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post",
    "SELECT d.title, c.id FROM (SELECT p.id, p.title FROM p, u WHERE p.owner = u.id LIMIT 2) AS d LEFT JOIN c ON d.id = c.post",
    "SELECT d.title, c.id FROM (SELECT p.id, p.title FROM p, u WHERE p.owner = u.id ORDER BY p.id) AS d LEFT JOIN c ON d.id = c.post",
    "SELECT d.title, c.id FROM (SELECT p.id, p.title FROM p LEFT JOIN u ON p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post",  # an outer join inside
    "SELECT d.title, c.id FROM (SELECT p.id, MAX(p.title) AS title FROM p, u WHERE p.owner = u.id GROUP BY p.id) AS d LEFT JOIN c ON d.id = c.post",
    "SELECT d.title, c.id FROM (SELECT p.id, p.title, ROW_NUMBER() OVER (ORDER BY p.id) AS r FROM p, u WHERE p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post",
    "SELECT d.title, c.id FROM (SELECT p.id, p.title, RANDOM() AS r FROM p, u WHERE p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post",
    "SELECT d.title, c.id FROM (SELECT p.id, p.title FROM p, u WHERE p.owner = u.id AND EXISTS (SELECT 1 FROM c WHERE c.post = p.id)) AS d LEFT JOIN c ON d.id = c.post",
    "SELECT d.title, c.id FROM (SELECT p.id, title FROM p, u WHERE p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post",  # a bare column in the body
    "SELECT d.title, u.id FROM (SELECT p.id, p.title FROM p, u WHERE p.owner = u.id) AS d LEFT JOIN u ON d.id = u.id",  # the alias u twice
    "SELECT d.title, c.id FROM (SELECT p.id, p.title FROM p, u WHERE p.owner = u.id) AS d(a, b) LEFT JOIN c ON d.a = c.post",  # a column list
    "SELECT d.title, c.id FROM (SELECT p.id, p.title FROM p, u WHERE p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post WHERE x.k = 1",  # an outer query's column
])
def test_other_shapes_are_left_alone(sql):
    assert merged(sql) is None


GROUPED = "SELECT p.title, COUNT(c.id) AS n FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post WHERE p.kind = 1 GROUP BY p.id, p.title, u.name"
TARGETS = [
    # an aggregate over the join, the shape Calcite prints
    (GROUPED, f"SELECT d.title, COUNT(c.id) AS n FROM {DERIVED} LEFT JOIN c ON d.id = c.post GROUP BY d.id, d.title, d.name"),
    # with a row tail
    (GROUPED + " ORDER BY p.title DESC LIMIT 3", f"SELECT d.title, COUNT(c.id) AS n FROM {DERIVED} LEFT JOIN c ON d.id = c.post GROUP BY d.id, d.title, d.name ORDER BY d.title DESC LIMIT 3"),
    # two nested layers, each a join
    (
        "SELECT p.title, u.name, MAX(c.score) AS m FROM p JOIN u ON p.owner = u.id JOIN u AS v ON p.kind = v.id LEFT JOIN c ON p.id = c.post GROUP BY p.id, p.title, u.name",
        "SELECT e.title, e.name, MAX(c.score) AS m FROM (SELECT d.id, d.title, d.name FROM (SELECT p.id, p.title, p.kind, u.name FROM p, u WHERE p.owner = u.id) AS d, u AS v WHERE d.kind = v.id) AS e"
        " LEFT JOIN c ON e.id = c.post GROUP BY e.id, e.title, e.name",
    ),
    # no aggregate
    ("SELECT p.title, c.score FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post AND c.score > 2", "SELECT d.title, c.score FROM (SELECT p.id, p.title FROM p JOIN u ON p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post AND c.score > 2"),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_derived_joins_prove(left, right):
    assert proved(left, right)


ROWS = {
    "p": [(1, 1, 1, 10), (2, 1, 1, 20), (3, None, 1, 30), (4, 2, 2, 40), (5, 9, 1, 50), (2, 1, 1, 20)],
    "u": [(1, 7, 5), (2, 8, 6), (2, 8, 6)],
    "c": [(1, 1, 3), (2, 1, 0), (3, 4, 5), (4, None, 1), (5, 8, 2)],
}
JOINED = "(SELECT p.id, p.title, p.kind, u.name FROM p, u WHERE p.owner = u.id) AS d"
NEAR_MISSES = [
    # the filter on the padded side: in the ON it keeps the left row, in the WHERE it drops it
    (f"SELECT d.title, c.score FROM {JOINED} LEFT JOIN c ON d.id = c.post AND c.score > 2", f"SELECT d.title, c.score FROM {JOINED} LEFT JOIN c ON d.id = c.post WHERE c.score > 2"),
    # the derived table on the padded side: its filter cannot become a WHERE of the select
    (f"SELECT c.id, d.title FROM c LEFT JOIN {JOINED} ON d.id = c.post", "SELECT c.id, p.title FROM c LEFT JOIN p ON p.id = c.post CROSS JOIN u WHERE p.owner = u.id"),
    (f"SELECT c.id, d.title FROM {JOINED} RIGHT JOIN c ON d.id = c.post", "SELECT c.id, p.title FROM p CROSS JOIN u RIGHT JOIN c ON p.id = c.post WHERE p.owner = u.id"),
    # duplicates removed, or rows cut, before the join
    ("SELECT d.title, c.score FROM (SELECT DISTINCT p.id, p.title FROM p, u WHERE p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post", "SELECT p.title, c.score FROM p CROSS JOIN u LEFT JOIN c ON p.id = c.post WHERE p.owner = u.id"),
    ("SELECT COUNT(*) AS n FROM (SELECT p.id, p.title FROM p, u WHERE p.owner = u.id ORDER BY p.id, p.title LIMIT 1) AS d LEFT JOIN c ON d.id = c.post", "SELECT COUNT(*) AS n FROM p CROSS JOIN u LEFT JOIN c ON p.id = c.post WHERE p.owner = u.id"),
    # an outer join inside is not the inner join
    ("SELECT d.title, c.score FROM (SELECT p.id, p.title FROM p LEFT JOIN u ON p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post", "SELECT p.title, c.score FROM p JOIN u ON p.owner = u.id LEFT JOIN c ON p.id = c.post"),
    # the aggregate over the join against the join without its filter
    (GROUPED, "SELECT d.title, COUNT(c.id) AS n FROM (SELECT p.id, p.title, u.name FROM p, u WHERE p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post GROUP BY d.id, d.title, d.name"),
    # COUNT(*) counts the padded row, COUNT(c.id) does not
    (GROUPED, f"SELECT d.title, COUNT(*) AS n FROM {DERIVED} LEFT JOIN c ON d.id = c.post GROUP BY d.id, d.title, d.name"),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right, ROWS)


MERGED_QUERIES = [
    f"SELECT d.title, c.id FROM {DERIVED} LEFT JOIN c ON d.id = c.post",
    f"SELECT d.title, COUNT(c.id) AS n, SUM(c.score) AS s FROM {DERIVED} LEFT JOIN c ON d.id = c.post GROUP BY d.id, d.title, d.name",
    f"SELECT d.name, COUNT(*) AS n FROM {JOINED} LEFT JOIN c ON d.id = c.post AND c.score > 0 JOIN u AS v ON v.id = d.kind WHERE d.title > 10 GROUP BY d.name HAVING COUNT(*) > 0",
    "SELECT d.s, d.s * 2 AS twice, c.score FROM (SELECT p.id, p.kind + u.rep AS s FROM p JOIN u ON p.owner = u.id) AS d LEFT JOIN c ON d.id = c.post",
    "SELECT e.title, e.name, MAX(c.score) AS m FROM (SELECT d.id, d.title, d.name FROM (SELECT p.id, p.title, p.kind, u.name FROM p, u WHERE p.owner = u.id OR p.kind = 2) AS d, u AS v WHERE d.kind = v.id) AS e"
    " LEFT JOIN c ON e.id = c.post GROUP BY e.id, e.title, e.name",
    f"SELECT d.title, RANK() OVER (PARTITION BY d.name ORDER BY d.title, c.score) AS r FROM {JOINED} LEFT JOIN c ON d.id = c.post",
]


def random_rows(rng):
    def table(width, size):
        rows = [tuple(rng.choice([None, 1, 2, 3]) for _ in range(width)) for _ in range(rng.randint(0, size))]
        return rows + rows[:2]  # with duplicate rows

    return {"p": table(4, 6), "u": table(3, 4), "c": table(3, 6)}


@pytest.mark.parametrize("sql", MERGED_QUERIES)
def test_merged_queries_give_the_same_rows(sql):
    out = merged(sql)
    normalized = normalize(sql, schema=SCHEMA, types=TYPES, dialect="postgres")
    assert out is not None and out != sql  # one layer per application; the normalizer applies it until none is left
    assert ") AS d " not in normalized and ") AS e " not in normalized
    rng = random.Random(len(sql))
    for rows in [{}, ROWS] + [random_rows(rng) for _ in range(14)]:
        assert not differ_on_duckdb(sql, out, rows)
        assert not differ_on_duckdb(sql, normalized, rows)
