"""Derived tables the prover shows equal are read as one text (``kumo_sql_prover.derived_table_proofs``)."""

import pytest

pytest.importorskip("z3")

from _made_up_posts import differ_on_duckdb, proved
from kumo_sql_prover.derived_table_proofs import unify

COUNTS = "SELECT p.id, p.day, COUNT(c.id) AS n, COUNT(DISTINCT c.score) AS m FROM p LEFT JOIN c ON c.post = p.id WHERE p.kind = 1 GROUP BY p.id, p.day"
OWNERS = "SELECT u.id, COUNT(DISTINCT p.id) AS posts FROM u LEFT JOIN p ON p.owner = u.id GROUP BY u.id"


def test_unify_installs_the_left_body_under_the_right_names_only_after_a_proof():
    left = "SELECT a.k, a.n FROM (SELECT p.kind AS k, COUNT(*) AS n FROM p GROUP BY p.kind) AS a JOIN u ON u.id = a.k"
    right = "SELECT b.cnt, b.kind FROM (SELECT COUNT(*) AS cnt, t.kind FROM p AS t GROUP BY t.kind) AS b, u WHERE u.id = b.kind"
    asked = []

    def yes(a, b):
        asked.append((a, b))
        return True

    # the bodies are compared with their outputs in the order each reader uses them: k, n against cnt, kind
    out = unify(left, right, dialect="postgres", prove=yes)
    assert asked == [("SELECT p.kind AS k, COUNT(*) AS n FROM p GROUP BY p.kind", "SELECT COUNT(*) AS cnt, t.kind FROM p AS t GROUP BY t.kind")]
    assert out == (left, "SELECT b.cnt, b.kind FROM (SELECT p.kind AS cnt, COUNT(*) AS kind FROM p GROUP BY p.kind) AS b, u WHERE u.id = b.kind")
    assert unify(left, right, dialect="postgres", prove=lambda a, b: False) is None


def test_the_prover_declines_outputs_that_do_not_line_up():
    # read in the order (count, key) on one side and (key, count) on the other: not the same rows column by column
    left = "SELECT a.k, a.n FROM (SELECT p.kind AS k, COUNT(*) AS n FROM p GROUP BY p.kind) AS a JOIN u ON u.id = a.k"
    right = "SELECT b.cnt, b.kind FROM (SELECT COUNT(*) AS cnt, t.kind FROM p AS t GROUP BY t.kind) AS b, u WHERE u.id = b.kind"
    assert not proved(left, right)
    assert differ_on_duckdb(left, right)


@pytest.mark.parametrize("right", [
    "SELECT b.kind, b.cnt FROM (SELECT t.kind, COUNT(*) AS cnt FROM p AS t GROUP BY t.kind) AS b(kind, cnt) JOIN u ON u.id = b.kind",  # a column list
    "SELECT * FROM (SELECT t.kind, COUNT(*) AS cnt FROM p AS t GROUP BY t.kind) AS b JOIN u ON u.id = b.kind",
    "SELECT b.kind, b.cnt FROM (SELECT t.kind, COUNT(*) AS cnt FROM p AS t GROUP BY t.kind) AS b JOIN u USING (kind)",
    "SELECT b.kind, b.cnt FROM (SELECT t.kind, COUNT(*) AS cnt, RANK() OVER (ORDER BY t.kind) AS r FROM p AS t GROUP BY t.kind) AS b JOIN u ON u.id = b.kind",  # a window: window_inputs' case
    "SELECT b.kind, b.cnt FROM (SELECT t.kind, t.day AS cnt FROM p AS t WHERE t.day > 1) AS b JOIN u ON u.id = b.kind",  # a plain filter
    "SELECT b.kind, b.cnt FROM (SELECT t.kind, COUNT(*) AS cnt FROM p AS t GROUP BY t.kind HAVING RANDOM() < 2) AS b JOIN u ON u.id = b.kind",
    "SELECT b.kind, b.cnt FROM (SELECT t.kind + 1 AS kind, COUNT(*) AS cnt FROM p AS t GROUP BY kind) AS b JOIN u ON u.id = b.kind",  # GROUP BY may name the output
    "SELECT u.id, (SELECT MAX(b.cnt) FROM (SELECT t.kind, COUNT(*) AS cnt FROM p AS t WHERE t.owner = u.id GROUP BY t.kind) AS b) AS m, 1 AS one FROM u",  # reads the enclosing row
    "WITH w AS (SELECT 1 AS one) SELECT b.kind, b.cnt FROM (SELECT t.kind, COUNT(*) AS cnt FROM p AS t GROUP BY t.kind) AS b JOIN u ON u.id = b.kind",
])
def test_other_shapes_are_never_compared(right):
    left = "SELECT a.k, a.n FROM (SELECT p.kind AS k, COUNT(*) AS n FROM p GROUP BY p.kind) AS a JOIN u ON u.id = a.k"
    assert unify(left, right, dialect="postgres", prove=lambda a, b: True) is None


TARGETS = [
    # two grouped derived tables, each spelled another way: aliases, the group keys' order, the outputs' order
    (
        f"SELECT ps.id, ps.n, ps.m, us.posts FROM ({COUNTS}) AS ps JOIN ({OWNERS}) AS us ON ps.id = us.id ORDER BY ps.day DESC, ps.id LIMIT 5",
        "SELECT t9.id, t9.n, t9.m, t12.posts FROM (SELECT COUNT(DISTINCT c1.score) AS m, COUNT(c1.id) AS n, p1.day, p1.id FROM p AS p1 LEFT JOIN c AS c1 ON p1.id = c1.post WHERE p1.kind = 1 GROUP BY p1.day, p1.id) AS t9"
        " INNER JOIN (SELECT COUNT(DISTINCT p2.id) AS posts, u1.id FROM u AS u1 LEFT JOIN p AS p2 ON u1.id = p2.owner GROUP BY u1.id) AS t12 ON t9.id = t12.id ORDER BY t9.day DESC, t9.id LIMIT 5",
    ),
    # a DISTINCT derived table under an outer join
    (
        "SELECT u.id, d.kind FROM u LEFT JOIN (SELECT DISTINCT p.owner, p.kind FROM p WHERE p.day > 5 AND p.kind IS NOT NULL) AS d ON d.owner = u.id",
        "SELECT u.id, e.k FROM u LEFT JOIN (SELECT DISTINCT x.kind AS k, x.owner AS o FROM p AS x WHERE NOT x.kind IS NULL AND 5 < x.day) AS e ON u.id = e.o",
    ),
]


@pytest.mark.parametrize("left, right", TARGETS)
def test_equal_derived_tables_prove(left, right):
    assert proved(left, right)


NEAR_MISSES = [
    # one of the derived tables differs: another filter, COUNT(*) for COUNT(c.id), an inner join for the outer one
    (f"SELECT ps.id, ps.n, us.posts FROM ({COUNTS}) AS ps JOIN ({OWNERS}) AS us ON ps.id = us.id", f"SELECT ps.id, ps.n, us.posts FROM ({COUNTS.replace('p.kind = 1', 'p.kind = 2')}) AS ps JOIN ({OWNERS}) AS us ON ps.id = us.id"),
    (f"SELECT ps.id, ps.n, us.posts FROM ({COUNTS}) AS ps JOIN ({OWNERS}) AS us ON ps.id = us.id", f"SELECT ps.id, ps.n, us.posts FROM ({COUNTS.replace('COUNT(c.id)', 'COUNT(*)')}) AS ps JOIN ({OWNERS}) AS us ON ps.id = us.id"),
    (f"SELECT ps.id, ps.n, us.posts FROM ({COUNTS}) AS ps JOIN ({OWNERS}) AS us ON ps.id = us.id", f"SELECT ps.id, ps.n, us.posts FROM ({COUNTS}) AS ps JOIN ({OWNERS.replace('LEFT JOIN', 'JOIN')}) AS us ON ps.id = us.id"),
    # the two counts read the other way round
    (f"SELECT ps.id, ps.n, ps.m FROM ({COUNTS}) AS ps JOIN ({OWNERS}) AS us ON ps.id = us.id", f"SELECT ps.id, ps.m, ps.n FROM ({COUNTS}) AS ps JOIN ({OWNERS}) AS us ON ps.id = us.id"),
    # DISTINCT against no DISTINCT
    (
        "SELECT u.id, d.kind FROM u LEFT JOIN (SELECT DISTINCT p.owner, p.kind FROM p) AS d ON d.owner = u.id",
        "SELECT u.id, d.kind FROM u LEFT JOIN (SELECT p.owner, p.kind FROM p GROUP BY p.owner, p.kind, p.id) AS d ON d.owner = u.id",
    ),
]


@pytest.mark.parametrize("left, right", NEAR_MISSES)
def test_near_misses_do_not_prove(left, right):
    assert not proved(left, right)
    assert differ_on_duckdb(left, right)
