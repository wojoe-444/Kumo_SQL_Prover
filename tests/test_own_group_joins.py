"""A ``LEFT JOIN`` from a table to its own groups is an inner join (``own_group_joins``)."""

import random

import pytest
import sqlglot

from kumo_sql_prover.own_group_joins import inner_join_own_groups

NOT_NULL = {"salaries": frozenset({"company_id", "grade"})}
GROUPS = "(SELECT company_id, MAX(salary) AS top FROM salaries GROUP BY company_id) AS m"
LEFT = f"SELECT s.employee_id, m.top FROM salaries AS s LEFT JOIN {GROUPS} ON s.company_id = m.company_id"


def rewritten(sql: str, not_null=NOT_NULL, types=None) -> str | None:
    result = inner_join_own_groups(sqlglot.parse_one(sql, read="mysql"), not_null, types)
    return None if result is None else result.sql(dialect="mysql")


def test_a_row_always_finds_its_own_group():
    assert rewritten(LEFT) == LEFT.replace("LEFT JOIN", "JOIN")
    assert rewritten(LEFT.replace("s.company_id = m.company_id", "m.company_id = s.company_id")) is not None
    # the key under another output name, two grouping columns, and a row reached through an inner join
    renamed = "SELECT s.salary FROM salaries AS s LEFT JOIN (SELECT t.company_id AS c, COUNT(*) AS n FROM salaries AS t GROUP BY t.company_id) AS m ON m.c = s.company_id"
    assert rewritten(renamed) == renamed.replace("LEFT JOIN", "JOIN")
    two = "SELECT s.salary FROM offices AS o JOIN salaries AS s ON o.id = s.company_id LEFT JOIN (SELECT company_id, grade, AVG(salary) AS a FROM salaries GROUP BY company_id, grade) AS m ON s.company_id = m.company_id AND s.grade = m.grade"
    assert rewritten(two) == two.replace("LEFT JOIN", "JOIN")


@pytest.mark.parametrize(
    "sql, not_null",
    [
        (LEFT, None),
        (LEFT, {"salaries": frozenset({"salary"})}),  # a NULL company_id is padded by LEFT and dropped by an inner join
        (LEFT.replace("GROUP BY company_id)", "WHERE salary > 0 GROUP BY company_id)"), NOT_NULL),  # the row's own group can be filtered away
        (LEFT.replace("GROUP BY company_id)", "GROUP BY company_id HAVING COUNT(*) > 1)"), NOT_NULL),
        (LEFT.replace("GROUP BY company_id)", "GROUP BY company_id LIMIT 5)"), NOT_NULL),
        (LEFT.replace("FROM salaries GROUP BY", "FROM payroll GROUP BY"), {**NOT_NULL, "payroll": frozenset({"company_id"})}),  # another table's groups
        (LEFT.replace("ON s.company_id = m.company_id", "ON s.company_id = m.company_id AND m.top > 5"), NOT_NULL),  # an extra condition
        (LEFT.replace("ON s.company_id = m.company_id", "ON s.employee_id = m.company_id"), NOT_NULL),  # another column
        (LEFT.replace("ON s.company_id = m.company_id", "ON s.company_id >= m.company_id"), NOT_NULL),
        (LEFT.replace("ON s.company_id = m.company_id", "ON s.company_id = m.top"), NOT_NULL),  # not the grouping column
        (LEFT.replace("SELECT company_id, MAX(salary) AS top FROM salaries GROUP BY company_id", "SELECT company_id, MAX(salary) AS top FROM salaries GROUP BY company_id, grade"), NOT_NULL),  # a grouping column the join does not tie
        (LEFT.replace("SELECT company_id, MAX(salary) AS top FROM salaries GROUP BY company_id", "SELECT company_id + 0 AS company_id, MAX(salary) AS top FROM salaries GROUP BY company_id"), NOT_NULL),
        (LEFT.replace("LEFT JOIN", "RIGHT JOIN"), NOT_NULL),  # the groups are kept, not the rows
        (LEFT.replace("LEFT JOIN", "FULL JOIN"), NOT_NULL),
        # the row may itself be padded by an earlier outer join
        ("SELECT s.salary FROM offices AS o LEFT JOIN salaries AS s ON o.id = s.company_id LEFT JOIN " + GROUPS + " ON s.company_id = m.company_id", NOT_NULL),
        # the row is not a base-table row
        ("SELECT s.salary FROM (SELECT company_id, salary FROM salaries WHERE salary > 0) AS s LEFT JOIN " + GROUPS + " ON s.company_id = m.company_id", NOT_NULL),
        (LEFT.replace("LEFT JOIN", "JOIN"), NOT_NULL),  # already inner
    ],
)
def test_other_joins_are_left_as_written(sql, not_null):
    assert rewritten(sql, not_null) is None


def test_a_floating_point_grouping_column_is_left_alone():
    assert rewritten(LEFT, types={"salaries": {"company_id": "DOUBLE"}}) is None
    assert rewritten(LEFT, types={"salaries": {"company_id": "INT"}}) is not None


BODIES = [
    "SELECT company_id, MAX(salary) AS top FROM salaries GROUP BY company_id",
    "SELECT t.company_id AS company_id, COUNT(*) AS top FROM salaries AS t GROUP BY t.company_id",
    "SELECT company_id, grade, SUM(salary) AS top FROM salaries GROUP BY company_id, grade",
]


@pytest.mark.parametrize("seed", range(3))
def test_left_and_inner_return_the_same_rows_on_duckdb_when_the_columns_are_not_null(seed):
    duckdb = pytest.importorskip("duckdb")
    rng = random.Random(seed)
    differs_with_nulls = 0
    for body in BODIES:
        on = "s.company_id = m.company_id" + (" AND s.grade = m.grade" if "grade," in body else "")
        sql = f"SELECT s.employee_id, s.salary, m.top FROM salaries AS s LEFT JOIN ({body}) AS m ON {on}"
        inner = rewritten(sql)
        assert inner is not None, sql
        for nullable in (False, False, False, True):
            con = duckdb.connect()
            con.execute("CREATE TABLE salaries (company_id INT, grade INT, employee_id INT, salary INT)")
            key = (lambda: rng.choice([None, 1, 2])) if nullable else (lambda: rng.choice([1, 2, 3]))
            for row in range(rng.randint(1, 8)):
                con.execute("INSERT INTO salaries VALUES (?, ?, ?, ?)", [key(), key(), row, rng.choice([None, 10, 20])])
            same = sorted(con.execute(sql).fetchall(), key=repr) == sorted(con.execute(inner).fetchall(), key=repr)
            if nullable:
                differs_with_nulls += not same
            else:
                assert same, sql
    assert differs_with_nulls > 0  # the NOT NULL facts are what make it true


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import TableConstraints  # noqa: E402

SCHEMA = {"salaries": ["company_id", "grade", "employee_id", "salary"]}


def test_the_two_spellings_are_one_query_under_not_null():
    inner = LEFT.replace("LEFT JOIN", "JOIN")
    prove = lambda constraints: prove_equivalent_algebraic(LEFT, inner, schema=SCHEMA, dialect="mysql", constraints=constraints).proven  # noqa: E731
    assert prove({"salaries": TableConstraints(not_null=frozenset({"company_id"}))})
    assert not prove(None)
    assert not prove({"salaries": TableConstraints(not_null=frozenset({"salary"}))})


def test_a_common_table_expression_of_the_same_name_is_read_as_what_it_is():
    # the name is the CTE, whose column can be NULL whatever is declared on a table of that name
    schema = {**SCHEMA, "raw": ["company_id", "salary", "employee_id"]}
    cte = "WITH salaries AS (SELECT NULLIF(company_id, 1) AS company_id, salary, employee_id FROM raw) " + LEFT
    constraints = {name: TableConstraints(not_null=frozenset({"company_id"})) for name in ("salaries", "raw")}
    assert not prove_equivalent_algebraic(cte, cte.replace("LEFT JOIN", "JOIN"), schema=schema, dialect="mysql", constraints=constraints).proven


def test_the_conditional_verdict_names_the_column():
    result = prove_equivalent_algebraic(LEFT, LEFT.replace("LEFT JOIN", "JOIN"), schema=SCHEMA, dialect="mysql", conditional=True)
    assert result.status.value == "proven_conditionally"
    assert [(c.kind, tuple(c.columns)) for c in result.conditions] == [("not_null", ("company_id",))]
