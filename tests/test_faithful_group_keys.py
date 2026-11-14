"""A ``GROUP BY`` key that repeats an aliased output prints as itself, not as the alias (``ast_utils.faithful_sql``)."""

import pytest
import sqlglot
from sqlglot import exp

from kumo_sql_prover.ast_utils import LossySql, _group_keys_apart, faithful_sql

# quoted names and UNION DISTINCT: no neighbouring dialect's text reads back as BigQuery, so only BigQuery can print it
SOURCE = "(SELECT `a`.`x` FROM `a` UNION DISTINCT SELECT `b`.`x` FROM `b`) AS `t`"
ORDERED = f"SELECT CAST(`t`.`x` / 50 AS INT64) AS seg, COUNT(*) AS n FROM {SOURCE} GROUP BY CAST(`t`.`x` / 50 AS INT64) ORDER BY n"


def tree_of(sql: str) -> exp.Expression:
    return sqlglot.parse_one(sql, read="bigquery")


def test_bigquery_prints_the_key_as_the_alias_and_the_faithful_text_does_not():
    tree = tree_of(ORDERED)
    assert "GROUP BY seg" in tree.sql(dialect="bigquery")  # the writer's own text: another tree
    text = faithful_sql(tree, "bigquery")
    assert "GROUP BY (CAST(" in text
    back = tree_of(text)
    key = back.args["group"].expressions[0]
    assert isinstance(key.unnest(), exp.Cast) and key.unnest() == back.expressions[0].this
    assert tree.args["group"].expressions[0] == tree_of(ORDERED).args["group"].expressions[0]  # the tree is not edited


def test_a_nested_select_is_printed_the_same_way():
    nested = f"SELECT `m`.`seg` FROM ({ORDERED} LIMIT 3) AS `m`"
    assert "GROUP BY (CAST(" in faithful_sql(tree_of(nested), "bigquery")


@pytest.mark.parametrize(
    "sql",
    [
        ORDERED.replace(" ORDER BY n", ""),  # no ORDER BY: the writer keeps the key
        ORDERED.replace("GROUP BY CAST(`t`.`x` / 50 AS INT64)", "GROUP BY `t`.`x`"),  # the key is no aliased output
        ORDERED.replace("GROUP BY CAST(`t`.`x` / 50 AS INT64)", "GROUP BY 1"),
        "SELECT `t`.`x`, COUNT(*) AS n FROM `t` GROUP BY `t`.`x` ORDER BY n",
    ],
)
def test_text_the_writer_already_prints_faithfully_is_unchanged(sql):
    tree = tree_of(sql)
    assert faithful_sql(tree, "bigquery") == tree.sql(dialect="bigquery")


def test_only_keys_that_repeat_an_aliased_output_of_an_ordered_select_are_set_apart():
    assert _group_keys_apart(tree_of(ORDERED.replace(" ORDER BY n", ""))) is None
    assert _group_keys_apart(tree_of(ORDERED.replace("GROUP BY CAST(`t`.`x` / 50 AS INT64)", "GROUP BY `t`.`x`"))) is None
    apart = _group_keys_apart(tree_of(ORDERED))
    assert isinstance(apart.args["group"].expressions[0], exp.Paren)


def test_a_string_where_a_name_belongs_is_a_failed_reading_not_a_crash():
    # the neighbouring writers quote names with double quotes, which BigQuery reads as strings; with nothing to set
    # apart the query has no faithful text, and that is reported as such
    tree = tree_of(ORDERED)
    tree.args["group"].expressions[0].replace(exp.Paren(this=tree.args["group"].expressions[0].copy()))  # keys already apart
    assert "GROUP BY (CAST(" in faithful_sql(tree, "bigquery")
    odd = tree_of("SELECT `t`.`x` FROM `t`")
    odd.find(exp.Table).set("this", exp.Literal.string("t"))  # a table that is a string literal reads back as nothing BigQuery has
    with pytest.raises(LossySql):
        faithful_sql(odd, "bigquery")


z3 = pytest.importorskip("z3")

from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic  # noqa: E402

SCHEMA = {"a": ["x"], "b": ["x"]}
TYPES = {"a": {"x": "INT64"}, "b": {"x": "INT64"}}
GROUPED = "SELECT CAST(t.x / 50 AS INT64) AS seg, COUNT(*) AS n FROM (SELECT `a`.`x` FROM `a` UNION DISTINCT SELECT `b`.`x` FROM `b`) AS t GROUP BY seg ORDER BY seg, n LIMIT 10"


def test_an_ordered_grouping_over_a_distinct_union_is_still_read():
    wrapped = "WITH m AS (SELECT CAST(t.x / 50 AS INT64) AS seg FROM (SELECT `a`.`x` FROM `a` UNION DISTINCT SELECT `b`.`x` FROM `b`) AS t) SELECT seg, COUNT(*) AS n FROM m GROUP BY seg ORDER BY seg, n LIMIT 10"
    prove = lambda left, right: prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect="bigquery").proven  # noqa: E731
    assert prove(GROUPED, wrapped)
    assert not prove(GROUPED, wrapped.replace("/ 50", "/ 51"))
