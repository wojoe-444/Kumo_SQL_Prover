"""The VeriEQL worker rules are one group of the narrow second attempt (``kumo_sql_prover.worker_rules``)."""

import sqlglot

from kumo_sql_prover import shape_fallback, worker_rules
from kumo_sql_prover.algebraic_equivalence import prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import TableConstraints

SCHEMA = {"t": ["a", "b", "c"]}
CONSTRAINTS = {"t": TableConstraints(not_null=frozenset({"a"}), keys=(("a",),))}


def test_a_step_that_changes_the_select_is_recorded_and_off_when_the_group_is_off():
    select = sqlglot.parse_one("SELECT a FROM t")

    def changes(sel):
        return sel

    with shape_fallback.tracking() as fired:
        assert worker_rules.step(changes, select) is select
    assert fired == {worker_rules.NAME}
    with shape_fallback.tracking() as fired, shape_fallback.without(worker_rules.NAME):
        assert worker_rules.step(changes, select) is None
    assert not fired


def test_a_step_that_changes_nothing_is_not_recorded():
    select = sqlglot.parse_one("SELECT a FROM t")
    with shape_fallback.tracking() as fired:
        assert worker_rules.step(lambda sel: None, select) is None
    assert not fired


def test_a_prepass_is_recorded_only_when_it_changes_the_tree():
    def rename(tree):
        tree.find(sqlglot.exp.Column).set("this", sqlglot.exp.to_identifier("b"))
        return tree

    with shape_fallback.tracking() as fired:
        worker_rules.prepass(lambda tree: tree, sqlglot.parse_one("SELECT a FROM t"))
    assert not fired
    with shape_fallback.tracking() as fired:
        worker_rules.prepass(rename, sqlglot.parse_one("SELECT a FROM t"))
    assert fired == {worker_rules.NAME}
    with shape_fallback.tracking() as fired, shape_fallback.without(worker_rules.NAME):
        tree = sqlglot.parse_one("SELECT a FROM t")
        assert worker_rules.prepass(rename, tree) is tree and tree.sql() == "SELECT a FROM t"
    assert not fired


def test_a_pair_the_batch_proves_is_still_proved_and_one_it_cannot_is_not():
    left = "SELECT t.a FROM t WHERE t.c IN (SELECT c FROM t GROUP BY c HAVING COUNT(*) > 1)"
    same = prove_equivalent_algebraic(left, left, schema=SCHEMA, constraints=CONSTRAINTS, dialect="mysql")
    assert same.proven
    other = prove_equivalent_algebraic(left, "SELECT t.a FROM t", schema=SCHEMA, constraints=CONSTRAINTS, dialect="mysql")
    assert not other.proven
