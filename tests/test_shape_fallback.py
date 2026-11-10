"""The second attempt without the shape rules (``kumo_sql_prover.shape_fallback``).

Every pair here is made up. The first two have the shape of proofs that a shape rule cost on a benchmark when it
fired on one query only; the checks with the second attempt switched off show that the fallback is what proves them.
"""

from contextlib import contextmanager

import pytest

pytest.importorskip("z3")

from kumo_sql_prover import algebraic_equivalence, shape_fallback
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic
from kumo_sql_prover.smt_equivalence import SmtStatus, TableConstraints

SCHEMA = {"l": ["k", "n", "q"], "o": ["k", "c"], "t": ["a", "b"]}
CONSTRAINTS = {
    "l": TableConstraints(not_null=frozenset({"k", "n", "q"}), keys=(("k", "n"),)),
    "o": TableConstraints(not_null=frozenset({"k", "c"}), keys=(("k",),)),
}

# A filter above a FULL JOIN, against the same rows written as a UNION ALL of the two one-sided joins.
# ``wrapped_join_filters`` pushes the right query's filter into its derived table and has nothing to push on the left.
FULL_JOIN = (
    "SELECT COUNT(*) FROM l FULL JOIN o ON l.k = o.k AND l.q > 5 WHERE l.k IS NULL OR o.k IS NULL",
    "SELECT COUNT(*) FROM (SELECT l.k AS k1, o.k AS k2 FROM l LEFT JOIN o ON l.k = o.k AND l.q > 5 "
    "UNION ALL SELECT l.k AS k1, o.k AS k2 FROM l RIGHT JOIN o ON l.k = o.k AND l.q > 5 WHERE l.k IS NULL) "
    "WHERE k1 IS NULL OR k2 IS NULL",
    {"schema": SCHEMA, "constraints": CONSTRAINTS, "dialect": "bigquery"},
)
# The same text twice. ``disjoint_union`` writes the UNION of two different constants as UNION ALL, and the
# derived table of a UNION ALL with unnamed columns is declined by a later step.
SAME_TEXT = (
    "SELECT CONCAT((SELECT x FROM (SELECT 'a' AS x) AS d1), (SELECT y FROM (SELECT 'b' AS y) AS d2)) FROM (SELECT 1 UNION SELECT 2) AS d3",
    "SELECT CONCAT((SELECT x FROM (SELECT 'a' AS x) AS d1), (SELECT y FROM (SELECT 'b' AS y) AS d2)) FROM (SELECT 1 UNION SELECT 2) AS d3",
    {"schema": {}, "dialect": "mysql"},
)
LOST_WITHOUT_FALLBACK = {"filter above a full join": FULL_JOIN, "same text over a union of constants": SAME_TEXT}

# Proofs that need a shape rule: the first attempt finds them, and with the rules off they are not found.
NEEDS_SHAPE_RULES = {
    "counted key test": (
        "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 0",
        "SELECT a FROM t WHERE a IS NOT NULL GROUP BY a",
        {"schema": SCHEMA, "dialect": "mysql"},
    ),
}


def prove(pair):
    left, right, options = pair
    return prove_equivalent_algebraic(left, right, compare_names=False, **options)


@contextmanager
def _rules_stay_on():
    yield


@pytest.mark.parametrize("name", LOST_WITHOUT_FALLBACK)
def test_second_attempt_proves_a_pair_the_shape_rules_hide(name, monkeypatch):
    assert prove(LOST_WITHOUT_FALLBACK[name]).proven
    monkeypatch.setattr(shape_fallback, "disabled", _rules_stay_on)
    assert not prove(LOST_WITHOUT_FALLBACK[name]).proven


@pytest.mark.parametrize("name", NEEDS_SHAPE_RULES)
def test_first_attempt_still_uses_the_shape_rules(name):
    assert prove(NEEDS_SHAPE_RULES[name]).proven
    with shape_fallback.disabled():
        assert not prove(NEEDS_SHAPE_RULES[name]).proven
    assert prove(NEEDS_SHAPE_RULES[name]).proven


def test_rules_off_leaves_their_inputs_as_written():
    sql = "SELECT a FROM t GROUP BY a HAVING COUNT(a) > 0"
    shaped = normalize(sql, schema=SCHEMA, dialect="mysql")
    with shape_fallback.disabled():
        plain = normalize(sql, schema=SCHEMA, dialect="mysql")
    assert "IS NULL" in shaped and "COUNT" not in shaped
    assert "COUNT" in plain
    assert normalize(sql, schema=SCHEMA, dialect="mysql") == shaped


def test_switch_is_restored_after_an_error():
    assert shape_fallback.enabled()
    with pytest.raises(RuntimeError), shape_fallback.disabled():
        assert not shape_fallback.enabled()
        raise RuntimeError("stop")
    assert shape_fallback.enabled()
    prove(LOST_WITHOUT_FALLBACK["same text over a union of constants"])
    assert shape_fallback.enabled()


def test_second_attempt_never_overrides_a_refutation():
    result = prove_equivalent_algebraic("SELECT a FROM t WHERE a > 1", "SELECT a FROM t WHERE a > 2", schema=SCHEMA, dialect="mysql")
    assert result.status is SmtStatus.NOT_EQUIVALENT


def test_no_second_attempt_when_the_rules_change_nothing(monkeypatch):
    """An unproven pair no shape rule touches costs one more normalization per step, and no solver call."""

    calls = []
    real = algebraic_equivalence.prove_equivalent_smt

    def counted(left, right, **kwargs):
        calls.append(shape_fallback.enabled())
        return real(left, right, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "prove_equivalent_smt", counted)
    result = prove_equivalent_algebraic(
        "SELECT a FROM t WHERE b LIKE 'x%'", "SELECT a FROM t WHERE b LIKE 'x%' AND LENGTH(b) > 0", schema=SCHEMA, dialect="mysql"
    )
    assert not result.proven
    assert calls and all(calls)


def test_one_attempt_inside_a_second_attempt(monkeypatch):
    """A proof started while the rules are off does not start a second attempt of its own."""

    seen = []
    real = algebraic_equivalence._prove_algebraic

    def counted(*args, **kwargs):
        seen.append(shape_fallback.enabled())
        return real(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_algebraic", counted)
    with shape_fallback.disabled():
        prove(NEEDS_SHAPE_RULES["counted key test"])
    assert seen and not any(seen)
