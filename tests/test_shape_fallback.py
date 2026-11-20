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


def test_no_second_attempt_for_a_very_large_pair(monkeypatch):
    """Normalizing twice costs seconds on a very large query: such a pair gets the first attempt only."""

    left, right, options = LOST_WITHOUT_FALLBACK["filter above a full join"]
    assert len(left) + len(right) <= shape_fallback.MAX_CHARS
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(left) + len(right) - 1)
    assert not prove_equivalent_algebraic(left, right, compare_names=False, **options).proven
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(left) + len(right))
    assert prove_equivalent_algebraic(left, right, compare_names=False, **options).proven


def test_ninth_batch_rules_fire_only_where_no_older_attempt_is_changed():
    """They fire on the first attempt and on a narrow attempt that switches off rules of their own batch; the attempt
    without the shape rules, the attempts with the late rules and a narrow attempt with an older rule off read the pair
    as before the batch."""

    assert shape_fallback.ninth_batch_enabled()
    with shape_fallback.without("own_row_copies"), shape_fallback.without("distinct_match_joins"):
        assert shape_fallback.ninth_batch_enabled()
    with shape_fallback.without_all(shape_fallback.NINTH_BATCH):
        assert shape_fallback.ninth_batch_enabled() and not shape_fallback.rule_enabled("own_column_lists")
    with shape_fallback.disabled():
        assert not shape_fallback.ninth_batch_enabled()
    with shape_fallback.late():
        assert not shape_fallback.ninth_batch_enabled()
    with shape_fallback.without("own_key_membership"):
        assert not shape_fallback.ninth_batch_enabled()
    assert shape_fallback.ninth_batch_enabled() and shape_fallback.off() == frozenset()


def test_a_ninth_batch_rule_stays_out_of_an_attempt_with_an_older_rule_off():
    """With ``own_key_membership`` off, ``distinct_match_joins`` began to read the list of this query and cost a proof
    that attempt used to give (``tests/test_self_key_membership.py`` has the pair)."""

    sql = (
        "SELECT DISTINCT sales.buyer_id AS buyer_id FROM product INNER JOIN sales ON sales.product_id = product.product_id "
        "WHERE product_name = 'a' AND sales.buyer_id NOT IN (SELECT sales.buyer_id FROM product INNER JOIN sales ON sales.product_id = product.product_id WHERE product_name = 'b')"
    )
    options = {
        "schema": {"product": ["product_id", "product_name"], "sales": ["product_id", "buyer_id"]},
        "keys": {"product": [("product_id",)], "sales": [("buyer_id",)]},
        "not_null": {"sales": frozenset({"buyer_id"})},
        "dialect": "mysql",
    }
    with shape_fallback.tracking() as fired:
        normalize(sql, **options)
    assert "distinct_match_joins" not in fired  # ``own_key_membership`` reads the list first
    with shape_fallback.without("own_key_membership"), shape_fallback.tracking() as fired:
        older = normalize(sql, **options)
    assert not fired & shape_fallback.NINTH_BATCH
    with shape_fallback.without("own_key_membership"), shape_fallback.without_all(shape_fallback.NINTH_BATCH):
        assert normalize(sql, **options) == older


# --- the two stages of a proof (the ninth batch runs second) --------------------------------------------------

from kumo_sql_prover import worker_rules  # noqa: E402

PATRONS = {"patron": ["patron_id", "patron_name"], "loan": ["patron_id", "title"]}
PATRON_FACTS = {
    "patron": TableConstraints(not_null=frozenset({"patron_id", "patron_name"}), keys=(("patron_id",),)),
    "loan": TableConstraints(not_null=frozenset({"patron_id", "title"})),
}
# Patrons who borrowed two titles and never a third, with joins under DISTINCT: ``distinct_match_joins`` of the ninth
# batch reads the joins.
JOINED = (
    "SELECT DISTINCT p.patron_id, p.patron_name FROM patron AS p JOIN loan AS o1 ON p.patron_id = o1.patron_id AND o1.title = 'A' "
    "JOIN loan AS o2 ON p.patron_id = o2.patron_id AND o2.title = 'B' WHERE p.patron_id NOT IN (SELECT patron_id FROM loan WHERE title = 'C')"
)
PATRON_OPTIONS = {"schema": PATRONS, "constraints": PATRON_FACTS, "compare_names": False, "dialect": "mysql"}
# Buyers of one label only, as a test of the group and as a list: ``single_value_groups`` of the ninth batch reads the
# first, and nothing older proves the pair.
SALES = {"sale": ["buyer", "label", "paid"]}
ONE_LABEL = "SELECT buyer FROM sale GROUP BY buyer HAVING COUNT(DISTINCT label) = 1 AND MIN(label) = 'S'"
NEVER_ANOTHER = "SELECT DISTINCT buyer FROM sale WHERE label = 'S' AND buyer NOT IN (SELECT buyer FROM sale WHERE label <> 'S')"
SALE_OPTIONS = {
    "schema": SALES, "constraints": {"sale": TableConstraints(not_null=frozenset({"buyer", "label"}))}, "types": {"sale": {"label": "STRING"}},
    "compare_names": False, "dialect": "mysql",
}
# A value looked up by a bare key name against a list: ``lookup_names``, the batch's pre-pass over the whole tree.
ITEMS = {"item": ["item_id", "label"], "sale": ["buyer", "item_id", "paid"]}
ITEM_FACTS = {"item": TableConstraints(keys=(("item_id",), ("label",))), "sale": TableConstraints()}
BARE = "SELECT buyer FROM sale WHERE item_id = (SELECT item_id FROM item WHERE label = 'S')"
LISTED = "SELECT s.buyer FROM sale AS s WHERE s.item_id IN (SELECT i.item_id FROM item AS i WHERE i.label = 'S')"
ITEM_OPTIONS = {"schema": ITEMS, "constraints": ITEM_FACTS, "compare_names": False, "dialect": "mysql"}


TENTH_SEEN: list = []  # for each step of the last proof ``stages_of`` ran, whether the tenth batch was on in it


def stages_of(monkeypatch, left, right, **options):
    """The proof's verdict, and for each step whether the ninth batch was on in it."""

    seen = []
    real = algebraic_equivalence._prove_algebraic

    def watched(*args, **kwargs):
        seen.append(shape_fallback.ninth_stage())
        TENTH_SEEN.append(shape_fallback.stage_key()[1])
        return real(*args, **kwargs)

    TENTH_SEEN.clear()
    monkeypatch.setattr(algebraic_equivalence, "_prove_algebraic", watched)
    return prove_equivalent_algebraic(left, right, **options), seen


def test_the_stage_switches():
    assert not shape_fallback.ninth_staged() and shape_fallback.ninth_batch_enabled()  # outside a proof the batch is on
    with shape_fallback.ninth_deferred() as seen:
        assert shape_fallback.ninth_staged() and not shape_fallback.ninth_batch_enabled() and shape_fallback.ninth_watching()
        with shape_fallback.without_all(shape_fallback.NINTH_BATCH):
            assert not shape_fallback.ninth_batch_enabled()
        for older in (shape_fallback.disabled, shape_fallback.late, lambda: shape_fallback.without("own_key_membership")):
            with older():
                assert not shape_fallback.ninth_watching()  # stage two would not let the batch fire here either
        with shape_fallback.without(worker_rules.NAME):
            assert shape_fallback.ninth_watching()
        shape_fallback.note_ninth_candidate()
        assert seen == [True] and not shape_fallback.ninth_watching()  # one yes is enough
        with shape_fallback.ninth_on():
            assert shape_fallback.ninth_batch_enabled() and not shape_fallback.ninth_watching()
            with shape_fallback.without(worker_rules.NAME):  # measured without the worker group, so it fires with the group off
                assert shape_fallback.ninth_batch_enabled()
                with shape_fallback.without("own_key_membership"):
                    assert not shape_fallback.ninth_batch_enabled()
    assert not shape_fallback.ninth_staged()
    assert shape_fallback._BESIDE_NINTH == {worker_rules.NAME}


def test_a_pair_proved_before_the_batch_never_sees_it(monkeypatch):
    """Stage one is the prover as it was: a rule of the batch would fire on both queries, and no step has the batch on."""

    with shape_fallback.tracking() as fired:
        normalize(JOINED, schema=PATRONS, dialect="mysql")
    assert "distinct_match_joins" in fired
    result, seen = stages_of(monkeypatch, JOINED, JOINED.replace(" AS o1", " AS first").replace("o1.", "first."), **PATRON_OPTIONS)
    assert result.proven and seen and not any(seen)


@pytest.mark.parametrize("pair", [(ONE_LABEL, NEVER_ANOTHER, SALE_OPTIONS), (BARE, LISTED, ITEM_OPTIONS)], ids=["a rule of one select", "the pre-pass"])
def test_a_pair_only_the_batch_proves_is_proved_in_stage_two(monkeypatch, pair):
    left, right, options = pair
    result, seen = stages_of(monkeypatch, left, right, **options)
    assert result.proven
    assert seen[0] is False and seen[-1] is True and seen == sorted(seen)  # all of stage one, then stage two
    monkeypatch.undo()
    with shape_fallback.ninth_deferred() as candidates:  # a proof inside a stage makes no stages of its own
        assert not prove_equivalent_algebraic(left, right, **options).proven
    assert candidates == [True]


def test_no_stage_two_when_no_rule_of_the_batch_would_fire(monkeypatch):
    result, seen = stages_of(
        monkeypatch, "SELECT a FROM t WHERE b LIKE 'x%'", "SELECT a FROM t WHERE b LIKE 'x%' AND LENGTH(b) > 0", schema=SCHEMA, dialect="mysql"
    )
    assert not result.proven and seen and not any(seen)


@pytest.mark.parametrize("sql,options", [(JOINED, {"schema": PATRONS}), (BARE, {"schema": ITEMS, "keys": {"item": [("label",)]}})], ids=["a rule of one select", "the pre-pass"])
def test_stage_one_asks_the_rules_and_changes_nothing(sql, options):
    """Trying a rule where it would fire tells stage one that stage two has work, and leaves the query as it reads with
    the batch off."""

    with shape_fallback.without_all(shape_fallback.NINTH_BATCH):
        off = normalize(sql, dialect="mysql", **options)
    with shape_fallback.ninth_deferred() as seen:
        assert normalize(sql, dialect="mysql", **options) == off
    assert seen == [True]
    assert normalize(sql, dialect="mysql", **options) != off
    with shape_fallback.ninth_deferred() as seen:
        normalize("SELECT a FROM t WHERE a > 1", schema=SCHEMA, dialect="mysql")
    assert seen == []


def test_tenth_batch_rules_fire_only_where_no_older_attempt_is_changed():
    """The same confinement one level further out: on the first attempt and on a narrow attempt that switches off
    rules of the tenth batch only. With a ninth-batch rule off the tenth batch stays out, as that attempt was made
    before it; with tenth-batch rules off the ninth batch stays on."""

    assert shape_fallback.tenth_batch_enabled()
    with shape_fallback.without("counted_sum_ratios"), shape_fallback.without("aggregate_scalars"):
        assert shape_fallback.tenth_batch_enabled() and shape_fallback.ninth_batch_enabled()
    with shape_fallback.without_all(shape_fallback.TENTH_BATCH):
        assert shape_fallback.tenth_batch_enabled() and shape_fallback.ninth_batch_enabled()
        assert not shape_fallback.rule_enabled("shifted_equalities") and shape_fallback.rule_enabled("own_row_copies")
    for older in ("own_row_copies", "own_key_membership"):
        with shape_fallback.without(older):
            assert not shape_fallback.tenth_batch_enabled()
    with shape_fallback.without_all(shape_fallback.NINTH_BATCH):
        assert not shape_fallback.tenth_batch_enabled() and shape_fallback.ninth_batch_enabled()
    with shape_fallback.disabled():
        assert not shape_fallback.tenth_batch_enabled()
    with shape_fallback.late():
        assert not shape_fallback.tenth_batch_enabled()
    assert not shape_fallback.TENTH_BATCH & shape_fallback.NINTH_BATCH


def test_with_the_tenth_batch_off_a_pair_is_read_as_before_it():
    sql = "SELECT ROUND(SUM(CASE WHEN b.paid = b.due - 1 THEN 1 ELSE 0 END) / COUNT(DISTINCT b.id), 2) AS share FROM bill AS b"
    options = {"schema": {"bill": ["id", "paid", "due"]}, "types": {"bill": {"id": "INT", "paid": "INT", "due": "INT"}}, "dialect": "mysql"}
    with shape_fallback.tracking() as fired:
        first = normalize(sql, **options)
    assert {"shifted_equalities", "counted_sum_ratios", "aggregate_scalars"} <= fired
    with shape_fallback.without_all(shape_fallback.TENTH_BATCH), shape_fallback.tracking() as fired:
        before = normalize(sql, **options)
    assert not fired & shape_fallback.TENTH_BATCH and before != first
    with shape_fallback.without("own_key_membership"), shape_fallback.tracking() as fired:
        older = normalize(sql, **options)
    assert not fired & shape_fallback.TENTH_BATCH
    with shape_fallback.without("own_key_membership"), shape_fallback.without_all(shape_fallback.TENTH_BATCH):
        assert normalize(sql, **options) == older


BILLS = {"bill": ["id", "owner", "paid", "due"]}
BILL_OPTIONS = {"schema": BILLS, "types": {"bill": {"id": "INT", "owner": "INT", "paid": "INT", "due": "INT"}}, "compare_names": False, "dialect": "mysql"}
BY_IF = "SELECT owner, SUM(IF(paid = due, 1, 0)) / COUNT(*) AS share FROM bill GROUP BY owner"
SHARE = "SELECT ROUND((SUM(paid = due) / COUNT(*)) * 100, 2) AS share FROM bill"
COUNTED = "SELECT ROUND(COUNT(*) / (SELECT COUNT(*) FROM bill) * 100, 2) AS share FROM bill WHERE paid = due"
SHARE_OPTIONS = {**BILL_OPTIONS, "constraints": {"bill": TableConstraints(not_null=frozenset({"id", "paid", "due"}), keys=(("id",),))}}


def test_the_tenth_stage_switches():
    assert shape_fallback.tenth_batch_enabled() and shape_fallback.stage_key()[:2] == (True, True)  # outside a proof both are on
    with shape_fallback.tenth_deferred() as seen:
        with shape_fallback.ninth_deferred():
            assert shape_fallback.stage_key()[:2] == (False, False) and shape_fallback.tenth_watching()  # stage one
            assert not shape_fallback.tenth_batch_enabled() and not shape_fallback.ninth_batch_enabled()
        with shape_fallback.ninth_on():
            assert shape_fallback.stage_key()[:2] == (True, False) and shape_fallback.tenth_watching()  # stage two
            assert not shape_fallback.tenth_batch_enabled() and shape_fallback.ninth_batch_enabled()
            for older in (shape_fallback.disabled, shape_fallback.late, lambda: shape_fallback.without("own_row_copies")):
                with older():
                    assert not shape_fallback.tenth_watching()  # stage three would not let the batch fire here either
            with shape_fallback.without(worker_rules.NAME):
                assert shape_fallback.tenth_watching()
        shape_fallback.note_tenth_candidate()
        assert seen == [True] and not shape_fallback.tenth_watching()  # one yes is enough
    assert not shape_fallback.tenth_stage_running()
    with shape_fallback.tenth_deferred():
        assert not shape_fallback.tenth_stage_running()
    with shape_fallback.tenth_on():
        assert shape_fallback.stage_key()[:2] == (True, True) and shape_fallback.ninth_staged()  # stage three
        assert shape_fallback.tenth_stage_running()
        assert shape_fallback.tenth_batch_enabled() and shape_fallback.ninth_batch_enabled() and not shape_fallback.tenth_watching()
        with shape_fallback.without("if_cases"), shape_fallback.without(worker_rules.NAME):
            assert shape_fallback.tenth_batch_enabled() and shape_fallback.ninth_batch_enabled()
        with shape_fallback.without("own_row_copies"):
            assert not shape_fallback.tenth_batch_enabled() and shape_fallback.ninth_batch_enabled()
    assert not shape_fallback.ninth_staged() and shape_fallback.stage_key() == (True, True, True, True)


def test_a_pair_only_the_tenth_batch_proves_is_proved_in_stage_three(monkeypatch):
    with shape_fallback.without_all(shape_fallback.TENTH_BATCH):
        assert not prove_equivalent_algebraic(SHARE, COUNTED, **SHARE_OPTIONS).proven
    result, ninth = stages_of(monkeypatch, SHARE, COUNTED, **SHARE_OPTIONS)
    tenth = list(TENTH_SEEN)
    assert result.proven
    first = tenth.index(True)
    assert first > 0 and not any(tenth[:first]) and all(tenth[first:])  # the prover as it was runs to its end first
    assert all(ninth[first:])  # the ninth batch is on beside the tenth


FLAGGED = "SELECT ROUND(100 * SUM(flag) / COUNT(*), 2) AS share FROM (SELECT *, IF(paid = due, 1, 0) AS flag FROM bill) t"
OTHER_ROUNDING = (
    "SELECT ROUND(SUM(IF(paid = due, 1, 0)) / COUNT(*) * 100, 2) AS share, MIN(due) AS first FROM bill",
    "SELECT ROUND(COUNT(*) / (SELECT COUNT(*) FROM bill) * 100, 1) AS share, (SELECT MIN(due) FROM bill) AS first FROM bill WHERE paid = due",
)


def tenth_steps(monkeypatch, left, right, **options):
    """The proof's verdict, and for each step that has the tenth batch on (sub-proofs too) the rules of it switched off."""

    steps = []
    real = algebraic_equivalence._prove_algebraic

    def watched(*args, **kwargs):
        if shape_fallback.tenth_stage_running():
            steps.append(frozenset(shape_fallback.off() & shape_fallback.TENTH_BATCH))
        return real(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_algebraic", watched)
    return prove_equivalent_algebraic(left, right, **options), steps


def test_stage_three_is_two_attempts_and_its_sub_proofs_one(monkeypatch):
    """A pair the batch reads and does not prove costs two attempts, and no retry with another rule of the batch off:
    with a retry for each rule that fired, such a pair went from seconds to minutes (the sub-proofs repeat)."""

    result, steps = tenth_steps(monkeypatch, *OTHER_ROUNDING, **SHARE_OPTIONS)
    assert not result.proven and steps
    assert set(steps) == {frozenset(), frozenset({"aggregate_scalars"})}
    first = [step for step in steps if not step]
    assert steps[: len(first)] == first  # the attempt with every rule on comes first, with its sub-proofs
    assert len(steps) <= 24  # two attempts of at most six steps, and one attempt for each sub-proof


def test_the_second_attempt_of_stage_three_proves_a_pair_the_scalar_reading_hides(monkeypatch):
    with shape_fallback.without_all(shape_fallback.TENTH_BATCH):
        assert not prove_equivalent_algebraic(SHARE, FLAGGED, **SHARE_OPTIONS).proven
    result, steps = tenth_steps(monkeypatch, SHARE, FLAGGED, **SHARE_OPTIONS)
    assert result.proven and steps[-1] == frozenset({"aggregate_scalars"}) and not any(steps[:-1])


def test_a_pair_proved_before_the_tenth_batch_never_sees_it(monkeypatch):
    """A rule of the batch would fire on both queries, and the pair is proved without it: no step has the batch on."""

    with shape_fallback.tracking() as fired:
        normalize(BY_IF, schema=BILLS, dialect="mysql")
    assert "if_cases" in fired
    result, _ = stages_of(monkeypatch, BY_IF, BY_IF.replace("AS share", "AS part"), **BILL_OPTIONS)
    assert result.proven and TENTH_SEEN and not any(TENTH_SEEN)


def test_no_stage_three_when_no_rule_of_the_tenth_batch_would_fire(monkeypatch):
    result, _ = stages_of(monkeypatch, "SELECT a FROM t WHERE b > 1", "SELECT a FROM t WHERE b > 2", schema={"t": ["a", "b"]}, dialect="mysql")
    assert not result.proven and TENTH_SEEN and not any(TENTH_SEEN)


def test_earlier_stages_ask_a_copy_and_change_nothing():
    """The trial of the tenth batch in an earlier stage runs on a copy: the text is what the batch switched off gives."""

    options = {"schema": BILLS, "types": BILL_OPTIONS["types"], "dialect": "mysql"}
    with shape_fallback.without_all(shape_fallback.TENTH_BATCH):
        before = normalize(BY_IF, **options)
    with shape_fallback.tenth_deferred() as seen, shape_fallback.ninth_on():
        assert normalize(BY_IF, **options) == before
    assert seen == [True]
    assert normalize(BY_IF, **options) != before


# The eleventh batch: a fourth stage.
ITEM_SALES = {"sale": ["id", "buyer", "item_id", "paid"], "item": ["item_id", "label"]}
ITEM_SALE_TYPES = {"sale": {"id": "INT", "buyer": "INT", "item_id": "INT", "paid": "INT"}, "item": {"item_id": "INT", "label": "VARCHAR"}}
ITEM_SALE_OPTIONS = {
    "schema": ITEM_SALES, "types": ITEM_SALE_TYPES, "compare_names": False, "dialect": "mysql",
    "constraints": {"sale": TableConstraints(not_null=frozenset({"id", "buyer", "item_id"}), keys=(("id",),)), "item": TableConstraints(not_null=frozenset({"item_id", "label"}), keys=(("item_id",),))},
}
SOLD = "FROM sale AS s JOIN item AS i ON s.item_id = i.item_id"
BY_SUMS = f"SELECT s.buyer {SOLD} GROUP BY s.buyer HAVING SUM(i.label = 'S') > 0 AND SUM(i.label = 'T') = 0"
BY_LIST = f"SELECT DISTINCT s.buyer {SOLD} WHERE i.label = 'S' AND s.buyer NOT IN (SELECT s2.buyer FROM sale AS s2 JOIN item AS i2 ON s2.item_id = i2.item_id WHERE i2.label = 'T')"
JOINED_SHARE = f"SELECT ROUND(SUM(i.label = 'S') / COUNT(*) * 100, 2) AS share {SOLD}"
JOINED_COUNTED = f"SELECT ROUND(COUNT(CASE WHEN i.label = 'S' THEN 1 END) * 100 / COUNT(*), 2) AS share {SOLD}"


def eleventh_steps(monkeypatch, left, right, **options):
    """The proof's verdict, and for each step which of the three staged batches were on in it."""

    steps = []
    real = algebraic_equivalence._prove_algebraic

    def watched(*args, **kwargs):
        steps.append(shape_fallback.stage_key()[:3])
        return real(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_algebraic", watched)
    return prove_equivalent_algebraic(left, right, **options), steps


def test_the_eleventh_stage_switches():
    assert shape_fallback.eleventh_batch_enabled() and shape_fallback.stage_key()[:3] == (True, True, True)  # outside a proof all are on
    with shape_fallback.without("if_cases"):
        assert shape_fallback.eleventh_batch_enabled()  # a rule of the tenth batch off: the stage's second attempt
    for older in (shape_fallback.disabled, shape_fallback.late, lambda: shape_fallback.without("own_row_copies"), lambda: shape_fallback.without("own_key_membership")):
        with older():
            assert not shape_fallback.eleventh_batch_enabled()
    with shape_fallback.eleventh_deferred() as seen:
        with shape_fallback.tenth_deferred(), shape_fallback.ninth_deferred():
            assert shape_fallback.stage_key()[:3] == (False, False, False) and shape_fallback.eleventh_watching() and not shape_fallback.eleventh_batch_enabled()
        with shape_fallback.tenth_on():  # stage three is as it was before the batch
            assert shape_fallback.stage_key()[:3] == (True, True, False) and not shape_fallback.eleventh_batch_enabled() and shape_fallback.tenth_batch_enabled()
            for older in (shape_fallback.disabled, shape_fallback.late, lambda: shape_fallback.without("own_row_copies")):
                with older():
                    assert not shape_fallback.eleventh_watching()  # stage four would not let the batch fire here either
        shape_fallback.note_eleventh_candidate()
        assert seen == [True] and not shape_fallback.eleventh_watching()  # one yes is enough
    assert not shape_fallback.eleventh_stage_running()
    with shape_fallback.eleventh_alone():  # the first attempt of stage four: the prover before the ninth batch, and this batch
        assert shape_fallback.stage_key()[:3] == (False, False, True) and shape_fallback.eleventh_stage_running()
        assert shape_fallback.eleventh_batch_enabled() and not shape_fallback.tenth_batch_enabled() and not shape_fallback.ninth_batch_enabled()
        assert not shape_fallback.ninth_watching() and not shape_fallback.tenth_watching()
    with shape_fallback.eleventh_on():
        assert shape_fallback.stage_key()[:3] == (True, True, True) and shape_fallback.eleventh_stage_running() and shape_fallback.tenth_stage_running()
        assert shape_fallback.eleventh_batch_enabled() and shape_fallback.tenth_batch_enabled() and shape_fallback.ninth_batch_enabled()
        with shape_fallback.without("aggregate_scalars"):
            assert shape_fallback.eleventh_batch_enabled() and shape_fallback.tenth_batch_enabled()
    assert not shape_fallback.ninth_staged() and shape_fallback.stage_key() == (True, True, True, True)
    assert not shape_fallback.ELEVENTH_BATCH & (shape_fallback.TENTH_BATCH | shape_fallback.NINTH_BATCH)


def test_a_pair_the_eleventh_batch_proves_alone_is_proved_by_the_first_attempt_of_stage_four(monkeypatch):
    with shape_fallback.without_all(shape_fallback.ELEVENTH_BATCH):
        assert not prove_equivalent_algebraic(BY_SUMS, BY_LIST, **ITEM_SALE_OPTIONS).proven
    result, steps = eleventh_steps(monkeypatch, BY_SUMS, BY_LIST, **ITEM_SALE_OPTIONS)
    assert result.proven
    first = next(index for index, step in enumerate(steps) if step[2])
    assert first > 0 and not any(step[2] for step in steps[:first])  # the prover as it was runs to its end first
    assert all(step == (False, False, True) for step in steps[first:])  # then the batch beside the prover before the ninth


def test_a_pair_that_needs_the_tenth_batch_too_is_proved_by_the_later_attempts_of_stage_four(monkeypatch):
    with shape_fallback.without_all(shape_fallback.ELEVENTH_BATCH):
        assert not prove_equivalent_algebraic(JOINED_SHARE, JOINED_COUNTED, **ITEM_SALE_OPTIONS).proven
    result, steps = eleventh_steps(monkeypatch, JOINED_SHARE, JOINED_COUNTED, **ITEM_SALE_OPTIONS)
    assert result.proven and steps[-1] == (True, True, True)
    assert (False, False, True) in steps and steps.index((False, False, True)) < steps.index((True, True, True))


def test_a_pair_proved_before_the_eleventh_batch_never_sees_it(monkeypatch):
    """A rule of the batch would fire on both queries, and the pair is proved without it: no step has the batch on."""

    with shape_fallback.tracking() as fired:
        normalize(BY_SUMS, schema=ITEM_SALES, dialect="mysql", types=ITEM_SALE_TYPES)
    assert "boolean_sums" in fired
    result, steps = eleventh_steps(monkeypatch, BY_SUMS, BY_SUMS.replace("s.buyer", "s.buyer AS b", 1), **ITEM_SALE_OPTIONS)
    assert result.proven and steps and not any(step[2] for step in steps)


def test_no_stage_four_when_no_rule_of_the_eleventh_batch_would_fire(monkeypatch):
    result, steps = eleventh_steps(monkeypatch, "SELECT a FROM t WHERE b > 1", "SELECT a FROM t WHERE b > 2", schema={"t": ["a", "b"]}, dialect="mysql")
    assert not result.proven and steps and not any(step[2] for step in steps)


def test_a_pair_the_eleventh_batch_reads_and_does_not_prove_costs_three_attempts_at_most(monkeypatch):
    other = BY_LIST.replace("i2.label = 'T'", "i2.label = 'U'")
    result, steps = eleventh_steps(monkeypatch, BY_SUMS, other, **ITEM_SALE_OPTIONS)
    assert not result.proven
    fourth = [step for step in steps if step[2]]
    assert fourth and set(fourth) <= {(False, False, True), (True, True, True)}


def test_earlier_stages_ask_the_eleventh_batch_and_change_nothing():
    options = {"schema": ITEM_SALES, "types": ITEM_SALE_TYPES, "dialect": "mysql", "not_null": {"sale": frozenset({"buyer", "item_id"}), "item": frozenset({"item_id", "label"})}}
    for sql in (BY_SUMS, f"SELECT s.item_id, COUNT(*) AS n {SOLD} GROUP BY item_id", f"SELECT SUM(s.buyer + i.item_id) AS n {SOLD}"):
        with shape_fallback.without_all(shape_fallback.ELEVENTH_BATCH), shape_fallback.without_all(shape_fallback.TENTH_BATCH):
            try:
                before = normalize(sql, **options)
            except Exception as error:  # a query the prover declines without the batch
                before = str(error)
        with shape_fallback.eleventh_deferred() as seen, shape_fallback.tenth_deferred(), shape_fallback.ninth_on():
            try:
                now = normalize(sql, **options)
            except Exception as error:
                now = str(error)
        assert now == before and seen == [True], sql
        with shape_fallback.tracking() as fired:
            normalize(sql, **options)
        assert fired & shape_fallback.ELEVENTH_BATCH, sql
    with shape_fallback.eleventh_deferred() as seen:
        normalize("SELECT a FROM t WHERE a > 1", schema=SCHEMA, dialect="mysql")
    assert seen == []


# The twelfth batch: a fifth stage.
DATED_SALES = {"sale": ["id", "buyer", "item_id", "paid", "sold"], "item": ["item_id", "label"]}
DATED_SALE_TYPES = {"sale": {"id": "INT", "buyer": "INT", "item_id": "INT", "paid": "INT", "sold": "DATE"}, "item": {"item_id": "INT", "label": "VARCHAR"}}
DATED_SALE_OPTIONS = {**ITEM_SALE_OPTIONS, "schema": DATED_SALES, "types": DATED_SALE_TYPES}
IN_JULY = "SELECT s.buyer, COUNT(DISTINCT s.item_id) AS items FROM sale AS s WHERE s.sold BETWEEN '2019-07-01' AND '2019-07-31' GROUP BY s.buyer"
IN_JULY_COUNTED = "SELECT s.buyer, COUNT(DISTINCT s.item_id) AS items FROM sale AS s WHERE DATEDIFF('2019-07-31', s.sold) BETWEEN 0 AND 30 GROUP BY s.buyer"
JULY_SHARE = f"SELECT ROUND(SUM(i.label = 'S') / COUNT(*) * 100, 2) AS share {SOLD} WHERE s.sold > '2019-06-30'"
JULY_COUNTED = f"SELECT ROUND(COUNT(CASE WHEN i.label = 'S' THEN 1 END) * 100 / COUNT(*), 2) AS share {SOLD} WHERE s.sold >= DATE_ADD('2019-06-30', INTERVAL 1 DAY)"


def twelfth_steps(monkeypatch, left, right, **options):
    """The proof's verdict, and for each step which of the four staged batches were on in it."""

    steps = []
    real = algebraic_equivalence._prove_algebraic

    def watched(*args, **kwargs):
        steps.append(shape_fallback.stage_key())
        return real(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_algebraic", watched)
    return prove_equivalent_algebraic(left, right, **options), steps


def test_the_twelfth_stage_switches():
    assert shape_fallback.twelfth_batch_enabled() and shape_fallback.stage_key() == (True, True, True, True)  # outside a proof all are on
    with shape_fallback.without("if_cases"):
        assert shape_fallback.twelfth_batch_enabled()  # a rule of the tenth batch off: the stage's last attempt
    for older in (shape_fallback.disabled, shape_fallback.late, lambda: shape_fallback.without("own_row_copies"), lambda: shape_fallback.without("boolean_sums")):
        with older():
            assert not shape_fallback.twelfth_batch_enabled()
    with shape_fallback.twelfth_deferred() as seen:
        with shape_fallback.eleventh_deferred(), shape_fallback.tenth_deferred(), shape_fallback.ninth_deferred():
            assert shape_fallback.stage_key() == (False, False, False, False) and shape_fallback.twelfth_watching() and not shape_fallback.twelfth_batch_enabled()
        with shape_fallback.eleventh_alone():  # stage four is as it was before the batch
            assert shape_fallback.stage_key() == (False, False, True, False) and shape_fallback.eleventh_batch_enabled() and not shape_fallback.twelfth_batch_enabled()
        with shape_fallback.eleventh_on():
            assert shape_fallback.stage_key() == (True, True, True, False) and shape_fallback.eleventh_batch_enabled() and not shape_fallback.twelfth_batch_enabled()
            for older in (shape_fallback.disabled, shape_fallback.late, lambda: shape_fallback.without("own_row_copies")):
                with older():
                    assert not shape_fallback.twelfth_watching()  # stage five would not let the batch fire here either
        shape_fallback.note_twelfth_candidate()
        assert seen == [True] and not shape_fallback.twelfth_watching()  # one yes is enough
    assert not shape_fallback.twelfth_stage_running()
    with shape_fallback.twelfth_alone():  # the first attempt of stage five: the prover before the ninth batch, and this batch
        assert shape_fallback.stage_key() == (False, False, False, True) and shape_fallback.twelfth_stage_running()
        assert shape_fallback.twelfth_batch_enabled() and not shape_fallback.eleventh_batch_enabled() and not shape_fallback.tenth_batch_enabled() and not shape_fallback.ninth_batch_enabled()
        assert not shape_fallback.ninth_watching() and not shape_fallback.tenth_watching() and not shape_fallback.eleventh_watching()
    with shape_fallback.twelfth_on():
        assert shape_fallback.stage_key() == (True, True, True, True) and shape_fallback.twelfth_stage_running()
        assert shape_fallback.twelfth_batch_enabled() and shape_fallback.eleventh_batch_enabled() and shape_fallback.tenth_batch_enabled() and shape_fallback.ninth_batch_enabled()
        with shape_fallback.without("aggregate_scalars"):
            assert shape_fallback.twelfth_batch_enabled() and shape_fallback.eleventh_batch_enabled()
    assert not shape_fallback.ninth_staged() and shape_fallback.stage_key() == (True, True, True, True)
    assert not shape_fallback.TWELFTH_BATCH & (shape_fallback.ELEVENTH_BATCH | shape_fallback.TENTH_BATCH | shape_fallback.NINTH_BATCH)


def test_a_pair_the_twelfth_batch_proves_alone_is_proved_by_the_first_attempt_of_stage_five(monkeypatch):
    with shape_fallback.without_all(shape_fallback.TWELFTH_BATCH):
        assert not prove_equivalent_algebraic(IN_JULY, IN_JULY_COUNTED, **DATED_SALE_OPTIONS).proven
    result, steps = twelfth_steps(monkeypatch, IN_JULY, IN_JULY_COUNTED, **DATED_SALE_OPTIONS)
    assert result.proven
    first = next(index for index, step in enumerate(steps) if step[3])
    assert first > 0 and not any(step[3] for step in steps[:first])  # the prover as it was runs to its end first
    assert all(step == (False, False, False, True) for step in steps[first:])  # then the batch beside the prover before the ninth


def test_a_pair_that_needs_older_batches_too_is_proved_by_the_later_attempts_of_stage_five(monkeypatch):
    with shape_fallback.without_all(shape_fallback.TWELFTH_BATCH):
        assert not prove_equivalent_algebraic(JULY_SHARE, JULY_COUNTED, **DATED_SALE_OPTIONS).proven
    result, steps = twelfth_steps(monkeypatch, JULY_SHARE, JULY_COUNTED, **DATED_SALE_OPTIONS)
    assert result.proven and steps[-1] == (True, True, True, True)
    assert (False, False, False, True) in steps and steps.index((False, False, False, True)) < steps.index((True, True, True, True))


def test_a_pair_proved_before_the_twelfth_batch_never_sees_it(monkeypatch):
    """The rule would fire on both queries, and the pair is proved without it: no step has the batch on."""

    with shape_fallback.tracking() as fired:
        normalize(IN_JULY_COUNTED, schema=DATED_SALES, dialect="mysql", types=DATED_SALE_TYPES)
    assert "date_windows" in fired
    result, steps = twelfth_steps(monkeypatch, IN_JULY_COUNTED, IN_JULY_COUNTED.replace("s.buyer", "s.buyer AS b", 1), **DATED_SALE_OPTIONS)
    assert result.proven and steps and not any(step[3] for step in steps)


def test_no_stage_five_when_the_rule_of_the_twelfth_batch_would_not_fire(monkeypatch):
    result, steps = twelfth_steps(monkeypatch, "SELECT id FROM sale WHERE paid > 1", "SELECT id FROM sale WHERE paid > 2", **DATED_SALE_OPTIONS)
    assert not result.proven and steps and not any(step[3] for step in steps)
    # a date written out already is nothing to rewrite
    result, steps = twelfth_steps(monkeypatch, "SELECT id FROM sale WHERE sold >= '2019-07-01'", "SELECT id FROM sale WHERE sold >= '2019-07-02'", **DATED_SALE_OPTIONS)
    assert not result.proven and steps and not any(step[3] for step in steps)


def test_a_pair_the_twelfth_batch_reads_and_does_not_prove_costs_three_attempts_at_most(monkeypatch):
    other = IN_JULY_COUNTED.replace("BETWEEN 0 AND 30", "BETWEEN 0 AND 29")
    result, steps = twelfth_steps(monkeypatch, IN_JULY, other, **DATED_SALE_OPTIONS)
    assert not result.proven
    fifth = [step for step in steps if step[3]]
    assert fifth and set(fifth) <= {(False, False, False, True), (True, True, True, True)}


def test_earlier_stages_ask_the_twelfth_batch_and_change_nothing():
    options = {"schema": DATED_SALES, "types": DATED_SALE_TYPES, "dialect": "mysql"}
    with shape_fallback.without_all(shape_fallback.TWELFTH_BATCH):
        before = normalize(IN_JULY_COUNTED, **options)
    with shape_fallback.twelfth_deferred() as seen, shape_fallback.eleventh_on():
        assert normalize(IN_JULY_COUNTED, **options) == before
    assert seen == [True]
    assert normalize(IN_JULY_COUNTED, **options) != before
    with shape_fallback.twelfth_deferred() as seen:
        normalize("SELECT id FROM sale WHERE paid > 1", **options)
    assert seen == []
