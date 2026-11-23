"""The twenty-fourth batch is one more step of a proof, after every other (``shape_fallback.TWENTY_FOURTH_BATCH``)."""

import re

import pytest

from kumo_sql_prover import algebraic_equivalence, distinct_window_groups, shape_fallback, shared_scan_aggregates
from kumo_sql_prover.algebraic_equivalence import normalize, prove_equivalent_algebraic

SCHEMA = {"projects": ["team", "person"], "staff": ["person", "years"], "people": ["id", "mail"], "visits": ["visit", "guest", "booked", "came"]}
TYPES = {
    "projects": {"team": "INT", "person": "INT"},
    "staff": {"person": "INT", "years": "INT"},
    "people": {"id": "INT", "mail": "VARCHAR"},
    "visits": {"visit": "INT", "guest": "INT", "booked": "INT", "came": "INT"},
}
OPTIONS = {"schema": SCHEMA, "keys": {}, "not_null": {}, "types": TYPES, "dialect": "mysql"}

JOINED = "FROM projects AS p JOIN staff AS e ON p.person = e.person"
GROUPED = f"SELECT p.team, ROUND(AVG(e.years), 2) AS mean {JOINED} GROUP BY p.team"
WINDOWED = f"SELECT DISTINCT p.team, ROUND(AVG(e.years) OVER (PARTITION BY p.team), 2) AS mean {JOINED}"
REPEATED = "SELECT mail FROM people GROUP BY mail HAVING COUNT(*) > 1"
REPEATED_WINDOWED = "SELECT DISTINCT mail FROM (SELECT mail, COUNT(*) OVER (PARTITION BY mail) AS n FROM people) AS t WHERE n > 1"
REPEATED_WITH = "WITH counted AS (SELECT mail, COUNT(*) OVER (PARTITION BY mail) AS n FROM people) SELECT DISTINCT mail FROM counted WHERE n >= 2"
TOTALS = f"SELECT p.team, SUM(e.years) AS total, COUNT(*) AS n {JOINED} WHERE e.years > 0 GROUP BY p.team"
TOTALS_WINDOWED = f"SELECT DISTINCT p.team, SUM(e.years) OVER (PARTITION BY p.team) AS total, COUNT(*) OVER (PARTITION BY p.team) AS n {JOINED} WHERE e.years > 0"

# shared_scan_aggregates: a share with one of its counts asked apart
SHARE = "SELECT ROUND(COUNT(CASE WHEN booked = came THEN 1 END) / COUNT(*) * 100, 2) AS share FROM visits"
SHARE_SUBQUERY = "SELECT ROUND(SUM(CASE WHEN booked = came THEN 1 ELSE 0 END) / (SELECT COUNT(*) FROM visits) * 100, 2) AS share FROM visits"
SHARE_COPY = "SELECT ROUND(COUNT(*) / (SELECT COUNT(*) FROM visits) * 100, 2) AS share FROM (SELECT * FROM visits WHERE booked = came) AS same"
COUNTED = "SELECT CASE WHEN COUNT(*) > 0 THEN COUNT(*) END AS n, COUNT(*) AS found FROM visits"
COUNTED_JOINED = "SELECT x.n AS n, COUNT(*) AS found FROM visits AS v, (SELECT COUNT(*) AS n FROM visits) AS x"
COUNTED_WITH = "WITH x AS (SELECT COUNT(*) AS n FROM visits) SELECT x.n AS n, COUNT(*) AS found FROM visits, x"

PAIRS = [
    (GROUPED, WINDOWED),
    (REPEATED, REPEATED_WINDOWED),
    (REPEATED, REPEATED_WITH),
    (TOTALS, TOTALS_WINDOWED),
    (REPEATED_WINDOWED, REPEATED_WITH),
    (SHARE, SHARE_SUBQUERY),
    (SHARE, SHARE_COPY),
    (COUNTED, COUNTED_JOINED),
    (COUNTED, COUNTED_WITH),
]


def prove(left: str, right: str, **more):
    return prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, compare_names=False, dialect="mysql", exact_arithmetic=True, timeout_ms=5000, **more)


def switch_off(monkeypatch) -> None:
    """Take the rule out and leave every other rule and step as it is (``shape_fallback.without`` also changes the
    steps before)."""

    monkeypatch.setattr(distinct_window_groups, "read_distinct_window_groups", lambda tree: (tree, False))
    monkeypatch.setattr(shared_scan_aggregates, "read_shared_scans", lambda tree, schema, dialect=None: (tree, False))


def test_the_batch_names_its_rules_and_has_its_own_entry_in_the_stage_key():
    assert shape_fallback.TWENTY_FOURTH_BATCH == frozenset({"distinct_window_groups", "shared_scan_aggregates"})
    assert len(shape_fallback.stage_key()) == 11 and shape_fallback.stage_key()[10] is False
    with shape_fallback.twenty_fourth_deferred():
        assert shape_fallback.stage_key()[10] is False
    with shape_fallback.twentieth_on():
        assert shape_fallback.stage_key()[5:] == (True, True, True, True, True, False)  # the step before leaves this one off
    with shape_fallback.twenty_fourth_on():
        assert shape_fallback.stage_key()[5:] == (True, True, True, True, True, True)  # the step runs with the batches before it on too
    assert shape_fallback.stage_key()[10] is False


def normal(sql: str) -> str:
    """The normal form, with the numbers of the names the normalizer makes up taken out (they count up per call)."""

    return re.sub(r"\bkq([a-z]+)\d+", r"kq\1", normalize(sql, **OPTIONS))


def test_the_rule_fires_only_in_the_last_step_of_a_proof():
    sql = WINDOWED
    assert not shape_fallback.twenty_fourth_batch_enabled()
    with shape_fallback.tracking() as fired:
        outside = normal(sql)  # outside a proof the rule is off
    assert not fired & shape_fallback.TWENTY_FOURTH_BATCH
    with shape_fallback.twenty_fourth_deferred() as seen:
        assert shape_fallback.twenty_fourth_watching() and not shape_fallback.twenty_fourth_batch_enabled()
        assert normal(sql) == outside  # an earlier step asks the rule and changes nothing
        assert seen == [True] and not shape_fallback.twenty_fourth_watching()
    with shape_fallback.twenty_fourth_deferred() as seen:
        normal(GROUPED)  # nothing to read
        with shape_fallback.disabled():
            normal(sql)
        assert seen == []
    with shape_fallback.twenty_fourth_deferred(watch=False) as seen:
        assert seen is None and not shape_fallback.twenty_fourth_watching()
        assert normal(sql) == outside
    with shape_fallback.twenty_fourth_on(), shape_fallback.tracking() as fired:
        assert shape_fallback.twenty_fourth_batch_enabled() and shape_fallback.twentieth_batch_enabled() and shape_fallback.eighteenth_batch_enabled()
        assert normal(sql) != outside and normal(sql) == normal(GROUPED)
        with shape_fallback.without("distinct_window_groups"), shape_fallback.tracking() as inner:
            assert normal(sql) == outside
        assert "distinct_window_groups" not in inner
        for off in (shape_fallback.disabled, shape_fallback.late):
            with off():
                assert not shape_fallback.twenty_fourth_batch_enabled()
    assert "distinct_window_groups" in fired
    assert not shape_fallback.twenty_fourth_batch_enabled() and not shape_fallback.twenty_fourth_watching()


def runs_of(monkeypatch) -> list:
    runs: list = []
    whole = algebraic_equivalence._prove_equivalent_algebraic_whole

    def counted(*args, **kwargs):
        runs.append((shape_fallback.twentieth_batch_enabled(), shape_fallback.twenty_fourth_batch_enabled()))
        return whole(*args, **kwargs)

    monkeypatch.setattr(algebraic_equivalence, "_prove_equivalent_algebraic_whole", counted)
    return runs


@pytest.mark.parametrize("position", range(len(PAIRS)))
def test_a_pair_is_proved_in_one_more_run_and_only_with_the_rule(monkeypatch, position):
    left, right = PAIRS[position]
    runs = runs_of(monkeypatch)
    assert prove(left, right).proven
    # every run before the last has the batch off; the proofs inside the last run are part of it
    first = runs.index((True, True))
    assert first > 0 and not any(last for _, last in runs[:first]) and all(run == (True, True) for run in runs[first:])
    with monkeypatch.context() as patch:
        switch_off(patch)
        assert not prove(left, right).proven
    assert prove(left, right).proven


def test_pairs_that_differ_stay_unproved():
    # without DISTINCT every row of a team shows the team's mean
    assert not prove(GROUPED, WINDOWED.replace("DISTINCT ", "")).proven
    # the mean of another partition
    assert not prove(GROUPED, WINDOWED.replace("PARTITION BY p.team", "PARTITION BY p.person")).proven
    # a count of the mails that are not NULL is not a count of rows
    assert not prove("SELECT mail FROM people GROUP BY mail HAVING COUNT(id) > 1", REPEATED_WINDOWED).proven
    # another bound
    assert not prove(REPEATED, REPEATED_WINDOWED.replace("n > 1", "n > 2")).proven
    # the filter is part of the partition's rows
    assert not prove(TOTALS, TOTALS_WINDOWED.replace(" WHERE e.years > 0", "")).proven
    # a total per team shown without the team: two teams with one total are one row
    assert not prove(f"SELECT SUM(e.years) AS total {JOINED} GROUP BY p.team", f"SELECT DISTINCT SUM(e.years) OVER (PARTITION BY p.team) AS total {JOINED}").proven
    # a share of the visits of some guests only
    assert not prove(SHARE, SHARE_SUBQUERY.replace("FROM visits)", "FROM visits WHERE guest > 1)")).proven
    # another test in the copy
    assert not prove(SHARE, SHARE_COPY.replace("booked = came", "booked < came")).proven
    # over no rows the joined count is NULL, not 0
    assert not prove("SELECT COUNT(*) AS n, COUNT(*) AS found FROM visits", COUNTED_JOINED).proven
    assert not prove(COUNTED, COUNTED_JOINED.replace("AS v,", "AS v CROSS JOIN").replace("FROM visits)", "FROM visits WHERE guest > 1)")).proven
    # and with the DISTINCT on both sides they are the same totals
    assert prove(f"SELECT DISTINCT SUM(e.years) AS total {JOINED} GROUP BY p.team", f"SELECT DISTINCT SUM(e.years) OVER (PARTITION BY p.team) AS total {JOINED}").proven


def test_a_pair_the_rule_does_not_read_and_a_pair_proved_before_get_no_run_of_the_step(monkeypatch):
    runs = runs_of(monkeypatch)
    assert not prove("SELECT mail FROM people WHERE id > 2", "SELECT mail FROM people WHERE id > 3").proven
    assert runs and not any(last for _, last in runs)
    runs.clear()
    assert prove(WINDOWED, WINDOWED.replace("AS e", "AS s").replace("e.", "s.")).proven  # proved without the rule
    assert runs and not any(last for _, last in runs)
    runs.clear()
    assert not prove(GROUPED, WINDOWED.replace("AVG(e.years)", "MAX(e.years)")).proven  # read, and still different
    assert any(last for _, last in runs)


def test_a_pair_over_the_size_limit_does_not_get_the_step(monkeypatch):
    assert shape_fallback.last_steps_worth_trying(GROUPED, WINDOWED)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(GROUPED) + len(WINDOWED) - 1)
    runs = runs_of(monkeypatch)
    assert not prove(GROUPED, WINDOWED).proven
    assert runs and not any(last for _, last in runs)
    monkeypatch.setattr(shape_fallback, "MAX_CHARS", len(GROUPED) + len(WINDOWED))
    assert prove(GROUPED, WINDOWED).proven


def test_the_step_gives_proofs_only(monkeypatch):
    # The run of the step looks for no counterexample: a pair it does not prove keeps the answer of the steps before.
    different = WINDOWED.replace("AVG(e.years)", "MAX(e.years)")
    after = prove(GROUPED, different)
    switch_off(monkeypatch)
    before = prove(GROUPED, different)
    assert not after.proven and after.status == before.status and after.reason == before.reason
