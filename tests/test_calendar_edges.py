"""A date at either end of the calendar never raises (hardest-cases batch 31).

Python's dates run from 0001-01-01 to 9999-12-31 and adding a day past either end raises ``OverflowError``. Five
places took "the day before and the day after" of a date the queries name without that in mind: the counterexample
search of the LeetCode harness stopped a whole run on one pair with such a date, and the provers themselves raised
on ``DATE_ADD(DATE '9999-12-31', INTERVAL 1 DAY)`` or on a text column compared with ``'9999-12-31'``, although
input they cannot read is "not proven", never an exception.
"""

from __future__ import annotations

import datetime

import pytest

pytest.importorskip("z3")
pytest.importorskip("duckdb")

from kumo_sql_prover import counterexample as cx  # noqa: E402
from kumo_sql_prover import targeted_data  # noqa: E402
from kumo_sql_prover.algebraic_equivalence import _shift, prove_equivalent_algebraic  # noqa: E402
from kumo_sql_prover.smt_equivalence import SmtStatus, prove_equivalent_smt  # noqa: E402

SCHEMA = {"t": ["id", "d", "s", "ts"]}
TYPES = {"t": {"id": "INT64", "d": "DATE", "s": "STRING", "ts": "TIMESTAMP"}}
ENDS = ("9999-12-31", "0001-01-01", "9999-12-30", "0001-01-02")


def _pairs(end: str, dialect: str) -> list[tuple[str, str]]:
    day = f"'{end}'" if dialect in ("mysql", "sqlite") else f"DATE '{end}'"
    if dialect in ("bigquery", "mysql"):
        later, earlier = f"DATE_ADD({day}, INTERVAL 1 DAY)", f"DATE_SUB({day}, INTERVAL 2 WEEK)"
    else:
        later, earlier = f"{day} + INTERVAL 1 DAY", f"{day} - INTERVAL 2 WEEK"
    return [
        (f"SELECT id FROM t WHERE d < {later}", f"SELECT id FROM t WHERE d <= {day}"),
        (f"SELECT id FROM t WHERE d > {earlier}", "SELECT id FROM t"),
        (f"SELECT id FROM t WHERE d = {day}", f"SELECT id FROM t WHERE d >= {day} AND id > 0"),
        (f"SELECT id FROM t WHERE s = '{end}'", f"SELECT id FROM t WHERE s >= '{end}' AND id > 0"),
        (f"SELECT id FROM t WHERE s BETWEEN '{end}' AND '{end}'", f"SELECT id FROM t WHERE s > '{end}'"),
        (f"SELECT id FROM t WHERE ts >= '{end} 00:00:00'", f"SELECT id FROM t WHERE ts > '{end} 23:59:59'"),
    ]


@pytest.mark.parametrize("end", ENDS)
@pytest.mark.parametrize("dialect", ["bigquery", "mysql", "duckdb"])
def test_the_provers_give_a_verdict_for_a_date_at_the_end_of_the_calendar(end, dialect):
    for left, right in _pairs(end, dialect):
        options = dict(schema=SCHEMA, types=TYPES, dialect=dialect)
        for result in (
            prove_equivalent_algebraic(left, right, search_counterexample=True, **options),
            prove_equivalent_smt(left, right, **options),
        ):
            assert isinstance(result.status, SmtStatus), (left, right)


def test_the_true_pairs_near_the_ends_are_still_proved():
    # one day inside the calendar the shifted date exists: `d < day + 1` is `d <= day` for a DATE column
    for dialect, day in (("mysql", "'9999-12-30'"), ("bigquery", "DATE '0001-01-02'")):
        left, right = f"SELECT id FROM t WHERE d < DATE_ADD({day}, INTERVAL 1 DAY)", f"SELECT id FROM t WHERE d <= {day}"
        assert prove_equivalent_algebraic(left, right, schema=SCHEMA, types=TYPES, dialect=dialect).proven
    # and a true difference at the very end is still found
    differ = prove_equivalent_algebraic(
        "SELECT id FROM t WHERE d = DATE '9999-12-31'", "SELECT id FROM t WHERE d >= DATE '9999-12-31' AND id > 0",
        schema=SCHEMA, types=TYPES, dialect="bigquery", search_counterexample=True,
    )
    assert differ.status is SmtStatus.NOT_EQUIVALENT


def test_a_shift_past_the_calendar_leaves_the_call_as_written():
    last, first = datetime.date.max, datetime.date.min
    assert _shift(last, 1, "DAY") is None and _shift(last, 1, "WEEK") is None
    assert _shift(first, -1, "DAY") is None and _shift(first, -3, "WEEK") is None
    assert _shift(last, -1, "DAY") == datetime.date(9999, 12, 30) and _shift(first, 1, "WEEK") == datetime.date(1, 1, 8)


def test_the_neighbours_of_the_first_and_the_last_day():
    for text in ("9999-12-31", "0001-01-01"):
        assert targeted_data._neighbours(text, "STRING") == [text]
        day = datetime.date.fromisoformat(text)
        assert targeted_data._neighbours(day, "DATE") == [day]
    assert targeted_data._neighbours(datetime.datetime(9999, 12, 31, 23, 30), "TIMESTAMP") == [datetime.datetime(9999, 12, 31, 23, 30)]
    assert targeted_data._neighbours("2024-03-01", "STRING") == ["2024-02-29", "2024-03-01", "2024-03-02"]


@pytest.mark.parametrize("end, neighbour", [("9999-12-31", "9999-12-30"), ("0001-01-01", "0001-01-02")])
def test_the_counterexample_search_runs_on_a_pair_that_names_such_a_date(end, neighbour):
    spec = cx.Spec({"EVENTS": cx.Table("EVENTS", [cx.Column("ID", "INT", not_null=True), cx.Column("DAY", "DATE")], primary_key=("ID",))})
    searcher = cx.Searcher(spec, f"SELECT ID FROM EVENTS WHERE DAY = '{end}'", f"SELECT ID FROM EVENTS WHERE DAY >= '{end}' AND DAY <= '{end}' AND ID > 1")
    domain = cx._domain(spec.tables["EVENTS"].column("DAY"), searcher.constants)
    assert end in domain and neighbour in domain
    assert searcher.search(200, seed=1) is not None  # the row with ID 1 on that day
