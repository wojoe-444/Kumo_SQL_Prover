"""Whether a pair of queries uses dates and times for nothing but their order (``tools/bench_order_only.py``)."""

from __future__ import annotations

import datetime
import random
import sys
from pathlib import Path

import pytest
from sqlglot import parse_one

duckdb = pytest.importorskip("duckdb")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from bench_order_only import TEMPORAL_TYPES, compared_only  # noqa: E402

KINDS = {
    ("STAYS", "GUEST"): "INT",
    ("STAYS", "ROOM"): "INT",
    ("STAYS", "ARRIVAL"): "DATE",
    ("STAYS", "DEPARTURE"): "DATE",
    ("CLEANINGS", "ROOM"): "INT",
    ("CLEANINGS", "DAY"): "DATE",
    ("CLEANINGS", "MINUTES"): "INT",
}

# queries that only compare, count, pick and select the dates
ORDER_ONLY = [
    "SELECT DISTINCT a.guest FROM stays AS a JOIN stays AS b ON a.guest = b.guest AND a.room <> b.room AND b.arrival BETWEEN a.arrival AND a.departure",
    "SELECT DISTINCT a.guest FROM stays AS a, stays AS b WHERE a.guest = b.guest AND a.room < b.room AND GREATEST(a.arrival, b.arrival) <= LEAST(a.departure, b.departure)",
    "SELECT guest, MIN(arrival) AS first_day, MAX(departure), COUNT(DISTINCT arrival), COUNT(departure) FROM stays WHERE arrival IS NOT NULL GROUP BY guest",
    "SELECT d.guest FROM (SELECT guest, MIN(arrival) AS first_day, MAX(departure) AS last_day FROM stays GROUP BY guest) AS d WHERE d.first_day <> d.last_day",
    "WITH firsts AS (SELECT guest, MIN(arrival) AS first_day FROM stays GROUP BY guest) SELECT s.guest, s.room FROM stays AS s JOIN firsts AS f ON f.guest = s.guest AND s.arrival = f.first_day",
    "SELECT s.guest FROM stays AS s WHERE s.arrival <= (SELECT MAX(c.day) FROM cleanings AS c WHERE c.room = s.room)",
    "SELECT s.guest FROM stays AS s WHERE s.arrival IN (SELECT c.day FROM cleanings AS c) OR s.departure IN (s.arrival, (SELECT MIN(c.day) FROM cleanings AS c))",
    "SELECT s.guest FROM stays AS s WHERE EXISTS (SELECT c.day FROM cleanings AS c WHERE c.day BETWEEN s.arrival AND s.departure AND c.room = s.room)",
    "SELECT d.guest FROM (SELECT *, LEAD(arrival) OVER (PARTITION BY guest ORDER BY arrival, room) AS next_arrival FROM stays) AS d WHERE d.next_arrival < d.departure",
    "SELECT guest, arrival, ROW_NUMBER() OVER (PARTITION BY arrival ORDER BY departure, guest, room) AS n FROM stays",
    "SELECT arrival AS day FROM stays UNION SELECT day FROM cleanings",
    "SELECT guest, COALESCE(departure, arrival) AS last_known FROM stays ORDER BY arrival, guest",
    "SELECT guest FROM stays GROUP BY guest, arrival HAVING COUNT(*) > 1 AND MIN(departure) = MAX(departure)",
    "SELECT guest, room, arrival <=> departure AS same_day, departure IS NULL AS open_stay FROM stays",
    "SELECT guest FROM stays WHERE minutes_of(room) > 3",  # a function of other columns is not this module's business
]

# one use the prover's numbers would not follow, each
OTHER_USES = {
    "arithmetic": "SELECT guest FROM stays WHERE arrival + 1 = departure",
    "a difference of dates": "SELECT guest FROM stays WHERE DATEDIFF(departure, arrival) > 2",
    "a date function": "SELECT guest FROM stays WHERE YEAR(arrival) = 2024",
    "a text written out": "SELECT guest FROM stays WHERE arrival < '2024-03-01'",
    "a text in a list": "SELECT guest FROM stays WHERE arrival IN ('2024-03-01')",
    "a number": "SELECT guest FROM stays WHERE arrival > 20240301",
    "a column of another type": "SELECT guest FROM stays WHERE arrival = room",
    "another type from a subquery": "SELECT guest FROM stays WHERE arrival > (SELECT MAX(minutes) FROM cleanings)",
    "the date as a condition": "SELECT guest FROM stays WHERE arrival",
    "IS TRUE": "SELECT guest FROM stays WHERE arrival IS TRUE",
    "a pattern": "SELECT guest FROM stays WHERE arrival LIKE '2024%'",
    "a text made of it": "SELECT CONCAT(arrival, '') FROM stays",
    "a cast": "SELECT CAST(arrival AS CHAR) FROM stays",
    "a sum": "SELECT SUM(arrival) FROM stays",
    "an average": "SELECT AVG(departure) FROM stays",
    "a choice the reader does not follow": "SELECT IF(room > 1, arrival, departure) FROM stays",
    "a fallback of another type": "SELECT COALESCE(departure, '2099-01-01') FROM stays",
    "a neighbour with a default": "SELECT LAG(arrival, 1, 0) OVER (ORDER BY arrival) FROM stays",
    "a named value used as a number": "SELECT d.first_day + 1 FROM (SELECT MIN(arrival) AS first_day FROM stays) AS d",
    "a value without a name in a derived table": "SELECT * FROM (SELECT MAX(arrival) FROM stays) AS d",
    "a column list on a table name": "WITH c (a, b) AS (SELECT arrival, departure FROM stays) SELECT a + b FROM c",
    "a name given to something else too": "SELECT room AS arrival FROM stays",
    "a union with another type": "SELECT arrival FROM stays UNION SELECT room FROM stays",
    "a union of stars": "SELECT * FROM stays UNION SELECT * FROM stays",
    "a window frame over the order": "SELECT COUNT(*) OVER (ORDER BY arrival RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM stays",
    "a comparison with a number in a scalar subquery": "SELECT guest FROM stays WHERE 5 > (SELECT MAX(arrival) FROM stays)",
    "between a text and a date": "SELECT guest FROM stays WHERE arrival BETWEEN '2024-01-01' AND departure",
    "a text that does not parse": "SELECT guest FROM (stays WHERE",
}
PLAIN = "SELECT guest FROM stays WHERE room > 1"


@pytest.mark.parametrize("position", range(len(ORDER_ONLY)))
def test_a_pair_that_only_compares_the_dates(position):
    assert compared_only([PLAIN, ORDER_ONLY[position]], KINDS, "DATE")
    assert compared_only([ORDER_ONLY[position].upper(), PLAIN], KINDS, "DATE")


@pytest.mark.parametrize("why", sorted(OTHER_USES))
def test_another_use_in_either_query_is_a_refusal(why):
    assert not compared_only([PLAIN, OTHER_USES[why]], KINDS, "DATE"), why
    assert not compared_only([OTHER_USES[why], ORDER_ONLY[0]], KINDS, "DATE"), why


def test_names_and_types():
    assert TEMPORAL_TYPES == ("DATE", "TIME", "DATETIME", "TIMESTAMP")
    arithmetic = "SELECT guest FROM stays WHERE arrival + 1 = departure"
    assert compared_only([arithmetic], KINDS, "TIME")  # no value of that type is read
    assert not compared_only([arithmetic], {**KINDS, ("STAYS", "ARRIVAL"): "TIME"}, "TIME")
    # a name the schema declares with two types is not followed
    assert not compared_only([PLAIN], {**KINDS, ("CLEANINGS", "ARRIVAL"): "INT"}, "DATE")
    assert compared_only([PLAIN], {**KINDS, ("CLEANINGS", "ARRIVAL"): "DATE"}, "DATE")
    # a name a query gives to a date is a date wherever it is read
    named = "SELECT d.guest FROM (SELECT guest, arrival AS came FROM stays) AS d WHERE d.came"
    assert not compared_only([named], KINDS, "DATE")
    assert compared_only([named.replace("d.came", "d.came IS NULL")], KINDS, "DATE")


# --- what the refusals are for: an accepted query cannot tell dates from numbers in the same order ---


def _databases(rng: random.Random):
    days = sorted(rng.sample(range(1, 400), 6))
    as_date = {n: datetime.date(2024, 1, 1) + datetime.timedelta(days=n) for n in days}
    as_number = {n: position * 7 + 3 for position, n in enumerate(days)}  # other numbers in the same order

    def day():
        return rng.choice([*days, None])

    stays = [(rng.choice([1, 1, 2, 3]), rng.choice([1, 2, 3, None]), day(), day()) for _ in range(rng.randint(0, 6))]
    cleanings = [(rng.choice([1, 2, 3]), day(), rng.choice([10, 20, None])) for _ in range(rng.randint(0, 4))]
    out = []
    for kind, values in (("DATE", as_date), ("INTEGER", as_number)):
        con = duckdb.connect()
        con.execute(f"CREATE TABLE stays (guest INT, room INT, arrival {kind}, departure {kind})")
        con.execute(f"CREATE TABLE cleanings (room INT, day {kind}, minutes INT)")
        if stays:
            con.executemany("INSERT INTO stays VALUES (?, ?, ?, ?)", [(g, r, values.get(a), values.get(d)) for g, r, a, d in stays])
        if cleanings:
            con.executemany("INSERT INTO cleanings VALUES (?, ?, ?)", [(r, values.get(d), m) for r, d, m in cleanings])
        out.append(con)
    back = {number: as_date[n] for n, number in as_number.items()}
    return out[0], out[1], back


@pytest.mark.parametrize("position", range(len(ORDER_ONLY) - 1))
def test_an_accepted_query_returns_the_same_rows_over_numbers_in_the_same_order(position):
    sql = parse_one(ORDER_ONLY[position], read="mysql").sql(dialect="duckdb")
    rng = random.Random(position)
    date_columns = None
    for _ in range(25):
        dates, numbers, back = _databases(rng)
        on_dates = dates.execute(sql).fetchall()
        if date_columns is None:
            date_columns = [index for index, described in enumerate(dates.description) if str(described[1]).upper() == "DATE"]
        on_numbers = [tuple(back.get(value, value) if index in date_columns else value for index, value in enumerate(row)) for row in numbers.execute(sql).fetchall()]
        assert sorted(on_dates, key=repr) == sorted(on_numbers, key=repr)
        dates.close()
        numbers.close()
