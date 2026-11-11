"""Refute a pair of DuckDB-dialect queries by running them on DuckDB as they are written.

:mod:`kumo_sql_prover.executed_refutation` searches for a database on which two BigQuery queries differ and
needs a BigQuery-faithful reading of each query on DuckDB. A pair written *for* DuckDB needs no such
reading: DuckDB is its own reference. So ``search`` builds the same candidate databases (the corner-case,
targeted and random ones, then the z3 candidates of :mod:`kumo_sql_prover.bounded_refutation`, all from the
BigQuery spelling sqlglot gives the two queries, which only steers the search) and accepts one only when the
two queries **as written** return different bags on it, on tables created with the declared column types.

A counterexample is observed, not derived, so it needs no trust in the prover. What could still make it
false is a query whose rows depend on something other than the database, so the search runs only when both
queries stay inside a fixed list of constructs with one answer per database:

* joins, set operations, subqueries, comparisons, arithmetic, ``CASE``/``COALESCE``/``NULLIF``, casts, the plain
  aggregates and aggregate windows with DuckDB's default frame;
* no ``LIMIT``/``OFFSET``/``FETCH``, no ``DISTINCT ON``, no ``ROWS`` or ``GROUPS`` window frame, no ranking or
  navigation window function (a tie or a missing ``ORDER BY`` lets the engine pick), no function outside the
  list (nothing random, nothing reading the clock or a file);
* every table is a declared table or a CTE of the query (DuckDB would otherwise look for a file or a
  variable of that name), and the connection has external access switched off.

A difference counts when it survives rounding floats to 6 digits, each side returns the same bag with every
table's rows reversed, and both queries return the same rows with DuckDB's optimizer off (DuckDB 1.5 returns
wrong rows for some correlated subqueries with it on). Finding nothing proves nothing.
"""

from __future__ import annotations

import itertools
import re
import threading
import time
from typing import Mapping

import sqlglot
from sqlglot import exp

from .smt_equivalence import Counterexample, TableConstraints

# DuckDB column types the candidate generator has a value domain for (its names are BigQuery's).
_GENERATOR_TYPES = {
    "TINYINT": "INT64", "SMALLINT": "INT64", "INT": "INT64", "INTEGER": "INT64", "BIGINT": "INT64",
    "INT1": "INT64", "INT2": "INT64", "INT4": "INT64", "INT8": "INT64", "INT64": "INT64",
    "UTINYINT": "INT64", "USMALLINT": "INT64", "UINTEGER": "INT64", "UBIGINT": "INT64",
    "UINT8": "INT64", "UINT16": "INT64", "UINT32": "INT64", "UINT64": "INT64",
    "DOUBLE": "FLOAT64", "FLOAT": "FLOAT64", "REAL": "FLOAT64", "FLOAT8": "FLOAT64", "FLOAT4": "FLOAT64",
    "VARCHAR": "STRING", "TEXT": "STRING", "STRING": "STRING", "CHAR": "STRING",
    "BOOLEAN": "BOOL", "BOOL": "BOOL",
    "DATE": "DATE",
}
_TYPE_TEXT = re.compile(r"[A-Za-z][A-Za-z0-9]*(\s*\(\s*\d+\s*(,\s*\d+\s*)?\))?")
_REFUSED = (
    exp.Unnest, exp.Array, exp.Struct, exp.PropertyEQ, exp.Interval, exp.Var, exp.DateAdd, exp.DateSub, exp.DateDiff,
)
_WINDOW_AGGREGATES = (exp.Count, exp.Sum, exp.Min, exp.Max, exp.Avg, exp.CountIf)
_SET_OPERATIONS = (exp.Union, exp.Intersect, exp.Except)


def refusal(tree: exp.Expression, tables: set[str]) -> str | None:
    """Why ``tree`` is outside what the search runs, or ``None``. ``tables`` holds the declared names, lower-case."""

    from .executed_refutation import _ALLOWED

    ctes = {cte.alias.lower() for cte in tree.find_all(exp.CTE) if cte.alias}
    for node in tree.walk():
        if isinstance(node, exp.DataTypeParam) and isinstance(node.parent, exp.DataType):
            continue  # the precision of a cast's target type
        if not isinstance(node, _ALLOWED) or isinstance(node, _REFUSED):
            return type(node).__name__
        if isinstance(node, exp.Distinct) and node.args.get("on"):
            return "DISTINCT ON"
        if isinstance(node, (exp.Select, *_SET_OPERATIONS)) and (node.args.get("limit") or node.args.get("offset")):
            return "LIMIT"
        if isinstance(node, exp.Select) and (node.args.get("sample") or node.args.get("windows") or node.args.get("qualify")):
            return "SAMPLE, WINDOW or QUALIFY"
        if isinstance(node, _SET_OPERATIONS) and (node.args.get("by_name") or node.args.get("side") or node.args.get("kind")):
            return "set operation by name"
        if isinstance(node, exp.Window):
            if not isinstance(node.this, _WINDOW_AGGREGATES):
                return "window function"
            if node.args.get("spec") is not None or node.args.get("first") is not None or isinstance(node.args.get("alias"), exp.Identifier):
                return "window frame"
        if isinstance(node, exp.WindowSpec):
            return "window frame"
        if isinstance(node, exp.Join) and node.args.get("method"):
            return "join form"
        if isinstance(node, exp.Table):
            if node.args.get("db") or node.args.get("catalog") or not isinstance(node.this, exp.Identifier):
                return "table reference"
            if node.name.lower() not in tables and node.name.lower() not in ctes:
                return f"table {node.name} is not declared"
            if any(node.args.get(arg) for arg in ("sample", "pivots", "version", "when", "changes")):
                return "table modifier"
    return None


def _type_sql(raw: str) -> str | None:
    """``raw`` if it is a plain type name (with an optional precision), else ``None``: it goes into a ``CREATE TABLE``."""

    text = raw.strip()
    return text if _TYPE_TEXT.fullmatch(text) else None


def _declared(read: set[str], schema, types) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]] | None:
    """For each table read: the generator's type and the declared DuckDB type of every column, or ``None``."""

    by_lower = {k.lower(): k for k in schema or {}}
    types_lower = {k.lower(): {c.lower(): t for c, t in v.items()} for k, v in (types or {}).items()}
    typed: dict[str, dict[str, str]] = {}
    native: dict[str, dict[str, str]] = {}
    for name in sorted(read):
        key = by_lower.get(name)
        declared = types_lower.get(name)
        if key is None or declared is None or not schema[key]:
            return None
        typed[key], native[key] = {}, {}
        for column in schema[key]:
            raw = declared.get(column.lower())
            text = _type_sql(raw) if isinstance(raw, str) else None
            if text is None or not re.fullmatch(r"\w+", column) or not re.fullmatch(r"\w+", key):
                return None
            generated = _GENERATOR_TYPES.get(text.split("(")[0].strip().upper())
            if generated is None:
                return None
            typed[key][column] = generated
            native[key][column] = text
    return typed, native


class _NativeRunner:
    """The tables, created with their declared types on one DuckDB connection; ``run`` swaps their rows and
    executes a query as written (the interface :class:`executed_refutation._Search` asks of a runner)."""

    def __init__(self, native: Mapping[str, Mapping[str, str]], settings: tuple[str, ...] = ()):
        from .duckdb_load import small_database
        from .result_equivalence import ExecutionError

        self._connection = small_database(enable_external_access=False)
        try:
            for statement in settings:
                self._connection.execute(statement)
            for table, columns in native.items():
                declared = ", ".join(f'"{name}" {sql_type}' for name, sql_type in columns.items())
                self._connection.execute(f'CREATE TABLE "{table}" ({declared})')
        except Exception as error:  # noqa: BLE001 - a type DuckDB does not know means no search
            self._connection.close()
            raise ExecutionError(f"DuckDB could not create the declared tables: {error}") from error
        self._loaded = None

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "_NativeRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _load(self, dataset) -> None:
        from .duckdb_load import insert_rows

        if dataset is self._loaded:
            return
        self._loaded = None
        for table, content in dataset.tables.items():
            self._connection.execute(f'DELETE FROM "{table}"')
            insert_rows(self._connection, f'"{table}"', content.rows)
        self._loaded = dataset

    def run(self, sql: str, dataset, *, timeout: float | None = None):
        from .result_equivalence import ExecutionError, QueryOutput

        timer = None
        try:
            self._load(dataset)  # a value the declared type rejects (a negative UINT8) fails here: not a candidate
            if timeout is not None:
                timer = threading.Timer(timeout, self._connection.interrupt)
                timer.start()
            cursor = self._connection.execute(sql)
            columns = tuple(column[0] for column in cursor.description)
            rows = tuple(tuple(row) for row in cursor.fetchall())
        except Exception as error:  # noqa: BLE001 - DuckDB raises several error types
            raise ExecutionError(f"DuckDB failed: {error}") from error
        finally:
            if timer is not None:
                timer.cancel()
        return QueryOutput(columns=columns, rows=rows)


def _candidates(left: str, right: str, typed, rules, foreign_keys, accept, deadline):
    """The first database ``accept`` confirms, built as :mod:`executed_refutation` builds them from BigQuery text."""

    from . import bounded_refutation
    from .executed_refutation import _DATASET_LIMIT, _datasets

    try:
        built = itertools.islice(_datasets(left, right, typed, rules), _DATASET_LIMIT)
        for dataset in itertools.chain(built, bounded_refutation.isolated_datasets(left, right, typed, rules)):
            if time.monotonic() > deadline:
                break
            if accept(dataset):
                return dataset
        return bounded_refutation.find(left, right, typed, rules, foreign_keys, accept)
    except Exception:  # noqa: BLE001 - the BigQuery spelling only steers the search; a gap in it costs candidates
        return None


def search(
    left_sql: str,
    right_sql: str,
    *,
    schema: Mapping[str, list[str]] | None,
    types: Mapping[str, Mapping[str, str]] | None,
    constraints: Mapping[str, TableConstraints] | None = None,
    time_limit: float = 3.0,
) -> Counterexample | None:
    """A database on which the two DuckDB queries, run as written, return different bags, or ``None``."""

    try:
        import duckdb  # noqa: F401
    except ImportError:
        return None
    from .executed_refutation import _export, _legal, _rules, _Search, _tables_read
    from .result_equivalence import ExecutionError, SyntheticDataset, compare_outputs

    try:
        parsed = [sqlglot.parse(sql, read="duckdb") for sql in (left_sql, right_sql)]
    except sqlglot.errors.SqlglotError:
        return None
    if any(len(statements) != 1 or not isinstance(statements[0], (exp.Select, *_SET_OPERATIONS, exp.Subquery)) for statements in parsed):
        return None
    trees = [statements[0] for statements in parsed]
    declared_names = {name.lower() for name in schema or {}}
    if any(refusal(tree, declared_names) for tree in trees):
        return None
    read = _tables_read(trees[0]) | _tables_read(trees[1])
    declared = _declared(read, schema, types)
    if declared is None:
        return None
    typed, native = declared
    constraints = constraints or {}
    rules = _rules(typed, constraints)

    def legal(dataset) -> bool:
        return _legal(dataset, typed, constraints)

    deadline = time.monotonic() + time_limit
    try:
        runner = _NativeRunner(native)
    except ExecutionError:
        return None
    with runner:
        search_ = _Search(runner, left_sql, right_sql)

        def accept(dataset) -> bool:
            return legal(dataset) and search_.differs(dataset)

        if not typed:
            # No table is read: each query has one answer, and one run of each decides.
            found = SyntheticDataset(0, {})
            if not search_.differs(found):
                return None
        else:
            try:
                spelled = [tree.sql(dialect="bigquery") for tree in trees]
            except Exception:  # noqa: BLE001
                return None
            foreign_keys = {k.lower(): list(v.foreign_keys) for k, v in constraints.items() if k.lower() in {t.lower() for t in typed}}
            found = _candidates(*spelled, typed, rules, foreign_keys, accept, deadline)
            if found is None:
                return None
            found = search_.shrink(found, legal, time.monotonic() + max(1.0, time_limit / 2))
        outputs = search_.outputs(found)
        if outputs is None:
            return None
        try:
            with _NativeRunner(native, settings=("PRAGMA disable_optimizer",)) as plain:
                for sql, output in zip((left_sql, right_sql), outputs):
                    if not compare_outputs(output, plain.run(sql, found, timeout=5), check_column_names=False, float_digits=12)[0]:
                        return None
        except ExecutionError:
            return None
        a, b = outputs
        tables = {
            key: [{name: _export(v) for (name, _), v in zip(table.columns, row)} for row in table.rows]
            for key, table in found.tables.items()
        }
        return Counterexample(
            tables=tables,
            left_rows=sorted((tuple(_export(v) for v in r) for r in a.rows), key=repr),
            right_rows=sorted((tuple(_export(v) for v in r) for r in b.rows), key=repr),
        )
