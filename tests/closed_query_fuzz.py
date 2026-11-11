"""Random queries that read no table, in the fragment ``kumo_sql_prover.closed_queries`` interprets.

Used by ``tests/test_closed_queries.py`` to compare the interpreter with DuckDB. Every query is valid in both
BigQuery and DuckDB and means the same in both: small integers, booleans, NULLs, three-valued logic, joins of
inline rows, grouping, set operations and subqueries.
"""

from __future__ import annotations

import random

VALUES = ["NULL", "0", "1", "2", "3"]


class Generator:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.counter = 0

    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def rows(self, names: list[str]) -> str:
        """A derived table body: zero to three rows of small integers and NULLs."""

        count = self.rng.choice([0, 1, 1, 2, 2, 3])
        if count == 0:
            return "SELECT " + ", ".join(f"1 AS {n}" for n in names) + " WHERE FALSE"
        return " UNION ALL ".join("SELECT " + ", ".join(f"{self.rng.choice(VALUES)} AS {n}" for n in names) for _ in range(count))

    def source(self) -> tuple[str, str, list[str]]:
        alias = self.fresh("t")
        names = [self.fresh("c") for _ in range(self.rng.choice([1, 2, 2]))]
        return f"({self.rows(names)}) AS {alias}", alias, names

    def number(self, columns: list[str], depth: int = 0) -> str:
        roll = self.rng.random()
        if depth > 1 or roll < 0.45:
            return self.rng.choice(columns) if columns and self.rng.random() < 0.75 else self.rng.choice(VALUES[1:])
        if roll < 0.65:
            return f"({self.number(columns, depth + 1)} {self.rng.choice(['+', '-', '*'])} {self.number(columns, depth + 1)})"
        if roll < 0.75:
            return f"COALESCE({self.number(columns, depth + 1)}, {self.number(columns, depth + 1)})"
        if roll < 0.83:
            return f"IF({self.boolean(columns, depth + 1)}, {self.number(columns, depth + 1)}, {self.number(columns, depth + 1)})"
        if roll < 0.9:
            return f"CASE WHEN {self.boolean(columns, depth + 1)} THEN {self.number(columns, depth + 1)} ELSE {self.number(columns, depth + 1)} END"
        if roll < 0.95:
            return f"NULLIF({self.number(columns, depth + 1)}, {self.number(columns, depth + 1)})"
        return f"CASE {self.number(columns, depth + 1)} WHEN 1 THEN {self.number(columns, depth + 1)} WHEN 2 THEN 7 END"

    def boolean(self, columns: list[str], depth: int = 0, subqueries: bool = True) -> str:
        roll = self.rng.random()
        if depth > 1 or roll < 0.4:
            return f"{self.number(columns, 2)} {self.rng.choice(['=', '!=', '<', '<=', '>', '>='])} {self.number(columns, 2)}"
        if roll < 0.55:
            return f"({self.boolean(columns, depth + 1)} {self.rng.choice(['AND', 'OR'])} {self.boolean(columns, depth + 1)})"
        if roll < 0.63:
            return f"NOT ({self.boolean(columns, depth + 1)})"
        if roll < 0.72:
            return f"{self.number(columns, 2)} IS {self.rng.choice(['', 'NOT '])}NULL"
        if roll < 0.78:
            return f"{self.number(columns, 2)} BETWEEN {self.number(columns, 2)} AND {self.number(columns, 2)}"
        if roll < 0.85:
            return f"{self.number(columns, 2)} {self.rng.choice(['', 'NOT '])}IN ({', '.join(self.rng.choice(VALUES) for _ in range(self.rng.choice([1, 2, 3])))})"
        if roll < 0.88:
            return f"({self.boolean(columns, depth + 1)}) IS {self.rng.choice(['', 'NOT '])}{self.rng.choice(['TRUE', 'FALSE'])}"
        if not subqueries:
            return self.rng.choice(["TRUE", "FALSE", "1 = 1"])
        source, alias, names = self.source()
        inner = [f"{alias}.{n}" for n in names]
        filter_ = f" WHERE {self.boolean(columns + inner, depth + 1, subqueries=False)}" if self.rng.random() < 0.6 else ""
        if roll < 0.94:
            return f"{self.number(columns, 2)} {self.rng.choice(['', 'NOT '])}IN (SELECT {self.number(inner, 1)} FROM {source}{filter_})"
        return f"{self.rng.choice(['', 'NOT '])}EXISTS (SELECT 1 FROM {source}{filter_})"

    def select(self, width: int | None = None) -> str:
        source, alias, names = self.source()
        from_ = source
        columns = [f"{alias}.{n}" for n in names]
        bare = list(names)
        for _ in range(self.rng.choice([0, 1, 1, 2])):
            source, alias, names = self.source()
            visible = columns + [f"{alias}.{n}" for n in names]
            kind = self.rng.choice([",", "CROSS JOIN", "JOIN", "INNER JOIN", "LEFT JOIN", "LEFT OUTER JOIN", "RIGHT JOIN", "FULL JOIN", "FULL OUTER JOIN"])
            if kind == ",":
                from_ += f", {source}"
            elif kind == "CROSS JOIN":
                from_ += f" CROSS JOIN {source}"
            else:
                from_ += f" {kind} {source} ON {self.boolean(visible, 0, subqueries=self.rng.random() < 0.2)}"
            columns = visible
            bare += names
        readable = columns + bare if self.rng.random() < 0.5 else columns
        where = f" WHERE {self.boolean(readable)}" if self.rng.random() < 0.5 else ""
        shape = self.rng.random()
        count = width or self.rng.choice([1, 2, 3])
        if shape < 0.12 and width is None:
            return f"SELECT * FROM {from_}{where}"
        if shape < 0.3:
            aggregates = [self.rng.choice(["COUNT(*)", f"COUNT({self.number(readable, 1)})", f"COUNT(DISTINCT {self.number(readable, 1)})", f"SUM({self.number(readable, 1)})", f"MIN({self.number(readable, 1)})", f"MAX({self.number(readable, 1)})"]) for _ in range(count)]
            return f"SELECT {', '.join(aggregates)} FROM {from_}{where}"
        if shape < 0.5:
            keys = [self.number(readable, 1) for _ in range(self.rng.choice([1, 2]))]
            keys = [k for k in dict.fromkeys(keys) if not k.isdigit() and k != "NULL"] or [readable[0]]
            items = [self.rng.choice(keys + ["COUNT(*)", f"SUM({self.number(readable, 1)})", f"MAX({self.number(readable, 1)})", f"COUNT(*) + {keys[0]}"]) for _ in range(count)]
            return f"SELECT {', '.join(items)} FROM {from_}{where} GROUP BY {', '.join(keys)}"
        distinct = "DISTINCT " if self.rng.random() < 0.25 else ""
        items = [self.number(readable) if self.rng.random() < 0.8 else self.boolean(readable, 1) for _ in range(count)]
        if distinct or self.rng.random() < 0.5:
            items = [self.number(readable) for _ in range(count)]
        return f"SELECT {distinct}{', '.join(items)} FROM {from_}{where}"

    def numbers_only(self, width: int) -> str:
        """A select whose columns are all integers, so set operations and DISTINCT never mix kinds."""

        for _ in range(50):
            sql = self.select(width)
            if " = " not in sql.split(" FROM ")[0] and all(op not in sql.split(" FROM ")[0] for op in (" < ", " > ", " IS ", " IN ", " BETWEEN ", "EXISTS", "!=", "<=", ">=", "NOT (")) or "COUNT" in sql.split(" FROM ")[0]:
                return sql
        return "SELECT " + ", ".join("1" for _ in range(width))

    def query(self) -> str:
        roll = self.rng.random()
        if roll < 0.7:
            return self.select()
        width = self.rng.choice([1, 2])
        operator = self.rng.choice(["UNION ALL", "UNION DISTINCT", "INTERSECT DISTINCT", "EXCEPT DISTINCT"])
        left, right = self.numbers_only(width), self.numbers_only(width)
        if roll < 0.9:
            return f"{left} {operator} {right}"
        alias = self.fresh("d")
        return f"SELECT * FROM ({left} {operator} {right}) AS {alias}"
