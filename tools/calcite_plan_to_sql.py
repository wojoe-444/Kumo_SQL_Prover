"""Mine Apache Calcite's optimizer rule tests into SQL equivalence pairs.

Source: https://github.com/apache/calcite (Apache-2.0). Each ``<TestCase>`` of
``RelOptRulesTest.xml`` (and of the smaller per-rule test files listed in
``XML_FILES``) records the plan a rule received (``planBefore``) and the plan it
produced (``planAfter``); the test asserts the two are interchangeable. This
script parses both plan dumps (Calcite's ``explain`` text: ``LogicalProject``,
``LogicalFilter``, ``LogicalJoin``, ... with ``$n`` ordinal RexNode
expressions), infers Calcite's types bottom-up from the mock catalog, and emits
a pair of MySQL-flavoured SQL strings ``(sql_a, sql_b)`` = (planBefore,
planAfter), or skips the test with an explicit reason. The test's original SQL
(Calcite dialect) is kept as ``sql_calcite`` for reference only. It never
guesses semantics: anything it does not model exactly is a skip.

    python tools/calcite_plan_to_sql.py --calcite /path/to/calcite \
        [--out tests/fixtures/calcite_mined] [--sqlsolver-names calcite_tests.json]

Why planBefore rather than the original SQL: the original SQL is Calcite's
dialect (integer ``/``, implicit coercions, CHAR padding, Calcite-only
functions, double-quoted identifiers) and 70+ tests have no SQL at all (they
are built with RelBuilder); planBefore is exactly what the rule received, so
(planBefore, planAfter) is precisely the equivalence the test asserts.

Conventions of the generated SQL (same as tools/qed_to_sql.py)
  * scans select the table's own column names and alias them ``c0..cN``;
    every other operator is a derived table whose columns are ``c0..cN``, so
    Calcite's ``$i`` becomes ``alias.c<i>``;
  * correlation variables (``$cor0.DEPTNO``) are resolved by field name against
    the row they are bound to (Filter/Project ``variablesSet``, Correlate) and
    become references to that row's alias;
  * ``ORDER BY`` keys carry an explicit null-ordering key (MySQL has no
    ``NULLS FIRST/LAST``): Calcite sorts ASC with nulls last, DESC nulls first
    unless the plan says otherwise;
  * Calcite's integer ``/`` and integer ``AVG`` truncate, so they become
    ``DIV`` / ``SUM(x) DIV COUNT(x)``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


class Skip(Exception):
    """Raised when a construct cannot be converted faithfully."""


XML_DIR = "core/src/test/resources/org/apache/calcite/test"
JAVA_DIR = "core/src/test/java/org/apache/calcite/test"
# (xml file, name prefix). RelOptRulesTest names are kept bare so they line up
# with SQLSolver / QED / R-Bot, which all use RelOptRulesTest method names.
XML_FILES = [
    ("RelOptRulesTest", ""),
    ("AggregateFilterToFilteredAggregateRuleTest", "AggregateFilterToFilteredAggregateRuleTest."),
    ("AggregateReduceFunctionsOnGroupKeysRuleTest", "AggregateReduceFunctionsOnGroupKeysRuleTest."),
    ("AggregateRemoveDuplicateKeysRuleTest", "AggregateRemoveDuplicateKeysRuleTest."),
    ("CombineRelOptRulesTest", "CombineRelOptRulesTest."),
    ("JoinAggregateTransposeRuleTest", "JoinAggregateTransposeRuleTest."),
    ("OuterJoinToAntiJoinRuleTest", "OuterJoinToAntiJoinRuleTest."),
]


# ============================================================== types
@dataclass(frozen=True)
class T:
    base: str  # INTEGER BIGINT SMALLINT TINYINT DECIMAL DOUBLE REAL FLOAT VARCHAR CHAR BOOLEAN DATE TIMESTAMP NULL
    p: int | None = None
    s: int | None = None

    @property
    def fam(self) -> str:
        return FAMILY[self.base]

    def ddl(self) -> str:
        if self.base in ("VARCHAR", "CHAR"):
            return f"{self.base}({self.p})"
        if self.base == "DECIMAL":
            return f"DECIMAL({self.p}, {self.s})"
        return self.base


INT_RANK = {"TINYINT": 1, "SMALLINT": 2, "INTEGER": 3, "BIGINT": 4}
INT_DIGITS = {"TINYINT": 3, "SMALLINT": 5, "INTEGER": 10, "BIGINT": 19}
FAMILY = {
    **{t: "int" for t in INT_RANK}, "DECIMAL": "dec", "DOUBLE": "approx", "REAL": "approx",
    "FLOAT": "approx", "VARCHAR": "str", "CHAR": "str", "BOOLEAN": "bool", "DATE": "date",
    "TIMESTAMP": "ts", "NULL": "null",
}
NUMERIC = {"int", "dec", "approx"}
INTEGER = T("INTEGER")
BIGINT = T("BIGINT")
BOOLEAN = T("BOOLEAN")
DOUBLE = T("DOUBLE")
NULLT = T("NULL")

TYPE_RE = re.compile(
    r"^(INTEGER|BIGINT|SMALLINT|TINYINT|DECIMAL|DOUBLE|REAL|FLOAT|VARCHAR|CHAR|BOOLEAN|DATE|TIMESTAMP)"
    r"(?:\((\d+)(?:, (\d+))?\))?"
    r"(?: CHARACTER SET \"[A-Za-z0-9_-]+\")?(?: NOT NULL)?$")


def parse_type(text: str) -> T:
    if text.strip() == "NULL":
        return NULLT
    m = TYPE_RE.match(text.strip())
    if not m:
        raise Skip(f"type {text.strip()}")
    base, p, s = m.group(1), m.group(2), m.group(3)
    if base == "TIMESTAMP":
        if p not in (None, "0"):
            raise Skip(f"type {text.strip()}")
        return T("TIMESTAMP")
    if base == "DECIMAL":
        if p is None:
            raise Skip("DECIMAL without precision")
        return T("DECIMAL", int(p), int(s or 0))
    if base in ("VARCHAR", "CHAR"):
        return T(base, int(p) if p else None)
    if p is not None:
        raise Skip(f"type {text.strip()}")
    return T(base)


def dec_shape(t: T) -> tuple[int, int]:
    """(integer digits, scale) of an exact numeric type."""
    if t.base == "DECIMAL":
        return t.p - t.s, t.s
    return INT_DIGITS[t.base], 0


def least_restrictive(ts: list[T], what: str) -> T:
    ts2 = [t for t in ts if t.base != "NULL"]
    if not ts2:
        return NULLT
    fams = {t.fam for t in ts2}
    if fams <= NUMERIC:
        if "approx" in fams:
            return DOUBLE
        if "dec" in fams:
            shapes = [dec_shape(t) for t in ts2]
            i, s = max(x[0] for x in shapes), max(x[1] for x in shapes)
            return T("DECIMAL", min(i + s, 19), s)
        return max(ts2, key=lambda t: INT_RANK[t.base])
    if len(fams) != 1:
        raise Skip(f"{what} mixes type families {sorted(fams)}")
    fam = fams.pop()
    if fam == "str":
        if any(t.p is None for t in ts2):
            return T("VARCHAR", None)
        n = max(t.p for t in ts2)
        if all(t.base == "CHAR" for t in ts2):
            if len({t.p for t in ts2}) != 1:
                raise Skip(f"{what} combines CHAR values of different lengths (Calcite pads them)")
            return T("CHAR", n)
        return T("VARCHAR", n)
    return ts2[0]


# ============================================================== catalogs
def _cols(*spec):
    return [(n, parse_type(t), nl) for n, t, nl in spec]


EMP_COLS = _cols(("EMPNO", "INTEGER", False), ("ENAME", "VARCHAR(20)", False), ("JOB", "VARCHAR(10)", False),
                 ("MGR", "INTEGER", True), ("HIREDATE", "TIMESTAMP", False), ("SAL", "INTEGER", False),
                 ("COMM", "INTEGER", False), ("DEPTNO", "INTEGER", False), ("SLACKER", "BOOLEAN", False))
EMPNULL_COLS = _cols(("EMPNO", "INTEGER", False), ("ENAME", "VARCHAR(20)", True), ("JOB", "VARCHAR(10)", True),
                     ("MGR", "INTEGER", True), ("HIREDATE", "TIMESTAMP", True), ("SAL", "INTEGER", True),
                     ("COMM", "INTEGER", True), ("DEPTNO", "INTEGER", True), ("SLACKER", "BOOLEAN", True))
EMPDEF_COLS = _cols(("EMPNO", "INTEGER", False), ("ENAME", "VARCHAR(20)", False), ("JOB", "VARCHAR(10)", True),
                    ("MGR", "INTEGER", True), ("HIREDATE", "TIMESTAMP", True), ("SAL", "INTEGER", True),
                    ("COMM", "INTEGER", True), ("DEPTNO", "INTEGER", True), ("SLACKER", "BOOLEAN", True))

# path -> (table name, columns [(name, T, nullable)], keys [[col, ...]])
# Columns/types/nullability/keys transcribed from Calcite's
# testkit/.../catalog/MockCatalogReaderSimple.java (+ MockCatalogReader.java) and,
# for the ``scott`` schema used by RelBuilder-built tests, from scott-data-hsqldb's
# scott.script (unique indexes I_EMP_PK / I_DEPT_PK).
CATALOG = {
    ("CATALOG", "SALES", "EMP"): ("EMP", EMP_COLS, [["EMPNO"]]),
    ("CATALOG", "SALES", "EMPNULLABLES"): ("EMPNULLABLES", EMPNULL_COLS, [["EMPNO"]]),
    ("CATALOG", "SALES", "EMPDEFAULTS"): ("EMPDEFAULTS", EMPDEF_COLS, [["EMPNO"]]),
    ("CATALOG", "SALES", "EMP_B"): ("EMP_B", EMP_COLS + _cols(("BIRTHDATE", "DATE", False)), [["EMPNO"]]),
    ("CATALOG", "SALES", "DEPT"): ("DEPT", _cols(("DEPTNO", "INTEGER", False), ("NAME", "VARCHAR(10)", False)),
                                   [["DEPTNO"]]),
    ("CATALOG", "SALES", "DEPTNULLABLES"): ("DEPTNULLABLES", _cols(("DEPTNO", "INTEGER", True),
                                                                   ("NAME", "VARCHAR(10)", True)), [["DEPTNO"]]),
    ("CATALOG", "SALES", "BONUS"): ("BONUS", _cols(("ENAME", "VARCHAR(20)", False), ("JOB", "VARCHAR(10)", False),
                                                   ("SAL", "INTEGER", False), ("COMM", "INTEGER", False)), []),
    ("CATALOG", "SALES", "SALGRADE"): ("SALGRADE", _cols(("GRADE", "INTEGER", False), ("LOSAL", "INTEGER", False),
                                                         ("HISAL", "INTEGER", False)), [["GRADE"]]),
    ("CATALOG", "SALES", "PRODUCTS"): ("PRODUCTS", _cols(("PRODUCTID", "INTEGER", False),
                                                         ("NAME", "VARCHAR(20)", False),
                                                         ("SUPPLIERID", "INTEGER", False)), []),
    ("CATALOG", "SALES", "SUPPLIERS"): ("SUPPLIERS", _cols(("SUPPLIERID", "INTEGER", False),
                                                           ("NAME", "VARCHAR(20)", False),
                                                           ("CITY", "INTEGER", False)), []),
    ("CATALOG", "SALES", "DOUBLE_PK"): ("DOUBLE_PK", _cols(("ID1", "INTEGER", False), ("ID2", "VARCHAR(20)", False),
                                                           ("NAME", "VARCHAR(20)", False), ("AGE", "INTEGER", False)),
                                       [["ID1"], ["ID1", "ID2"]]),
    ("CATALOG", "CUSTOMER", "ACCOUNT"): ("ACCOUNT", _cols(("ACCTNO", "INTEGER", False), ("TYPE", "VARCHAR(20)", False),
                                                          ("BALANCE", "INTEGER", False)), []),
    ("scott", "EMP"): ("EMP", _cols(("EMPNO", "SMALLINT", False), ("ENAME", "VARCHAR(10)", True),
                                    ("JOB", "VARCHAR(9)", True), ("MGR", "SMALLINT", True), ("HIREDATE", "DATE", True),
                                    ("SAL", "DECIMAL(7, 2)", True), ("COMM", "DECIMAL(7, 2)", True),
                                    ("DEPTNO", "TINYINT", True)), [["EMPNO"]]),
    ("scott", "DEPT"): ("DEPT", _cols(("DEPTNO", "TINYINT", False), ("DNAME", "VARCHAR(14)", True),
                                      ("LOC", "VARCHAR(13)", True)), [["DEPTNO"]]),
    ("scott", "BONUS"): ("BONUS", _cols(("ENAME", "VARCHAR(10)", True), ("JOB", "VARCHAR(9)", True),
                                        ("SAL", "DECIMAL(7, 2)", True), ("COMM", "DECIMAL(7, 2)", True)), []),
    ("scott", "SALGRADE"): ("SALGRADE", _cols(("GRADE", "INTEGER", True), ("LOSAL", "DECIMAL(7, 2)", True),
                                              ("HISAL", "DECIMAL(7, 2)", True)), []),
}
KNOWN_UNSUPPORTED_TABLES = {
    "EMPTY_PRODUCTS": "table is empty only by catalog convention (not expressible in a schema)",
    "EMP_20": "view table with a hidden constraint", "EMPNULLABLES_20": "view table with a hidden constraint",
    "ORDERS": "streaming table", "SHIPMENTS": "streaming table",
    "PRODUCTS_TEMPORAL": "temporal table",
}

try:
    from sqlglot.tokens import Tokenizer as _Tok

    _RESERVED = {k.upper() for k in _Tok.KEYWORDS}
except Exception:  # pragma: no cover
    _RESERVED = set()


def qident(name: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and name.upper() not in _RESERVED:
        return name
    return "`" + name.replace("`", "``") + "`"


def sql_string(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"


# ============================================================== plan parsing
@dataclass
class PNode:
    kind: str
    attrs: list  # [(name, raw value)]
    children: list


def match_close(s: str, i: int, open_ch: str, close_ch: str) -> int:
    """Index of the bracket closing s[i] (== open_ch); quote-aware; tracks (), [] and {}."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    q = False
    j = i
    while j < len(s):
        ch = s[j]
        if not q and s.startswith("Sarg[", j):
            j = sarg_end(s, j + 4) + 1
            if not stack:
                raise Skip("unbalanced brackets in plan text")
            continue
        if q:
            if ch == "'":
                if j + 1 < len(s) and s[j + 1] == "'":
                    j += 2
                    continue
                q = False
        elif ch == "'":
            q = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in ")]}":
            if not stack or stack[-1] != ch:
                raise Skip("unbalanced brackets in plan text")
            stack.pop()
            if not stack:
                return j
        j += 1
    raise Skip("unbalanced brackets in plan text")


def sarg_end(s: str, i: int) -> int:
    """Index of the ']' closing the Sarg whose '[' is at s[i]; ranges inside mix [ ( ] )."""
    depth = 0
    q = False
    j = i
    while j < len(s):
        ch = s[j]
        if q:
            if ch == "'":
                if s.startswith("''", j):
                    j += 2
                    continue
                q = False
        elif ch == "'":
            q = True
        elif ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
            if depth == 0:
                if ch != "]":
                    raise Skip("unbalanced brackets in plan text")
                return j
        j += 1
    raise Skip("unbalanced brackets in plan text")


class PlanParser:
    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def parse(self) -> PNode:
        self.s = self.s.strip()
        node = self.rel(0)
        if self.i != len(self.s):
            raise Skip("trailing plan text")
        return node

    def rel(self, indent: int) -> PNode:
        m = re.compile(r"[A-Za-z][A-Za-z0-9_$]*").match(self.s, self.i)
        if not m:
            raise Skip("malformed plan line")
        if m and not self.s.startswith("(", m.end()):
            if m.group(0) == "LogicalProject":
                raise Skip("projection with no columns")
            raise Skip(f"unsupported relational operator {m.group(0)}")
        kind = m.group(0)
        close = match_close(self.s, m.end(), "(", ")")
        attrs = self.split_attrs(self.s[m.end() + 1:close])
        self.i = close + 1
        children = []
        prefix = "\n" + " " * (indent + 2)
        while self.s.startswith(prefix, self.i) and self.i + len(prefix) < len(self.s) \
                and self.s[self.i + len(prefix)].isalpha():
            self.i += len(prefix)
            children.append(self.rel(indent + 2))
        return PNode(kind, attrs, children)

    @staticmethod
    def split_attrs(body: str) -> list:
        out = []
        i = 0
        while i < len(body):
            m = re.compile(r"\s*([^=\[\]]+?)=\[").match(body, i)
            if not m:
                raise Skip("malformed attribute list")
            close = match_close(body, m.end() - 1, "[", "]")
            out.append((m.group(1), body[m.end():close]))
            i = close + 1
            if body.startswith(", ", i):
                i += 2
            elif i != len(body):
                raise Skip("malformed attribute list")
        return out


def parse_plan(text: str) -> PNode:
    return PlanParser(text).parse()


# ============================================================== rex parsing
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:E-?\d+)?")
TS_RE = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?")
DATE_RE = re.compile(r"\d{4}-\d\d-\d\d")
OP_RE = re.compile(r"[^(),\[\]{}']+?(?=\()")


class RexParser:
    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def parse_all(self):
        e = self.expr()
        if self.i != len(self.s):
            raise Skip(f"unparsed expression text {self.s[self.i:self.i + 30]!r}")
        return e

    def peek(self, t: str) -> bool:
        return self.s.startswith(t, self.i)

    def type_suffix(self, stop_dots: bool = False):
        if not self.peek(":"):
            return None
        j = self.i + 1
        depth = 0
        while j < len(self.s):
            ch = self.s[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            elif ch in ",]" and depth == 0:
                break
            elif stop_dots and depth == 0 and self.s.startswith("..", j):
                break
            j += 1
        text = self.s[self.i + 1:j]
        self.i = j
        return text

    def expr(self, stop_dots: bool = False):
        e = self.atom(stop_dots)
        t = self.type_suffix(stop_dots)
        if t is not None:
            e["suffix"] = t
        return e

    def string(self):
        assert self.s[self.i] == "'"
        j = self.i + 1
        buf = []
        while True:
            if j >= len(self.s):
                raise Skip("unterminated string literal")
            if self.s[j] == "'":
                if self.s.startswith("''", j):
                    buf.append("'")
                    j += 2
                    continue
                break
            buf.append(self.s[j])
            j += 1
        self.i = j + 1
        return "".join(buf)

    def atom(self, stop_dots: bool):
        s, i = self.s, self.i
        if s.startswith("$", i) and i + 1 < len(s) and s[i + 1].isdigit():
            m = re.compile(r"\$(\d+)").match(s, i)
            self.i = m.end()
            return {"k": "ref", "i": int(m.group(1))}
        m = re.compile(r"\$cor\d+").match(s, i)
        if m:
            self.i = m.end()
            if not self.peek("."):
                raise Skip("bare correlation variable")
            fm = re.compile(r"\.([A-Za-z_$#][A-Za-z0-9_$#]*)").match(s, self.i)
            if not fm:
                raise Skip("correlation field access shape")
            self.i = fm.end()
            if self.peek("."):
                raise Skip("nested field access")
            return {"k": "cor", "var": m.group(0), "field": fm.group(1)}
        if s.startswith("{", i):
            j = match_close(s, i, "{", "}")
            inner = s[i + 1:j]
            self.i = j + 1
            return {"k": "plan", "plan": parse_plan(inner)}
        if s.startswith("'", i):
            return {"k": "lit", "kind": "str", "v": self.string()}
        m = re.compile(r"_[A-Za-z0-9-]+(?=')").match(s, i)
        if m:
            if m.group(0) not in ("_ISO-8859-1", "_UTF-8"):
                raise Skip(f"string literal charset {m.group(0)}")
            self.i = m.end()
            return {"k": "lit", "kind": "str", "v": self.string()}
        if s.startswith("Sarg[", i):
            j = sarg_end(s, i + 4)
            self.i = j + 1
            return {"k": "sarg", "body": s[i + 5:j]}
        m = TS_RE.match(s, i)
        if m:
            self.i = m.end()
            return {"k": "lit", "kind": "ts", "v": m.group(0)}
        m = DATE_RE.match(s, i)
        if m and not (m.end() < len(s) and s[m.end()].isdigit()):
            self.i = m.end()
            return {"k": "lit", "kind": "date", "v": m.group(0)}
        m = NUM_RE.match(s, i)
        if m and not s.startswith("(", m.end()):
            txt = m.group(0)
            self.i = m.end()
            kind = "dbl" if "E" in txt else ("dec" if "." in txt else "int")
            return {"k": "lit", "kind": kind, "v": txt}
        for w in ("true", "false", "null"):
            if s.startswith(w, i) and not (i + len(w) < len(s) and (s[i + len(w)].isalnum() or s[i + len(w)] in "_(")):
                self.i = i + len(w)
                return {"k": "lit", "kind": "bool" if w != "null" else "null", "v": w}
        if s.startswith("?", i):
            raise Skip("dynamic parameter")
        m = OP_RE.match(s, i)
        if not m:
            w = re.compile(r"[A-Za-z_]+").match(s, i)
            if w and w.group(0) in NONDET:
                raise Skip(f"non-deterministic function {w.group(0)}")
            if w and w.group(0) in ("Infinity", "NaN"):
                raise Skip("non-finite floating-point literal")
            if s.startswith("(", i) and "->" in s[i:i + 40]:
                raise Skip("lambda expression")
            raise Skip(f"unparsed expression {s[i:i + 30]!r}")
        op = m.group(0)
        if op.startswith(" ") or op.endswith(" "):
            raise Skip(f"operator shape {op!r}")
        close = match_close(s, m.end(), "(", ")")
        self.i = m.end() + 1
        args = []
        if op == "FLAG":
            flag = s[self.i:close]
            self.i = close + 1
            return {"k": "flag", "v": flag}
        while self.i < close:
            args.append(self.expr())
            if self.peek(", "):
                self.i += 2
            elif self.i != close:
                raise Skip(f"argument list shape after {op}: {s[self.i:self.i + 20]!r}")
        self.i = close + 1
        if self.peek(" OVER ("):
            return {"k": "over", "op": op, "args": args, **self.window()}
        if self.peek(" OVER") or self.peek(" FILTER") or self.peek(" WITHIN") or self.peek(" IGNORE NULLS"):
            raise Skip("window function")
        return {"k": "call", "op": op, "args": args}

    def window(self) -> dict:
        """The window of ``op(args) OVER (...)`` as RexWindow prints it: partition keys, order keys (each with its
        direction words) and the frame text left over, which is empty for the default frame."""
        s = self.s
        close = match_close(s, self.i + len(" OVER "), "(", ")")
        self.i += len(" OVER (")
        partition, order = [], []
        if self.peek("PARTITION BY "):
            self.i += len("PARTITION BY ")
            while True:
                partition.append(self.expr())
                if not self.peek(", "):
                    break
                self.i += 2
            if self.peek(" "):
                self.i += 1
        if self.peek("ORDER BY "):
            self.i += len("ORDER BY ")
            while True:
                key = self.expr()
                words = ""
                for word in (" DESC", " NULLS FIRST", " NULLS LAST"):
                    if self.peek(word):
                        words += word
                        self.i += len(word)
                order.append((key, words))
                if not self.peek(", "):
                    break
                self.i += 2
            if self.peek(" "):
                self.i += 1
        if self.i > close:
            raise Skip("window function")
        frame = s[self.i:close]
        self.i = close + 1
        return {"partition": partition, "order": order, "frame": frame}


def parse_rex(raw: str):
    return RexParser(raw).parse_all()


# ============================================================== translation
@dataclass
class Rel:
    sql: str
    types: list  # [T]
    names: list  # [str] or None (unknown, e.g. Values)
    empty: bool = False  # empty LogicalValues of unknown row type (sql/types are None)


VIRTUAL_WIDTH = 512


class NeedHint(Exception):
    """The plan root is an empty relation whose row type the digest does not give."""


def uniquify(names: list) -> list:
    used, out = set(), []
    for n in names:
        if n is None:
            out.append(None)
            continue
        cand = n
        k = 0
        while cand in used:
            cand = f"{n}{k}"
            k += 1
        used.add(cand)
        out.append(cand)
    return out


CMP = {"=", "<>", "<", "<=", ">", ">="}
IS_OPS = {"IS NULL", "IS NOT NULL", "IS TRUE", "IS NOT TRUE", "IS FALSE", "IS NOT FALSE"}
NONDET = {"RAND", "RAND_INTEGER", "CURRENT_TIMESTAMP", "CURRENT_DATE", "CURRENT_TIME", "LOCALTIME",
          "LOCALTIMESTAMP", "USER", "CURRENT_USER", "SESSION_USER", "SYSTEM_USER", "UUID", "NOW"}
MEASURE_OPS = {"V2M", "M2V", "M2X", "AGG_M2V", "AGG_M2M", "SAME_PARTITION"}
SCALAR_AGGS = {"COUNT", "SUM", "$SUM0", "MIN", "MAX", "AVG", "ANY_VALUE", "SINGLE_VALUE"}
SUB_CMP_RE = re.compile(r"^(=|<>|<|<=|>|>=) (SOME|ALL)$")
AGG_RE = re.compile(r"^([A-Z_$][A-Z0-9_$]*)\((DISTINCT )?((?:\$\d+(?:, )?)*)\)(?: FILTER \$(\d+))?$")
ALLOWED_ATTRS = {
    "LogicalTableScan": {"table"},
    "LogicalFilter": {"condition", "variablesSet"},
    "LogicalJoin": {"condition", "joinType", "semiJoinDone", "variablesSet"},
    "LogicalCorrelate": {"correlation", "joinType", "requiredColumns"},
    "LogicalSort": None,  # validated in rel_sort
    "LogicalUnion": {"all"}, "LogicalIntersect": {"all"}, "LogicalMinus": {"all"},
    "LogicalValues": {"tuples"},
}


class Converter:
    def __init__(self, root_types_hint: list | None = None, empty_tables: list | None = None):
        self.empty_tables = list(empty_tables or [])  # catalog paths of the plan's empty LogicalValues, in order
        self.windows_allowed = False  # a windowed call is SQL only in a select list
        self.n = 0
        self.tables: dict = {}  # path -> catalog entry
        self.root_hint = root_types_hint
        self.depth = 0

    def alias(self) -> str:
        self.n += 1
        return f"t{self.n}"

    # ------------------------------------------------------------ literals
    def literal(self, e) -> tuple[str, T]:
        kind, v, suf = e["kind"], e["v"], e.get("suffix")
        st = parse_type(suf) if suf is not None else None
        if kind == "null":
            if st is None or st.base == "NULL":
                return "NULL", NULLT
            return f"CAST(NULL AS {self.cast_name(st)})", st
        if kind == "bool":
            if st is not None and st.base != "BOOLEAN":
                raise Skip("boolean literal with non-boolean type")
            return v.upper(), BOOLEAN
        if kind == "str":
            if v != v.rstrip(" ") and v.strip(" "):
                raise Skip("CHAR-padded string literal (Calcite CHAR semantics)")
            if "\\" in v:
                raise Skip("backslash in string literal")
            t = T("CHAR", len(v))
            if st is not None:
                if st.fam != "str":
                    raise Skip("string literal with non-string type")
                if st.base == "CHAR" and st.p != len(v):
                    raise Skip("CHAR-padded string literal (Calcite CHAR semantics)")
                if st.p is not None and st.p < len(v):
                    raise Skip("string literal longer than its type")
                t = st
            return sql_string(v), t
        if kind == "date":
            if st is not None and st.base != "DATE":
                raise Skip("date literal with other type")
            return f"DATE '{v}'", T("DATE")
        if kind == "ts":
            if st is not None and st.base != "TIMESTAMP":
                raise Skip("timestamp literal with other type")
            if "." in v:
                raise Skip("fractional-second timestamp literal")
            return f"TIMESTAMP '{v}'", T("TIMESTAMP")
        # numbers
        if kind == "int":
            t = INTEGER if abs(int(v)) < 2 ** 31 else BIGINT
            sql = f"({v})" if v.startswith("-") else v
        elif kind == "dec":
            ip, fp = v.lstrip("-").split(".")
            t = T("DECIMAL", max(len(ip.lstrip("0")) + len(fp), len(fp), 1), len(fp))
            sql = f"({v})" if v.startswith("-") else v
        else:
            t = DOUBLE
            sql = f"({v})" if v.startswith("-") else v
        if st is not None:
            if st.fam not in NUMERIC:
                raise Skip("numeric literal with non-numeric type")
            if st != t:
                if st.fam == "int" and kind != "int":
                    raise Skip("non-integral literal typed as integer")
                sql = f"CAST({sql} AS {self.cast_name(st)})"
                t = st
        return sql, t

    @staticmethod
    def cast_name(t: T) -> str:
        if t.base == "NULL":
            raise Skip("cast to NULL type")
        if t.base in ("VARCHAR", "CHAR"):
            return f"{t.base}({t.p})" if t.p is not None else "CHAR"
        if t.base == "DECIMAL":
            return f"DECIMAL({t.p}, {t.s})"
        if t.base == "REAL" or t.base == "FLOAT":
            return "FLOAT"
        return t.base

    # ------------------------------------------------------------ Sarg
    def sarg_bound(self, text: str) -> tuple[str, T]:
        e = RexParser(text.strip())
        x = e.expr(stop_dots=True)
        if e.i != len(e.s) or x["k"] != "lit":
            raise Skip("Sarg bound shape")
        return self.literal(x)

    def search(self, x: str, xt: T, body: str, suffix) -> tuple[str, T]:
        null_mode = None
        if "; NULL AS " in body:
            body, null_mode = body.rsplit("; NULL AS ", 1)
            if null_mode not in ("TRUE", "FALSE"):
                raise Skip("Sarg null mode")
        items = [] if body.strip() == "" else split_top(body, ", ")
        disj = []
        ts = []
        for item in items:
            item = item.strip()
            if item[:1] in "[(" and item[-1:] in "])" and ".." in item:
                inner = item[1:-1]
                parts_ = split_top(inner, "..")
                if len(parts_) != 2:
                    raise Skip("Sarg range shape")
                lo, hi = parts_
                parts = []
                if lo != "-\u221e":
                    v, t = self.sarg_bound(lo)
                    ts.append(t)
                    parts.append(f"{x} {'>=' if item[0] == '[' else '>'} {v}")
                if hi != "+\u221e":
                    v, t = self.sarg_bound(hi)
                    ts.append(t)
                    parts.append(f"{x} {'<=' if item[-1] == ']' else '<'} {v}")
                if not parts:
                    parts.append(f"{x} IS NOT NULL")
                disj.append("(" + " AND ".join(parts) + ")")
            else:
                v, t = self.sarg_bound(item)
                ts.append(t)
                disj.append(f"({x} = {v})")
        for t in ts:
            self.check_comparable(xt, t)
        core = "(" + " OR ".join(disj) + ")" if disj else "FALSE"
        if not disj and null_mode is None:
            raise Skip("empty Sarg")
        if null_mode == "TRUE":
            return f"(({x} IS NULL) OR {core})", BOOLEAN
        if null_mode == "FALSE":
            return f"(({x} IS NOT NULL) AND {core})", BOOLEAN
        return core, BOOLEAN

    # ------------------------------------------------------------ expressions
    @staticmethod
    def check_comparable(a: T, b: T):
        if a.base == "NULL" or b.base == "NULL":
            return
        if a.fam in NUMERIC and b.fam in NUMERIC:
            return
        if a.fam != b.fam:
            raise Skip(f"comparison across type families ({a.base} vs {b.base})")

    def expr(self, e, ctx: list, ctypes: list, env: dict) -> tuple[str, T]:
        k = e["k"]
        if k == "ref":
            i = e["i"]
            if not 0 <= i < len(ctx):
                raise Skip("input ref out of range (plan refers past its input row)")
            if e.get("suffix"):
                raise Skip("typed input ref")
            return ctx[i], ctypes[i]
        if k == "cor":
            if e["var"] not in env:
                raise Skip(f"unbound correlation variable {e['var']}")
            names, cols, types = env[e["var"]]
            if names is None or names.count(e["field"]) != 1:
                raise Skip("correlation field not resolvable by name")
            j = names.index(e["field"])
            return cols[j], types[j]
        if k == "lit":
            return self.literal(e)
        if k == "sarg":
            raise Skip("Sarg outside SEARCH")
        if k == "flag":
            raise Skip("flag outside call")
        if k == "plan":
            raise Skip("subquery outside call")
        if k == "over":
            return self.over(e, ctx, ctypes, env)
        op, args = e["op"], e["args"]
        suf = e.get("suffix")
        out = self.call(op, args, suf, ctx, ctypes, env)
        return out

    WHOLE_FRAMES = ("ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
                    "RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING")

    def over(self, e, ctx, ctypes, env) -> tuple[str, T]:
        """``op(args) OVER (window)`` in a projection, as RexOver prints it (see ``window_call``)."""
        if not self.windows_allowed:
            raise Skip("window function outside a projection")
        if e.get("suffix"):
            raise Skip("typed window function")
        partition = [self.expr(key, ctx, ctypes, env)[0] for key in e["partition"]]
        order = []
        for key, words in e["order"]:
            desc = " DESC" in words
            nulls_last = " NULLS LAST" in words or (not desc and " NULLS FIRST" not in words)
            order.append((self.expr(key, ctx, ctypes, env)[0], desc, nulls_last))
        frame = e["frame"]
        if frame and (frame not in self.WHOLE_FRAMES or not order):
            # RexWindow prints no frame for the default one; with no ORDER BY it prints any ROWS frame
            raise Skip("window frame (its result depends on the order of ties)")
        args = [self.expr(a, ctx, ctypes, env) for a in e["args"]]
        return self.window_call(e["op"], args, partition, order, whole=bool(frame) or not order)

    def window_call(self, op: str, args: list, partition: list, order: list, whole: bool) -> tuple[str, T]:
        """A windowed call whose value does not depend on how tied rows are ordered.

        ``whole``: the frame is the whole partition (no ``ORDER BY``, or a frame unbounded on both sides);
        otherwise it is the default frame of an ``ORDER BY``, every row up to the current row's last peer, which
        MySQL and DuckDB share. Translated: SUM, $SUM0, COUNT, MIN, MAX and integer or approximate AVG over
        either frame, and RANK / DENSE_RANK. Not translated: ROW_NUMBER, LAG, LEAD, FIRST_VALUE and the like and
        every other ``ROWS`` or ``RANGE`` frame: their result changes with the order of tied rows, so two plans
        need not agree row for row. ``order`` is ``(sql, descending, nulls_last)`` per key; the null position is
        written as a key of its own because MySQL has no ``NULLS FIRST/LAST``.
        """
        keys = []
        for x, desc, nulls_last in order:
            keys.append(f"({x} IS NULL){'' if nulls_last else ' DESC'}")
            keys.append(f"{x}{' DESC' if desc else ''}")
        by = "PARTITION BY " + ", ".join(partition) if partition else ""
        ordered = " ".join(part for part in (by, "ORDER BY " + ", ".join(keys) if keys else "") if part)
        if op in ("RANK", "DENSE_RANK"):
            if args or not order:
                raise Skip(f"window function {op} shape")
            return f"{op}() OVER ({ordered})", BIGINT
        if op not in ("SUM", "$SUM0", "COUNT", "MIN", "MAX", "AVG"):
            raise Skip(f"window function {op} (not modelled, or its result depends on the order of ties)")
        # over the whole partition these aggregates do not read the order at all
        w = f"OVER ({by if whole else ordered})"
        if op == "COUNT":
            if not args:
                return f"COUNT(*) {w}", BIGINT
            if len(args) > 1:
                nn = " AND ".join(f"{x} IS NOT NULL" for x, _ in args)
                return f"COUNT(CASE WHEN {nn} THEN 1 END) {w}", BIGINT
            return f"COUNT({args[0][0]}) {w}", BIGINT
        if len(args) != 1:
            raise Skip(f"window function {op} with {len(args)} arguments")
        x, t = args[0]
        if op in ("MIN", "MAX"):
            return f"{op}({x}) {w}", t
        if op in ("SUM", "$SUM0"):
            if t.fam not in NUMERIC and t.base != "NULL":
                raise Skip("SUM of non-numeric")
            rt = T("DECIMAL", 19, t.s) if t.base == "DECIMAL" else t
            return (f"SUM({x}) {w}" if op == "SUM" else f"COALESCE(SUM({x}) {w}, 0)"), rt
        if t.fam == "int":
            return f"(SUM({x}) {w} DIV COUNT({x}) {w})", t
        if t.fam == "approx":
            return f"AVG({x}) {w}", DOUBLE
        raise Skip("AVG of DECIMAL (Calcite result scale not modelled)")

    WINDOW_GROUP_RE = re.compile(r"window\((?:partition \{([\d, ]*)\})? ?(?:order by \[([^\]]*)\])? ?"
                                 r"(?:(rows|range) between (.+?) and (.+?))? ?(?:aggs \[(.*)\])?\)")

    def rel_LogicalWindow(self, node, env):
        """The input's columns followed by every windowed call of every ``window#i`` group, in order.

        A group prints as ``window(partition {7} order by [0 DESC] aggs [COUNT($0), SUM($5)])`` (Window.Group):
        no frame text is the whole partition without ``order by`` and the default frame with it.
        """
        src = self.nonempty(self.one_child(node, env), "a window")
        a = self.alias()
        cols, ctypes = self.cols(a, len(src.types)), list(src.types)
        exprs, types, names = list(cols), list(ctypes), list(src.names) if src.names else [None] * len(cols)
        for k, v in node.attrs:
            if not re.fullmatch(r"window#\d+", k):
                raise Skip(f"LogicalWindow attribute {k}")
            m = self.WINDOW_GROUP_RE.fullmatch(v.strip())
            if not m or not m.group(6):
                raise Skip("LogicalWindow group shape")
            part_txt, order_txt, unit, lower, upper, aggs_txt = m.groups()
            partition = [int(x) for x in re.findall(r"\d+", part_txt or "")]
            order = []
            for item in [x.strip() for x in (order_txt or "").split(",") if x.strip()]:
                om = re.fullmatch(r"(\d+)(?: (ASC|DESC))?(?: (FIRST|LAST))?", item)
                if not om:
                    raise Skip(f"window order key {item!r}")
                desc = om.group(2) == "DESC"
                nulls_last = om.group(3) == "LAST" if om.group(3) else not desc
                order.append((int(om.group(1)), desc, nulls_last))
            if any(not 0 <= i < len(cols) for i in partition + [i for i, _, _ in order]):
                raise Skip("window key out of range")
            whole = not order
            if unit:
                bounds = (lower, upper)
                if bounds == ("UNBOUNDED PRECEDING", "UNBOUNDED FOLLOWING"):
                    whole = True
                elif not (unit == "range" and not order and bounds == ("UNBOUNDED PRECEDING", "CURRENT ROW")):
                    # (RANGE up to the current row with no ORDER BY: every row is a peer, the whole partition)
                    raise Skip("window frame (its result depends on the order of ties)")
            for call in split_top(aggs_txt, ","):
                e = parse_rex(call.strip())
                if e.get("k") != "call" or e.get("suffix"):
                    raise Skip("windowed call shape")
                args = [self.expr(arg, cols, ctypes, env) for arg in e["args"]]
                x, t = self.window_call(e["op"], args, [cols[i] for i in partition],
                                        [(cols[i], desc, nulls_last) for i, desc, nulls_last in order], whole)
                exprs.append(x)
                types.append(t)
                names.append(None)
        return Rel(f"SELECT {self.select_list(exprs)} FROM ({src.sql}) AS {a}", types, names)

    def call(self, op, args, suf, ctx, ctypes, env) -> tuple[str, T]:
        if op in NONDET:
            raise Skip(f"non-deterministic function {op}")
        if op in MEASURE_OPS:
            raise Skip("measure expression (Calcite-specific)")
        if op in SCALAR_AGGS:
            raise Skip(f"aggregate function {op} in scalar position")
        if any(a["k"] == "plan" for a in args):
            return self.subquery(op, args, ctx, ctypes, env)
        if op == "SEARCH":
            if len(args) != 2 or args[1]["k"] != "sarg":
                raise Skip("SEARCH shape")
            x, xt = self.expr(args[0], ctx, ctypes, env)
            return self.search(x, xt, args[1]["body"], args[1].get("suffix"))
        if op == "CAST":
            if len(args) != 1 or suf is None:
                raise Skip("CAST shape")
            x, xt = self.expr(args[0], ctx, ctypes, env)
            return self.cast(x, xt, parse_type(suf))
        if op == "EXTRACT":
            if len(args) != 2 or args[0]["k"] != "flag":
                raise Skip("EXTRACT shape")
            unit = args[0]["v"]
            if unit not in ("YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND", "QUARTER"):
                raise Skip(f"EXTRACT unit {unit}")
            x, xt = self.expr(args[1], ctx, ctypes, env)
            if xt.base not in ("DATE", "TIMESTAMP"):
                raise Skip("EXTRACT from non-datetime")
            if xt.base == "DATE" and unit in ("HOUR", "MINUTE", "SECOND"):
                raise Skip("EXTRACT time unit from DATE")
            return f"EXTRACT({unit} FROM {x})", BIGINT
        if op == "TRIM":
            if len(args) != 3 or args[0]["k"] != "flag" or args[0]["v"] not in ("BOTH", "LEADING", "TRAILING"):
                raise Skip("TRIM shape")
            c, ct = self.expr(args[1], ctx, ctypes, env)
            x, xt = self.expr(args[2], ctx, ctypes, env)
            if ct.fam != "str" or xt.fam != "str":
                raise Skip("TRIM of non-string")
            return f"TRIM({args[0]['v']} {c} FROM {x})", T("VARCHAR", xt.p)
        if any(a["k"] == "flag" for a in args):
            raise Skip(f"flag argument to {op}")
        a = [self.expr(x, ctx, ctypes, env) for x in args]
        sq = [x for x, _ in a]
        ts = [t for _, t in a]
        if suf is not None and op not in ("CASE", "COALESCE"):
            raise Skip(f"typed call {op}")
        if op in CMP and len(a) == 2:
            self.check_comparable(ts[0], ts[1])
            return f"({sq[0]} {op} {sq[1]})", BOOLEAN
        if op in ("IS NOT DISTINCT FROM", "IS DISTINCT FROM") and len(a) == 2:
            self.check_comparable(ts[0], ts[1])
            s = f"({sq[0]} <=> {sq[1]})"
            return (s if op == "IS NOT DISTINCT FROM" else f"(NOT {s})"), BOOLEAN
        if op in ("AND", "OR") and len(a) >= 2:
            self.need_bool(ts, op)
            return "(" + f" {op} ".join(sq) + ")", BOOLEAN
        if op == "NOT" and len(a) == 1:
            self.need_bool(ts, op)
            return f"(NOT {sq[0]})", BOOLEAN
        if op in IS_OPS and len(a) == 1:
            if op != "IS NULL" and op != "IS NOT NULL":
                self.need_bool(ts, op)
            return f"({sq[0]} {op})", BOOLEAN
        if op in ("+", "-", "*") and len(a) == 2:
            return self.arith(op, sq, ts)
        if op == "-" and len(a) == 1:
            if ts[0].fam not in NUMERIC:
                raise Skip("unary minus on non-numeric")
            return f"(-{sq[0]})", ts[0]
        if op == "+" and len(a) == 1:
            if ts[0].fam not in NUMERIC:
                raise Skip("unary plus on non-numeric")
            return sq[0], ts[0]
        if op in ("/", "/INT") and len(a) == 2:
            fams = {t.fam for t in ts if t.base != "NULL"}
            if not fams <= NUMERIC:
                raise Skip("division of non-numeric")
            if fams <= {"int"}:
                t = max((t for t in ts if t.base != "NULL"), key=lambda t: INT_RANK[t.base], default=INTEGER)
                return f"({sq[0]} DIV {sq[1]})", t
            if "approx" in fams:
                return f"({sq[0]} / {sq[1]})", DOUBLE
            raise Skip("DECIMAL division (Calcite result scale not modelled)")
        if op == "MOD" and len(a) == 2:
            if not all(t.fam == "int" or t.base == "NULL" for t in ts):
                raise Skip("MOD of non-integers")
            return f"MOD({sq[0]}, {sq[1]})", ts[1] if ts[1].base != "NULL" else ts[0]
        if op == "CASE" and len(a) >= 3 and len(a) % 2 == 1:
            conds = sq[0:-1:2]
            self.need_bool(ts[0:-1:2], "CASE")
            vals = sq[1::2] + [sq[-1]]
            vts = ts[1::2] + [ts[-1]]
            rt = least_restrictive(vts, "CASE")
            if suf is not None:
                st = parse_type(suf)
                if st.fam != rt.fam and rt.base != "NULL":
                    raise Skip("CASE with type suffix of other family")
                rt = st
            whens = "".join(f" WHEN {c} THEN {v}" for c, v in zip(conds, vals[:-1]))
            return f"(CASE{whens} ELSE {vals[-1]} END)", rt
        if op == "COALESCE" and len(a) >= 2:
            rt = least_restrictive(ts, "COALESCE")
            return f"COALESCE({', '.join(sq)})", rt
        if op == "POWER" and len(a) == 2:
            if not all(t.fam in NUMERIC for t in ts):
                raise Skip("POWER of non-numeric")
            return f"POWER({sq[0]}, {sq[1]})", DOUBLE
        if op in ("UPPER", "LOWER") and len(a) == 1:
            if ts[0].fam != "str":
                raise Skip(f"{op} of non-string")
            return f"{op}({sq[0]})", ts[0]
        if op in ("CHAR_LENGTH", "CHARACTER_LENGTH") and len(a) == 1:
            if ts[0].fam != "str":
                raise Skip(f"{op} of non-string")
            return f"CHAR_LENGTH({sq[0]})", INTEGER
        if op == "ABS" and len(a) == 1:
            if ts[0].fam not in NUMERIC:
                raise Skip("ABS of non-numeric")
            return f"ABS({sq[0]})", ts[0]
        if op == "||" and len(a) == 2:
            if not all(t.fam == "str" for t in ts):
                raise Skip("|| of non-strings")
            n = None if any(t.p is None for t in ts) else ts[0].p + ts[1].p
            base = "CHAR" if all(t.base == "CHAR" for t in ts) else "VARCHAR"
            return f"CONCAT({sq[0]}, {sq[1]})", T(base, n)
        if op == "LIKE" and len(a) in (2, 3):
            if not all(t.fam == "str" for t in ts):
                raise Skip("LIKE of non-strings")
            esc = f" ESCAPE {sq[2]}" if len(a) == 3 else ""
            return f"({sq[0]} LIKE {sq[1]}{esc})", BOOLEAN
        if op == "SUBSTRING" and len(a) in (2, 3):
            for j, x in enumerate(args[1:]):
                if not (x["k"] == "lit" and x["kind"] == "int" and not x.get("suffix") and int(x["v"]) >= 1 - j):
                    raise Skip("SUBSTRING with non-literal or non-positive bounds")
            if ts[0].fam != "str":
                raise Skip("SUBSTRING of non-string")
            return f"SUBSTRING({', '.join(sq)})", T("VARCHAR", ts[0].p)
        raise Skip(f"unsupported operator {op}")

    @staticmethod
    def need_bool(ts, op):
        for t in ts:
            if t.base not in ("BOOLEAN", "NULL"):
                raise Skip(f"{op} of non-boolean")

    def arith(self, op, sq, ts) -> tuple[str, T]:
        fams = {t.fam for t in ts if t.base != "NULL"}
        if not fams <= NUMERIC:
            raise Skip(f"arithmetic {op} on {sorted(fams)}")
        nn = [t for t in ts if t.base != "NULL"]
        if "approx" in fams:
            rt = DOUBLE
        elif "dec" in fams:
            (i1, s1), (i2, s2) = [dec_shape(t) for t in (nn if len(nn) == 2 else nn * 2)]
            if op == "*":
                rt = T("DECIMAL", min(i1 + i2 + s1 + s2, 19), s1 + s2)
            else:
                s = max(s1, s2)
                rt = T("DECIMAL", min(max(i1, i2) + 1 + s, 19), s)
        else:
            rt = max(nn, key=lambda t: INT_RANK[t.base]) if nn else INTEGER
            # Calcite types INT op INT as the wider operand type; widen explicitly
            sq = [x if t.base in (rt.base, "NULL") else f"CAST({x} AS {rt.base})" for x, t in zip(sq, ts)]
        return f"({sq[0]} {op} {sq[1]})", rt

    def cast(self, x: str, xt: T, tt: T) -> tuple[str, T]:
        if xt.base == "NULL":
            return f"CAST({x} AS {self.cast_name(tt)})", tt
        if xt == tt:
            return x, tt
        f, g = xt.fam, tt.fam
        if f == "int" and g == "int":
            return f"CAST({x} AS {tt.base})", tt
        if f == "int" and g == "dec":
            # exact; a value too wide for the target is an overflow error in both engines
            return f"CAST({x} AS {self.cast_name(tt)})", tt
        if f == "dec" and g == "dec":
            if tt.s < xt.s:
                raise Skip("DECIMAL cast reducing scale (rounding mode not modelled)")
            return f"CAST({x} AS {self.cast_name(tt)})", tt
        if f in NUMERIC and g == "approx":
            return f"CAST({x} AS DOUBLE)", DOUBLE
        if f == "str" and g == "str":
            if tt.base == "VARCHAR" and (tt.p is None or (xt.p is not None and xt.p <= tt.p)):
                return x, tt
            if tt.base == "CHAR" and xt.base == "CHAR" and xt.p == tt.p:
                return x, tt
            raise Skip("string CAST that may truncate or pad")
        if f == g and f in ("bool", "date", "ts"):
            return x, tt
        if f == "date" and g == "ts":
            return f"CAST({x} AS TIMESTAMP)", tt
        raise Skip(f"CAST from {xt.base} to {tt.base}")

    def subquery(self, op, args, ctx, ctypes, env) -> tuple[str, T]:
        plans = [a for a in args if a["k"] == "plan"]
        if len(plans) != 1 or args[-1]["k"] != "plan":
            raise Skip(f"subquery operator shape {op}")
        sub = self.rel(plans[0]["plan"], env)
        if sub.empty:
            if op == "EXISTS" and len(args) == 1:
                return "EXISTS (SELECT 1 WHERE 1 = 0)", BOOLEAN
            self.nonempty(sub, f"a {op} subquery")
        operands = [self.expr(a, ctx, ctypes, env) for a in args[:-1]]
        if op == "EXISTS" and not operands:
            return f"EXISTS ({sub.sql})", BOOLEAN
        if op == "$SCALAR_QUERY" and not operands:
            if len(sub.types) != 1:
                raise Skip("scalar subquery with several columns")
            return f"({sub.sql})", sub.types[0]
        if op == "IN" and operands:
            if len(operands) != len(sub.types):
                raise Skip("IN arity mismatch")
            for (_, t), st in zip(operands, sub.types):
                self.check_comparable(t, st)
            lhs = operands[0][0] if len(operands) == 1 else "(" + ", ".join(x for x, _ in operands) + ")"
            return f"({lhs} IN ({sub.sql}))", BOOLEAN
        m = SUB_CMP_RE.match(op)
        if m and len(operands) == 1 and len(sub.types) == 1:
            self.check_comparable(operands[0][1], sub.types[0])
            q = "ANY" if m.group(2) == "SOME" else "ALL"
            return f"({operands[0][0]} {m.group(1)} {q} ({sub.sql}))", BOOLEAN
        raise Skip(f"unsupported subquery operator {op}")

    # ------------------------------------------------------------ aggregates
    def agg(self, raw: str, cols: list, types: list, has_keys: bool, keys: list | None = None,
            sets: list | None = None) -> tuple[str, T]:
        gm = re.fullmatch(r"(GROUPING|GROUPING_ID)\(((?:\$\d+(?:, )?)+)\)", raw.strip())
        if gm:
            # GROUPING(k1, .., kn): one bit per argument, leftmost the most significant, set when the
            # argument is not a key of the row's grouping set (always 0 without grouping sets).
            args = [int(x) for x in re.findall(r"\$(\d+)", gm.group(2))]
            if any(i not in (keys or []) for i in args):
                raise Skip("GROUPING of a column that is not a group key")
            if not sets:
                return "0", BIGINT
            return f"GROUPING({', '.join(cols[i] for i in args)})", BIGINT
        lm = re.fullmatch(r"LITERAL_AGG\((.*)\)", raw.strip())
        if lm:
            if not has_keys:
                raise Skip("LITERAL_AGG without group keys")
            e = parse_rex(lm.group(1))
            if e["k"] != "lit":
                raise Skip("LITERAL_AGG of non-literal")
            # one output row per non-empty group, carrying the literal
            return self.literal(e)
        m = AGG_RE.match(raw.strip())
        if not m:
            if " WITHIN DISTINCT" in raw:
                raise Skip("aggregate WITHIN DISTINCT")
            if " WITHIN GROUP" in raw:
                raise Skip("aggregate WITHIN GROUP")
            raise Skip(f"aggregate call shape {raw[:40]!r}")
        fn, distinct, argtxt, filt = m.group(1), bool(m.group(2)), m.group(3), m.group(4)
        idx = [int(x) for x in re.findall(r"\$(\d+)", argtxt)]
        for i in idx + ([int(filt)] if filt else []):
            if not 0 <= i < len(cols):
                raise Skip("aggregate argument out of range")
        a = [cols[i] for i in idx]
        at = [types[i] for i in idx]
        fcond = None
        if filt is not None:
            if types[int(filt)].base != "BOOLEAN":
                raise Skip("aggregate FILTER on non-boolean")
            fcond = cols[int(filt)]
        dis = "DISTINCT " if distinct else ""

        def guard(x):
            return x if fcond is None else f"CASE WHEN {fcond} THEN {x} END"

        if fn == "COUNT":
            if not a:
                if distinct:
                    raise Skip("COUNT(DISTINCT) without operand")
                return ("COUNT(*)" if fcond is None else f"COUNT({guard('1')})"), BIGINT
            if len(a) > 1:
                if distinct:
                    # MySQL's COUNT(DISTINCT a, b) is Calcite's: distinct rows with no NULL among the arguments
                    return f"COUNT(DISTINCT {', '.join(guard(x) for x in a)})", BIGINT
                nn = " AND ".join(f"{x} IS NOT NULL" for x in a)
                cond = nn if fcond is None else f"{fcond} AND {nn}"
                return f"COUNT(CASE WHEN {cond} THEN 1 END)", BIGINT
            return f"COUNT({dis}{guard(a[0])})", BIGINT
        if len(a) != 1:
            raise Skip(f"aggregate {fn} with {len(a)} arguments")
        x, t = guard(a[0]), at[0]
        if fn in ("MIN", "MAX"):
            return f"{fn}({dis}{x})", t
        if fn in ("SUM", "$SUM0"):
            if t.fam not in NUMERIC and t.base != "NULL":
                raise Skip("SUM of non-numeric")
            rt = T("DECIMAL", 19, t.s) if t.base == "DECIMAL" else t
            s = f"SUM({dis}{x})"
            return (s if fn == "SUM" else f"COALESCE({s}, 0)"), rt
        if fn == "AVG":
            if t.fam == "int":
                return f"(SUM({dis}{x}) DIV COUNT({dis}{x}))", t
            if t.fam == "approx":
                return f"AVG({dis}{x})", DOUBLE
            raise Skip("AVG of DECIMAL (Calcite result scale not modelled)")
        if fn in ("ANY_VALUE", "SINGLE_VALUE", "LITERAL_AGG", "GROUPING", "GROUP_ID", "GROUPING_ID"):
            raise Skip(f"unsupported aggregate {fn}")
        raise Skip(f"unsupported aggregate {fn}")

    # ------------------------------------------------------------ relations
    @staticmethod
    def cols(alias: str, n: int) -> list:
        return [f"{alias}.c{i}" for i in range(n)]

    @staticmethod
    def select_list(exprs: list) -> str:
        return ", ".join(f"{e} AS c{i}" for i, e in enumerate(exprs))

    def rel(self, node: PNode, env: dict) -> Rel:
        self.depth += 1
        windows, self.windows_allowed = self.windows_allowed, False  # a nested plan starts outside any select list
        try:
            if self.depth > 60:
                raise Skip("plan too deep")
            fn = getattr(self, "rel_" + node.kind, None)
            if fn is None:
                raise Skip(f"unsupported relational operator {node.kind}")
            allowed = ALLOWED_ATTRS.get(node.kind)
            if allowed:
                for k, _ in node.attrs:
                    if k not in allowed:
                        raise Skip(f"{node.kind} attribute {k}")
            return fn(node, env)
        finally:
            self.depth -= 1
            self.windows_allowed = windows

    @staticmethod
    def attr(node: PNode, name: str, default=None):
        for k, v in node.attrs:
            if k == name:
                return v
        return default

    def one_child(self, node: PNode, env) -> Rel:
        if len(node.children) != 1:
            raise Skip(f"{node.kind} with {len(node.children)} inputs")
        return self.rel(node.children[0], env)

    def nonempty(self, r: Rel, what: str) -> Rel:
        if r.empty:
            raise Skip(f"empty LogicalValues under {what} (row width not in the digest)")
        return r

    def input_ctx(self, src: Rel, a: str):
        """Column references and types of an input; an empty input of unknown width gets a
        wide virtual row of NULL-typed columns (no row ever reaches them)."""
        if src.empty:
            return self.cols(a, VIRTUAL_WIDTH), [NULLT] * VIRTUAL_WIDTH
        return self.cols(a, len(src.types)), src.types

    @staticmethod
    def input_sql(src: Rel, a: str, sql_after: str) -> str:
        if not src.empty:
            return src.sql
        used = [int(x) for x in re.findall(rf"\b{a}\.c(\d+)\b", sql_after)]
        k = max(used, default=0) + 1
        return f"SELECT {', '.join(f'NULL AS c{i}' for i in range(k))} WHERE 1 = 0"

    def top(self, node: PNode) -> Rel:
        r = self.rel(node, {})
        if r.empty:
            if self.root_hint is None:
                raise NeedHint()
            exprs = [f"CAST(NULL AS {self.cast_name(t)})" if t.base != "NULL" else "NULL" for t in self.root_hint]
            return Rel(f"SELECT {self.select_list(exprs)} WHERE 1 = 0", list(self.root_hint), None)
        return r

    def rel_LogicalTableScan(self, node, env):
        tbl = self.attr(node, "table")
        path = tuple(x.strip() for x in tbl.strip("[]").split(","))
        if path not in CATALOG:
            last = path[-1]
            raise Skip(KNOWN_UNSUPPORTED_TABLES.get(last, f"table {'.'.join(path)} outside the modelled catalogs"))
        name, cols, _ = CATALOG[path]
        prev = [p for p, e in self.tables.items() if e[0] == name and p != path]
        if prev:
            raise Skip("two catalogs' tables share a name")
        self.tables[path] = CATALOG[path]
        sel = self.select_list([qident(c[0]) for c in cols])
        return Rel(f"SELECT {sel} FROM {qident(name)}", [c[1] for c in cols], [c[0] for c in cols])

    def bind(self, node, env, r: Rel, a: str) -> dict:
        vs = self.attr(node, "variablesSet")
        if vs is None:
            return env
        self.nonempty(r, "a correlated operator")
        vars_ = re.findall(r"\$cor\d+", vs)
        env = dict(env)
        for v in vars_:
            env[v] = (r.names, self.cols(a, len(r.types)), r.types)
        return env

    def rel_LogicalProject(self, node, env):
        src = self.one_child(node, env)
        a = self.alias()
        env2 = self.bind(node, env, src, a)
        cols, ctypes = self.input_ctx(src, a)
        exprs, types, names = [], [], []
        for k, v in node.attrs:
            if k == "variablesSet":
                continue
            self.windows_allowed = True
            try:
                x, t = self.expr(parse_rex(v), cols, ctypes, env2)
            finally:
                self.windows_allowed = False
            exprs.append(x)
            types.append(t)
            names.append(k)
        if not exprs:
            raise Skip("projection with no columns")
        sel = self.select_list(exprs)
        return Rel(f"SELECT {sel} FROM ({self.input_sql(src, a, sel)}) AS {a}", types, names)

    def rel_LogicalFilter(self, node, env):
        src = self.one_child(node, env)
        if src.empty:
            return src
        a = self.alias()
        env2 = self.bind(node, env, src, a)
        cond, t = self.expr(parse_rex(self.attr(node, "condition")), self.cols(a, len(src.types)), src.types, env2)
        self.need_bool([t], "filter condition")
        sel = self.select_list(self.cols(a, len(src.types)))
        return Rel(f"SELECT {sel} FROM ({src.sql}) AS {a} WHERE {cond}", src.types, src.names)

    def rel_LogicalJoin(self, node, env):
        if self.attr(node, "variablesSet") is not None:
            raise Skip("join with correlation variables")
        if len(node.children) != 2:
            raise Skip("join arity")
        left = self.nonempty(self.rel(node.children[0], env), "a join")
        right = self.nonempty(self.rel(node.children[1], env), "a join")
        la, ra = self.alias(), self.alias()
        ctx = self.cols(la, len(left.types)) + self.cols(ra, len(right.types))
        cond, t = self.expr(parse_rex(self.attr(node, "condition")), ctx, left.types + right.types, env)
        self.need_bool([t], "join condition")
        kind = self.attr(node, "joinType")
        if kind in ("semi", "anti"):
            neg = "NOT " if kind == "anti" else ""
            sel = self.select_list(self.cols(la, len(left.types)))
            return Rel(f"SELECT {sel} FROM ({left.sql}) AS {la} WHERE {neg}EXISTS "
                       f"(SELECT 1 FROM ({right.sql}) AS {ra} WHERE {cond})", left.types, left.names)
        jk = {"inner": "INNER", "left": "LEFT", "right": "RIGHT", "full": "FULL"}.get(kind)
        if jk is None:
            raise Skip(f"join type {kind}")
        sel = self.select_list(ctx)
        names = uniquify(left.names + right.names) if left.names and right.names else None
        return Rel(f"SELECT {sel} FROM ({left.sql}) AS {la} {jk} JOIN ({right.sql}) AS {ra} ON {cond}",
                   left.types + right.types, names)

    def rel_LogicalCorrelate(self, node, env):
        if len(node.children) != 2:
            raise Skip("correlate arity")
        var = self.attr(node, "correlation")
        left = self.nonempty(self.rel(node.children[0], env), "a correlate")
        la = self.alias()
        env2 = dict(env)
        env2[var] = (left.names, self.cols(la, len(left.types)), left.types)
        right = self.nonempty(self.rel(node.children[1], env2), "a correlate")
        ra = self.alias()
        kind = self.attr(node, "joinType")
        if kind in ("semi", "anti"):
            neg = "NOT " if kind == "anti" else ""
            sel = self.select_list(self.cols(la, len(left.types)))
            return Rel(f"SELECT {sel} FROM ({left.sql}) AS {la} WHERE {neg}EXISTS ({right.sql})",
                       left.types, left.names)
        sel = self.select_list(self.cols(la, len(left.types)) + self.cols(ra, len(right.types)))
        names = uniquify(left.names + right.names) if left.names and right.names else None
        types = left.types + right.types
        if kind == "inner":
            return Rel(f"SELECT {sel} FROM ({left.sql}) AS {la} CROSS JOIN LATERAL ({right.sql}) AS {ra}",
                       types, names)
        if kind == "left":
            return Rel(f"SELECT {sel} FROM ({left.sql}) AS {la} LEFT JOIN LATERAL ({right.sql}) AS {ra} ON TRUE",
                       types, names)
        raise Skip(f"correlate type {kind}")

    def rel_LogicalAggregate(self, node, env):
        src = self.one_child(node, env)
        a = self.alias()
        cols, ctypes = self.input_ctx(src, a)
        keys_txt = None
        sets_txt = None
        aggs = []
        for k, v in node.attrs:
            if k == "group":
                keys_txt = v
            elif k == "groups":
                sets_txt = v
            else:
                aggs.append((k, v))
        if keys_txt is None or not (keys_txt.startswith("{") and keys_txt.endswith("}")):
            raise Skip("aggregate group shape")
        keys = [int(x) for x in re.findall(r"\d+", keys_txt)]
        if any(not 0 <= i < len(cols) for i in keys):
            raise Skip("group key out of range")
        sets = None
        if sets_txt is not None:
            # groups=[[{0, 7}, {0}, {}]]: one output row per group of every grouping set; a key the
            # set leaves out is NULL in that row (standard GROUPING SETS, which the SQL states as is)
            if not re.fullmatch(r"\[\{[\d, ]*\}(?:, \{[\d, ]*\})*\]", sets_txt):
                raise Skip("grouping sets shape")
            sets = [[int(x) for x in re.findall(r"\d+", part)] for part in re.findall(r"\{([\d, ]*)\}", sets_txt)]
            if any(i not in keys for one in sets for i in one) or {i for one in sets for i in one} != set(keys):
                raise Skip("grouping sets do not cover the group keys")
            if len({tuple(sorted(one)) for one in sets}) != len(sets):
                raise Skip("duplicate grouping sets")
            if src.empty:
                raise Skip("grouping sets over an empty LogicalValues")
        if not keys and sets is None and len(aggs) == 1 and not src.empty:
            single = re.fullmatch(r"SINGLE_VALUE\(\$(\d+)\)", aggs[0][1].strip())
            if single and int(single.group(1)) < len(cols):
                # with no keys SINGLE_VALUE is NULL over no rows, the value over one row and an error over
                # more: what a scalar subquery is, so the one-row relation is a select of that subquery
                i = int(single.group(1))
                return Rel(f"SELECT (SELECT {cols[i]} FROM ({src.sql}) AS {a}) AS c0", [ctypes[i]], [aggs[0][0]])
        exprs = [cols[i] for i in keys]
        types = [ctypes[i] for i in keys]
        names = [src.names[i] if src.names else None for i in keys]
        for name, raw in aggs:
            x, t = self.agg(raw, cols, ctypes, bool(keys), keys, sets)
            exprs.append(x)
            types.append(t)
            names.append(name)
        if not exprs:
            raise Skip("aggregate with no keys and no calls (zero-column relation)")
        if sets is not None:
            gb = " GROUP BY GROUPING SETS (" + ", ".join("(" + ", ".join(cols[i] for i in one) + ")" for one in sets) + ")"
        else:
            gb = f" GROUP BY {', '.join(cols[i] for i in keys)}" if keys else ""
        sel = self.select_list(exprs)
        return Rel(f"SELECT {sel} FROM ({self.input_sql(src, a, sel + gb)}) AS {a}{gb}", types, names)

    def rel_LogicalSort(self, node, env):
        src = self.one_child(node, env)
        if src.empty:
            return src
        a = self.alias()
        n = len(src.types)
        attrs = dict(node.attrs)
        for k in attrs:
            if not re.fullmatch(r"sort\d+|dir\d+|fetch|offset", k):
                raise Skip(f"LogicalSort attribute {k}")
        keys = []
        i = 0
        while f"sort{i}" in attrs:
            m = re.fullmatch(r"\$(\d+)", attrs[f"sort{i}"])
            if not m or int(m.group(1)) >= n:
                raise Skip("sort key shape")
            d = attrs.get(f"dir{i}")
            col = f"{a}.c{m.group(1)}"
            dirs = {"ASC": (False, True), "DESC": (True, False), "ASC-nulls-first": (False, False),
                    "DESC-nulls-last": (True, True)}
            if d not in dirs:
                raise Skip(f"sort direction {d}")
            desc, nulls_last = dirs[d]
            keys.append(f"({col} IS NULL){'' if nulls_last else ' DESC'}")
            keys.append(f"{col}{' DESC' if desc else ''}")
            i += 1
        sql = f"SELECT {self.select_list(self.cols(a, n))} FROM ({src.sql}) AS {a}"
        if keys:
            sql += " ORDER BY " + ", ".join(keys)
        fetch, offset = attrs.get("fetch"), attrs.get("offset")
        for v in (fetch, offset):
            if v is not None and not re.fullmatch(r"\d+", v):
                raise Skip("non-literal LIMIT/OFFSET")
        if (fetch is not None or offset is not None) and not keys:
            raise Skip("LIMIT/OFFSET without ORDER BY (nondeterministic)")
        if fetch is not None:
            sql += f" LIMIT {fetch}"
        elif offset is not None:
            sql += " LIMIT 9223372036854775807"
        if offset is not None:
            sql += f" OFFSET {offset}"
        return Rel(sql, src.types, src.names)

    def setop(self, node, env, keyword_all, keyword_distinct):
        allv = self.attr(node, "all")
        if allv not in ("true", "false"):
            raise Skip("set operation without all flag")
        if len(node.children) == 1:
            # Calcite allows a one-input set operation: the input itself (ALL) or its distinct rows
            p = self.nonempty(self.rel(node.children[0], env), "a set operation")
            if allv == "true":
                return p
            a = self.alias()
            n = len(p.types)
            return Rel(f"SELECT DISTINCT {self.select_list(self.cols(a, n))} FROM ({p.sql}) AS {a}", p.types, p.names)
        if len(node.children) < 2:
            raise Skip("set operation with no inputs")
        parts = [self.nonempty(self.rel(c, env), "a set operation") for c in node.children]
        w = {len(p.types) for p in parts}
        if len(w) != 1:
            raise Skip("set operation inputs with different widths")
        n = w.pop()
        types = [least_restrictive([p.types[i] for p in parts], "set operation") for i in range(n)]
        kw = keyword_all if allv == "true" else keyword_distinct
        a = self.alias()
        body = f" {kw} ".join(f"({p.sql})" for p in parts)
        return Rel(f"SELECT {self.select_list(self.cols(a, n))} FROM ({body}) AS {a}", types, parts[0].names)

    def rel_LogicalUnion(self, node, env):
        return self.setop(node, env, "UNION ALL", "UNION")

    def rel_LogicalIntersect(self, node, env):
        return self.setop(node, env, "INTERSECT ALL", "INTERSECT")

    def rel_LogicalMinus(self, node, env):
        return self.setop(node, env, "EXCEPT ALL", "EXCEPT")

    def rel_LogicalValues(self, node, env):
        if node.children:
            raise Skip("values with inputs")
        raw = self.attr(node, "tuples").strip()
        if not (raw.startswith("[") and raw.endswith("]")):
            raise Skip("values shape")
        inner = raw[1:-1].strip()
        if inner == "":
            if self.empty_tables:
                # the test's Java names the scan this empty relation was made from (java_empty_tables)
                _, cols, _ = CATALOG[self.empty_tables.pop(0)]
                exprs = [f"CAST(NULL AS {self.cast_name(t)})" for _, t, _ in cols]
                return Rel(f"SELECT {self.select_list(exprs)} WHERE 1 = 0", [t for _, t, _ in cols], [n for n, _, _ in cols])
            return Rel(None, None, None, empty=True)
        rows = []
        for tup in split_top(inner, ", "):
            tup = tup.strip()
            if not (tup.startswith("{") and tup.endswith("}")):
                raise Skip("values tuple shape")
            body = tup[1:-1].strip()
            if body == "":
                raise Skip("VALUES with no columns")
            row = []
            for item in split_top(body, ", "):
                e = parse_rex(item.strip())
                if e["k"] != "lit":
                    raise Skip("non-literal in VALUES")
                row.append(self.literal(e))
            rows.append(row)
        n = len(rows[0])
        if any(len(r) != n for r in rows):
            raise Skip("VALUES row width mismatch")
        types = [least_restrictive([r[i][1] for r in rows], "VALUES") for i in range(n)]
        if any(t.base == "NULL" for t in types):
            raise Skip("VALUES column of untyped NULLs")
        sels = []
        for r in rows:
            vals = [x if t2 == t or t2.base != "NULL" else f"CAST({x} AS {self.cast_name(t)})"
                    for (x, t2), t in zip(r, types)]
            sels.append(f"SELECT {self.select_list(vals)}")
        if len(sels) == 1:
            return Rel(sels[0], types, None)
        a = self.alias()
        return Rel(f"SELECT {self.select_list(self.cols(a, n))} FROM ({' UNION ALL '.join(sels)}) AS {a}",
                   types, None)


def split_top(s: str, sep: str) -> list:
    """Split on ``sep`` outside quotes and brackets."""
    out, cur, depth, q, i = [], [], 0, False, 0
    while i < len(s):
        ch = s[i]
        if q:
            cur.append(ch)
            if ch == "'":
                if s.startswith("''", i):
                    cur.append("'")
                    i += 2
                    continue
                q = False
            i += 1
            continue
        if ch == "'":
            q = True
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and s.startswith(sep, i):
            out.append("".join(cur))
            cur = []
            i += len(sep)
            continue
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


# ============================================================== DDL
def table_ddl(entry) -> str:
    name, cols, keys = entry
    lines = [f"  {qident(c)} {t.ddl()}{' NOT NULL' if not nl else ''}" for c, t, nl in cols]
    nullable = {c: nl for c, _, nl in cols}
    pk_used = False
    for key in keys:
        names = ", ".join(qident(c) for c in key)
        if not pk_used and not any(nullable[c] for c in key):
            lines.append(f"  PRIMARY KEY ({names})")
            pk_used = True
        else:
            lines.append(f"  UNIQUE ({names})")
    return f"CREATE TABLE {qident(name)} (\n" + ",\n".join(lines) + "\n);"


def tables_struct(entries) -> list:
    return [{"table": name, "columns": [{"name": c, "type": t.base, "ddl_type": t.ddl(), "nullable": nl}
                                        for c, t, nl in cols], "keys": keys}
            for name, cols, keys in entries]


# ============================================================== test discovery
def java_bodies(path: Path) -> dict:
    """Map test method name -> source text up to the next method."""
    if not path.exists():
        return {}
    src = path.read_text()
    out = {}
    ms = list(re.finditer(r"(?:void|RelOptFixture|Sql|RelOptFixture\.?\w*)\s+(test\w+)\s*\(", src))
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(src)
        out.setdefault(m.group(1), src[m.start():end])
    return out


def _java_block_end(src: str, start: int) -> int:
    """Index just past the ``}`` that closes the ``{`` at ``start`` (string and char literals and comments skipped)."""
    depth, i, n = 0, start, len(src)
    while i < n:
        ch = src[i]
        if ch in "\"'":
            i += 1
            while i < n and src[i] != ch:
                i += 2 if src[i] == "\\" else 1
        elif src.startswith("//", i):
            i = src.find("\n", i)
            if i < 0:
                return n
        elif src.startswith("/*", i):
            i = src.find("*/", i) + 1
            if i <= 0:
                return n
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


JAVA_METHOD_RE = re.compile(r"\n  (?:@\w+(?:\([^)]*\))?\s+)*(?:(?:private|protected|public|static|final)\s+)*"
                            r"[\w<>\[\], .?]+?\s+(\w+)\s*\(([^()]*)\)\s*(?:throws [\w, .]+)?\s*\{")


def java_methods(path: Path) -> dict:
    """Every method of the test class: name -> (parameter names, body text). Overloads are left out."""
    if not path.exists():
        return {}
    src = path.read_text()
    out, seen = {}, set()
    for m in JAVA_METHOD_RE.finditer(src):
        name = m.group(1)
        if name in seen:
            out.pop(name, None)
            continue
        seen.add(name)
        params = [part.split()[-1] for part in m.group(2).split(",") if part.strip()]
        out[name] = (params, src[m.end() - 1:_java_block_end(src, m.end() - 1)])
    return out


# RelBuilder calls that leave the relation on top of the stack with the row type it had
JAVA_ROW_PRESERVING = {"filter", "variable", "field", "equals", "literal", "add", "get", "set", "and", "or", "not",
                       "call", "isNull", "isNotNull", "as", "id"}
JAVA_CALL_RE = re.compile(r"\.scan\(\"(\w+)\"\)|\.empty\(\)|\.(\w+)\(")


def java_inline_helpers(body: str, methods: dict, self_name: str, depth: int = 3) -> str:
    """The body with calls of the class's own helper methods replaced by the helper's body; a boolean literal
    argument decides the helper's ``if (parameter) { .. }`` blocks."""
    for _ in range(depth):
        changed = False
        for m in list(re.finditer(r"(?<![.\w])(\w+)\(", body))[::-1]:
            name = m.group(1)
            if name == self_name or name not in methods:
                continue
            end = match_close(body, m.end() - 1, "(", ")")
            args = [a.strip() for a in split_top(body[m.end():end], ",")] if body[m.end():end].strip() else []
            params, text = methods[name]
            if len(args) != len(params):
                continue
            for param, arg in zip(params, args):
                if arg not in ("true", "false"):
                    continue
                for cond in list(re.finditer(rf"if \((!?){re.escape(param)}\) \{{", text))[::-1]:
                    stop = _java_block_end(text, cond.end() - 1)
                    keep = (arg == "true") != bool(cond.group(1))
                    if text[stop:].lstrip().startswith("else"):
                        continue  # not modelled: left as written, which makes the scan below decline
                    text = text[:cond.start()] + (text[cond.end():stop - 1] if keep else "") + text[stop:]
            body = body[:m.start()] + " " + text + " " + body[end + 1:]
            changed = True
        if not changed:
            break
    return body


def java_empty_tables(name: str, methods: dict) -> list | None:
    """The scott tables a RelBuilder test empties with ``.empty()``, in the order it builds them, or ``None``.

    ``b.scan("EMP").empty()`` is a ``LogicalValues`` with EMP's row type and no rows; the plan dump prints it as
    ``LogicalValues(tuples=[[]])`` and drops the row type, so the Java is the only place that states it. RelBuilder
    builds leaves left to right, which is the order the plan lists them. Anything this does not follow exactly (an
    ``.empty()`` that is not on a scan, or on a scan with more than filters above it, a conditional it cannot
    decide) gives ``None``.
    """
    if name not in methods:
        return None
    body = java_inline_helpers(methods[name][1], methods, name).replace("Holder.empty()", "")  # an empty variable holder
    if "relFn" not in body or re.search(r"if \(\w+\) \{\s*\w+\.empty\(\)", body):
        return None
    out, last, clean = [], None, False
    for m in JAVA_CALL_RE.finditer(body):
        if m.group(1):
            last, clean = m.group(1), True
        elif m.group(0) == ".empty()":
            if last is None or not clean or ("scott", last) not in CATALOG:
                return None
            out.append(("scott", last))
            last, clean = None, False
        elif m.group(2) not in JAVA_ROW_PRESERVING:
            clean = False
    return out


JAVA_SKIPS = [
    (re.compile(r"withCatalogReaderFactory"), "custom catalog reader (tables not in MockCatalogReaderSimple)"),
    (re.compile(r"withDynamicTable"), "dynamic-star table (schema unknown)"),
    (re.compile(r"TypeSystem|withTypeFactory"), "custom type system"),
]


def load_cases(calcite: Path):
    for stem, prefix in XML_FILES:
        xml = calcite / XML_DIR / f"{stem}.xml"
        bodies = java_bodies(calcite / JAVA_DIR / f"{stem}.java")
        methods = java_methods(calcite / JAVA_DIR / f"{stem}.java")
        root = ET.parse(xml).getroot()
        for tc in root:
            res = {r.get("name"): (r.text or "") for r in tc}
            yield {"name": prefix + tc.get("name"), "test": tc.get("name"), "file": stem,
                   "res": res, "java": bodies.get(tc.get("name"), ""),
                   "empty_tables": java_empty_tables(tc.get("name"), methods)}


def convert_case(case: dict) -> dict:
    res = case["res"]
    if "planBefore" not in res or set(res) - {"sql", "planBefore", "planAfter", "planMid"}:
        raise Skip("nonstandard test resources")
    for pat, why in JAVA_SKIPS:
        if pat.search(case["java"]):
            raise Skip(why)
    before_txt, after_txt = res["planBefore"], res["planAfter"]
    if "hints=" in before_txt + after_txt:
        raise Skip("hints")
    pb, pa = parse_plan(before_txt), parse_plan(after_txt)
    empties = case.get("empty_tables") or []
    if len(empties) != before_txt.count("LogicalValues(tuples=[[]])"):
        empties = []  # the Java does not account for every empty relation of the plan: none is typed from it
    try:
        cb = Converter(empty_tables=empties)
        rb = cb.top(pb)
        ca = Converter(root_types_hint=rb.types)
        ra = ca.top(pa)
    except NeedHint:
        ca = Converter()
        try:
            ra = ca.top(pa)
        except NeedHint:
            raise Skip("both plans are empty relations of unknown row type") from None
        cb = Converter(root_types_hint=ra.types, empty_tables=empties)
        rb = cb.top(pb)
    if len(rb.types) != len(ra.types):
        raise Skip("plans differ in width")
    for x, y in zip(rb.types, ra.types):
        if x.base != "NULL" and y.base != "NULL" and x.fam != y.fam and not {x.fam, y.fam} <= NUMERIC:
            raise Skip("plans differ in column type family")
    tables = {**cb.tables, **ca.tables}
    names = [e[0] for e in tables.values()]
    if len(set(names)) != len(names):
        raise Skip("two catalogs' tables share a name")
    entries = sorted(tables.values(), key=lambda e: e[0])
    ddl = "\n".join(table_ddl(e) for e in entries)
    return {"sql_a": rb.sql, "sql_b": ra.sql, "ddl": ddl, "tables": tables_struct(entries)}


def norm_plan(t: str) -> str:
    return "\n".join(l.rstrip() for l in t.strip().splitlines())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calcite", required=True, type=Path, help="Calcite checkout (sparse is enough)")
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--out", type=Path, default=root / "tests" / "fixtures" / "calcite_mined")
    ap.add_argument("--sqlsolver-names", type=Path,
                    help="SPES testData/calcite_tests.json (names of SQLSolver's 232 Calcite pairs)")
    args = ap.parse_args()
    sys.setrecursionlimit(10000)

    commit = subprocess.run(["git", "-C", str(args.calcite), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()

    sqlsolver = set()
    if args.sqlsolver_names and args.sqlsolver_names.exists():
        sqlsolver = {d["name"] for d in json.loads(args.sqlsolver_names.read_text())}
    qdir = root / "tests" / "fixtures" / "qed"
    qed = set()
    for f in ("qed_calcite_pairs.jsonl", "qed_calcite_skipped.jsonl"):
        qed |= {json.loads(l)["name"] for l in (qdir / f).read_text().splitlines() if l.strip()}
    rbot = set()
    for l in (root / "tests" / "fixtures" / "rbot" / "calcite.jsonl").read_text().splitlines():
        if l.strip():
            rbot |= {r["name"] for r in json.loads(l).get("rewrites", [])}

    pairs, skipped, unchanged = [], [], []
    ddl_ids: dict = {}
    total = 0
    for case in load_cases(args.calcite):
        total += 1
        name, res = case["name"], case["res"]
        bare = case["test"] if case["file"] == "RelOptRulesTest" else None
        prov = {"in_sqlsolver": bare in sqlsolver, "in_qed": bare in qed, "in_rbot": bare in rbot}
        prov["new"] = not any(prov.values())
        if "planBefore" in res and ("planAfter" not in res or norm_plan(res["planAfter"]) == norm_plan(res["planBefore"])) \
                and set(res) <= {"sql", "planBefore", "planAfter", "planMid"}:
            unchanged.append({"name": name, "source": case["file"], **prov})
            continue
        try:
            c = convert_case(case)
        except Skip as e:
            skipped.append({"name": name, "source": case["file"], "reason": str(e), **prov})
            continue
        except RecursionError:
            skipped.append({"name": name, "source": case["file"], "reason": "recursion limit", **prov})
            continue
        if c["sql_a"] == c["sql_b"]:
            # the plans differ only in field names or annotations (e.g. semiJoinDone): not a rewrite
            unchanged.append({"name": name, "source": case["file"], "identical_sql": True, **prov})
            continue
        if c["ddl"] not in ddl_ids:
            ddl_ids[c["ddl"]] = (f"s{len(ddl_ids) + 1}", c["tables"])
        sid = ddl_ids[c["ddl"]][0]
        pairs.append({"name": name, "source": case["file"], "calcite_commit": commit, "schema_id": sid,
                      "sql_a": c["sql_a"], "sql_b": c["sql_b"], "sql_calcite": res.get("sql", "").strip() or None,
                      **prov})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "pairs.jsonl").open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (args.out / "skipped.jsonl").open("w") as fh:
        for s in skipped:
            fh.write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")
        for u in unchanged:
            why = ("unchanged (plans differ only in names or annotations; identical SQL)" if u.get("identical_sql")
                   else "unchanged (no planAfter, or planAfter == planBefore)")
            row = {k: v for k, v in u.items() if k != "identical_sql"}
            fh.write(json.dumps({**row, "reason": why}, ensure_ascii=False, separators=(",", ":")) + "\n")
    (args.out / "schemas.json").write_text(json.dumps(
        {sid: {"ddl": ddl, "tables": tables} for ddl, (sid, tables) in ddl_ids.items()},
        indent=1, ensure_ascii=False) + "\n")

    def cover(rows):
        return {"new": sum(r["new"] for r in rows), "in_sqlsolver": sum(r["in_sqlsolver"] for r in rows),
                "in_qed": sum(r["in_qed"] for r in rows), "in_rbot": sum(r["in_rbot"] for r in rows)}

    summary = {
        "calcite_commit": commit,
        "sources": [f"{XML_DIR}/{s}.xml" for s, _ in XML_FILES],
        "tests": total, "translated": len(pairs), "unchanged": len(unchanged),
        "unchanged_identical_sql": sum(bool(u.get("identical_sql")) for u in unchanged), "skipped": len(skipped),
        "translated_coverage": cover(pairs), "unchanged_coverage": cover(unchanged),
        "skipped_coverage": cover(skipped),
        "skip_reasons": dict(Counter(s["reason"] for s in skipped).most_common()),
    }
    prev = args.out / "summary.json"
    if prev.exists():  # keep the DuckDB validation block written by validate_calcite_mined_pairs.py
        old = json.loads(prev.read_text())
        if "duckdb_validation" in old:
            summary["duckdb_validation"] = old["duckdb_validation"]
    prev.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "skip_reasons"}, indent=1))
    for r, n in Counter(s["reason"] for s in skipped).most_common():
        print(f"{n:5d}  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
