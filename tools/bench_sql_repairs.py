"""Repairs that make benchmark SQL (printed by Calcite, or written in a benchmark's own notation) mean the
same to KumoSQL's prover and to DuckDB as it did to its authors.

Each repair is a rewrite with one reading; anything ambiguous is left as written, so the pair stays
unknown rather than being scored under a guessed meaning.

* ``rebind_correlation_variables``: Calcite's ``$corN.col`` for a column of a LATERAL subquery.
* ``uniquify_star_columns``: ``SELECT *`` over a join with repeated column names inside a derived table.
* ``expand_row_predicates``: VeriEQL Literature's uninterpreted predicates over whole rows, ``B1(X)``.
* ``fold_table_names``: Cosette's case-insensitive table names (``A`` and ``a`` are one table).
* ``pipes_as_concat``: Calcite's ``||`` (string concatenation; MySQL reads it as OR).
* ``barewords_as_strings``: VeriEQL's LeetCode answers whose double-quoted strings lost their quotes
  (``THEN YES ELSE NO``).
* ``schema_table_names``: a LeetCode answer that spells the schema's table with underscores (``MY_NUMBERS``
  for ``MYNUMBERS``).
* ``html_ampersands``: a LeetCode answer whose ``&&`` reached the file as ``&AMP;&AMP;``.
* ``joined_strings``: a LeetCode answer whose strings were joined across ``THEN`` or ``ELSE``
  (``'S THEN SMALL ELSE LARGE'``).
* ``update_as_select``: a LeetCode answer that is an ``UPDATE``, read as the table it leaves.
"""

from __future__ import annotations

import re

import sqlglot
import sqlglot.expressions as exp


def _mysql(tree: exp.Expression) -> str:
    """The repaired query as MySQL text.

    sqlglot's MySQL writer spells FULL JOIN as a UNION of one-sided joins; that is another query for the
    prover to read, so a query with one is not repaired at all.
    """

    for join in tree.find_all(exp.Join):
        if (join.side or "").upper() == "FULL":
            raise ValueError("FULL JOIN would be rewritten by the MySQL writer")
    return tree.sql(dialect="mysql")


def _from_items(select) -> list:
    items = [select.args.get(k) for k in ("from", "from_") if select.args.get(k) is not None]
    return [i.this for i in items] + [j.this for j in select.args.get("joins") or []]


def _output_names(query, tables: dict[str, list[str]]) -> list[str] | None:
    """The output column names of a query, expanding ``*``; ``None`` when they cannot be read off."""

    while isinstance(query, (exp.Subquery, exp.Paren)):
        query = query.this
    if isinstance(query, exp.SetOperation):
        return _output_names(query.this, tables)
    if not isinstance(query, exp.Select):
        return None
    names: list[str] = []
    for item in query.expressions:
        if isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            wanted = item.table if isinstance(item, exp.Column) else None
            for source in _from_items(query):
                if wanted and source.alias_or_name != wanted:
                    continue
                inner = _source_names(source, tables)
                if inner is None:
                    return None
                names.extend(inner)
        else:
            names.append(item.alias_or_name)
    return names


def _source_names(source, tables: dict[str, list[str]]) -> list[str] | None:
    """The column names one FROM item provides, in order; ``None`` when unknown."""

    if isinstance(source, exp.Lateral):
        source = source.this
    if isinstance(source, exp.Table):
        return tables.get(source.name.lower())
    if isinstance(source, exp.Values):
        alias = source.args.get("alias")
        return [c.name for c in alias.columns] if alias is not None and alias.columns else None
    if isinstance(source, exp.Subquery):
        return _output_names(source.this, tables)
    return None


def rebind_correlation_variables(sql: str, tables: dict[str, list[str]]) -> str:
    """Calcite prints a column of a LATERAL subquery as ``$corN.col`` although ``$corN`` names the outer source.

    ``FROM EMP AS $cor0, LATERAL (SELECT MIN(TRUE) AS $f0 ...) AS t1 WHERE $cor0.$f0 IS NOT NULL`` means
    ``t1.$f0``. A ``$corN.col`` is rebound only when every source of that FROM has known columns, ``col``
    is not one of ``$corN``'s, and exactly one other source has it; anything else is left as written
    (DuckDB then rejects it and the pair stays unknown).
    """

    tree = sqlglot.parse_one(sql, read="mysql")
    changed = False
    for select in tree.find_all(exp.Select):
        sources = {}
        for source in _from_items(select):
            names = _source_names(source, tables)
            sources[source.alias_or_name] = None if names is None else {n.lower() for n in names}
        if not any(alias.startswith("$cor") for alias in sources) or any(v is None for v in sources.values()):
            continue
        for column in select.find_all(exp.Column):
            alias = column.table
            if not alias.startswith("$cor") or alias not in sources or column.name.lower() in sources[alias]:
                continue
            owners = [a for a, names in sources.items() if a != alias and column.name.lower() in names]
            if len(owners) == 1:
                column.set("table", exp.to_identifier(owners[0]))
                changed = True
    return _mysql(tree) if changed else sql


def expand_row_predicates(sql: str, tables: dict[str, list[str]]) -> tuple[str, dict[str, int]]:
    """Literature pairs call uninterpreted predicates on whole rows: ``B1(X)`` where ``X`` names a FROM item.

    Each such argument is spelled out as that row's columns (``B1(X.a, X.b)``), so the prover reads the
    predicate as an uninterpreted function of the row and DuckDB can run it as a macro. Returns the new
    SQL and each predicate's arity (empty when nothing names a row).
    """

    tree = sqlglot.parse_one(sql, read="mysql")
    arities: dict[str, int] = {}
    for call in list(tree.find_all(exp.Anonymous)):
        arguments, expanded = [], False
        for argument in call.expressions:
            row = None
            if isinstance(argument, exp.Column) and not argument.table:
                scope = call.find_ancestor(exp.Select)
                while scope is not None and row is None:
                    for source in _from_items(scope):
                        if source.alias_or_name == argument.name:
                            row = (argument.name, _source_names(source, tables))
                            break
                    scope = scope.find_ancestor(exp.Select)
            if row is None:
                arguments.append(argument)
                continue
            alias, names = row
            if not names:
                return sql, {}
            arguments.extend(exp.column(n, table=alias) for n in names)
            expanded = True
        if expanded:
            call.set("expressions", arguments)
            name = str(call.this)
            if arities.setdefault(name, len(arguments)) != len(arguments):
                return sql, {}  # one name used with two arities: leave the pair as written
    return (_mysql(tree), arities) if arities else (sql, {})


def uniquify_star_columns(sql: str, tables: dict[str, list[str]]) -> str:
    """Spell out ``SELECT *`` inside a derived table when the star repeats a column name.

    ``(SELECT * FROM EMP AS a, EMP AS b) AS t`` has two ``SAL`` columns; ``t.SAL`` means the first
    (Calcite renames later copies, and so does DuckDB, which calls the second ``SAL_1``). The star
    becomes ``a.SAL AS SAL, ..., b.SAL AS SAL_1, ...``, the DuckDB spelling, so the prover reads the
    same columns DuckDB runs. A star whose sources are unnamed or unknown is left alone.
    """

    tree = sqlglot.parse_one(sql, read="mysql")
    changed = False
    for select in reversed(list(tree.find_all(exp.Select))):  # inner derived tables first
        holder = select.parent
        while isinstance(holder, exp.SetOperation):
            holder = holder.parent
        if not isinstance(holder, exp.Subquery) or not any(isinstance(e, exp.Star) for e in select.expressions):
            continue
        sources = _from_items(select)
        aliases = [s.alias_or_name for s in sources]
        if len(set(aliases)) != len(aliases):
            continue
        columns = []
        for source in sources:
            names = _source_names(source, tables)
            if names is None:
                columns = None
                break
            columns.extend((source.alias_or_name, n) for n in names)
        if not columns:
            continue
        lowered = [n.lower() for _, n in columns]
        if len(set(lowered)) == len(lowered):
            continue
        seen: dict[str, int] = {}
        items = []
        for alias, name in columns:
            count = seen.get(name.lower(), 0)
            seen[name.lower()] = count + 1
            output = name if count == 0 else f"{name}_{count}"
            items.append(exp.alias_(exp.column(name, table=alias), output))
        outputs = [item.alias.lower() for item in items]
        if len(set(outputs)) != len(outputs):
            continue  # a renamed copy collides with a real column name: DuckDB's naming would differ
        expanded = []
        for item in select.expressions:
            expanded.extend(items if isinstance(item, exp.Star) else [item])
        select.set("expressions", expanded)
        changed = True
    return _mysql(tree) if changed else sql


def fold_table_names(*queries: str) -> list[str]:
    """Spell each table one way across a pair when the queries spell it in more than one case.

    Cosette, like Calcite, reads ``A`` and ``a`` as one table, but MySQL on Linux and the prover keep
    table names case-sensitive, so ``FROM A`` would be a second, unrelated table. The first spelling
    seen wins; queries that already agree are returned unchanged.
    """

    trees = [sqlglot.parse_one(sql, read="mysql") for sql in queries]
    spellings: dict[str, list[str]] = {}
    for tree in trees:
        for table in tree.find_all(exp.Table):
            seen = spellings.setdefault(table.name.lower(), [])
            if table.name not in seen:
                seen.append(table.name)
    out = []
    for sql, tree in zip(queries, trees):
        changed = False
        for table in tree.find_all(exp.Table):
            first = spellings[table.name.lower()][0]
            if table.name != first:
                table.set("this", exp.to_identifier(first))
                changed = True
        out.append(_mysql(tree) if changed else sql)
    return out


def pipes_as_concat(sql: str) -> str:
    """Calcite's ``a || b`` as ``CONCAT(a, b)``: the pairs are read as MySQL, where ``||`` means OR.

    Each ``||`` token is parsed as MySQL's ``|``, whose precedence sits where Calcite's ``||`` does (above
    comparisons, below arithmetic), and then becomes a concatenation. A query that also uses ``|`` itself is
    left as written. ``CONCAT`` returns NULL when an argument is NULL, as ``||`` does.
    """

    from sqlglot.dialects.mysql import MySQL
    from sqlglot.tokens import TokenType

    tokens = MySQL().tokenize(sql)
    pipes = [t for t in tokens if t.token_type == TokenType.DPIPE]
    if not pipes or any(t.token_type == TokenType.PIPE for t in tokens):
        return sql
    for token in reversed(pipes):
        sql = sql[: token.start] + "|" + sql[token.end + 1 :]
    tree = sqlglot.parse_one(sql, read="mysql")
    for node in reversed(list(tree.find_all(exp.BitwiseOr, bfs=False))):  # inner ones first
        node.replace(exp.Concat(expressions=[node.this.copy(), node.expression.copy()]))
    return _mysql(tree)


_PLAIN_WORD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _plain_word(word: str) -> bool:
    """Whether the word is letters, digits and underscores and no keyword to the MySQL tokenizer."""

    from sqlglot.dialects.mysql import MySQL
    from sqlglot.tokens import TokenType

    if not _PLAIN_WORD.match(word):
        return False
    tokens = MySQL().tokenize(word)
    return len(tokens) == 1 and tokens[0].token_type == TokenType.VAR


def _string_place(column: exp.Column) -> bool:
    """Whether the column stands where a value is chosen or compared (see ``barewords_as_strings``)."""

    node, parent = column, column.parent
    while isinstance(parent, exp.Paren):
        node, parent = parent, parent.parent
    key = node.arg_key
    if isinstance(parent, exp.If):  # IF(c, a, b), or a WHEN .. THEN .. of a CASE; the WHEN value of CASE x WHEN ..
        return key in ("true", "false") or (key == "this" and isinstance(parent.parent, exp.Case) and parent.parent.this is not None)
    if isinstance(parent, exp.Case):
        return key == "default"
    if isinstance(parent, exp.Alias):
        return key == "this" and isinstance(parent.parent, exp.Select) and parent.arg_key == "expressions"
    if isinstance(parent, exp.Select):
        return key == "expressions"
    return isinstance(parent, (exp.EQ, exp.NEQ, exp.NullSafeEQ, exp.Like, exp.In))


def barewords_as_strings(sql: str, columns: set[str]) -> str:
    """A LeetCode answer whose strings lost their quotes, with the quotes put back.

    The answers were written for MySQL, where ``"Yes"`` in double quotes is a string. VeriEQL's files hold
    them without the quotes (``CASE WHEN .. THEN YES ELSE NO END``), which no engine runs: ``YES`` is no
    column. The counterexample search has always read an unqualified name that is no column of the schema
    and no alias or table of the query as that string (``counterexample._barewords``); this is the same
    reading, decided by the same function, for the prover.

    Only the words are touched: each is quoted where it stands and the rest of the text is kept character for
    character (writing the parsed query out again would respell other things, ``DIV`` among them). The result
    is parsed again and must be the tree the search reads; otherwise the query is returned as written. So is a
    query that holds no such word.

    The prover gets the reading only where a lost string is the plain one: every such word of the query must
    stand where a value is chosen or compared (a ``THEN`` or ``ELSE`` result, an ``IF`` result, a select-list
    item, an operand of ``=``, ``<>``, ``<=>``, ``LIKE`` or ``IN``). A word that is the argument of an aggregate
    (``COUNT(GAME)``), a sort or grouping key, or part of a sum is more likely a column the file lost in some
    other way, and a query with one is returned as written; so is one with a word in backticks or a word the
    MySQL tokenizer knows as a keyword (``TABLE``). The pair then stays unproved, which is never wrong.
    """

    from kumo_sql_prover.counterexample import _barewords

    tree = sqlglot.parse_one(sql, read="mysql")
    before = [column for column in tree.find_all(exp.Column) if not column.table]
    spans = {id(column): (column.this, column.this.meta) for column in before if isinstance(column.this, exp.Identifier)}
    places = {id(column): _string_place(column) for column in before}  # read before the search's reader moves the nodes
    tree = _barewords(tree, {name.lower() for name in columns})
    alive = {id(column) for column in tree.find_all(exp.Column)}
    words = []
    for column in before:
        if id(column) in alive:
            continue
        identifier, meta = spans.get(id(column), (None, {}))
        if identifier is None or identifier.quoted or "start" not in meta or "end" not in meta:
            return sql
        if not places[id(column)] or not _plain_word(identifier.name):
            return sql
        start, end = meta["start"], meta["end"] + 1
        if sql[start:end] != identifier.name:
            return sql
        words.append((start, end))
    if not words:
        return sql
    out = sql
    for start, end in sorted(words, reverse=True):
        out = out[:start] + "'" + out[start:end] + "'" + out[end:]
    return out if sqlglot.parse_one(out, read="mysql") == tree else sql


def schema_table_names(sql: str, tables: set[str]) -> str:
    """A LeetCode answer that spells a table of the schema with other underscores, with the schema's spelling.

    VeriEQL's schema for one problem names its table ``MYNUMBERS`` while most of the problem's answers read
    ``MY_NUMBERS`` (the problem's table was renamed at some point). A table name of the query that is no table of
    the schema and no ``WITH`` name of the query, and that equals exactly one schema table once underscores are
    dropped from both (letter case aside), is that table: nothing else in the schema could be meant. Only the
    name is rewritten where it stands as a table (a qualifier ``MY_NUMBERS.NUM`` keeps the query as written);
    the rest of the text is kept character for character, and the result is parsed again and must be the query
    with those tables renamed. A query without such a name is returned as written.
    """

    def bare(name: str) -> str:
        return name.replace("_", "").lower()

    schema = {name.lower() for name in tables}
    folded: dict[str, list[str]] = {}
    for name in tables:
        folded.setdefault(bare(name), []).append(name)
    tree = sqlglot.parse_one(sql, read="mysql")
    local = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}
    spans = []
    for table in tree.find_all(exp.Table):
        name = table.name
        if not name or name.lower() in schema or name.lower() in local or table.args.get("db") or table.args.get("catalog"):
            continue
        matches = folded.get(bare(name), [])
        identifier = table.this
        if len(matches) != 1 or not isinstance(identifier, exp.Identifier) or identifier.quoted:
            continue
        meta = identifier.meta
        if "start" not in meta or "end" not in meta or sql[meta["start"] : meta["end"] + 1] != name:
            return sql
        if not table.alias and any(column.table.lower() == name.lower() for column in tree.find_all(exp.Column)):
            return sql  # the old name is also used as a qualifier
        spans.append((meta["start"], meta["end"] + 1, matches[0], table))
    if not spans:
        return sql
    out = sql
    for start, end, name, table in sorted(spans, key=lambda item: -item[0]):
        out = out[:start] + name + out[end:]
        table.set("this", exp.to_identifier(name))
    return out if sqlglot.parse_one(out, read="mysql") == tree else sql


# --- LeetCode answers the file holds in a form no parser reads --------------------------------

_STRING = re.compile(r"'[^']*'")
_ENTITY = re.compile(r"&amp;", re.IGNORECASE)
_SWALLOWED = re.compile(r" (THEN|ELSE) ", re.IGNORECASE)
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)$")


def _parses(sql: str) -> bool:
    try:
        return sqlglot.parse_one(sql, read="mysql") is not None
    except Exception:
        return False


def _plain_quotes(sql: str) -> bool:
    """Whether every string of the text is a plain ``'..'``: no double quote, backslash or doubled quote."""

    return '"' not in sql and "\\" not in sql and "''" not in sql and sql.count("'") % 2 == 0


def html_ampersands(sql: str) -> str:
    """A LeetCode answer whose ``&`` reached the file as the HTML entity ``&AMP;``, with the ``&`` put back.

    ``WHERE A = 1 &AMP;&AMP; B = 2`` was ``WHERE A = 1 && B = 2`` on the page the answers were collected from.
    As the file holds it the text is ``A = 1 & AMP`` followed by a ``;``, which no answer means: outside a string
    the entity has one reading. Only the entities outside strings are rewritten, the rest of the text is kept
    character for character, and the result must parse; otherwise the answer is returned as written. So is an
    answer with a double quote, a backslash or a doubled quote, whose strings this function does not find.
    """

    if not _ENTITY.search(sql) or not _plain_quotes(sql):
        return sql
    parts, last = [], 0
    for match in _STRING.finditer(sql):
        parts += [_ENTITY.sub("&", sql[last : match.start()]), match.group(0)]
        last = match.end()
    parts.append(_ENTITY.sub("&", sql[last:]))
    out = "".join(parts)
    return out if out != sql and _parses(out) else sql


class _OtherReading(Exception):
    pass


def joined_strings(sql: str, columns: set[str]) -> str:
    """A LeetCode answer whose neighbouring strings were joined across ``THEN`` or ``ELSE``, taken apart again.

    VeriEQL's files hold ``CASE WHEN SIZE = 'S' THEN 'SMALL' ELSE 'LARGE' END`` as
    ``CASE WHEN SIZE = 'S THEN SMALL ELSE LARGE' END``: where a string ended right before one of the two words and
    the next began right after it, the two quotes between them are gone. No parser reads the result (the ``CASE``
    has lost its ``THEN``). The repair applies only to an answer that does not parse as written: each string that
    holds `` THEN `` or `` ELSE `` is split there into strings, the rest of the text is kept character for
    character, and the result must parse. The answer is returned as written when a piece between two of the words
    is spelled like a number or like a column of the schema (it may not have been a string), when a piece is empty
    or begins or ends with a space, and when the text has a double quote, a backslash or a doubled quote.
    """

    if not _SWALLOWED.search(sql) or not _plain_quotes(sql) or _parses(sql):
        return sql
    known = {name.lower() for name in columns}

    def apart(match: re.Match) -> str:
        pieces = _SWALLOWED.split(match.group(0)[1:-1])
        if len(pieces) == 1:
            return match.group(0)
        values, words = pieces[0::2], pieces[1::2]
        if any(not value or value != value.strip() for value in values):
            raise _OtherReading
        if any(_NUMBER.match(value) or value.lower() in known for value in values[1:-1]):
            raise _OtherReading
        return f"'{values[0]}'" + "".join(f" {word} '{value}'" for word, value in zip(words, values[1:]))

    try:
        out = _STRING.sub(apart, sql)
    except _OtherReading:
        return sql
    return out if out != sql and _parses(out) else sql


_WHOLE_NUMBER_TYPES = ("INT", "INTEGER", "BIGINT", "SMALLINT")


def _kind(column) -> str:
    return column.type.split("(")[0].upper()


def _whole_number(node: exp.Expression) -> bool:
    if isinstance(node, exp.Neg):
        node = node.this
    return isinstance(node, exp.Literal) and not node.is_string and node.name.isdigit() and len(node.name) <= 9


def _fits(node: exp.Expression, column, by_name: dict) -> bool:
    """Whether ``node`` is a literal of the column's kind (for an ``ENUM``, one of its values), or a column of that kind."""

    kind = _kind(column)
    if isinstance(node, exp.Column):
        other = by_name[node.name.lower()]
        return other is column or (kind in _WHOLE_NUMBER_TYPES and _kind(other) in _WHOLE_NUMBER_TYPES)
    if kind == "ENUM":
        return isinstance(node, exp.Literal) and node.is_string and node.name in column.values
    return kind in _WHOLE_NUMBER_TYPES and _whole_number(node)


def _compares(node: exp.Expression, other: exp.Expression, by_name: dict, *, ordered: bool) -> bool:
    """Whether ``node`` and ``other`` are a column and a literal of its kind: a comparison the engine makes without
    converting either side. Texts are only compared for equality (an ``ENUM`` orders by position, not by spelling)."""

    if isinstance(other, exp.Column) and not isinstance(node, exp.Column):
        node, other = other, node
    if not isinstance(node, exp.Column):
        return False
    column = by_name[node.name.lower()]
    kind = _kind(column)
    if kind in _WHOLE_NUMBER_TYPES:
        return _whole_number(other) or (isinstance(other, exp.Column) and _kind(by_name[other.name.lower()]) in _WHOLE_NUMBER_TYPES)
    if ordered or kind not in ("ENUM", "VARCHAR", "CHAR", "TEXT"):
        return False
    return isinstance(other, exp.Literal) and other.is_string or other is not node and isinstance(other, exp.Column) and by_name[other.name.lower()] is column


def _plain_condition(node: exp.Expression, by_name: dict) -> bool:
    """Whether ``node`` is a test an ``UPDATE`` and a ``SELECT`` evaluate alike (see ``update_as_select``)."""

    if isinstance(node, exp.Paren):
        return _plain_condition(node.this, by_name)
    if isinstance(node, (exp.And, exp.Or)):
        return _plain_condition(node.this, by_name) and _plain_condition(node.expression, by_name)
    if isinstance(node, exp.Not):
        return _plain_condition(node.this, by_name)
    if isinstance(node, exp.Boolean):
        return True
    if isinstance(node, (exp.EQ, exp.NEQ, exp.NullSafeEQ)):
        return _compares(node.this, node.expression, by_name, ordered=False)
    if isinstance(node, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
        return _compares(node.this, node.expression, by_name, ordered=True)
    if isinstance(node, exp.In):
        values = node.expressions
        return bool(values) and set(node.args) <= {"this", "expressions"} and all(_compares(node.this, value, by_name, ordered=False) for value in values) and not any(isinstance(value, exp.Column) for value in values)
    if isinstance(node, exp.Like):
        return not node.args.get("escape") and isinstance(node.expression, exp.Literal) and _compares(node.this, node.expression, by_name, ordered=False)
    if isinstance(node, exp.Is):
        return isinstance(node.this, exp.Column) and isinstance(node.expression, exp.Null)
    return False


def _assigned_value(node: exp.Expression, column, by_name: dict, tests: list) -> bool:
    """Whether every value ``node`` can give is one the column stores as it is; the tests it makes go to ``tests``."""

    if isinstance(node, exp.Paren):
        return _assigned_value(node.this, column, by_name, tests)
    if isinstance(node, exp.Null):
        return not column.not_null
    if isinstance(node, exp.Case):
        operand = node.this
        for branch in node.args.get("ifs") or []:
            tests.append(branch.this if operand is None else exp.EQ(this=operand.copy(), expression=branch.this.copy()))
            if not _assigned_value(branch.args["true"], column, by_name, tests):
                return False
        default = node.args.get("default")
        return not column.not_null if default is None else _assigned_value(default, column, by_name, tests)
    if isinstance(node, exp.If):
        tests.append(node.this)
        other = node.args.get("false")
        return _assigned_value(node.args["true"], column, by_name, tests) and (not column.not_null if other is None else _assigned_value(other, column, by_name, tests))
    return _fits(node, column, by_name)


def update_as_select(sql: str, tables: dict, constrained: frozenset = frozenset()) -> str:
    """A LeetCode answer that is an ``UPDATE``, read as the table it leaves.

    One problem of VeriEQL's files asks for a statement, not a query ("swap two values of a column in place"). Two
    such statements are the same answer exactly when they leave the same table, and the table an ``UPDATE`` leaves
    is a query over the table it found: every column as it was, and the assigned column as

        CASE WHEN <the statement's WHERE> THEN <the assigned value> ELSE <the column> END

    (the value alone when there is no ``WHERE``; a row whose ``WHERE`` is NULL is not updated, which is the
    ``ELSE``). The harness then searches and proves the two queries like any other pair.

    The reading is taken only where storing the value cannot change or refuse it, and where the statement's tests
    are ones an ``UPDATE`` and a ``SELECT`` evaluate alike (MySQL's strict mode turns a conversion warning of a
    test into an error in an ``UPDATE`` only). So the statement must be: one table of the schema, with or
    without an alias; one assignment; no join, ``ORDER BY``, ``LIMIT`` or subquery; the assigned column an
    ``ENUM`` or a whole number, and not one of ``constrained`` (the harness names there, as lower-case
    ``(table, column)``, every column of a key, a foreign key or a declared check a new value could break); the
    value built from ``CASE``, ``IF`` and parentheses over values the column
    stores unchanged (one of the ``ENUM``'s declared values spelled exactly, a whole-number literal of at most
    nine digits for a whole-number column, a column of the same kind, and NULL only for a column that may be
    NULL); every test a comparison of a column with a literal of its own kind (``=``, ``<>``, ``<=>``, ``IN``,
    ``LIKE``, ``IS NULL``; an order comparison only between whole numbers) joined by ``AND``, ``OR`` and ``NOT``.
    Any other statement is returned as written and its pair stays unread, which is never wrong. ``tables`` maps a
    table's name to its columns (objects with ``name``, ``type``, ``values`` and ``not_null``).
    """

    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql
    if not isinstance(tree, exp.Update) or any(value for key, value in tree.args.items() if key not in ("this", "expressions", "where")):
        return sql
    target = tree.this
    if not isinstance(target, exp.Table) or not isinstance(target.this, exp.Identifier) or any(value for key, value in target.args.items() if key not in ("this", "alias")):
        return sql
    alias = target.args.get("alias")
    if alias is not None and (not isinstance(alias, exp.TableAlias) or alias.args.get("columns") or alias.this is None):
        return sql
    columns = {name.lower(): found for name, found in tables.items()}.get(target.name.lower())
    if columns is None or len(tree.expressions) != 1:
        return sql
    by_name = {column.name.lower(): column for column in columns}
    if len(by_name) != len(columns):
        return sql
    qualifier = (target.alias or target.name).lower()
    for column in tree.find_all(exp.Column):
        if column.args.get("db") or column.args.get("catalog") or column.name.lower() not in by_name or not isinstance(column.this, exp.Identifier):
            return sql
        if column.table and column.table.lower() != qualifier:
            return sql
    assignment = tree.expressions[0]
    if not isinstance(assignment, exp.EQ) or not isinstance(assignment.this, exp.Column):
        return sql
    assigned = by_name[assignment.this.name.lower()]
    if _kind(assigned) != "ENUM" and _kind(assigned) not in _WHOLE_NUMBER_TYPES:
        return sql
    if (target.name.lower(), assigned.name.lower()) in constrained:
        return sql
    tests: list[exp.Expression] = []
    if not _assigned_value(assignment.expression, assigned, by_name, tests):
        return sql
    where = tree.args.get("where")
    if where is not None:
        tests.append(where.this)
    if not all(_plain_condition(test, by_name) for test in tests):
        return sql
    items = []
    for column in columns:
        own = exp.column(column.name, table=target.alias or None)
        if column is not assigned:
            items.append(own)
            continue
        value = assignment.expression.copy()
        if where is not None:
            value = exp.Case(ifs=[exp.If(this=where.this.copy(), true=value)], default=own)
        items.append(exp.alias_(value, column.name))
    select = exp.Select(expressions=items).from_(target.copy())
    out = select.sql(dialect="mysql")
    try:
        return out if sqlglot.parse_one(out, read="mysql") == select else sql
    except Exception:
        return sql
