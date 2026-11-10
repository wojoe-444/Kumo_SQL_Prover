"""BigQuery syntax that sqlglot's BigQuery parser does not read, added once for every parse in this package.

``DROP TABLE FUNCTION`` is read as dropping a function of kind ``TABLE FUNCTION``. Pipe ``SET`` and ``DROP`` are read as the
``SELECT * REPLACE`` and ``SELECT * EXCEPT`` they stand for. ``GRAPH_TABLE(...)`` and the ``MODEL m`` argument of an ``ML.``
function sqlglot does not know are kept as their own text.

``FROM dataset.fn(TABLE dataset.input, option => value)`` passes a whole table to a table-valued function. sqlglot stops at
``TABLE`` ("Expecting )"), so the model was reported unparseable and everything it read was lost. Here the argument becomes a
:data:`TABLE_ARGUMENT` call holding the table, so the table is read like any other and the SQL prints back unchanged.

``x LIKE ALL UNNEST(array)`` (and ``LIKE SOME``), an aggregate with a ``WHERE`` filter inside its parentheses (``COUNT(* WHERE c)``),
the ``WITH(a AS 1, a + 1)`` expression and ``t.arr elem WITH OFFSET off`` are read the same way, each into nodes sqlglot already
has or a marker call that prints back as written. ``STRUCT<>()``, which sqlglot reads as the comparison ``STRUCT <> ()``, is refused.

It works the same on pure and compiled sqlglot: a compiled build ignores parser methods assigned after the fact and refuses
subclasses of its expressions, so SQL sqlglot rejects is parsed again with the dialect's tokens rewritten and the marker call is resolved after.
"""

from __future__ import annotations

import threading

from sqlglot import exp
from sqlglot import parser as sqlglot_parser
from sqlglot.dialects.bigquery import BigQuery
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import ParseError
from sqlglot.generator import Generator
from sqlglot.tokens import Token, TokenType

from .string_literals import _bytes_literal, _decode_bytes, _string_end


TABLE_ARGUMENT = "__KUMO_TABLE_ARGUMENT__"


VERBATIM = "__KUMO_VERBATIM__"
# Calls whose arguments are a language of their own, kept as their text: ``GRAPH_TABLE(graph MATCH ... RETURN ...)``.
_VERBATIM_CALLS = ("GRAPH_TABLE",)


WITH_EXPRESSION = "__KUMO_WITH__"
WITH_VARIABLE = "__KUMO_WITH_VARIABLE__"
LIKE_ALL = "__KUMO_LIKE_ALL__"


def _generate(self: Generator, expression: exp.Anonymous) -> str:
    if expression.name == WITH_EXPRESSION and len(expression.expressions) >= 2:
        return f"WITH({', '.join(self.sql(item) for item in expression.expressions)})"
    if expression.name == WITH_VARIABLE and len(expression.expressions) == 2 and expression.expressions[0].is_string:
        return f"{expression.expressions[0].this} AS {self.sql(expression.expressions[1])}"
    if expression.name == TABLE_ARGUMENT and len(expression.expressions) == 1:
        return f"TABLE {self.sql(expression.expressions[0])}"
    if expression.name == VERBATIM and len(expression.expressions) == 1 and expression.expressions[0].is_string:
        return expression.expressions[0].this
    return self.anonymous_sql(expression)


def _all_sql(self: Generator, expression: exp.All) -> str:
    """``ALL UNNEST(array)`` as BigQuery spells ``LIKE ALL UNNEST(array)``, without the parentheses sqlglot adds."""

    if isinstance(expression.this, exp.Unnest):
        return f"ALL {self.sql(expression, 'this')}"
    return Generator.all_sql(self, expression)


def _filter_sql(self: Generator, expression: exp.Filter) -> str:
    """``COUNT(x) FILTER (WHERE c)`` as BigQuery's ``COUNT(x WHERE c)``, the filter it allows inside an aggregate call.

    BigQuery rejects the filter inside a window call and next to ``ORDER BY``, ``LIMIT`` or ``HAVING``; those keep the standard form.
    """

    aggregate = expression.this
    where = expression.expression
    if (
        isinstance(where, exp.Where) and isinstance(aggregate, (exp.AggFunc, exp.IgnoreNulls, exp.RespectNulls))
        and not isinstance(expression.parent, exp.Window) and not aggregate.find(exp.Order, exp.Limit, exp.HavingMax)
    ):
        text = self.sql(aggregate)
        if text.endswith(")"):
            return f"{text[:-1]} WHERE {self.sql(where, 'this')})"
    return Generator.filter_sql(self, expression)


_installed = False


def _open_call(before: list) -> int | None:
    """The index of the ``(`` of the innermost call still open at the end of ``before``."""

    depth = 0
    for index in range(len(before) - 1, -1, -1):
        kind = before[index].token_type
        if kind == TokenType.R_PAREN:
            depth += 1
        elif kind == TokenType.L_PAREN:
            if depth == 0:
                return index
            depth -= 1
    return None


def _ml_call(before: list, index: int | None) -> str | None:
    """The function name when the ``(`` at ``index`` opens ``ML.fn(`` or ``AI.fn(``."""

    if index is None or index < 3 or before[index - 2].token_type != TokenType.DOT:
        return None
    if before[index - 3].text.upper() not in ("ML", "AI"):
        return None
    return before[index - 1].text.upper()


def _in_native_call(before: list) -> bool:
    """Inside an ``ML.fn(...)`` or ``AI.fn(...)`` that sqlglot reads itself (``ML.PREDICT``), with its own ``TABLE`` handling."""

    name = _ml_call(before, _open_call(before))
    return name is not None and name in BigQuery.Parser.FUNCTION_PARSERS


def _mark_table_arguments(tokens: list) -> list:
    """``TABLE name`` as an argument becomes ``__KUMO_TABLE_ARGUMENT__(name)``, which every sqlglot build parses as a call."""

    out: list = []
    i = 0
    count = len(tokens)
    while i < count:
        token = tokens[i]
        follows_open = bool(out) and out[-1].token_type in (TokenType.L_PAREN, TokenType.COMMA)
        if follows_open and _in_native_call(out):
            follows_open = False
        if (
            token.token_type == TokenType.TABLE
            and follows_open
            and i + 1 < count
            and tokens[i + 1].token_type not in (TokenType.COMMA, TokenType.R_PAREN)
        ):
            depth = 0
            end = i + 1
            while end < count:
                kind = tokens[end].token_type
                if kind == TokenType.L_PAREN:
                    depth += 1
                elif kind == TokenType.R_PAREN:
                    if depth == 0:
                        break
                    depth -= 1
                elif kind == TokenType.COMMA and depth == 0:
                    break
                end += 1
            if end < count:
                def made(kind, text, source=token):
                    return Token(kind, text, line=source.line, col=source.col, start=source.start, end=source.end)

                out.append(made(TokenType.VAR, TABLE_ARGUMENT))
                out.append(made(TokenType.L_PAREN, "("))
                out.extend(tokens[i + 1 : end])
                out.append(made(TokenType.R_PAREN, ")"))
                i = end
                continue
        out.append(token)
        i += 1
    return out


def _merge_table_function_kind(tokens: list) -> list:
    """``DROP TABLE FUNCTION`` becomes ``DROP`` and one ``FUNCTION`` token spelled ``TABLE FUNCTION``: sqlglot takes the kind
    of a ``DROP`` from that token's text, so it parses as a dropped function of kind ``TABLE FUNCTION`` and prints back."""

    out: list = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if (
            token.token_type == TokenType.TABLE
            and out and out[-1].token_type == TokenType.DROP
            and i + 1 < len(tokens) and tokens[i + 1].token_type == TokenType.FUNCTION
        ):
            end = tokens[i + 1]
            out.append(Token(TokenType.FUNCTION, "TABLE FUNCTION", line=token.line, col=token.col, start=token.start, end=end.end))
            i += 2
            continue
        out.append(token)
        i += 1
    return out


# ``RENAME a AS b`` is left out on purpose: it keeps ``b`` in ``a``'s place, which no ``SELECT`` can say without knowing the
# columns, and a translation that moved the column would let a prover equate queries whose columns are in different orders.
_PIPE_OPERATORS = (TokenType.SET, TokenType.DROP)
_PIPE = getattr(TokenType, "PIPE_GT", None)  # sqlglot 26.0.0 has no pipe syntax at all


def _pipe_items(tokens: list, start: int) -> tuple[list[list], int]:
    """The comma-separated items of a pipe operator whose arguments start at ``start``, and the index after them."""

    items: list[list] = [[]]
    depth = 0
    index = start
    while index < len(tokens):
        kind = tokens[index].token_type
        if depth == 0 and kind in (_PIPE, TokenType.SEMICOLON):
            break
        if kind in (TokenType.L_PAREN, TokenType.L_BRACKET):
            depth += 1
        elif kind in (TokenType.R_PAREN, TokenType.R_BRACKET):
            if depth == 0:
                break
            depth -= 1
        if depth == 0 and kind == TokenType.COMMA:
            items.append([])
        else:
            items[-1].append(tokens[index])
        index += 1
    return [item for item in items if item], index


def _pipe_operator_text(sql: str, operator, items: list[list]) -> str | None:
    """``|> SET c = e`` as ``|> SELECT * REPLACE (e AS c)`` and ``|> DROP c`` as ``|> SELECT * EXCEPT (c)``, which BigQuery
    defines them to mean, or None when an item is not plain."""

    def text(tokens: list) -> str:
        return sql[tokens[0].start : tokens[-1].end + 1]

    if not items:
        return None
    if operator == TokenType.DROP:
        if any(len(item) != 1 for item in items):
            return None
        return f"|> SELECT * EXCEPT ({', '.join(text(item) for item in items)})"
    parts = []
    for item in items:
        if len(item) < 3 or item[1].token_type != TokenType.EQ:
            return None
        parts.append(f"{text(item[2:])} AS {text(item[:1])}")
    return f"|> SELECT * REPLACE ({', '.join(parts)})"


def _rewrite_pipe_operators(sql: str, tokens: list) -> str:
    """Pipe ``SET`` and ``DROP``, which sqlglot does not read, written as the pipe ``SELECT`` that means the same."""

    pieces: list[str] = []
    position = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            _PIPE is not None
            and token.token_type == _PIPE
            and index + 1 < len(tokens)
            and tokens[index + 1].token_type in _PIPE_OPERATORS
        ):
            items, after = _pipe_items(tokens, index + 2)
            replacement = _pipe_operator_text(sql, tokens[index + 1].token_type, items)
            if replacement is not None:
                end = tokens[after - 1].end + 1
                pieces += [sql[position : token.start], replacement]
                position = end
                index = after
                continue
        index += 1
    return "".join(pieces) + sql[position:] if pieces else sql


def _rewrite_verbatim_calls(sql: str, tokens: list) -> str:
    """``GRAPH_TABLE(...)`` becomes ``__KUMO_VERBATIM__('<its text>')``, which prints back as the text it holds.

    The rest of the query is read as usual. The call is an unknown function of its own text, so it reads no table sqlglot
    can see and a prover can only equate it with the same text.
    """

    pieces: list[str] = []
    position = 0
    index = 0
    while index + 1 < len(tokens):
        token = tokens[index]
        if token.text.upper() in _VERBATIM_CALLS and tokens[index + 1].token_type == TokenType.L_PAREN:
            depth = 0
            end = index + 1
            while end < len(tokens):
                kind = tokens[end].token_type
                depth += kind == TokenType.L_PAREN
                depth -= kind == TokenType.R_PAREN
                if depth == 0:
                    break
                end += 1
            if end < len(tokens):
                text = sql[token.start : tokens[end].end + 1]
                pieces += [sql[position : token.start], f"{VERBATIM}({_quoted(text)})"]
                position = tokens[end].end + 1
                index = end + 1
                continue
        index += 1
    return "".join(pieces) + sql[position:] if pieces else sql


def _rewrite_model_arguments(sql: str, tokens: list) -> str:
    """``MODEL name`` as an argument of an ``ML.fn`` sqlglot does not read (``ML.EVALUATE``, ``ML.DETECT_ANOMALIES``) becomes
    ``__KUMO_VERBATIM__('MODEL name')``: it prints back as written and, a model not being a table, is no read."""

    pieces: list[str] = []
    position = 0
    for index, token in enumerate(tokens):
        if token.text.upper() != "MODEL" or index == 0 or index + 1 >= len(tokens):
            continue
        if tokens[index - 1].token_type not in (TokenType.L_PAREN, TokenType.COMMA):
            continue
        name = _ml_call(tokens, _open_call(tokens[:index]))
        if name is None or name in BigQuery.Parser.FUNCTION_PARSERS:
            continue
        end = index + 1
        while end + 1 < len(tokens) and tokens[end + 1].token_type not in (TokenType.COMMA, TokenType.R_PAREN):
            end += 1
        text = sql[token.start : tokens[end].end + 1]
        pieces += [sql[position : token.start], f"{VERBATIM}({_quoted(text)})"]
        position = tokens[end].end + 1
    return "".join(pieces) + sql[position:] if pieces else sql


def _rewrite_raw_bytes(sql: str, tokens: list) -> str:
    """``rb'..'`` and ``br'..'``, which sqlglot reads as a name and a string, written as the ``b'..'`` with the same bytes."""

    pieces: list[str] = []
    position = 0
    for prefix, literal in zip(tokens, tokens[1:]):
        if (
            prefix.token_type == TokenType.VAR and prefix.text.lower() in ("rb", "br")
            and literal.token_type == TokenType.STRING and prefix.end + 1 == literal.start
        ):
            end, _, body = _string_end(sql, literal.start)
            if body is not None and end == literal.end + 1:
                pieces += [sql[position : prefix.start], _bytes_literal(_decode_bytes(body, raw=True))]
                position = end
    return "".join(pieces) + sql[position:] if pieces else sql


def _query_start(tokens: list, index: int) -> int:
    """Index of the first token of the query that holds ``tokens[index]``: just after the parenthesis it sits in, or after
    the statement's start."""

    depth = 0
    for i in range(index - 1, -1, -1):
        kind = tokens[i].token_type
        if kind in (TokenType.R_PAREN, TokenType.R_BRACKET):
            depth += 1
        elif kind in (TokenType.L_PAREN, TokenType.L_BRACKET):
            if depth == 0:
                return i + 1
            depth -= 1
        elif kind == TokenType.SEMICOLON and depth == 0:
            return i + 1
    return 0


def _rewrite_pipe_with(sql: str, tokens: list) -> str:
    """``FROM t |> WITH y AS (q) |> ...`` written as ``WITH y AS (q) FROM t |> ...``, which sqlglot reads.

    A pipe ``WITH`` names its queries for the operators after it. Moved to the front, a name would also cover the part
    before it, so the move is made only when nothing before its definition spells the name (later definitions and operators
    may), and only for a query that has no ``WITH`` of its own. The innermost query goes first; the reparse moves the rest.
    """

    if _PIPE is None:
        return sql
    found: dict[int, list[int]] = {}
    for index, token in enumerate(tokens[:-1]):
        if token.token_type == _PIPE and tokens[index + 1].token_type == TokenType.WITH:
            found.setdefault(_query_start(tokens, index), []).append(index)
    for start in sorted(found, reverse=True):
        moved = _move_pipe_with(sql, tokens, start, found[start])
        if moved is not None:
            return moved
    return sql


def _move_pipe_with(sql: str, tokens: list, start: int, pipes: list[int]) -> str | None:
    if tokens[start].token_type not in (TokenType.FROM, TokenType.SELECT):
        return None
    definitions: list[str] = []
    cuts: list[tuple[int, int]] = []
    aliases: dict[str, int] = {}  # name -> index of the token that defines it
    for index in pipes:
        items, after = _pipe_items(tokens, index + 2)
        if not items or items[0][0].text.upper() == "RECURSIVE":
            return None
        for item in items:
            if (
                len(item) < 4 or item[0].token_type not in (TokenType.VAR, TokenType.IDENTIFIER)
                or item[1].token_type != TokenType.ALIAS or item[2].token_type != TokenType.L_PAREN
                or item[-1].token_type != TokenType.R_PAREN or item[0].text.upper() in aliases
            ):
                return None
            aliases[item[0].text.upper()] = tokens.index(item[0], start)
            definitions.append(sql[item[0].start : item[-1].end + 1])
        cuts.append((tokens[index].start, tokens[after - 1].end + 1))
    if any(
        token.token_type in (TokenType.VAR, TokenType.IDENTIFIER) and i < aliases.get(token.text.upper(), -1)
        for i, token in enumerate(tokens[start:], start)
    ):
        return None
    head = tokens[start].start
    kept, position = [], head
    for cut_start, cut_end in cuts:
        kept.append(sql[position:cut_start])
        position = cut_end
    return f"{sql[:head]}WITH {', '.join(definitions)} {''.join(kept)}{sql[position:]}"


def _call_groups(tokens: list) -> list[tuple[int, int]]:
    """(open, close) token indexes of every parenthesis pair, innermost first."""

    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for index, token in enumerate(tokens):
        if token.token_type == TokenType.L_PAREN:
            stack.append(index)
        elif token.token_type == TokenType.R_PAREN and stack:
            pairs.append((stack.pop(), index))
    return pairs


def _top_level(tokens: list, start: int, end: int):
    """(index, token) of the tokens between ``start`` and ``end`` that sit outside every nested bracket."""

    depth = 0
    for index in range(start, end):
        kind = tokens[index].token_type
        if kind in (TokenType.R_PAREN, TokenType.R_BRACKET):
            depth -= 1
        if depth == 0:
            yield index, tokens[index]
        if kind in (TokenType.L_PAREN, TokenType.L_BRACKET):
            depth += 1


def _apply_edits(sql: str, edits: list[tuple[int, int, str]]) -> str:
    """``edits`` are (start, end, text) character ranges, replaced left to right; overlapping ones are dropped."""

    pieces: list[str] = []
    position = 0
    for start, end, text in sorted(edits):
        if start < position:
            continue
        pieces += [sql[position:start], text]
        position = end
    return "".join(pieces) + sql[position:] if pieces else sql


def _rewrite_like_quantifiers(sql: str, tokens: list) -> str:
    """``x LIKE ALL UNNEST(arr)`` becomes ``x LIKE ANY UNNEST(__KUMO_LIKE_ALL__(arr))``, which every build reads; the marker
    turns the ``ANY`` back into ``ALL`` once parsed. ``LIKE SOME`` is ``LIKE ANY``."""

    edits: list[tuple[int, int, str]] = []
    for index in range(len(tokens) - 2):
        if tokens[index].text.upper() not in ("LIKE", "ILIKE"):
            continue
        quantifier, after = tokens[index + 1], tokens[index + 2]
        word = quantifier.text.upper()
        if word == "SOME":
            edits.append((quantifier.start, quantifier.end + 1, "ANY"))
        elif word == "ALL" and after.text.upper() == "UNNEST" and index + 3 < len(tokens) and tokens[index + 3].token_type == TokenType.L_PAREN:
            close = next((close for opened, close in _call_groups(tokens) if opened == index + 3), None)
            if close is not None:
                edits.append((quantifier.start, quantifier.end + 1, "ANY"))
                edits.append((tokens[index + 3].end + 1, tokens[index + 3].end + 1, f"{LIKE_ALL}("))
                edits.append((tokens[close].start, tokens[close].start, ")"))
    return _apply_edits(sql, edits)


_NOT_AGGREGATE_ARGUMENT = (TokenType.SELECT, TokenType.WITH, TokenType.FROM)


def _rewrite_aggregate_where(sql: str, tokens: list) -> str:
    """``COUNT(* WHERE c)`` becomes ``COUNT(*) FILTER (WHERE c)``, the standard filter sqlglot reads as ``exp.Filter``.

    BigQuery allows the filter after the arguments (and ``IGNORE NULLS``) and rejects it next to ``ORDER BY``, ``LIMIT`` or
    ``HAVING``, inside a window call and in a subquery, so those are left alone.
    """

    matches: list[tuple[int, int, int]] = []
    for opened, closed in _call_groups(tokens):
        where = None
        for index, token in _top_level(tokens, opened + 1, closed):
            kind, word = token.token_type, token.text.upper()
            if kind in _NOT_AGGREGATE_ARGUMENT or word in ("ORDER", "LIMIT", "HAVING", "QUALIFY", "GROUP", "UNION"):
                where = None
                break
            if kind == TokenType.WHERE and where is None:
                where = index
        if where is None or where == opened + 1:
            continue
        following = tokens[closed + 1].text.upper() if closed + 1 < len(tokens) else ""
        if following == "OVER" or opened == 0 or tokens[opened - 1].token_type in (
            TokenType.IN, TokenType.EXISTS, TokenType.ALIAS, TokenType.L_PAREN, TokenType.COMMA
        ):
            continue
        matches.append((opened, where, closed))
    edits = []
    for opened, where, closed in matches:
        if any(opened < o < closed for o, _, _ in matches):
            continue  # the inner filter goes first; the next pass takes this one
        condition = sql[tokens[where].end + 1 : tokens[closed].start].strip()
        edits.append((tokens[where].start, tokens[closed].end + 1, f") FILTER (WHERE {condition})"))
    return _apply_edits(sql, edits)


def _rewrite_with_expressions(sql: str, tokens: list) -> str:
    """``WITH(a AS 1, a + 1)`` becomes ``__KUMO_WITH__(__KUMO_WITH_VARIABLE__('a', 1), a + 1)``: sqlglot has no node for it."""

    edits: list[tuple[int, int, str]] = []
    pairs = {opened: closed for opened, closed in _call_groups(tokens)}
    for index in range(1, len(tokens) - 1):
        token = tokens[index]
        if (
            token.token_type != TokenType.WITH or tokens[index + 1].token_type != TokenType.L_PAREN
            or tokens[index - 1].token_type == TokenType.SEMICOLON or index + 1 not in pairs
        ):
            continue
        opened, closed = index + 1, pairs[index + 1]
        items: list[list] = [[]]
        for position, inner in _top_level(tokens, opened + 1, closed):
            if inner.token_type == TokenType.COMMA:
                items.append([])
            else:
                items[-1].append(position)
        if len(items) < 2 or any(not item for item in items):
            continue
        parts: list[str] = []
        for item in items[:-1]:
            name = tokens[item[0]]
            if (
                len(item) < 3 or name.token_type not in (TokenType.VAR, TokenType.IDENTIFIER)
                or tokens[item[1]].token_type != TokenType.ALIAS
            ):
                parts = []
                break
            label = f"`{name.text}`" if name.token_type == TokenType.IDENTIFIER else name.text
            value = sql[tokens[item[2]].start : tokens[item[-1]].end + 1]
            parts.append(f"{WITH_VARIABLE}({_quoted(label)}, {value})")
        if parts:
            parts.append(sql[tokens[items[-1][0]].start : tokens[items[-1][-1]].end + 1])
            edits.append((token.start, tokens[closed].end + 1, f"{WITH_EXPRESSION}({', '.join(parts)})"))
    # An outer expression holds the text of an inner one, so the inner goes first and the next pass takes the outer.
    return _apply_edits(sql, [e for e in edits if not any(e[0] < o[0] < e[1] for o in edits)])


def _rewrite_unnest_offset(sql: str, tokens: list) -> str:
    """``FROM t, t.arr elem WITH OFFSET off`` becomes ``FROM t, UNNEST(t.arr) elem WITH OFFSET off``, which BigQuery defines
    ``t.arr`` in a FROM clause to mean when it is a path into an earlier table."""

    identifiers = (TokenType.VAR, TokenType.IDENTIFIER)
    edits: list[tuple[int, int, str]] = []
    for index in range(1, len(tokens) - 1):
        if tokens[index].token_type != TokenType.WITH or tokens[index + 1].text.upper() != "OFFSET":
            continue
        end = index - 1
        if end >= 1 and tokens[end].token_type in identifiers and tokens[end - 1].token_type == TokenType.ALIAS:
            end -= 2
        elif end >= 1 and tokens[end].token_type in identifiers and tokens[end - 1].token_type in identifiers:
            end -= 1
        start = end
        while start >= 2 and tokens[start - 1].token_type == TokenType.DOT and tokens[start - 2].token_type in identifiers:
            start -= 2
        if (
            end - start < 2 or start < 1 or tokens[end].token_type not in identifiers or tokens[start].token_type not in identifiers
            or not (tokens[start - 1].token_type in (TokenType.FROM, TokenType.COMMA) or tokens[start - 1].text.upper() == "JOIN")
        ):
            continue
        path = sql[tokens[start].start : tokens[end].end + 1]
        edits.append((tokens[start].start, tokens[end].end + 1, f"UNNEST({path})"))
    return _apply_edits(sql, edits)


def _check_empty_struct(sql: str, tokens: list) -> None:
    """Refuse ``STRUCT<>(...)``: sqlglot reads it as the comparison ``STRUCT <> (...)``, and BigQuery rejects an empty struct type."""

    for token, after in zip(tokens, tokens[1:]):
        if token.token_type == TokenType.STRUCT and after.token_type == TokenType.NEQ and after.start == token.end + 1:
            raise ParseError("STRUCT<>() is an empty struct type, which BigQuery rejects; it is not read as a comparison")


def _quoted(text: str) -> str:
    """``text`` as a BigQuery string literal that reads back as exactly ``text``."""

    escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
    return f"'{escaped}'"


def _table_argument(call: exp.Anonymous) -> exp.Expression:
    arguments = call.expressions
    inner = arguments[0] if len(arguments) == 1 else None
    if isinstance(inner, (exp.Column, exp.Dot)):
        try:
            table = exp.to_table(inner.sql("bigquery"), dialect="bigquery")
        except Exception:
            return call
        return exp.Anonymous(this=TABLE_ARGUMENT, expressions=[table])
    return call


def _resolve_table_arguments(trees):
    for tree in trees:
        if tree is None:
            continue
        for call in list(tree.find_all(exp.Anonymous)):
            if call.name == TABLE_ARGUMENT:
                replacement = _table_argument(call)
                if call.parent is None:
                    tree = replacement
                elif replacement is not call:
                    call.replace(replacement)
        yield tree


def _resolve_markers(trees):
    """Turn the marker calls the rewrites left into the nodes they stand for."""

    for tree in trees:
        if tree is not None:
            tree = _resolve_like_all(tree)
            tree = _resolve_with_expressions(tree)
        yield tree


def _resolve_like_all(tree: exp.Expression) -> exp.Expression:
    for quantifier in list(tree.find_all(exp.Any)):
        unnest = quantifier.this
        if (
            isinstance(unnest, exp.Unnest) and len(unnest.expressions) == 1
            and isinstance(unnest.expressions[0], exp.Anonymous) and unnest.expressions[0].name == LIKE_ALL
            and len(unnest.expressions[0].expressions) == 1
        ):
            unnest.expressions[0].replace(unnest.expressions[0].expressions[0])
            replacement = exp.All(this=unnest)
            if quantifier.parent is None:
                return replacement
            quantifier.replace(replacement)
    return tree


def _resolve_with_expressions(tree: exp.Expression) -> exp.Expression:
    """Inside ``WITH(a AS e, ..., result)`` a bare ``a`` is the variable, not a column, so it becomes ``exp.Var``: nothing then
    counts it as a read of a column ``a``. A name also used inside a subquery of the expression is refused, because
    which ``a`` the subquery means is not something sqlglot can tell."""

    for call in list(tree.find_all(exp.Anonymous)):
        if call.name != WITH_EXPRESSION:
            continue
        *definitions, result = call.expressions
        names: list[str] = []
        for definition in definitions:
            raw = definition.expressions[0].this
            name = raw.strip("`").lower()
            if name in names:
                raise ParseError(f"WITH expression defines {raw} twice")
            names.append(name)
        scopes = [(definition.expressions[1], names[:position]) for position, definition in enumerate(definitions)]
        scopes.append((result, names))
        for scope, visible in scopes:
            for column in list(scope.find_all(exp.Column)):
                if column.table or column.name.lower() not in visible:
                    continue
                if column.find_ancestor(exp.Select, exp.Subquery) is not scope.find_ancestor(exp.Select, exp.Subquery):
                    raise ParseError(f"WITH expression variable {column.name} is used inside a subquery")
                raw = next(d.expressions[0].this for d, n in zip(definitions, names) if n == column.name.lower())
                column.replace(exp.Var(this=raw))
    return tree


class TooDeeplyNested(ParseError):
    """The query nests further than ``MAX_NESTING`` or ``MAX_TREE_DEPTH``, or than the interpreter's stack allows."""


# sqlglot's parser and its SQL writer recurse once per level of nesting. Pure-Python sqlglot ends the interpreter's stack
# at some fifty nested parentheses (a ``RecursionError``, turned into ``TooDeeplyNested`` below). The compiled build
# (sqlglotc) has no such check: it overflows the C stack and the process dies. Measured with sqlglotc 30.21.0 on a
# 512 KB stack: reading 400 nested parentheses, subqueries or CASEs, or 400 of ``@``, ``ANY`` or ``INTERVAL`` in a row;
# writing a chain of 300 ``a->'k'->'k'..`` or lambdas (250 of each of 88 constructs in four dialects are written). A
# 1 MB stack, the smallest a CPython thread gets unasked, takes about twice that. Three things keep it from happening,
# the same on every build, for any stack of 1 MB or more:
#
# - what opens a level is counted in the tokens and refused past ``MAX_NESTING`` (``nesting_weight``);
# - a statement that could take the parser deep some other way is read on a thread with a stack sized for it
#   (``_with_room``), because no list of tokens covers every construct of every dialect that sqlglot reads by recursing;
# - the tree that comes back is measured and refused past ``MAX_TREE_DEPTH`` (``_too_deep``), so what reaches the
#   writer, the optimizer and the provers fits the stack they run on.
MAX_NESTING = 100  # brackets, CASEs, nested ARRAY<..> types and prefix operators inside each other
MAX_TREE_DEPTH = 200  # nodes below each other in the parsed tree; a chain of + - * / || AND OR or set operations counts once
# A different cost of depth, on every build: sqlglot reads what is inside ``DATE(..)``, ``STRING(..)``, ``STRUCT(..)`` or any
# other call of a type's name twice, once as a type's parameters and once as arguments, so each such call inside
# another doubles the time. Sixteen take half a second with one column inside and seven with fifty; twenty-four take
# ninety seconds; forty do not finish.
MAX_TYPE_CALLS = 10  # calls of a type's name inside each other
# The same doubling for a join with no ON or USING (and not CROSS or NATURAL): sqlglot tries the joins after it as
# nested inside it, and reads them again when no ON follows. Each such join doubles the time, and with the compiled build
# the memory: twenty ``JOIN t`` in a row take six seconds and 1.1 GB, twenty-four more memory than most machines have.
MAX_OPEN_JOINS = 10  # joins still waiting for their ON

_PREFIX_OPERATORS = frozenset({TokenType.NOT, TokenType.DASH, TokenType.PLUS, TokenType.TILDE})
_ANGLE_TYPES = frozenset({TokenType.ARRAY, TokenType.STRUCT, TokenType.MAP, TokenType.RANGE, TokenType.LIST, TokenType.NESTED, TokenType.OBJECT, TokenType.UNION})
_TYPE_NAMES = frozenset(sqlglot_parser.Parser.TYPE_TOKENS)
_JOINS = frozenset(kind for kind in (TokenType.JOIN, getattr(TokenType, "STRAIGHT_JOIN", None)) if kind is not None)
_JOINS_WITHOUT_ON = frozenset({TokenType.CROSS, TokenType.NATURAL})
# Chains sqlglot reads and writes in a loop, however long (checked to 30,000 operands on a 512 KB stack): a sum of 3,000
# columns or 1,000 UNION ALL branches is one level. Other operators (``%``, ``IN``, ``a.b``, ``::``) recurse per operand.
_FLAT_CHAINS = (exp.Connector, exp.Add, exp.Sub, exp.Mul, exp.Div, exp.DPipe, exp.SetOperation)
_TOO_DEEP = f"the query is nested too deeply to read (more than {MAX_NESTING} brackets inside each other, or {MAX_TREE_DEPTH} levels of operators)"
_TOO_MANY_TYPE_CALLS = (
    f"the query is nested too deeply to read (more than {MAX_TYPE_CALLS} calls of a type's name, such as DATE(..) or STRUCT(..), "
    "inside each other: each one doubles the time it takes to read)"
)
_TOO_MANY_OPEN_JOINS = (
    f"the query cannot be read (more than {MAX_OPEN_JOINS} joins in a row with no ON or USING: each one doubles the time "
    "and the memory it takes to read)"
)

# What a token does to the count, for the few kinds that do anything.
_OPEN, _CLOSE, _BLOCK, _END, _LESS, _GREATER, _PREFIX, _COMMA, _JOIN, _ON, _SEMICOLON, _NOT_IN_A_TYPE = range(1, 13)
_ROLES = {
    TokenType.L_PAREN: _OPEN, TokenType.L_BRACKET: _OPEN, TokenType.L_BRACE: _OPEN,
    TokenType.R_PAREN: _CLOSE, TokenType.R_BRACKET: _CLOSE, TokenType.R_BRACE: _CLOSE,
    TokenType.CASE: _BLOCK, TokenType.BEGIN: _BLOCK, TokenType.END: _END,
    TokenType.LT: _LESS, TokenType.GT: _GREATER, TokenType.COMMA: _COMMA,
    TokenType.ON: _ON, TokenType.USING: _ON, **{kind: _JOIN for kind in _JOINS}, TokenType.SEMICOLON: _SEMICOLON,
    # ``list < map AND ..`` is a comparison, not the start of ``LIST<MAP..>``: these end a type that was never one
    **{kind: _NOT_IN_A_TYPE for kind in (TokenType.AND, TokenType.OR, TokenType.SELECT, TokenType.FROM, TokenType.WHERE, TokenType.THEN, TokenType.WHEN, TokenType.ELSE)},
    **{kind: _PREFIX for kind in _PREFIX_OPERATORS},
}

# The compiled parser's worst measured appetite is 1.3 KB of stack for a token (a run of ``@`` or ``ANY``). A statement
# whose weight (``nesting_weight``) is ``_DIRECT_WEIGHT`` or less is read where the caller stands: it cannot take the
# parser more than 104 KB deep, nor make a tree deeper than ``MAX_TREE_DEPTH`` (a node and a half below each other per
# token at the most, measured over 101 constructs in five dialects). A heavier one gets ``_STACK_PER_TOKEN`` for each token, between the
# two bounds (Windows allows no thread stack of 256 MiB or more), and its tree is measured. Stack is address space until
# it is touched, so the size costs nothing; the thread costs a third of a millisecond, which is why not every statement
# gets one.
_COMPILED = not (getattr(sqlglot_parser, "__file__", None) or "").endswith(".py")
_DIRECT_WEIGHT = 80
_STACK_PER_TOKEN = 4096
_SMALLEST_STACK, _LARGEST_STACK = 16 << 20, 255 << 20
_stack_size_lock = threading.Lock()
_reading = threading.local()


def nesting_weight(sql: str, tokens: list[Token]) -> int:
    """How deep ``tokens`` could take a parser that recurses, at the most; ``TooDeeplyNested`` past the fixed limits.

    A parser goes a level deeper only by reading a token, and a comma or a semicolon ends whatever was begun since the
    last one at its own bracket level (a list is read in a loop). So the weight is the largest number of tokens that can be open
    at once: for each bracket, CASE, block or ``ARRAY<`` still open, the tokens read directly inside it since its last
    comma. ``f(a, g(b, c))`` weighs 5 however long the lists are; ``NOT NOT NOT a``, ``((a))`` and ``a[0][0]`` weigh
    what they are long.

    The fixed limits are counted on the way: ``MAX_NESTING`` brackets, CASEs, ``ARRAY<`` types and prefix operators
    inside each other, ``MAX_TREE_DEPTH`` subscripts in a row, ``MAX_TYPE_CALLS`` calls of a type's name inside each
    other, ``MAX_OPEN_JOINS`` joins waiting for their ON.
    """

    count = len(tokens)
    if (
        count <= _DIRECT_WEIGHT
        and (sql.count("(") <= MAX_TYPE_CALLS or sum(1 for token in tokens if token.token_type in _TYPE_NAMES) <= MAX_TYPE_CALLS)
        and (count <= 2 * MAX_OPEN_JOINS or sum(1 for token in tokens if token.token_type in _JOINS) <= MAX_OPEN_JOINS)
    ):
        return count  # too short to weigh more, and without enough calls or joins to trip the other limits
    roles = _ROLES.get
    # per open level: tokens since its last comma, joins waiting for their ON, and what opened it (0 for the statement)
    runs, waiting, openers = [0], [0], [0]
    total = peak = nested = prefix = subscripts = type_calls = open_joins = 0
    previous = None
    for index, token in enumerate(tokens):
        kind = token.token_type
        role = roles(kind)
        if role is None:
            runs[-1] += 1
            total += 1
            if total > peak:
                peak = total
            prefix = 0
            previous = kind
            continue
        if role == _COMMA or role == _SEMICOLON:
            total -= runs[-1]
            runs[-1] = 0
            if role == _SEMICOLON:  # the next statement's joins are its own
                open_joins -= waiting[-1]
                waiting[-1] = 0
            prefix = 0
            previous = kind
            continue
        runs[-1] += 1
        total += 1
        if total > peak:
            peak = total
        if role == _PREFIX:
            prefix += 1
            if nested + prefix > MAX_NESTING:
                raise TooDeeplyNested(_TOO_DEEP)
            previous = kind
            continue
        prefix = 0
        if role == _NOT_IN_A_TYPE or role == _JOIN or role == _ON:
            while openers[-1] == _LESS:
                openers.pop()
                total -= runs.pop()
                open_joins -= waiting.pop()
                nested -= 1
            if role == _NOT_IN_A_TYPE:
                previous = kind
                continue
        if role == _JOIN:
            if previous not in _JOINS_WITHOUT_ON:
                waiting[-1] += 1
                open_joins += 1
                if open_joins > MAX_OPEN_JOINS:
                    raise TooDeeplyNested(_TOO_MANY_OPEN_JOINS)
            previous = kind
            continue
        if role == _ON:
            if waiting[-1]:
                waiting[-1] -= 1
                open_joins -= 1
            previous = kind
            continue
        if role == _OPEN:
            if kind is TokenType.L_BRACKET:
                # ``a[0][0]..``: each subscript is a level of the tree, and DuckDB's parser takes time quadratic in the chain
                subscripts = subscripts + 1 if previous is TokenType.R_BRACKET else 1
                if subscripts > MAX_TREE_DEPTH:
                    raise TooDeeplyNested(_TOO_DEEP)
            opener = _OPEN
            if kind is TokenType.L_PAREN and previous in _TYPE_NAMES:
                opener = -_OPEN  # the call of a type's name
                type_calls += 1
                if type_calls > MAX_TYPE_CALLS:
                    raise TooDeeplyNested(_TOO_MANY_TYPE_CALLS)
        elif role == _BLOCK:
            opener = _BLOCK
        elif role == _LESS and previous in _ANGLE_TYPES and any(after.token_type in _TYPE_NAMES for after in tokens[index + 1 : index + 3]):
            opener = _LESS  # ``ARRAY<INT64>`` or ``STRUCT<x INT64>``; ``range < 5`` is a comparison
        else:
            opener = 0
            if role == _CLOSE:
                # A closing bracket also ends the ``ARRAY<`` types and CASEs opened inside it and never closed.
                if _OPEN in openers or -_OPEN in openers:
                    while True:
                        closed = openers.pop()
                        total -= runs.pop()
                        open_joins -= waiting.pop()
                        nested -= 1
                        if closed == -_OPEN:
                            type_calls -= 1
                        if closed == _OPEN or closed == -_OPEN:
                            break
            elif (role == _END and openers[-1] == _BLOCK) or (role == _GREATER and openers[-1] == _LESS):
                openers.pop()
                total -= runs.pop()
                open_joins -= waiting.pop()
                nested -= 1
        if opener:
            openers.append(opener)
            runs.append(0)
            waiting.append(0)
            nested += 1
            if nested > MAX_NESTING:
                raise TooDeeplyNested(_TOO_DEEP)
        previous = kind
    return peak


def check_nesting(sql: str, tokens: list[Token]) -> None:
    """Raise ``TooDeeplyNested`` when ``tokens`` pass one of the fixed limits: before the parser recurses into them."""

    nesting_weight(sql, tokens)


def _too_deep(trees) -> bool:
    """Whether a parsed tree is deeper than ``MAX_TREE_DEPTH``, found without recursing.

    Reading ``a::INT::INT..`` or ``a % a % a..`` is a loop, so no bracket count sees them, but writing the tree recurses."""

    level = [tree for tree in trees if tree is not None]
    depth = 1
    while level:
        if depth > MAX_TREE_DEPTH:
            return True
        below = []
        for node in level:  # ``level`` grows as it is read: the left operand of a flat chain is on its parent's level
            if isinstance(node, _FLAT_CHAINS):
                left = node.args.get("this")
                if isinstance(left, _FLAT_CHAINS):
                    level.append(left)
                    below.extend(child for child in node.iter_expressions() if child is not left)
                    continue
            below.extend(node.iter_expressions())
        level = below
        depth += 1
    return False


def _with_room(work, token_count: int):
    """``work()``, on a thread whose stack is sized for ``token_count`` tokens when the compiled parser could need it.

    Pure-Python sqlglot stops at the interpreter's recursion limit on any stack, so it never needs the thread. Where a
    thread of that size cannot be had, the work is done in place, as it was before."""

    if not _COMPILED or getattr(_reading, "has_room", False):
        return work()
    outcome = []

    def run():
        _reading.has_room = True  # a parse started from inside this one is already where it needs to be
        try:
            outcome.append((work(), None))
        except BaseException as error:  # handed to the caller, whatever it is
            outcome.append((None, error))

    size = min(_LARGEST_STACK, max(_SMALLEST_STACK, token_count * _STACK_PER_TOKEN))
    try:
        with _stack_size_lock:  # the size is one setting for the whole process: set, start, put back
            previous = threading.stack_size(size)
            try:
                thread = threading.Thread(target=run, name="deep-sql-parse", daemon=True)
                thread.start()
            finally:
                threading.stack_size(previous)
    except (ValueError, RuntimeError, OSError):
        return work()
    thread.join()
    result, error = outcome[0]
    if error is not None:
        raise error
    return result


_REFUSED = object()


def read_within_limits(sql: str, tokens: list[Token], parse):
    """The statements ``parse()`` returns for ``tokens``, or ``TooDeeplyNested``: never a process that ran out of stack."""

    light = nesting_weight(sql, tokens) <= _DIRECT_WEIGHT

    def work():
        try:
            trees = parse()
        except RecursionError:
            # Pure-Python sqlglot recurses about twenty frames per nesting level, so some fifty nested parentheses end
            # the interpreter's stack. Callers handle text that cannot be read (ParseError); few expect this.
            raise TooDeeplyNested("the query is nested too deeply to read") from None
        return _REFUSED if not light and _too_deep(trees) else trees  # a tree that deep is also let go of here

    trees = work() if light else _with_room(work, len(tokens))
    if trees is _REFUSED:
        raise TooDeeplyNested(_TOO_DEEP)
    return trees


class UnclosedLiteral(ParseError):
    """A string, bytes literal or quoted name that is not triple-quoted runs onto another line, which BigQuery rejects."""


_ONE_LINE = tuple(
    kind for kind in (getattr(TokenType, name, None) for name in ("STRING", "BYTE_STRING", "RAW_STRING", "NATIONAL_STRING", "IDENTIFIER"))
    if kind is not None
)


def _check_literals(sql: str, tokens: list) -> None:
    """Raise ``UnclosedLiteral`` for a quoted token that is not triple-quoted and holds a line break.

    sqlglot reads ``'a<line break>b'`` as the string ``a\\nb``; BigQuery stops with "Unclosed string literal". Reading it
    would let a prover equate an invalid query with a valid one.
    """

    if "\n" not in sql and "\r" not in sql:
        return
    for token in tokens:
        if token.token_type in _ONE_LINE:
            text = sql[token.start : token.end + 1].lstrip("rRbB")  # a command's remaining text is a STRING token too
            if text[:1] in ("'", '"', "`") and ("\n" in text or "\r" in text) and not text.startswith(("'''", '"""')):
                raise UnclosedLiteral(f"Unclosed literal: a quoted string or name that is not triple-quoted runs past the end of line {token.line}")


def _limit_depth_in_every_dialect() -> None:
    """The same two limits for the dialects sqlglot reads without this module's help (MySQL, PostgreSQL, DuckDB, ..):
    the provers read those too, and the compiled parser ends the process on them just the same."""

    if getattr(Dialect.parse, "limits_depth", False):
        return  # a second package with this module has been imported

    def parse(self, sql, **opts):
        tokens = self.tokenize(sql)
        return read_within_limits(sql, tokens, lambda: self.parser(**opts).parse(tokens, sql))  # the dialect's own parse

    def parse_into(self, expression_type, sql, **opts):
        tokens = self.tokenize(sql)
        return read_within_limits(sql, tokens, lambda: self.parser(**opts).parse_into(expression_type, tokens, sql))

    parse.limits_depth = parse_into.limits_depth = True
    Dialect.parse = parse
    Dialect.parse_into = parse_into


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    # The parser is not touched: a compiled sqlglot (sqlglotc) dispatches parser methods through a table built when the class
    # is defined, ignores a method assigned afterwards and refuses a subclass. SQL it rejects is parsed again with the
    # tokens rewritten, and the marker call is turned into a table argument after, which works the same on every build.
    def parse_with_table_arguments(self, sql, **opts):
        # Whatever sqlglot reads on its own is left alone; only SQL it rejects is retried with the table arguments marked.
        tokens = self.tokenize(sql)
        return read_within_limits(sql, tokens, lambda: parse_tokens(self, sql, tokens, opts))

    def parse_tokens(self, sql, tokens, opts):
        try:
            _check_literals(sql, tokens)
            _check_empty_struct(sql, tokens)
            return self.parser(**opts).parse(tokens, sql)  # what the dialect's own parse does
        except UnclosedLiteral:
            raise
        except ParseError as error:
            rewritten = sql
            for rewrite in (
                _rewrite_raw_bytes, _rewrite_verbatim_calls, _rewrite_model_arguments, _rewrite_pipe_operators, _rewrite_pipe_with,
                _rewrite_like_quantifiers, _rewrite_aggregate_where, _rewrite_with_expressions, _rewrite_unnest_offset,
            ):
                rewritten = rewrite(rewritten, tokens if rewritten == sql else self.tokenize(rewritten))
            if rewritten != sql:
                try:
                    return list(_resolve_markers(parse_with_table_arguments(self, rewritten, **opts)))
                except (UnclosedLiteral, TooDeeplyNested):
                    raise
                except ParseError:
                    raise error from None
            marked = _mark_table_arguments(_merge_table_function_kind(tokens))
            if len(marked) == len(tokens) and all(a is b for a, b in zip(marked, tokens)):
                raise
            return list(_resolve_markers(_resolve_table_arguments(self.parser(**opts).parse(marked, sql))))

    BigQuery.parse = parse_with_table_arguments
    _limit_depth_in_every_dialect()
    # Releases differ in how a generator finds its handler (a method named after the class, a per-class table, a cache of both),
    # so every generator that exists gets the handler in its own table and any cache is dropped.
    pending = [Generator]
    while pending:
        generator = pending.pop()
        generator.TRANSFORMS[exp.Anonymous] = _generate
        generator.TRANSFORMS[exp.All] = _all_sql
        pending.extend(generator.__subclasses__())
    from sqlglot import generator as generator_module

    BigQuery.Generator.TRANSFORMS[exp.Filter] = _filter_sql
    getattr(generator_module, "_DISPATCH_CACHE", {}).clear()

install()
