"""A derived table written ``SELECT *`` over a join lists only the columns the select around it reads.

    SELECT MIN(t.a), MIN(t.x) FROM (SELECT * FROM r, d AS d1, d AS d2 WHERE ..) AS t, u WHERE t.b = u.b
    SELECT MIN(t.a), MIN(t.x) FROM (SELECT r.a, r.x, r.b FROM r, d AS d1, d AS d2 WHERE ..) AS t, u WHERE t.b = u.b

A star over a join that reads one table twice returns two columns of each of that table's names. PostgreSQL
accepts the derived table as long as nobody reads one of the repeated names, and Calcite's planner writes every
join level this way (``SELECT * FROM (SELECT * FROM a, b ..) AS t0, c ..``). The normalizer spells a star out
only when the names are distinct (``_expand_stars``), so such a query kept its stars and the prover declined it
(``SELECT * inside a derived relation``).

A derived table's output columns that nobody reads can be left out: the table has the same rows in the same
number, and the select around it sees the same values in the columns it does read. So this pass replaces the
star by the columns read through the table's alias, each named by the one source that has it. It runs from
the outside in, because the select around a derived table must have its own list written out before the
columns it reads are known. It runs after ``_expand_stars``, so only the stars that pass left are read; the
level that pass did spell out, the one whose written list has a name twice, loses its unread items the same
way (plain items only: no aggregate, window, subquery or unknown function).

Read only when:

* the derived table is one ``SELECT *`` (a single bare star, no ``EXCEPT`` or ``REPLACE``) with ``FROM``,
  joins and ``WHERE`` alone: no ``DISTINCT`` (fewer columns would merge more rows), grouping, set operation,
  ``ORDER BY``, ``LIMIT``, ``USING`` or ``NATURAL`` join (which list a merged column once), and no lateral
  source;
* every source of it has a name of its own and known columns: a table of the schema, a derived table with a
  written list, or another such star (its columns are its sources' columns, in order);
* the select around it has no star of its own, names every column it reads with a source (a bare column could
  be one of the derived table's), reads the derived table by its alias, which no other source of that select
  or of a subquery in it shares, and has no lateral source and no alias column list on the derived table;
* every column read exists exactly once among the derived table's sources (a repeated one is an error in
  PostgreSQL: the query is left as written), and at least one column is read.
"""

from __future__ import annotations

from sqlglot import exp

from .ast_utils import select_sources

_ALLOWED = frozenset({"expressions", "from_", "from", "joins", "where", "kind", "hint"})


def _is_star_select(node: exp.Expression) -> bool:
    if not isinstance(node, exp.Select) or len(node.expressions) != 1:
        return False
    star = node.expressions[0]
    if type(star) is not exp.Star or any(star.args.values()):
        return False
    if any(value for key, value in node.args.items() if key not in _ALLOWED and value):
        return False
    return not any(j.args.get("using") or (j.args.get("method") or "") for j in node.args.get("joins") or [])


def _columns(source: exp.Expression, schema: dict[str, list[str]], depth: int = 0) -> list[str] | None:
    """The output names of a source, in order and with repeats; ``None`` when they are not known."""

    if isinstance(source, exp.Table):
        alias = source.args.get("alias")
        if (alias is not None and alias.args.get("columns")) or any(v for k, v in source.args.items() if k not in ("this", "db", "catalog", "alias") and v):
            return None  # ``r AS x(a, b)`` renames the columns; a pivot, sample or function table has its own
        parts = [p.name for p in (source.args.get("catalog"), source.args.get("db"), source.this) if p is not None]
        return schema.get(".".join(parts).lower())
    if not isinstance(source, exp.Subquery) or depth > 40:
        return None
    alias = source.args.get("alias")
    if (alias is not None and alias.args.get("columns")) or any(v for k, v in source.args.items() if k not in ("this", "alias") and v):
        return None
    inner = source.this
    if _is_star_select(inner):
        pairs = _source_columns(inner, schema, depth + 1)
        return None if pairs is None else [name for _, name in pairs]
    if not isinstance(inner, exp.Select):
        return None
    names = []
    for item in inner.expressions:
        name = (item.alias_or_name or "").lower()
        if not name or isinstance(item, exp.Star) or (isinstance(item, exp.Column) and isinstance(item.this, exp.Star)):
            return None
        names.append(name)
    return names


def _source_columns(select: exp.Select, schema: dict[str, list[str]], depth: int = 0) -> list[tuple[str, str]] | None:
    """``(source name, column)`` for every column a ``SELECT *`` of ``select``'s sources returns, in order."""

    sources = select_sources(select)
    names = [(s.alias_or_name or "").lower() for s in sources]
    if not sources or "" in names or len(set(names)) != len(names):
        return None
    found = []
    for source, name in zip(sources, names):
        columns = _columns(source, schema, depth)
        if columns is None:
            return None
        found.extend((name, column) for column in columns)
    return found


def _inside(node: exp.Expression, top: exp.Expression, ids: set[int]) -> bool:
    node = node.parent
    while node is not None and node is not top:
        if id(node) in ids:
            return True
        node = node.parent
    return False


def _narrow(select: exp.Select, schema: dict[str, list[str]]) -> None:
    sources = select_sources(select)
    if not any(isinstance(s, exp.Subquery) and isinstance(s.this, exp.Select) and (_is_star_select(s.this) or _repeats_a_name(s.this)) for s in sources):
        return
    if any(isinstance(s, (exp.Lateral, exp.Unnest)) for s in sources) or select.args.get("laterals"):
        return
    source_ids = {id(s) for s in sources}
    outside = [n for n in select.walk() if isinstance(n, (exp.Column, exp.Star, exp.Subquery, exp.Table)) and not _inside(n, select, source_ids)]
    outside = [n for n in outside if id(n) not in source_ids]
    if any(isinstance(n, exp.Star) and not isinstance(n.parent, exp.Count) for n in outside):
        return  # a star reads every column
    columns = [n for n in outside if isinstance(n, exp.Column)]
    if any(not c.table for c in columns):
        return  # a bare column may be any source's
    aliases = [(s.alias_or_name or "").lower() for s in sources]
    nested = {(n.alias_or_name or "").lower() for n in outside if isinstance(n, (exp.Subquery, exp.Table))}
    for source in sources:
        if not isinstance(source, exp.Subquery) or not isinstance(source.this, exp.Select):
            continue
        alias_node = source.args.get("alias")
        alias = (source.alias or "").lower()
        if not alias or aliases.count(alias) != 1 or alias in nested or (alias_node is not None and alias_node.args.get("columns")):
            continue
        if any(value for key, value in source.args.items() if key not in ("this", "alias") and value):
            continue
        inner = source.this
        read = {c.name.lower() for c in columns if c.table.lower() == alias}
        if not read:
            continue
        if _is_star_select(inner):
            available = _source_columns(inner, schema)
            if available is None:
                continue
            names = [column for _, column in available]
            items = [exp.column(column, table=owner) for owner, column in available]
        elif _repeats_a_name(inner):
            names = [(item.alias_or_name or "").lower() for item in inner.expressions]
            items = list(inner.expressions)
        else:
            continue
        if any(names.count(name) != 1 for name in read):
            continue  # a column that is not there, or is there twice: left as written
        inner.set("expressions", [item for item, name in zip(items, names) if name in read])


def _repeats_a_name(inner: exp.Select) -> bool:
    """A written select list over ``FROM``, joins and ``WHERE`` alone, two of whose plain items share a name."""

    if any(value for key, value in inner.args.items() if key not in _ALLOWED and value):
        return False
    names = []
    for item in inner.expressions:
        name = (item.alias_or_name or "").lower()
        if not name or any(isinstance(n, (exp.Star, exp.AggFunc, exp.Window, exp.Anonymous, exp.Subquery, exp.Exists)) for n in item.walk()):
            return False  # an aggregate decides how many rows the table has: its list stays whole
        names.append(name)
    return len(set(names)) != len(names)


def list_read_star_columns(tree: exp.Expression, schema: dict[str, list[str]] | None) -> exp.Expression:
    """Replace the star of each derived ``SELECT *`` over known sources by the columns read from it."""

    if not schema:
        return tree
    schema = {key.lower(): [c.lower() for c in columns] for key, columns in schema.items()}
    for select in list(tree.find_all(exp.Select)):  # breadth first: the select around a derived table comes before it
        _narrow(select, schema)
    return tree
