"""Nothing in this package imports KumoSQL: the import graph runs one way (product -> prover)."""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "src" / "kumo_sql_prover"


def test_no_module_imports_kumosql():
    offenders = []
    for path in sorted(SOURCE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            offenders += [f"{path.relative_to(SOURCE)}:{node.lineno} imports {name}" for name in names if name == "kumosql" or name.startswith("kumosql.")]
    assert not offenders, "\n".join(offenders)


def test_every_relative_import_resolves_inside_the_package():
    modules = {p.relative_to(SOURCE).with_suffix("").parts for p in SOURCE.rglob("*.py")}
    names = {".".join(parts[:-1] if parts[-1] == "__init__" else parts) for parts in modules}
    missing = []
    for path in sorted(SOURCE.rglob("*.py")):
        package = path.relative_to(SOURCE).parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = list(package[: len(package) - (node.level - 1)]) + (node.module.split(".") if node.module else [])
                target = ".".join(base)
                candidates = {target, *(f"{target}.{a.name}".strip(".") for a in node.names)}
                if not candidates & names and target:
                    missing.append(f"{path.relative_to(SOURCE)}:{node.lineno} imports {target}")
    assert not missing, "\n".join(missing)
