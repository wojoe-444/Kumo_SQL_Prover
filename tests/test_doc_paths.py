"""A path the docs name in code font is a file of this repository, or a link to the file in KumoSQL."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = re.compile(r"(?<!\[)`((?:tests|tools|src/kumo_sql_prover|benchmarks)/[\w./-]+)`")


def test_every_path_the_docs_name_exists():
    pages = [*sorted((ROOT / "docs").rglob("*.md")), *sorted((ROOT / "docs_simple").rglob("*.md")), ROOT / "README.md", ROOT / "CLAUDE.md", ROOT / "benchmarks" / "README.md"]
    missing = []
    for page in pages:
        for path in PATH.findall(page.read_text(encoding="utf-8")):
            if "*" in path or path.endswith("/"):
                continue
            if not (ROOT / path.rstrip(".")).exists():
                missing.append(f"{page.relative_to(ROOT)}: {path}")
    assert not missing, "name a file that exists here, or link the KumoSQL file:\n" + "\n".join(missing)


def test_the_results_files_name_only_files_that_exist():
    missing = []
    for results in sorted((ROOT / "benchmarks" / "results").glob("*.json")):
        for path in re.findall(r"(?:tests|tools|src/kumo_sql_prover)/[\w./-]*\w", results.read_text(encoding="utf-8")):
            if not (ROOT / path).exists():
                missing.append(f"{results.name}: {path}")
    assert not missing, "\n".join(missing)
