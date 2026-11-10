"""docs/proof-safeguards.md names every proof family of the registry (the rules that use them are KumoSQL's)."""

from pathlib import Path

from kumo_sql_prover.proof_registry import FAMILIES

DOCS = Path(__file__).resolve().parent.parent / "docs" / "proof-safeguards.md"


def test_the_docs_list_every_family():
    text = DOCS.read_text(encoding="utf-8")
    for name in FAMILIES:
        assert f"`{name}`" in text, f"docs/proof-safeguards.md does not mention the family {name}"
