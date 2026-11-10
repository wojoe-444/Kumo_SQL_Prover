import importlib
import tomllib
from pathlib import Path

from kumo_sql_prover import __main__ as entry

ROOT = Path(__file__).resolve().parent.parent


def test_every_command_resolves_to_a_function():
    for target in entry.COMMANDS.values():
        module, _, attribute = target.partition(":")
        assert callable(getattr(importlib.import_module(module), attribute))


def test_console_scripts_match_the_commands():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert sorted(scripts.values()) == sorted(entry.COMMANDS.values())


def test_unknown_command_exits_2_and_help_exits_0(capsys):
    assert entry.main(["nonsense"]) == 2
    assert entry.main(["--help"]) == 0
    assert "smt" in capsys.readouterr().out
