"""``--help`` on a benchmark tool prints usage and changes nothing; an unknown suite is refused, not scored."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
HELPFUL = "parse_check_sweep rule_fuzz_gen sqlsolver_bench scoreboard cosette_bench".split()  # not quoted: these are not eval floors


def _run(tool, *args):
    return subprocess.run([sys.executable, str(TOOLS / f"{tool}.py"), *args], capture_output=True, text=True, cwd=ROOT, timeout=120)


@pytest.mark.parametrize("tool", HELPFUL)
def test_help_prints_usage_and_exits_cleanly(tool):
    result = _run(tool, "--help")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage:")


def test_scoreboard_help_does_not_rewrite_the_readme():
    before = (ROOT / "README.md").read_bytes()
    _run(HELPFUL[3], "--help")
    assert (ROOT / "README.md").read_bytes() == before


@pytest.mark.parametrize("tool", "cosette_bench sqlsolver_bench".split())
def test_an_unknown_suite_is_refused(tool):
    result = _run(tool, "no-such-suite")
    assert result.returncode == 2 and "unknown suite" in result.stderr


def test_a_sweep_argument_must_name_a_dialect():
    result = _run(HELPFUL[0], "spider2")
    assert result.returncode == 2 and "DIR:DIALECT" in result.stderr
