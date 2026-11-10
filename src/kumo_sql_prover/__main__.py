"""``python -m kumo_sql_prover COMMAND ...``: the prover's command-line entry points without a launcher.

``smt`` is the SMT prover (``kumo-prove-smt``) and ``sqlsolver`` the algebraic prover with the optional SQLSolver jar
(``kumo-prove-sqlsolver``). Both read two SQL files and print a verdict.
"""

from __future__ import annotations

import importlib
import sys

# Keep in step with [project.scripts] in pyproject.toml (tests/test_main_module.py checks).
COMMANDS = {
    "smt": "kumo_sql_prover.smt_equivalence:main",
    "sqlsolver": "kumo_sql_prover.sqlsolver_backend:main",
}


def usage() -> str:
    lines = ["usage: python -m kumo_sql_prover COMMAND [ARGS...]", "", "commands:"]
    lines += [f"  {name}" for name in COMMANDS]
    lines += ["", "Run `python -m kumo_sql_prover COMMAND --help` for a command's options. Both commands only read and print."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    target = COMMANDS.get(args[0])
    if target is None:
        print(f"kumo_sql_prover: unknown command {args[0]!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module, _, attribute = target.partition(":")
    function = getattr(importlib.import_module(module), attribute)
    sys.argv = [f"python -m kumo_sql_prover {args[0]}", *args[1:]]
    result = function()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
