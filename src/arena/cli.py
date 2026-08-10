"""Command line entry point.

Kept deliberately small. Subcommands are added as the capability behind them
starts working, so `arena --help` never advertises something that is not there
yet.
"""

import argparse
import sys
from collections.abc import Sequence

from arena import __version__
from arena.gateway.contract import Action, ErrorCode, default_code_for


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arena",
        description="Measure what LLM agents do with payment authority.",
    )
    parser.add_argument("--version", action="version", version=f"arena {__version__}")

    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "contract",
        help="print the gateway contract both backends are held to",
    )
    return parser


def _print_contract() -> None:
    print("Decision space (shared with the reference gateway)")
    for action in Action:
        code = default_code_for(action)
        if code is None:
            print(f"  {action.value}")
        else:
            print(f"  {action.value:<8}  -> {code.value}")

    print()
    print("Refusal codes")
    for error_code in ErrorCode:
        print(f"  {error_code.value}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector excluding the program name. Defaults to
            `sys.argv[1:]`.

    Returns:
        A process exit status.
    """
    args = _build_parser().parse_args(argv)

    if args.command == "contract":
        _print_contract()
        return 0

    # argparse rejects unknown subcommands before reaching here.
    raise AssertionError(f"unhandled command: {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
