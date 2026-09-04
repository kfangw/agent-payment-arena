"""Command line entry point.

Kept deliberately small. Subcommands are added as the capability behind them
starts working, so `arena --help` never advertises something that is not there
yet.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from arena import __version__
from arena.evaluation import load_result, run_attack_suite, run_mcp_demo, run_minimum_suite
from arena.experiments.artifacts import write_json_once
from arena.frontier import PolicyGrid, build_frontier, run_policy_grid, write_frontier
from arena.gateway.contract import Action, ErrorCode, default_code_for
from arena.report import build_report, render_markdown, write_report


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
    demo = subcommands.add_parser("demo", help="run the minimum offline evaluation")
    demo.add_argument("--seed", type=int, default=1)
    run = subcommands.add_parser("run", help="run an evaluation suite")
    run.add_argument("--suite", choices=("minimum", "attack-catalog"), default="minimum")
    run.add_argument("--repetitions", type=int, default=2)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--out", type=Path, required=True)
    report = subcommands.add_parser("report", help="aggregate an evaluation artifact")
    report.add_argument("result", type=Path)
    report.add_argument("--json-out", type=Path, required=True)
    report.add_argument("--markdown-out", type=Path, required=True)
    frontier = subcommands.add_parser("frontier", help="run a policy grid and plot its frontier")
    frontier.add_argument("--limits", default="25,50,100")
    frontier.add_argument("--ask-thresholds", default="off,20,50")
    frontier.add_argument("--bond-thresholds", default="off,50")
    frontier.add_argument("--repetitions", type=int, default=2)
    frontier.add_argument("--seed", type=int, default=1)
    frontier.add_argument("--json-out", type=Path, required=True)
    frontier.add_argument("--svg-out", type=Path, required=True)
    frontier.add_argument("--otlp-endpoint")
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
    if args.command == "demo":
        first, second = run_mcp_demo(args.seed)
        print(f"MCP payment flow: {first} -> {second}")
        print(render_markdown(build_report(run_minimum_suite(2, args.seed))), end="")
        return 0
    if args.command == "run":
        result = (
            run_minimum_suite(args.repetitions, args.seed)
            if args.suite == "minimum"
            else run_attack_suite(args.repetitions, args.seed)
        )
        write_json_once(args.out, result.to_dict())
        print(args.out)
        return 0
    if args.command == "report":
        report = build_report(load_result(args.result))
        write_report(report, args.json_out, args.markdown_out)
        print(args.markdown_out)
        return 0
    if args.command == "frontier":
        sink = None
        if args.otlp_endpoint:
            from arena.telemetry import OpenTelemetryTraceSink

            sink = OpenTelemetryTraceSink(args.otlp_endpoint)
        grid = PolicyGrid(
            tuple(int(value) for value in args.limits.split(",")),
            _optional_ints(args.ask_thresholds),
            _optional_ints(args.bond_thresholds),
        )
        frontier = build_frontier(
            run_policy_grid(grid, args.repetitions, args.seed, trace_sink=sink)
        )
        write_frontier(frontier, args.json_out, args.svg_out)
        if sink is not None:
            sink.close()
        print(args.svg_out)
        return 0

    # argparse rejects unknown subcommands before reaching here.
    raise AssertionError(f"unhandled command: {args.command!r}")


def _optional_ints(raw: str) -> tuple[int | None, ...]:
    return tuple(None if value.strip() == "off" else int(value) for value in raw.split(","))


if __name__ == "__main__":
    sys.exit(main())
