"""The CLI is the only surface a reader touches first, so it is covered."""

import pytest

from arena.cli import main


def test_contract_command_lists_the_decision_space(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["contract"]) == 0

    out = capsys.readouterr().out
    for action in ("approve", "reject", "defer", "ask", "bond"):
        assert action in out
    assert "mandate_exceeded" in out


def test_no_subcommand_is_an_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code != 0


def test_version_flag_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "arena" in capsys.readouterr().out
