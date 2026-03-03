from __future__ import annotations

from click.testing import CliRunner

from prscope.cli import main


def test_cli_help_shows_planning_commands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "plan" in result.output
    assert "repos" in result.output
    assert "upstream" in result.output
    assert "recall" in result.output
    assert "sync" not in result.output  # top-level sync hidden in second-pass CLI


def test_cli_no_legacy_prd_command():
    runner = CliRunner()
    result = runner.invoke(main, ["prd"])
    assert result.exit_code != 0
    assert "No such command 'prd'" in result.output


def test_plan_start_help_includes_no_recall_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["plan", "start", "--help"])
    assert result.exit_code == 0
    assert "--no-recall" in result.output


def test_recall_help_shows_core_options():
    runner = CliRunner()
    result = runner.invoke(main, ["recall", "--help"])
    assert result.exit_code == 0
    assert "--repo" in result.output
    assert "--all-repos" in result.output
    assert "--full" in result.output
