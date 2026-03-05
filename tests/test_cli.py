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


def test_web_help_shows_separate_terminals_option():
    runner = CliRunner()
    result = runner.invoke(main, ["web", "--help"])
    assert result.exit_code == 0
    assert "--separate-terminals" in result.output
    assert "--terminal-tabs" in result.output


def test_web_separate_terminals_requires_dev():
    runner = CliRunner()
    result = runner.invoke(main, ["web", "--separate-terminals"])
    assert result.exit_code != 0
    assert "--separate-terminals requires --dev." in result.output


def test_web_terminal_tabs_requires_dev():
    runner = CliRunner()
    result = runner.invoke(main, ["web", "--terminal-tabs"])
    assert result.exit_code != 0
    assert "--terminal-tabs requires --dev." in result.output


def test_web_terminal_flags_are_mutually_exclusive():
    runner = CliRunner()
    result = runner.invoke(main, ["web", "--dev", "--separate-terminals", "--terminal-tabs"])
    assert result.exit_code != 0
    assert "Choose only one of --separate-terminals or --terminal-tabs." in result.output


def test_web_separate_terminals_invokes_split_launcher(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_open_dev_in_separate_terminals(
        host: str, port: int, repo_name: str | None, repo_root: str | None
    ) -> None:
        calls["host"] = host
        calls["port"] = port
        calls["repo_name"] = repo_name
        calls["repo_root"] = repo_root

    monkeypatch.setattr("prscope.cli._open_dev_in_separate_terminals", _fake_open_dev_in_separate_terminals)

    runner = CliRunner()
    result = runner.invoke(main, ["web", "--dev", "--separate-terminals", "--repo", "demo"])
    assert result.exit_code == 0
    assert calls == {
        "host": "127.0.0.1",
        "port": 8420,
        "repo_name": "demo",
        "repo_root": None,
    }


def test_web_terminal_tabs_invokes_tabs_launcher(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_open_dev_in_terminal_tabs(host: str, port: int, repo_name: str | None, repo_root: str | None) -> None:
        calls["host"] = host
        calls["port"] = port
        calls["repo_name"] = repo_name
        calls["repo_root"] = repo_root

    monkeypatch.setattr("prscope.cli._open_dev_in_terminal_tabs", _fake_open_dev_in_terminal_tabs)

    runner = CliRunner()
    result = runner.invoke(main, ["web", "--dev", "--terminal-tabs", "--repo", "demo"])
    assert result.exit_code == 0
    assert calls == {
        "host": "127.0.0.1",
        "port": 8420,
        "repo_name": "demo",
        "repo_root": None,
    }
