from __future__ import annotations

from prscope.planning.scanners import get_scanner, list_scanners
from prscope.planning.scanners.grep import GrepScanner
from prscope.planning.scanners.repomap import RepoMapScanner


def test_get_scanner_unknown_falls_back_to_grep() -> None:
    s = get_scanner("definitely_not_a_real_scanner_name")
    assert isinstance(s, GrepScanner)
    assert s.name == "grep"


def test_list_scanners_includes_registered_backends() -> None:
    rows = list_scanners()
    names = {str(r["name"]) for r in rows}
    assert names >= {"grep", "repomap", "repomix"}


def test_repomap_resolves_to_grep_when_aider_unavailable() -> None:
    if RepoMapScanner().is_available():
        s = get_scanner("repomap")
        assert isinstance(s, RepoMapScanner)
    else:
        s = get_scanner("repomap")
        assert isinstance(s, GrepScanner)
