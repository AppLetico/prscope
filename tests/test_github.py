from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from prscope.github import GitHubClient, parse_since


def test_parse_since_relative_days():
    result = parse_since("30d")
    now = datetime.now(timezone.utc)
    expected = now - timedelta(days=30)
    # Allow 1 second tolerance
    assert abs((result - expected).total_seconds()) < 1


def test_parse_since_relative_months():
    result = parse_since("6m")
    now = datetime.now(timezone.utc)
    expected = now - timedelta(days=180)  # 6 * 30
    assert abs((result - expected).total_seconds()) < 1


def test_parse_since_relative_years():
    result = parse_since("1y")
    now = datetime.now(timezone.utc)
    expected = now - timedelta(days=365)
    assert abs((result - expected).total_seconds()) < 1


def test_parse_since_iso_date():
    result = parse_since("2024-06-15")
    assert result.year == 2024
    assert result.month == 6
    assert result.day == 15


def test_parse_since_iso_datetime():
    result = parse_since("2024-06-15T12:30:00Z")
    assert result.year == 2024
    assert result.month == 6
    assert result.day == 15
    assert result.hour == 12


def test_parse_since_invalid_defaults():
    result = parse_since("invalid")
    now = datetime.now(timezone.utc)
    expected = now - timedelta(days=90)
    assert abs((result - expected).total_seconds()) < 1


def test_list_pulls_filters_merged():
    """Test that state='merged' filters to only merged PRs."""
    client = GitHubClient(token="test-token")

    # Mock the _paginate method
    mock_items = [
        {
            "number": 1,
            "state": "closed",
            "title": "Merged PR",
            "body": None,
            "user": {"login": "alice"},
            "labels": [],
            "updated_at": "2024-01-01T00:00:00Z",
            "merged_at": "2024-01-02T00:00:00Z",  # This one is merged
            "head": {"sha": "sha1"},
            "html_url": "https://example.com/pr/1",
        },
        {
            "number": 2,
            "state": "closed",
            "title": "Closed but not merged",
            "body": None,
            "user": {"login": "bob"},
            "labels": [],
            "updated_at": "2024-01-01T00:00:00Z",
            "merged_at": None,  # This one is not merged
            "head": {"sha": "sha2"},
            "html_url": "https://example.com/pr/2",
        },
    ]

    with patch.object(client, "_paginate", return_value=iter(mock_items)):
        prs = client.list_pulls("owner/repo", state="merged")

    # Should only return the merged PR
    assert len(prs) == 1
    assert prs[0].number == 1
    assert prs[0].merged_at is not None


def test_list_pulls_respects_since():
    """Test that since parameter filters by date."""
    client = GitHubClient(token="test-token")

    mock_items = [
        {
            "number": 1,
            "state": "closed",
            "title": "Recent PR",
            "body": None,
            "user": {"login": "alice"},
            "labels": [],
            "updated_at": "2024-12-01T00:00:00Z",
            "merged_at": "2024-12-01T00:00:00Z",
            "head": {"sha": "sha1"},
            "html_url": "https://example.com/pr/1",
        },
        {
            "number": 2,
            "state": "closed",
            "title": "Old PR",
            "body": None,
            "user": {"login": "bob"},
            "labels": [],
            "updated_at": "2024-01-01T00:00:00Z",  # Old
            "merged_at": "2024-01-01T00:00:00Z",
            "head": {"sha": "sha2"},
            "html_url": "https://example.com/pr/2",
        },
    ]

    with patch.object(client, "_paginate", return_value=iter(mock_items)):
        # Only get PRs since June 2024
        prs = client.list_pulls("owner/repo", state="merged", since="2024-06-01")

    # Should only return the recent PR
    assert len(prs) == 1
    assert prs[0].number == 1
