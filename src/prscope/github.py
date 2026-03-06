"""
GitHub REST API client for Prscope.

Fetches pull requests and their files from upstream repositories.
Uses GITHUB_TOKEN environment variable for authentication.

Supports:
- Date-based windowing (since parameter)
- Incremental sync (watermark tracking)
- Rate limit handling
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .store import Store

GITHUB_API_BASE = "https://api.github.com"
DEFAULT_PER_PAGE = 100
MAX_RETRIES = 3
RETRY_DELAY = 1.0


def parse_since(since: str) -> datetime:
    """
    Parse a 'since' value into a datetime.

    Supports:
    - ISO date: "2024-01-01"
    - Relative days: "30d", "90d"
    - Relative months: "3m", "6m"
    - Relative years: "1y"

    Returns:
        datetime in UTC
    """
    now = datetime.now(timezone.utc)

    # Try relative format first
    match = re.match(r"^(\d+)([dmy])$", since.lower())
    if match:
        value = int(match.group(1))
        unit = match.group(2)

        if unit == "d":
            return now - timedelta(days=value)
        elif unit == "m":
            return now - timedelta(days=value * 30)
        elif unit == "y":
            return now - timedelta(days=value * 365)

    # Try ISO date format
    try:
        dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    # Default to 90 days ago
    return now - timedelta(days=90)


@dataclass
class GitHubPR:
    """Parsed GitHub PR data."""

    number: int
    state: str
    title: str
    body: str | None
    author: str | None
    labels: list[str]
    updated_at: str | None
    merged_at: str | None
    head_sha: str | None
    html_url: str


@dataclass
class GitHubFile:
    """Parsed GitHub file change data."""

    path: str
    additions: int
    deletions: int
    status: str  # added, removed, modified, renamed


class GitHubAPIError(Exception):
    """Error from GitHub API."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(GitHubAPIError):
    """Rate limit exceeded."""

    def __init__(self, reset_time: int | None = None):
        super().__init__("GitHub API rate limit exceeded", 403)
        self.reset_time = reset_time


class GitHubClient:
    """GitHub REST API client with pagination and rate limit handling."""

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()

        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"

        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "prscope/0.1.0"

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Make an API request with retry and rate limit handling."""
        url = f"{GITHUB_API_BASE}{endpoint}"

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.request(method, url, params=params, **kwargs)

                # Check rate limit
                if response.status_code == 403:
                    remaining = response.headers.get("X-RateLimit-Remaining")
                    if remaining == "0":
                        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                        raise RateLimitError(reset_time)

                # Check for errors
                if response.status_code >= 400:
                    raise GitHubAPIError(
                        f"GitHub API error: {response.status_code} - {response.text}",
                        response.status_code,
                    )

                return response

            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise GitHubAPIError(f"Request failed: {e}")

        raise GitHubAPIError("Max retries exceeded")

    def _paginate(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate through paginated API results."""
        params = params or {}
        params.setdefault("per_page", DEFAULT_PER_PAGE)
        page = 1

        while True:
            if max_pages and page > max_pages:
                break

            params["page"] = page
            response = self._request("GET", endpoint, params=params)
            items = response.json()

            if not items:
                break

            yield from items

            # Check if there are more pages
            if len(items) < params["per_page"]:
                break

            page += 1

    def list_pulls(
        self,
        repo: str,
        state: str = "merged",
        sort: str = "updated",
        direction: str = "desc",
        max_prs: int | None = None,
        since: str | datetime | None = None,
    ) -> list[GitHubPR]:
        """
        List pull requests for a repository.

        Args:
            repo: Full repository name (owner/repo)
            state: PR state filter (merged, open, closed, all)
                   Note: "merged" fetches closed PRs and filters to only merged ones
            sort: Sort field (created, updated, popularity, long-running)
            direction: Sort direction (asc, desc)
            max_prs: Maximum number of PRs to fetch
            since: Only fetch PRs updated/merged after this date
                   Accepts ISO date string, relative format ("90d", "6m"), or datetime

        Returns:
            List of GitHubPR objects
        """
        endpoint = f"/repos/{repo}/pulls"

        # Parse since parameter
        since_dt: datetime | None = None
        if since:
            if isinstance(since, str):
                since_dt = parse_since(since)
            else:
                since_dt = since

        # GitHub API only supports open, closed, all
        # For "merged", we fetch closed and filter
        filter_merged_only = state == "merged"
        api_state = "closed" if filter_merged_only else state

        params = {
            "state": api_state,
            "sort": sort,
            "direction": direction,
        }

        prs = []
        # Fetch extra if filtering merged (some closed PRs won't be merged)
        fetch_multiplier = 2 if filter_merged_only else 1
        max_pages = ((max_prs * fetch_multiplier) // DEFAULT_PER_PAGE + 1) if max_prs else None

        for item in self._paginate(endpoint, params, max_pages=max_pages):
            pr = self._parse_pr(item)

            # Filter to only merged PRs if requested
            if filter_merged_only and not pr.merged_at:
                continue

            # Date filter - check merged_at or updated_at
            if since_dt:
                pr_date = pr.merged_at or pr.updated_at
                if pr_date:
                    try:
                        pr_dt = datetime.fromisoformat(pr_date.replace("Z", "+00:00"))
                        if pr_dt < since_dt:
                            # PRs are sorted by updated desc, so we can stop here
                            break
                    except ValueError:
                        pass

            prs.append(pr)

            if max_prs and len(prs) >= max_prs:
                break

        return prs

    def get_pull(self, repo: str, number: int) -> GitHubPR:
        """Get a specific pull request."""
        endpoint = f"/repos/{repo}/pulls/{number}"
        response = self._request("GET", endpoint)
        return self._parse_pr(response.json())

    def get_pull_files(self, repo: str, number: int) -> list[GitHubFile]:
        """
        Get files changed in a pull request.

        Args:
            repo: Full repository name (owner/repo)
            number: PR number

        Returns:
            List of GitHubFile objects
        """
        endpoint = f"/repos/{repo}/pulls/{number}/files"

        files = []
        for item in self._paginate(endpoint):
            files.append(
                GitHubFile(
                    path=item.get("filename", ""),
                    additions=item.get("additions", 0),
                    deletions=item.get("deletions", 0),
                    status=item.get("status", "modified"),
                )
            )

        return files

    def _parse_pr(self, data: dict[str, Any]) -> GitHubPR:
        """Parse raw PR data into GitHubPR object."""
        user = data.get("user", {})
        head = data.get("head", {})
        labels = data.get("labels", [])

        return GitHubPR(
            number=data.get("number", 0),
            state=data.get("state", ""),
            title=data.get("title", ""),
            body=data.get("body"),
            author=user.get("login") if user else None,
            labels=[label.get("name", "") for label in labels if label.get("name")],
            updated_at=data.get("updated_at"),
            merged_at=data.get("merged_at"),
            head_sha=head.get("sha") if head else None,
            html_url=data.get("html_url", ""),
        )

    def check_rate_limit(self) -> dict[str, Any]:
        """Check current rate limit status."""
        response = self._request("GET", "/rate_limit")
        return response.json()


def sync_repo_prs(
    client: GitHubClient,
    store: Store,
    repo_name: str,
    state: str = "merged",
    max_prs: int = 100,
    fetch_files: bool = True,
    since: str | None = None,
    incremental: bool = True,
    progress_callback: callable | None = None,
) -> tuple[int, int, int]:
    """
    Sync PRs from a GitHub repository to the store.

    Args:
        client: GitHub API client
        store: Prscope store
        repo_name: Full repository name (owner/repo)
        state: PR state filter
        max_prs: Maximum PRs to sync
        fetch_files: Whether to fetch file lists
        since: Only sync PRs after this date (ISO or relative like "90d")
        incremental: If True, use last sync watermark as cutoff
        progress_callback: Optional callback(stage, current, total, message)

    Returns:
        Tuple of (new_count, updated_count, skipped_count)
    """

    def report(stage: str, current: int, total: int, message: str = ""):
        if progress_callback:
            progress_callback(stage, current, total, message)

    # Ensure repo exists in store
    repo = store.upsert_upstream_repo(repo_name)

    # Determine the effective 'since' date
    effective_since: str | datetime | None = None

    if incremental and repo.last_synced_at:
        # Use watermark from last sync (fetch slightly earlier for safety)
        try:
            watermark = datetime.fromisoformat(repo.last_synced_at.replace("Z", "+00:00"))
            # Go back 1 day for safety margin
            effective_since = watermark - timedelta(days=1)
        except ValueError:
            effective_since = since
    else:
        effective_since = since

    report("fetch", 0, 0, "Fetching PR list from GitHub API...")

    # Fetch PRs from GitHub with date filter
    prs = client.list_pulls(
        repo_name,
        state=state,
        max_prs=max_prs,
        since=effective_since,
    )

    total_prs = len(prs)
    report("fetch", total_prs, total_prs, f"Found {total_prs} PRs")

    new_count = 0
    updated_count = 0
    skipped_count = 0
    last_updated = None

    for i, gh_pr in enumerate(prs, 1):
        # Check if PR already exists
        existing = store.get_pull_request(repo.id, gh_pr.number)

        # Skip if no changes (same head_sha)
        if existing and existing.head_sha == gh_pr.head_sha:
            skipped_count += 1
            report("process", i, total_prs, f"PR #{gh_pr.number} unchanged")
            continue

        action = "Updating" if existing else "Adding"
        report("process", i, total_prs, f"{action} PR #{gh_pr.number}: {gh_pr.title[:40]}...")

        # Upsert PR
        pr = store.upsert_pull_request(
            repo_id=repo.id,
            number=gh_pr.number,
            state=gh_pr.state,
            title=gh_pr.title,
            body=gh_pr.body,
            author=gh_pr.author,
            labels=gh_pr.labels,
            updated_at=gh_pr.updated_at,
            merged_at=gh_pr.merged_at,
            head_sha=gh_pr.head_sha,
            html_url=gh_pr.html_url,
        )

        if existing:
            updated_count += 1
        else:
            new_count += 1

        # Track latest update time
        if gh_pr.updated_at:
            if last_updated is None or gh_pr.updated_at > last_updated:
                last_updated = gh_pr.updated_at

        # Fetch files if needed
        if fetch_files:
            report("files", i, total_prs, f"Fetching files for PR #{gh_pr.number}...")
            files = client.get_pull_files(repo_name, gh_pr.number)
            store.save_pr_files(
                pr.id,
                [{"path": f.path, "additions": f.additions, "deletions": f.deletions} for f in files],
            )

    # Update repo sync watermark
    store.update_repo_sync_time(repo.id, last_updated)

    report("done", total_prs, total_prs, "Sync complete")

    return new_count, updated_count, skipped_count
