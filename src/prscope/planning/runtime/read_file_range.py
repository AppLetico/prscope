"""
Line-windowed file reads for planning tools.

Small files use read_text + splitlines (fast path). Larger files use a streaming
line iterator so peak memory stays proportional to the returned window, not the
whole file (see Claude Code readFileInRange pattern).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadFileSliceResult:
    """Aligned with ToolExecutor.read_file return payload (minus path / read_history)."""

    truncated: bool
    line_count: int
    file_size_bytes: int
    start_line: int
    end_line: int
    content: str


def _normalize_line(line: str, strip_bom: bool) -> str:
    s = line.rstrip("\r\n")
    if strip_bom and s.startswith("\ufeff"):
        s = s[1:]
    return s


def _count_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def _read_lines_slice(path: Path, start_idx: int, end_idx: int) -> list[str]:
    """Read lines with 0-based indices in [start_idx, end_idx)."""
    if start_idx >= end_idx:
        return []
    out: list[str] = []
    i = 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = _normalize_line(line, i == 0)
            if start_idx <= i < end_idx:
                out.append(raw)
            i += 1
    return out


def _compute_indices(
    total_lines: int,
    max_lines: int,
    start_line: int | None,
    around_line: int | None,
    radius: int,
) -> tuple[int, int]:
    if around_line is not None:
        focus = max(1, int(around_line))
        win = max(1, int(radius))
        start_idx = max(0, focus - win - 1)
        end_idx = min(total_lines, focus + win)
    elif start_line is not None:
        start_idx = max(0, int(start_line) - 1)
        end_idx = min(total_lines, start_idx + max(1, int(max_lines)))
    else:
        start_idx = 0
        end_idx = min(total_lines, max(1, int(max_lines)))
    return start_idx, end_idx


def read_file_slice_fast(
    path: Path,
    max_lines: int = 200,
    start_line: int | None = None,
    around_line: int | None = None,
    radius: int = 80,
) -> ReadFileSliceResult:
    """read_text + splitlines — matches historical ToolExecutor behavior."""
    file_size_bytes = path.stat().st_size
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.splitlines()
    start_idx, end_idx = _compute_indices(len(lines), max_lines, start_line, around_line, radius)
    snippet = lines[start_idx:end_idx]
    return ReadFileSliceResult(
        truncated=start_idx > 0 or end_idx < len(lines),
        line_count=len(lines),
        file_size_bytes=file_size_bytes,
        start_line=start_idx + 1,
        end_line=end_idx,
        content="\n".join(snippet),
    )


def read_file_slice_streaming(
    path: Path,
    file_size_bytes: int,
    max_lines: int = 200,
    start_line: int | None = None,
    around_line: int | None = None,
    radius: int = 80,
) -> ReadFileSliceResult:
    """Single- or two-pass streaming read; bounded memory for snippet only."""
    if around_line is not None:
        total = _count_lines(path)
        start_idx, end_idx = _compute_indices(total, max_lines, start_line, around_line, radius)
        snippet = _read_lines_slice(path, start_idx, end_idx)
        truncated = start_idx > 0 or end_idx < total
        return ReadFileSliceResult(
            truncated=truncated,
            line_count=total,
            file_size_bytes=file_size_bytes,
            start_line=start_idx + 1,
            end_line=end_idx,
            content="\n".join(snippet),
        )

    max_n = max(1, int(max_lines))
    snippet: list[str] = []
    i = 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = _normalize_line(line, i == 0)
            if start_line is not None:
                start_idx = max(0, int(start_line) - 1)
                if i >= start_idx and len(snippet) < max_n:
                    snippet.append(raw)
            else:
                if i < max_n:
                    snippet.append(raw)
            i += 1
    total = i
    start_idx, end_idx = _compute_indices(total, max_lines, start_line, around_line, radius)
    truncated = start_idx > 0 or end_idx < total
    return ReadFileSliceResult(
        truncated=truncated,
        line_count=total,
        file_size_bytes=file_size_bytes,
        start_line=start_idx + 1,
        end_line=end_idx,
        content="\n".join(snippet),
    )


def read_file_slice(
    path: Path,
    *,
    max_lines: int = 200,
    start_line: int | None = None,
    around_line: int | None = None,
    radius: int = 80,
    fast_path_max_bytes: int,
) -> ReadFileSliceResult:
    """Choose fast (read whole file) vs streaming by on-disk size."""
    st = path.stat()
    size = st.st_size
    if size <= fast_path_max_bytes:
        return read_file_slice_fast(path, max_lines, start_line, around_line, radius)
    return read_file_slice_streaming(
        path,
        size,
        max_lines=max_lines,
        start_line=start_line,
        around_line=around_line,
        radius=radius,
    )
