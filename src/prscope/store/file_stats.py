from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class StoreFileStatsMixin:
    def _repo_stats_path(self, repo_name: str) -> Path:
        base = Path.home() / ".prscope" / "repos" / repo_name
        base.mkdir(parents=True, exist_ok=True)
        return base / "constraint_stats.json"

    def _rounds_log_path(self, repo_name: str) -> Path:
        base = Path.home() / ".prscope" / "repos" / repo_name
        base.mkdir(parents=True, exist_ok=True)
        return base / "rounds.jsonl"

    def update_constraint_stats(self, repo_name: str, constraint_ids: list[str], session_id: str) -> None:
        path = self._repo_stats_path(repo_name)
        data: dict[str, Any] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        stamp = datetime.utcnow().strftime("%Y-%m-%d")
        for cid in constraint_ids:
            current = data.get(cid, {"violations": 0, "last_seen": stamp, "sessions": []})
            current["violations"] = int(current.get("violations", 0)) + 1
            current["last_seen"] = stamp
            sessions = current.get("sessions", [])
            if session_id not in sessions:
                sessions.append(session_id)
            current["sessions"] = sessions
            data[cid] = current
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def get_constraint_stats(self, repo_name: str) -> dict[str, Any]:
        path = self._repo_stats_path(repo_name)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}
