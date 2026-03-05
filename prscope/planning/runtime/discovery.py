"""
Chat-first requirements discovery flow.

On first user message the LLM scans the codebase via tools before responding,
then asks only questions that cannot be answered from code.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable

from ...config import PlanningConfig
from ...memory import MemoryStore
from ...pricing import MODEL_CONTEXT_WINDOWS
from .telemetry import completion_telemetry
from .tools import CODEBASE_TOOLS, ToolExecutor

logger = logging.getLogger(__name__)

DISCOVERY_SYSTEM_PROMPT = """You are a planning assistant helping scope a software implementation plan.

## Your Process

**Step 1 — Research first (ALWAYS on the first user message):**
Use list_files, read_file, and grep_code to understand the project before asking anything.
Read: README, key source files, package manifests, existing patterns relevant to the request.
Do not skip this step. Hallucinating project structure is worse than asking.
If the request mentions endpoints/routes/APIs, inspect backend route handlers (not only frontend files) before asking.

**Step 2 — Ask only what code can't tell you:**
After scanning, ask ONLY 2-3 questions that require a human decision:
- Priorities and trade-offs
- Acceptance criteria or definition of done
- Business constraints not visible in code
- Architectural preferences where multiple valid approaches exist

Never ask questions whose answers are already in the codebase.
If code scan evidence clearly identifies the backend/API framework (for example FastAPI imports
or route decorators), do NOT ask which backend/framework is being used.

## Question Format (REQUIRED)

You MUST format every question with lettered multiple-choice options:

**Q1: <your question>**
A) <concrete option — the most common sensible default>
B) <concrete alternative>
C) <another concrete alternative>
D) Other — describe your preference

Rules:
- Always provide A–C or A–D options covering the realistic choices for this specific codebase.
- Always end with an "Other" option so the user can type a custom answer.
- Options must be specific and actionable (e.g. "A) Add a `lastSeen` column to the `users` table", not "A) Use the database").
- Do NOT ask open-ended questions without options.

**Step 3 — Complete discovery:**
After 1-2 exchanges when you have enough context, return ONLY this JSON (no extra text after it):
{"discovery":"complete","summary":"<comprehensive requirements summary including relevant file paths and constraints>"}

## Rules
- Research first, questions second.
- Max 3 questions per turn.
- Batch questions in one turn: when discovery is still open, ask all unresolved questions together
  in a single response (usually 3), not one-by-one across multiple turns.
- Do not ask "Q2" or "Q3" in later turns if those could have been asked in the prior question batch.
- Only ask a follow-up single question when a prior answer is genuinely ambiguous or contradictory.
- Never repeat a question the user already answered in a previous turn.
- If the user's message already answers all open questions, complete immediately.
- Reference specific file paths you found when relevant.
"""


@dataclass
class QuestionOption:
    letter: str  # "A", "B", "C", "D"
    text: str
    is_other: bool = False


@dataclass
class DiscoveryQuestion:
    index: int  # 1-based question number shown in the UI
    text: str
    options: list[QuestionOption]

    def option_text(self, letter: str) -> str:
        for opt in self.options:
            if opt.letter == letter:
                return opt.text
        return ""


def parse_questions(reply: str) -> list[DiscoveryQuestion]:
    """
    Parse structured Q&A blocks from an LLM reply.

    Recognises:
      **Q1: question text**    or    **Q: text**    or    **1. text** (bold)
    followed by lines:
      A) option text
      B) option text
      ...
    """
    questions: list[DiscoveryQuestion] = []
    lines = reply.splitlines()
    i = 0

    # Accept both bold and plain headers:
    #   **Q1: text** / Q1: text / Question 1: text / 1. text
    q_re = re.compile(
        r"^(?:Q\s*\d*|Question\s+\d+|\d+)\s*[:\.\)\-]\s*(.+)$",
        re.IGNORECASE,
    )
    # Accept option prefixes with markdown bullets and separators:
    #   A) text / A. text / - A) text / * B. text
    opt_re = re.compile(r"^(?:[-*]\s*)?([A-D])[\)\.\-:]\s*(.+)$", re.IGNORECASE)

    def normalize_line(line: str) -> str:
        normalized = line.strip()
        # Strip simple markdown wrappers and bullets.
        normalized = re.sub(r"^[-*]\s+", "", normalized)
        normalized = normalized.replace("**", "").strip()
        return normalized

    while i < len(lines):
        line = normalize_line(lines[i])
        q_match = q_re.match(line)
        if q_match:
            q_text = q_match.group(1).strip().rstrip(":?").strip() + "?"
            options: list[QuestionOption] = []
            i += 1
            while i < len(lines):
                stripped = normalize_line(lines[i])
                opt_match = opt_re.match(stripped)
                if opt_match:
                    letter = opt_match.group(1).upper()
                    text = opt_match.group(2).strip()
                    is_other = bool(re.match(r"other", text, re.IGNORECASE))
                    options.append(QuestionOption(letter=letter, text=text, is_other=is_other))
                    i += 1
                elif not stripped and options:
                    # blank line after options block — stop collecting
                    break
                elif stripped and not opt_match:
                    # non-option text after collecting at least one option — stop
                    if options:
                        break
                    i += 1  # pre-option prose — keep scanning
                else:
                    i += 1
            if options:
                questions.append(
                    DiscoveryQuestion(
                        index=len(questions) + 1,
                        text=q_text,
                        options=options,
                    )
                )
        else:
            i += 1

    return questions


@dataclass
class DiscoveryTurnResult:
    reply: str
    complete: bool
    summary: str | None = None
    questions: list[DiscoveryQuestion] = field(default_factory=list)


@dataclass
class FeatureIntent:
    label: str
    keywords: list[str]
    patterns: list[str]

    @functools.cached_property
    def compiled_patterns(self) -> list[re.Pattern]:
        return [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns]


@dataclass
class Framework:
    name: str
    route_patterns: list[re.Pattern]
    file_patterns: list[str]


@dataclass
class Evidence:
    path: str
    snippet: str
    confidence: int


@dataclass
class CodeSignal:
    name: str
    patterns: list[re.Pattern]


@dataclass
class IndexedMatch:
    path: str
    line_number: int
    line: str


SignalIndex = dict[str, list[IndexedMatch]]

FRAMEWORKS = [
    Framework(
        name="fastapi",
        route_patterns=[
            re.compile(r"fastapi\(", re.I),
            re.compile(r"apirouter\(", re.I),
            re.compile(r"@(app|router)\.(get|post|put|patch|delete)\(", re.I),
        ],
        file_patterns=["api.py", "routes.py", "router.py", "main.py", "app.py"],
    ),
    Framework(
        name="flask",
        route_patterns=[
            re.compile(r"from\s+flask", re.I),
            re.compile(r"flask\(", re.I),
            re.compile(r"@app\.route\(", re.I),
        ],
        file_patterns=["app.py", "routes.py", "views.py"],
    ),
    Framework(
        name="express",
        route_patterns=[
            re.compile(r"express\(", re.I),
            re.compile(r"app\.(get|post|put|patch|delete)\(", re.I),
            re.compile(r"router\.(get|post|put|patch|delete)\(", re.I),
        ],
        file_patterns=["server.ts", "server.js", "routes.ts", "routes.js", "router.ts", "router.js"],
    ),
    Framework(
        name="django",
        route_patterns=[
            re.compile(r"from\s+django", re.I),
            re.compile(r"urlpatterns", re.I),
            re.compile(r"\b(path|re_path)\(", re.I),
        ],
        file_patterns=["urls.py", "views.py"],
    ),
    Framework(
        name="rails",
        route_patterns=[
            re.compile(r"rails\.application", re.I),
            re.compile(r"\bresources\s+:", re.I),
            re.compile(r"\b(get|post|put|patch|delete)\s+['\"]", re.I),
        ],
        file_patterns=["routes.rb", "application.rb"],
    ),
    Framework(
        name="spring",
        route_patterns=[
            re.compile(r"@restcontroller", re.I),
            re.compile(r"@(get|post|put|patch|delete)mapping", re.I),
            re.compile(r"@requestmapping", re.I),
        ],
        file_patterns=["Controller.java", "Controller.kt"],
    ),
    Framework(
        name="gin",
        route_patterns=[
            re.compile(r"gin\.(default|new)\(", re.I),
            re.compile(r"\b\w+\.(GET|POST|PUT|PATCH|DELETE)\(", re.I),
        ],
        file_patterns=["main.go", "routes.go", "handler.go", "server.go"],
    ),
    Framework(
        name="aspnet",
        route_patterns=[
            re.compile(r"\[apicontroller\]", re.I),
            re.compile(r"\[(httpget|httppost|httpput|httppatch|httpdelete)\]", re.I),
            re.compile(r"map(get|post|put|patch|delete)\(", re.I),
        ],
        file_patterns=["Controller.cs", "Program.cs"],
    ),
]

_bootstrap_patterns = [pattern.pattern for framework in FRAMEWORKS for pattern in framework.route_patterns]
BOOTSTRAP_ROUTE_REGEX = re.compile("|".join(_bootstrap_patterns) if _bootstrap_patterns else r"$^", re.I)

CODE_SIGNALS = [
    CodeSignal(
        "route",
        [
            re.compile(r"@(app|router)\.(get|post|put|patch|delete)\(", re.I),
            re.compile(r"app\.(get|post|put|patch|delete)\(", re.I),
            re.compile(r"router\.(get|post|put|patch|delete)\(", re.I),
            re.compile(r"@(get|post|put|patch|delete)mapping", re.I),
        ],
    ),
    CodeSignal(
        "middleware",
        [
            re.compile(r"app\.use\(", re.I),
            re.compile(r"\bmiddleware\b", re.I),
        ],
    ),
    CodeSignal(
        "worker",
        [
            re.compile(r"celery\.task", re.I),
            re.compile(r"@shared_task", re.I),
            re.compile(r"rq\.queue", re.I),
        ],
    ),
    CodeSignal(
        "cron",
        [
            re.compile(r"\bcron\b", re.I),
            re.compile(r"schedule\.every", re.I),
        ],
    ),
    CodeSignal(
        "cli",
        [
            re.compile(r"@click\.(command|group)", re.I),
            re.compile(r"\bargparse\b", re.I),
            re.compile(r"commander\.(command|program)", re.I),
        ],
    ),
]

_INTENT_STOP_WORDS = {"a", "an", "the", "new", "feature", "endpoint", "support", "system"}
_VENDOR_DIRS = {"node_modules", "venv", ".env", "dist", "build", "__pycache__", ".git", ".tox", "egg-info"}
_BACKEND_DIR_NAMES = {"backend", "api", "server", "src", "services", "web", "app", "lib"}
_CODE_NOUNS = {
    "endpoint",
    "api",
    "route",
    "middleware",
    "handler",
    "controller",
    "model",
    "service",
    "worker",
    "auth",
    "authentication",
    "login",
}
MAX_INTENT_KEYWORDS = 3
_MAX_TRUSTWORTHY_LINES = 5000


class DiscoveryManager:
    def __init__(
        self,
        config: PlanningConfig,
        tool_executor: ToolExecutor,
        memory: MemoryStore,
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ):
        self.config = config
        self.tool_executor = tool_executor
        self.memory = memory
        self.turn_counts_by_session: dict[str, int] = {}
        self.bootstrap_insights_by_session: dict[str, dict[str, Any]] = {}
        self._active_discovery_session_id: str | None = None
        self.event_callback = event_callback

    def reset_session(self, session_id: str) -> None:
        self.turn_counts_by_session[session_id] = 0

    def clear_session(self, session_id: str) -> None:
        self.turn_counts_by_session.pop(session_id, None)
        if hasattr(self, "bootstrap_insights_by_session"):
            self.bootstrap_insights_by_session.pop(session_id, None)
        if getattr(self, "_active_discovery_session_id", None) == session_id:
            self._active_discovery_session_id = None

    def _next_turn_count(self, session_id: str) -> int:
        current = int(self.turn_counts_by_session.get(session_id, 0))
        next_count = current + 1
        self.turn_counts_by_session[session_id] = next_count
        return next_count

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        maybe = self.event_callback(event)
        if asyncio.iscoroutine(maybe):
            await maybe

    @staticmethod
    def _normalize_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Map internal role names to OpenAI-accepted roles (author → assistant)."""
        role_map = {"author": "assistant"}
        return [{**m, "role": role_map.get(m.get("role", ""), m.get("role", ""))} for m in messages]

    def _build_memory_context(self) -> str:
        """Inject pre-built memory blocks so LLM has architecture overview upfront."""
        blocks = self.memory.load_all_blocks()
        manifesto = self.memory.load_manifesto()
        parts = []
        if manifesto:
            parts.append(f"## Project Manifesto\n{manifesto}")
        for name in ("architecture", "modules", "patterns", "entrypoints", "context"):
            block = blocks.get(name, "").strip()
            if block:
                parts.append(f"## {name.title()}\n{block}")
        return "\n\n".join(parts)

    @staticmethod
    def _latest_user_message(conversation: list[dict[str, Any]]) -> str:
        for message in reversed(conversation):
            if str(message.get("role", "")).strip() == "user":
                return str(message.get("content", "")).strip()
        return ""

    @staticmethod
    def _pattern(word: str) -> str:
        return rf"\b{re.escape(word)}\b"

    @classmethod
    def _extract_feature_intent(cls, user_message: str) -> FeatureIntent | None:
        """Extract one primary feature intent for the current user turn."""
        text = str(user_message or "").strip().lower()
        if not text:
            return None
        text = re.sub(r"[^\w\s\-\.]", " ", text)
        match = re.search(r"\b(create|add|build|implement|introduce|make|setup)\b", text, flags=re.IGNORECASE)
        if not match:
            return None
        tail = text[match.end() :].strip()
        if not tail:
            return None
        words = re.findall(r"[a-z0-9][a-z0-9._-]*", tail)
        keywords: list[str] = []
        for word in words:
            normalized = word.strip("-_.")
            if not normalized or normalized in _INTENT_STOP_WORDS:
                continue
            keywords.append(normalized)
            if len(keywords) >= MAX_INTENT_KEYWORDS:
                break
        if not keywords:
            return None
        patterns: list[str] = []
        for keyword in keywords:
            patterns.append(cls._pattern(keyword))
            patterns.append(cls._pattern(keyword.replace("-", "_")))
            patterns.append(cls._pattern(keyword.replace("_", "-")))
        return FeatureIntent(label=" ".join(keywords), keywords=keywords, patterns=list(dict.fromkeys(patterns)))

    @classmethod
    def _should_bootstrap_scan(cls, user_message: str) -> bool:
        text = str(user_message or "").strip().lower()
        if not text:
            return False
        if re.search(r"\b(endpoint|route|routing|api|http|webhook|handler)\b", text):
            return True
        intent = cls._extract_feature_intent(text)
        if intent is None:
            return False
        return any(noun in text for noun in _CODE_NOUNS)

    @staticmethod
    def _parse_existing_endpoint_followup_choice(user_message: str) -> str | None:
        normalized = " ".join((user_message or "").strip().split()).lower()
        if not normalized:
            return None
        if re.fullmatch(r"(q1[:\-\s]*)?a", normalized):
            return "A"
        if re.fullmatch(r"(q1[:\-\s]*)?b", normalized):
            return "B"
        if re.fullmatch(r"(q1[:\-\s]*)?c", normalized):
            return "C"
        if "review current behavior" in normalized or "summarize it only" in normalized:
            return "A"
        if "propose targeted enhancements" in normalized or "without creating a new route" in normalized:
            return "B"
        if "leave it unchanged" in normalized or "no planning needed" in normalized:
            return "C"
        return None

    @staticmethod
    def _detect_framework(index: SignalIndex) -> str | None:
        candidates = index.get("framework", [])
        if not candidates:
            return None
        best_name: str | None = None
        best_score = 0
        for framework in FRAMEWORKS:
            score = 0
            for pattern in framework.route_patterns:
                if any(pattern.search(match.line) for match in candidates):
                    score += 1
            if score > best_score:
                best_name = framework.name
                best_score = score
        return best_name if best_score > 0 else None

    @staticmethod
    def _is_framework_identification_question(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return False
        framework_names = {framework.name for framework in FRAMEWORKS}
        if not any(token in normalized for token in tuple(framework_names) + ("backend", "framework")):
            return False
        framework_pattern = "|".join(re.escape(name) for name in sorted(framework_names))
        patterns = (
            r"\bwhat\s+backend\b",
            r"\bwhich\s+backend\b",
            r"\bwhat\s+framework\b",
            r"\bwhich\s+framework\b",
            rf"\bare\s+we\s+using\s+({framework_pattern})\b",
            rf"\b({framework_pattern})\s+or\s+({framework_pattern})\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @classmethod
    def _drop_redundant_framework_questions(
        cls,
        parsed: DiscoveryTurnResult,
        inferred_framework: str | None,
    ) -> DiscoveryTurnResult:
        if parsed.complete or not parsed.questions or not inferred_framework:
            return parsed
        filtered_questions = [
            question for question in parsed.questions if not cls._is_framework_identification_question(question.text)
        ]
        if len(filtered_questions) == len(parsed.questions):
            return parsed
        return DiscoveryTurnResult(
            reply=parsed.reply,
            complete=False,
            summary=parsed.summary,
            questions=filtered_questions,
        )

    @staticmethod
    def _route_file_score(path: str) -> int:
        normalized_path = str(path or "").replace("\\", "/")
        path_lower = normalized_path.lower()
        filename = path_lower.rsplit("/", 1)[-1]
        score = 0
        if any(token in path_lower for token in ("/api/", "/routes/", "/controllers/", "/handlers/")):
            score += 4
        if re.search(r"(routes?|router|controller|handler)\.", filename):
            score += 6
        if re.search(r"(server|app|main|urls|views)\.", filename):
            score += 2
        if "test" in path_lower:
            score -= 3
        return score

    @staticmethod
    def _location_score(path: str, file_line_count: int | None = None) -> int:
        score = 0
        lowered = str(path or "").replace("\\", "/").lower()
        if "/middleware/" in lowered:
            score += 5
        if "/routes/" in lowered:
            score += 4
        if "/api/" in lowered:
            score += 4
        if "/controllers/" in lowered:
            score += 4
        if "/handlers/" in lowered:
            score += 3
        if "/services/" in lowered:
            score += 3
        if "/utils/" in lowered:
            score -= 1
        if "/helpers/" in lowered:
            score -= 1
        if "/docs/" in lowered:
            score -= 3
        if "test" in lowered:
            score -= 3
        if "/examples/" in lowered:
            score -= 2
        if "/vendor/" in lowered:
            score -= 4
        if file_line_count and file_line_count > _MAX_TRUSTWORTHY_LINES:
            score -= 5
        return score

    @staticmethod
    def _detect_code_signals(index: SignalIndex) -> dict[str, int]:
        output: dict[str, int] = {}
        for signal in CODE_SIGNALS:
            count = len(index.get(signal.name, []))
            if count > 0:
                output[signal.name] = count
        return output

    @staticmethod
    def _detect_architecture(signal_scores: dict[str, int]) -> str | None:
        arch_scores = {
            "api_service": signal_scores.get("route", 0) * 3 + signal_scores.get("middleware", 0) * 2,
            "worker_service": signal_scores.get("worker", 0) * 3 + signal_scores.get("cron", 0) * 2,
            "cli_tool": signal_scores.get("cli", 0) * 3,
        }
        best = max(arch_scores, key=arch_scores.get)
        return best if arch_scores[best] > 0 else None

    @staticmethod
    def _build_signal_index(matches: list[dict[str, Any]]) -> SignalIndex:
        all_patterns: list[tuple[str, re.Pattern]] = []
        for signal in CODE_SIGNALS:
            for pattern in signal.patterns:
                all_patterns.append((signal.name, pattern))
        for framework in FRAMEWORKS:
            for pattern in framework.route_patterns:
                all_patterns.append(("framework", pattern))
        index: SignalIndex = {signal.name: [] for signal in CODE_SIGNALS}
        index["framework"] = []
        lines_per_path: dict[str, int] = {}
        for item in matches:
            path = str(item.get("path", "")).strip()
            line_number = int(item.get("line", 0) or 0)
            line = str(item.get("text", ""))
            if not path:
                continue
            lines_per_path[path] = max(lines_per_path.get(path, 0), line_number)
            if lines_per_path[path] > _MAX_TRUSTWORTHY_LINES:
                continue
            indexed = IndexedMatch(path=path, line_number=line_number, line=line)
            for signal_name, pattern in all_patterns:
                if pattern.search(line):
                    index[signal_name].append(indexed)
        return index

    @staticmethod
    def _aggregate_evidence(evidence_list: list[Evidence]) -> list[Evidence]:
        if not evidence_list:
            return []
        by_path: dict[str, Evidence] = {}
        for evidence in evidence_list:
            existing = by_path.get(evidence.path)
            if existing is None:
                by_path[evidence.path] = evidence
                continue
            if evidence.confidence > existing.confidence:
                by_path[evidence.path] = evidence
            elif evidence.confidence == existing.confidence and len(evidence.snippet) > len(existing.snippet):
                by_path[evidence.path] = evidence
        return sorted(by_path.values(), key=lambda item: item.confidence, reverse=True)

    @classmethod
    def _select_scan_directories(cls, entries: list[dict[str, Any]]) -> list[str]:
        if not entries:
            return ["."]
        scores: dict[str, int] = {}
        for entry in entries:
            path = str(entry.get("path", "")).strip().replace("\\", "/")
            if not path:
                continue
            depth = path.count("/")
            name = path.rsplit("/", 1)[-1]
            name_lower = name.lower()
            if name.startswith(".") or name_lower in _VENDOR_DIRS or depth > 2:
                continue
            kind = str(entry.get("type", "")).lower()
            if kind == "dir":
                scores.setdefault(path, 0)
                if name_lower in _BACKEND_DIR_NAMES:
                    scores[path] += 4
                continue
            parent = path.rsplit("/", 1)[0] if "/" in path else "."
            parent_lower = parent.rsplit("/", 1)[-1].lower()
            if parent_lower in _VENDOR_DIRS or parent.startswith(".") or parent.count("/") > 2:
                continue
            scores.setdefault(parent, 0)
            if re.search(r"\.(py|ts|tsx|js|jsx|go|rb|java|kt|cs)$", path, flags=re.IGNORECASE):
                scores[parent] += 3
            if cls._route_file_score(path) > 0:
                scores[parent] += 5
            if parent_lower in _BACKEND_DIR_NAMES:
                scores[parent] += 4
        if not scores:
            return ["."]
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [path for path, _ in ranked[:3]]

    @staticmethod
    def _format_grep_matches(matches: list[dict[str, Any]], limit: int = 14) -> str:
        if not matches:
            return "- (no matches)"
        preferred = sorted(
            matches,
            key=lambda item: (
                -DiscoveryManager._route_file_score(str(item.get("path", ""))),
                str(item.get("path", "")),
                int(item.get("line", 0)),
            ),
        )
        rendered: list[str] = []
        for item in preferred[:limit]:
            path = str(item.get("path", "")).strip()
            line = int(item.get("line", 0) or 0)
            text = str(item.get("text", "")).strip()
            if not path:
                continue
            rendered.append(f"- `{path}:{line}` {text[:120]}")
        if not rendered:
            return "- (no matches)"
        if len(preferred) > len(rendered):
            rendered.append(f"- ... and {len(preferred) - len(rendered)} more")
        return "\n".join(rendered)

    def _existing_feature_evidence_lines(self, session_id: str) -> list[str]:
        insights = getattr(self, "bootstrap_insights_by_session", {}).get(session_id, {})
        lines = [str(line).strip() for line in insights.get("matched_evidence", []) if str(line).strip()]
        if lines:
            return lines[:3]
        paths = [str(path).strip() for path in insights.get("matched_paths", []) if str(path).strip()]
        feature_label = str(insights.get("feature_label", "feature")).strip() or "feature"
        if paths:
            return [f"Found existing {feature_label} in `{paths[0]}`"]
        return [f"Found existing {feature_label} in the codebase"]

    @staticmethod
    def _parse_evidence_reference(line: str) -> tuple[str, int] | None:
        match = re.search(r"`([^`:]+):(\d+)`", str(line))
        if not match:
            return None
        path = match.group(1).strip()
        line_num = int(match.group(2))
        if not path or line_num <= 0:
            return None
        return path, line_num

    @staticmethod
    def _summarize_endpoint_snippet(snippet: str) -> str | None:
        if not snippet.strip():
            return None
        route_match = re.search(
            r"@(app|router)\.(get|post|put|patch|delete)\(([^)]*)\)",
            snippet,
            flags=re.IGNORECASE,
        )
        handler_match = re.search(r"^\s*(async\s+def|def)\s+([a-zA-Z0-9_]+)\(", snippet, flags=re.MULTILINE)
        return_match = re.search(r"^\s*return\s+(.+)$", snippet, flags=re.MULTILINE)
        details: list[str] = []
        if route_match:
            verb = route_match.group(2).upper()
            route_expr = route_match.group(3).strip()
            details.append(f"Detected endpoint shape: `{verb} {route_expr}`")
        if handler_match:
            details.append(f"Handler function: `{handler_match.group(2)}`")
        if return_match:
            details.append(f"Observed return behavior: `{return_match.group(1).strip()[:120]}`")
        return "\n".join(f"- {item}" for item in details) if details else None

    @staticmethod
    def _functional_summary_from_snippet(snippet: str) -> str | None:
        if not snippet.strip():
            return None
        route_match = re.search(
            r"@(app|router)\.(get|post|put|patch|delete)\(([^)]*)\)",
            snippet,
            flags=re.IGNORECASE,
        )
        return_match = re.search(r"^\s*return\s+(.+)$", snippet, flags=re.MULTILINE)
        if route_match and return_match:
            verb = route_match.group(2).upper()
            route_expr = route_match.group(3).strip().strip("'\"")
            return_expr = return_match.group(1).strip()
            return f"This route already exists: `{verb} {route_expr}` currently returns `{return_expr[:120]}`."
        return None

    async def _build_existing_endpoint_deep_summary(self, session_id: str) -> tuple[str | None, str | None]:
        evidence_lines = self._existing_feature_evidence_lines(session_id)
        for line in evidence_lines:
            parsed = self._parse_evidence_reference(line)
            if parsed is None:
                continue
            path, anchor_line = parsed
            await self._emit(
                {
                    "type": "tool_call",
                    "name": "read_file",
                    "session_stage": "discovery",
                    "path": path,
                    "query": f"around:{anchor_line}",
                }
            )
            started = asyncio.get_running_loop().time()
            try:
                if not hasattr(self, "tool_executor"):
                    continue
                payload = await asyncio.to_thread(
                    self.tool_executor.read_file,
                    path,
                    120,
                    None,
                    anchor_line,
                    24,
                )
            except Exception:
                continue
            finally:
                elapsed = (asyncio.get_running_loop().time() - started) * 1000.0
                await self._emit(
                    {
                        "type": "tool_result",
                        "name": "read_file",
                        "session_stage": "discovery",
                        "duration_ms": round(elapsed, 2),
                    }
                )
            snippet = str(payload.get("content", ""))
            summary = self._summarize_endpoint_snippet(snippet)
            functional = self._functional_summary_from_snippet(snippet)
            if summary or functional:
                return summary, functional
        return None, None

    def _merge_feature_evidence(
        self,
        *,
        session_id: str,
        feature: FeatureIntent,
        candidate_paths: list[str],
        evidence_lines: list[str] | None = None,
    ) -> None:
        if not hasattr(self, "bootstrap_insights_by_session"):
            self.bootstrap_insights_by_session = {}
        insights = self.bootstrap_insights_by_session.setdefault(
            session_id,
            {"existing_feature": False, "feature_label": feature.label, "matched_paths": [], "matched_evidence": []},
        )
        existing_paths = {str(path).strip() for path in insights.get("matched_paths", []) if str(path).strip()}
        existing_evidence = {str(line).strip() for line in insights.get("matched_evidence", []) if str(line).strip()}
        for path in candidate_paths:
            normalized = str(path).strip()
            if normalized:
                existing_paths.add(normalized)
        for line in evidence_lines or []:
            normalized_line = str(line).strip()
            if normalized_line:
                existing_evidence.add(normalized_line)
        insights["matched_paths"] = sorted(existing_paths)
        insights["matched_evidence"] = sorted(existing_evidence)
        insights["existing_feature"] = bool(existing_paths)
        insights["feature_label"] = feature.label
        insights["feature_keywords"] = list(feature.keywords)
        insights["feature_patterns"] = list(feature.patterns)

    async def _maybe_read_more_context(self, path: str, line_number: int) -> str:
        payload = await self._run_bootstrap_tool(tool_name="read_file", path=path)
        if not payload or not isinstance(payload, dict):
            return ""
        content = str(payload.get("content", ""))
        if not content:
            return ""
        if line_number <= 0:
            return content
        lines = content.splitlines()
        if line_number > len(lines):
            return content
        start = max(0, line_number - 6)
        end = min(len(lines), line_number + 5)
        return "\n".join(lines[start:end])

    async def _ingest_feature_evidence_from_tool(
        self,
        *,
        session_id: str,
        feature: FeatureIntent | None,
        tool_name: str,
        parsed_args: dict[str, Any],
        tool_result_payload: dict[str, Any],
    ) -> list[Evidence]:
        if tool_name not in {"grep_code", "read_file"} or feature is None:
            return []
        patterns = feature.compiled_patterns
        payload = tool_result_payload.get("result")
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            payload = payload.get("result")
        if not isinstance(payload, dict):
            payload = {}
        evidence_list: list[Evidence] = []
        candidate_paths: set[str] = set()
        if tool_name == "grep_code":
            results = payload.get("results", [])
            if isinstance(results, list):
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    path = str(item.get("path", "")).strip()
                    text = str(item.get("text", "")).strip()
                    line_num = int(item.get("line", 0) or 0)
                    if not path:
                        continue
                    matched = any(pattern.search(text) for pattern in patterns)
                    snippet = text
                    if not matched and snippet.count("\n") < 2:
                        expanded = await self._maybe_read_more_context(path, line_num)
                        if expanded and any(pattern.search(expanded) for pattern in patterns):
                            matched = True
                            snippet = expanded
                    if not matched:
                        continue
                    confidence = self._location_score(path)
                    evidence_list.append(Evidence(path=path, snippet=snippet[:240], confidence=confidence))
                    candidate_paths.add(path)
        else:
            path = str(parsed_args.get("path") or payload.get("path") or "").strip()
            content = str(payload.get("content", "")).strip()
            if path and any(pattern.search(content) for pattern in patterns):
                confidence = self._location_score(path, file_line_count=len(content.splitlines()))
                evidence_list.append(Evidence(path=path, snippet=content[:240], confidence=confidence))
                candidate_paths.add(path)
        aggregated = self._aggregate_evidence(evidence_list)
        if candidate_paths:
            evidence_lines = [f"`{item.path}` {item.snippet[:120]}" for item in aggregated[:5]]
            self._merge_feature_evidence(
                session_id=session_id,
                feature=feature,
                candidate_paths=sorted(candidate_paths),
                evidence_lines=evidence_lines,
            )
        return aggregated

    async def _run_bootstrap_tool(
        self,
        *,
        tool_name: str,
        path: str | None = None,
        pattern: str | None = None,
        max_entries: int = 120,
        max_results: int = 80,
    ) -> dict[str, Any] | None:
        await self._emit(
            {
                "type": "tool_call",
                "name": tool_name,
                "session_stage": "discovery",
                "path": path,
                "query": pattern,
            }
        )
        started = asyncio.get_running_loop().time()
        try:
            if tool_name == "list_files":
                result = self.tool_executor.list_files(path=path, max_entries=max_entries)
            elif tool_name == "read_file":
                if not path:
                    return None
                result = self.tool_executor.read_file(path=path, max_lines=220)
            elif tool_name == "grep_code":
                if not pattern:
                    return None
                result = self.tool_executor.grep_code(
                    pattern=pattern,
                    path=path,
                    max_results=max_results,
                )
            else:
                return None
            return result
        except Exception:
            return None
        finally:
            elapsed = (asyncio.get_running_loop().time() - started) * 1000.0
            await self._emit(
                {
                    "type": "tool_result",
                    "name": tool_name,
                    "session_stage": "discovery",
                    "duration_ms": round(elapsed, 2),
                }
            )

    @staticmethod
    def _extract_feature_evidence_from_content(
        path: str,
        content: str,
        feature: FeatureIntent,
        limit: int = 3,
    ) -> list[str]:
        evidence: list[str] = []
        patterns = feature.compiled_patterns
        for idx, raw_line in enumerate((content or "").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if any(pattern.search(line) for pattern in patterns):
                evidence.append(f"`{path}:{idx}` {line[:120]}")
                if len(evidence) >= limit:
                    break
        return evidence

    async def _verify_feature_in_candidate_files(
        self,
        *,
        session_id: str,
        feature: FeatureIntent,
        candidate_paths: list[str],
    ) -> list[Evidence]:
        patterns = feature.compiled_patterns
        evidence_list: list[Evidence] = []
        for path in candidate_paths:
            normalized = str(path or "").strip()
            if not normalized or self._route_file_score(normalized) <= 0:
                continue
            payload = await self._run_bootstrap_tool(
                tool_name="grep_code",
                path=normalized,
                pattern="|".join(feature.patterns),
                max_results=12,
            )
            if not payload or not isinstance(payload, dict):
                continue
            results = payload.get("results", [])
            if not isinstance(results, list):
                continue
            for item in results[:12]:
                if not isinstance(item, dict):
                    continue
                item_path = str(item.get("path", normalized)).strip()
                line_num = int(item.get("line", 0) or 0)
                text = str(item.get("text", "")).strip()
                if not item_path:
                    continue
                matched = any(pattern.search(text) for pattern in patterns)
                snippet = text
                if not matched and snippet.count("\n") < 2:
                    expanded = await self._maybe_read_more_context(item_path, line_num)
                    if expanded and any(pattern.search(expanded) for pattern in patterns):
                        matched = True
                        snippet = expanded
                if not matched:
                    continue
                evidence_list.append(
                    Evidence(
                        path=item_path,
                        snippet=snippet[:240],
                        confidence=self._location_score(item_path),
                    )
                )
        aggregated = self._aggregate_evidence(evidence_list)
        if aggregated:
            self._merge_feature_evidence(
                session_id=session_id,
                feature=feature,
                candidate_paths=[item.path for item in aggregated],
                evidence_lines=[f"`{item.path}` {item.snippet[:120]}" for item in aggregated[:5]],
            )
        return aggregated

    async def _build_first_turn_bootstrap_context(
        self,
        session_id: str,
        conversation: list[dict[str, Any]],
        turn_count: int,
    ) -> tuple[str, str | None]:
        if not hasattr(self, "bootstrap_insights_by_session"):
            self.bootstrap_insights_by_session = {}
        if turn_count != 1:
            return "", None
        if not hasattr(self, "tool_executor"):
            return "", None
        user_message = self._latest_user_message(conversation)
        if not self._should_bootstrap_scan(user_message):
            return "", None
        feature = self._extract_feature_intent(user_message)
        previous_insights = self.bootstrap_insights_by_session.get(session_id, {})
        prev_label = str(previous_insights.get("feature_label", "")).strip()
        if prev_label and feature and prev_label != feature.label:
            self.bootstrap_insights_by_session[session_id] = {}
        else:
            self.bootstrap_insights_by_session.pop(session_id, None)

        context_lines: list[str] = [
            "## Bootstrap Scan Evidence (automatic first-turn preflight)",
            "Use this before asking implementation clarification questions.",
        ]
        inferred_framework: str | None = None

        root_listing = await self._run_bootstrap_tool(tool_name="list_files", path=".", max_entries=120)
        if root_listing and isinstance(root_listing.get("entries"), list):
            top_level = [str(item.get("path", "")) for item in root_listing["entries"][:20]]
            visible = ", ".join(path for path in top_level if path) or "(none)"
            context_lines.append(f"Top-level entries: {visible}")
        root_entries = root_listing.get("entries", []) if isinstance(root_listing, dict) else []
        candidate_dirs = self._select_scan_directories(root_entries if isinstance(root_entries, list) else [])

        for candidate in candidate_dirs:
            listed = await self._run_bootstrap_tool(tool_name="list_files", path=candidate, max_entries=80)
            if listed and isinstance(listed.get("entries"), list):
                sample = [str(item.get("path", "")) for item in listed["entries"][:10]]
                if sample:
                    context_lines.append(f"- `{candidate}` sample: {', '.join(sample)}")

        bootstrap_patterns = [BOOTSTRAP_ROUTE_REGEX.pattern]
        if feature is not None:
            bootstrap_patterns.extend(feature.patterns)
        endpoint_matches = await self._run_bootstrap_tool(
            tool_name="grep_code",
            pattern="|".join(pattern for pattern in bootstrap_patterns if pattern),
            max_results=120,
        )
        if endpoint_matches and isinstance(endpoint_matches.get("results"), list):
            matches = [item for item in endpoint_matches["results"] if isinstance(item, dict)]
            signal_index = self._build_signal_index(matches)
            inferred_framework = self._detect_framework(signal_index)
            signal_scores = self._detect_code_signals(signal_index)
            architecture = self._detect_architecture(signal_scores)
            evidence_list: list[Evidence] = []
            if feature is not None:
                feature_patterns = feature.compiled_patterns
                for item in matches:
                    path = str(item.get("path", "")).strip()
                    line = int(item.get("line", 0) or 0)
                    text = str(item.get("text", "")).strip()
                    if not path or not any(pattern.search(text) for pattern in feature_patterns):
                        continue
                    evidence_list.append(
                        Evidence(
                            path=path,
                            snippet=text[:240],
                            confidence=self._location_score(path, file_line_count=line if line > 0 else None),
                        )
                    )
            evidence_list = self._aggregate_evidence(evidence_list)
            if feature is not None and not evidence_list:
                candidate_paths = sorted(
                    {str(item.get("path", "")).strip() for item in matches if str(item.get("path", "")).strip()}
                )
                evidence_list = await self._verify_feature_in_candidate_files(
                    session_id=session_id,
                    feature=feature,
                    candidate_paths=candidate_paths[:12],
                )
            if feature is not None:
                self._merge_feature_evidence(
                    session_id=session_id,
                    feature=feature,
                    candidate_paths=[item.path for item in evidence_list],
                    evidence_lines=[f"`{item.path}` {item.snippet[:120]}" for item in evidence_list[:6]],
                )
                insights = self.bootstrap_insights_by_session.setdefault(session_id, {})
                insights["architecture"] = architecture
                insights["signal_scores"] = signal_scores
                insights["best_path"] = evidence_list[0].path if evidence_list else None
                logger.debug(
                    "discovery feature=%s framework=%s arch=%s signals=%s best=%s paths=%s",
                    feature.label,
                    inferred_framework,
                    architecture,
                    signal_scores,
                    evidence_list[0].path if evidence_list else None,
                    [(item.path, item.confidence) for item in evidence_list],
                )
            context_lines.append("Endpoint/API-related code matches:")
            context_lines.append(self._format_grep_matches(matches, limit=16))
            if inferred_framework:
                context_lines.append(
                    f"Inferred backend framework from code scan: {inferred_framework}. Do not ask which backend framework is used."
                )
            if architecture:
                context_lines.append(f"Inferred repository architecture: {architecture}")
            if signal_scores:
                context_lines.append(f"Detected code signals: {signal_scores}")

        if len(context_lines) <= 2:
            return "", inferred_framework
        return "\n".join(context_lines)[:3500], inferred_framework

    async def _llm_call_with_tools(
        self,
        messages: list[dict[str, Any]],
        max_tool_rounds: int = 6,
        model_override: str | None = None,
    ) -> str:
        """LLM call loop that executes tool calls until LLM produces a text response."""
        import litellm

        litellm.drop_params = True
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False
        session_id = str(getattr(self, "_active_discovery_session_id", "default") or "default")
        conversation = list(messages)
        active_feature = self._extract_feature_intent(self._latest_user_message(messages))
        announced_scanning = False

        for _ in range(max_tool_rounds):
            response = await self._safe_completion_call(
                litellm=litellm,
                messages=self._normalize_roles(conversation),
                tools=CODEBASE_TOOLS,
                max_tokens=1800,
                model_override=model_override,
            )
            message = response.choices[0].message
            content = str(getattr(message, "content", None) or "").strip()
            tool_calls = getattr(message, "tool_calls", None) or []

            if tool_calls:
                if not announced_scanning:
                    announced_scanning = True
                    await self._emit(
                        {
                            "type": "thinking",
                            "message": "Scanning codebase and refining questions...",
                        }
                    )
                # Execute all tool calls and append results
                conversation.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": getattr(tc, "id", None),
                                "type": "function",
                                "function": {
                                    "name": getattr(tc.function, "name", ""),
                                    "arguments": getattr(tc.function, "arguments", "{}"),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    tool_name = getattr(getattr(tc, "function", None), "name", "")
                    raw_args = getattr(getattr(tc, "function", None), "arguments", "{}") or "{}"
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else {}
                    except json.JSONDecodeError:
                        parsed_args = {}
                    path_hint = parsed_args.get("path") or parsed_args.get("relative_path")
                    query_hint = parsed_args.get("query") or parsed_args.get("pattern")
                    await self._emit(
                        {
                            "type": "tool_call",
                            "name": tool_name,
                            "session_stage": "discovery",
                            "path": str(path_hint) if isinstance(path_hint, str) else None,
                            "query": str(query_hint) if isinstance(query_hint, str) else None,
                        }
                    )
                    try:
                        tool_started = asyncio.get_running_loop().time()
                        result = await asyncio.to_thread(self.tool_executor.execute, tc)
                        tool_elapsed_ms = (asyncio.get_running_loop().time() - tool_started) * 1000.0
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "tool_call_id": getattr(tc, "id", ""),
                            "name": "",
                            "result": {"error": str(exc)},
                        }
                        tool_elapsed_ms = 0.0
                    await self._emit(
                        {
                            "type": "tool_result",
                            "name": tool_name,
                            "session_stage": "discovery",
                            "duration_ms": round(tool_elapsed_ms, 2),
                        }
                    )
                    await self._ingest_feature_evidence_from_tool(
                        session_id=session_id,
                        feature=active_feature,
                        tool_name=tool_name,
                        parsed_args=parsed_args,
                        tool_result_payload=result["result"] if isinstance(result, dict) else {},
                    )
                    raw_content = json.dumps(result["result"])
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "name": result["name"],
                            "content": raw_content,
                        }
                    )
                continue

            return content

        return "I've scanned the codebase. What aspect of this would you like to focus on first?"

    async def _llm_call(self, messages: list[dict[str, Any]], model_override: str | None = None) -> str:
        """Simple LLM call without tools (used for force_summary)."""
        import litellm

        litellm.drop_params = True
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False
        response = await self._safe_completion_call(
            litellm=litellm,
            messages=self._normalize_roles(messages),
            max_tokens=900,
            model_override=model_override,
        )
        return str(response.choices[0].message.content or "").strip()

    async def _safe_completion_call(
        self,
        *,
        litellm: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1200,
        model_override: str | None = None,
    ) -> Any:
        """Call chat completion with graceful fallback for non-chat models."""
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        primary_model = model_override or self.config.author_model
        fallback_model = "gpt-4o-mini"
        models_to_try = [primary_model]
        if fallback_model != primary_model:
            models_to_try.append(fallback_model)

        last_error: Exception | None = None
        for idx, model in enumerate(models_to_try):
            try:
                llm_started = asyncio.get_running_loop().time()
                response = await asyncio.to_thread(
                    litellm.completion,
                    model=model,
                    **kwargs,
                )
                llm_elapsed_ms = (asyncio.get_running_loop().time() - llm_started) * 1000.0
                telemetry = completion_telemetry(response, model=model)
                context_window = MODEL_CONTEXT_WINDOWS.get(model)
                await self._emit(
                    {
                        "type": "token_usage",
                        "session_stage": "discovery",
                        "model": model,
                        "prompt_tokens": telemetry.usage.prompt_tokens,
                        "completion_tokens": telemetry.usage.completion_tokens,
                        "call_cost_usd": telemetry.cost.total_cost_usd,
                        "llm_call_latency_ms": round(llm_elapsed_ms, 2),
                        "context_window_tokens": context_window,
                        "context_usage_ratio": (
                            round(float(telemetry.usage.prompt_tokens) / float(context_window), 4)
                            if context_window and context_window > 0
                            else None
                        ),
                    }
                )
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                err_text = str(exc).lower()
                non_chat_model = (
                    "not a chat model" in err_text
                    or "v1/chat/completions" in err_text
                    or "did you mean to use v1/completions" in err_text
                )
                # Retry only for known model-compatibility issues.
                if not non_chat_model or idx == len(models_to_try) - 1:
                    break

        if last_error is not None:
            raise RuntimeError(
                "Configured planning model is incompatible with chat completions. "
                "Update planning.author_model or use a chat-capable model."
            ) from last_error
        raise RuntimeError("Unknown completion failure during discovery.")

    def opening_prompt(self) -> str:
        return (
            "Tell me what you want to plan. "
            "I'll scan the codebase first, then ask only about decisions and constraints "
            "that aren't already answered by the code."
        )

    @staticmethod
    def _strip_json_comments(raw: str) -> str:
        # Best-effort cleanup for model outputs that include JS-style comments.
        no_block = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
        no_line = re.sub(r"//[^\n\r]*", "", no_block)
        return no_line.strip()

    @classmethod
    def _parse_discovery_complete_candidate(cls, candidate: str) -> dict[str, Any] | None:
        for attempt in (candidate, cls._strip_json_comments(candidate)):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and str(parsed.get("discovery", "")).strip().lower() == "complete":
                return parsed
        return None

    @staticmethod
    def _extract_json_code_blocks(text: str) -> list[str]:
        blocks = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        return [block.strip() for block in blocks if block.strip()]

    @staticmethod
    def _extract_balanced_json_objects(text: str) -> list[str]:
        candidates: list[str] = []
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            end = -1
            for idx in range(start, len(text)):
                ch = text[idx]
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx
                        break
            if end != -1:
                candidate = text[start : end + 1].strip()
                if candidate:
                    candidates.append(candidate)
                start = text.find("{", end + 1)
            else:
                break
        return candidates

    def _extract_completion_payload(self, text: str) -> tuple[dict[str, Any], str] | None:
        candidates = self._extract_json_code_blocks(text)
        candidates.extend(self._extract_balanced_json_objects(text))
        for candidate in candidates:
            parsed = self._parse_discovery_complete_candidate(candidate)
            if parsed is not None:
                return parsed, candidate
        return None

    def _try_extract_completion(self, text: str) -> DiscoveryTurnResult:
        payload = self._extract_completion_payload(text)
        if payload is None:
            questions = parse_questions(text)
            return DiscoveryTurnResult(reply=text, complete=False, summary=None, questions=questions)
        parsed, payload_text = payload

        summary_raw = parsed.get("summary", "")
        if isinstance(summary_raw, str):
            summary = summary_raw.strip()
        elif summary_raw:
            summary = json.dumps(summary_raw, ensure_ascii=False)
        else:
            summary = ""
        # Remove machine-readable completion payload from chat-visible prose.
        prose = re.sub(r"```json[\s\S]*?```", "", text, flags=re.IGNORECASE)
        prose = prose.replace(payload_text, "", 1).strip()
        return DiscoveryTurnResult(
            reply=prose or "Discovery complete — drafting plan now.",
            complete=True,
            summary=summary or None,
        )

    async def handle_turn(
        self,
        conversation: list[dict[str, Any]],
        session_id: str = "default",
        model_override: str | None = None,
        extra_context: str = "",
    ) -> DiscoveryTurnResult:
        turn_count = self._next_turn_count(session_id)

        if turn_count >= self.config.discovery_max_turns:
            summary = await self.force_summary(conversation)
            return DiscoveryTurnResult(
                reply=("I have enough context to start drafting your plan now."),
                complete=True,
                summary=summary,
            )

        try:
            import litellm  # noqa: F401
        except ImportError:
            return DiscoveryTurnResult(
                reply="LLM unavailable; proceeding with provided requirements context.",
                complete=True,
                summary="\n".join(m.get("content", "") for m in conversation if m.get("role") == "user"),
            )

        latest_user_message = self._latest_user_message(conversation)
        prior_insights = getattr(self, "bootstrap_insights_by_session", {}).get(session_id, {})
        feature_label = str(prior_insights.get("feature_label", "feature")).strip() or "feature"
        if turn_count > 1 and bool(prior_insights.get("existing_feature")):
            choice = self._parse_existing_endpoint_followup_choice(latest_user_message)
            if choice == "A":
                evidence = "\n".join(f"- {line}" for line in self._existing_feature_evidence_lines(session_id))
                deep_summary, functional_summary = await self._build_existing_endpoint_deep_summary(session_id)
                details_block = f"What it currently does:\n{deep_summary}\n\n" if deep_summary else ""
                return DiscoveryTurnResult(
                    reply=(
                        "Summary of what already exists:\n"
                        f"- Functional overview: {functional_summary or 'This implementation is already present in the repository.'}\n"
                        f"{evidence}\n\n"
                        f"{details_block}"
                        f"If you want, I can propose targeted enhancements next for the existing {feature_label}, "
                        "but I won't create a duplicate implementation."
                    ),
                    complete=False,
                    summary=None,
                    questions=[],
                )
            if choice == "B":
                return DiscoveryTurnResult(
                    reply=(
                        f"Great — I can focus on enhancing the existing {feature_label}.\n"
                        "Tell me which area to prioritize first."
                    ),
                    complete=False,
                    summary=None,
                    questions=[],
                )
            if choice == "C":
                return DiscoveryTurnResult(
                    reply=(
                        f"Understood — we'll leave the existing {feature_label} unchanged. "
                        "No planning draft will be generated."
                    ),
                    complete=False,
                    summary=None,
                    questions=[],
                )

        memory_context = self._build_memory_context()
        system_content = DISCOVERY_SYSTEM_PROMPT
        if memory_context:
            system_content += f"\n\n## Pre-built Codebase Memory\n{memory_context}"
        try:
            bootstrap_context, inferred_framework = await self._build_first_turn_bootstrap_context(
                session_id,
                conversation,
                turn_count,
            )
        except TypeError:
            # Backwards-compatible fallback for tests/mocks that still provide the
            # older method signature (_conversation, _turn_count).
            bootstrap_context, inferred_framework = await self._build_first_turn_bootstrap_context(  # type: ignore[misc]
                conversation,
                turn_count,
            )
        if bootstrap_context:
            system_content += f"\n\n{bootstrap_context}"
        if extra_context:
            system_content += f"\n\n{extra_context}"

        messages = [{"role": "system", "content": system_content}, *conversation]
        await self._emit({"type": "thinking", "message": "Refining questions from available context..."})
        self._active_discovery_session_id = session_id
        try:
            response = await self._llm_call_with_tools(
                messages,
                max_tool_rounds=self.config.discovery_tool_rounds,
                model_override=model_override,
            )
        finally:
            self._active_discovery_session_id = None
        parsed = self._drop_redundant_framework_questions(
            self._try_extract_completion(response),
            inferred_framework,
        )
        insights = getattr(self, "bootstrap_insights_by_session", {}).get(session_id, {})
        existing_feature = bool(insights.get("existing_feature"))
        current_intent = self._extract_feature_intent(latest_user_message)
        current_feature_label = str(insights.get("feature_label", "feature")).strip() or "feature"
        if turn_count == 1 and current_intent is not None and existing_feature:
            evidence_lines = [str(line).strip() for line in insights.get("matched_evidence", []) if str(line).strip()]
            evidence_block = (
                "\n".join(f"- {line}" for line in evidence_lines[:3])
                or "- Found existing implementation in the codebase"
            )
            return DiscoveryTurnResult(
                reply=(
                    f"The {current_feature_label} already appears to exist in the codebase, so I won't draft a new creation plan.\n\n"
                    f"Evidence:\n{evidence_block}\n\n"
                    f"Do you want me to review and propose enhancements to the existing {current_feature_label} instead?"
                ),
                complete=False,
                summary=None,
                questions=[
                    DiscoveryQuestion(
                        index=1,
                        text=f"What should we do with the existing {current_feature_label}?",
                        options=[
                            QuestionOption("A", "Review current behavior and summarize it only."),
                            QuestionOption(
                                "B", "Propose targeted enhancements without creating a duplicate implementation."
                            ),
                            QuestionOption("C", "Leave it unchanged; no planning needed."),
                            QuestionOption("D", "Other — describe your preference", is_other=True),
                        ],
                    )
                ],
            )
        # Guardrail: discovery should ask a batched question set (2-3), not one-by-one.
        if not parsed.complete and len(parsed.questions) == 1:
            expanded = await self._expand_question_batch(
                original_response=response,
                model_override=model_override,
            )
            expanded_parsed = self._drop_redundant_framework_questions(
                self._try_extract_completion(expanded),
                inferred_framework,
            )
            if not expanded_parsed.complete and len(expanded_parsed.questions) >= 2:
                return expanded_parsed
            # Retry once more to reduce one-by-one drift from weaker model replies.
            expanded_retry = await self._expand_question_batch(
                original_response=expanded,
                model_override=model_override,
            )
            expanded_retry_parsed = self._drop_redundant_framework_questions(
                self._try_extract_completion(expanded_retry),
                inferred_framework,
            )
            if not expanded_retry_parsed.complete and len(expanded_retry_parsed.questions) >= 2:
                return expanded_retry_parsed
        # Guardrail: if the model returned prose (no questions, not complete) it
        # likely decided discovery is done but forgot to emit the completion JSON.
        # Ask it to either emit the JSON payload or provide more questions.
        if not parsed.complete and len(parsed.questions) == 0:
            reformatted = await self._force_completion_or_questions(
                original_response=response,
                model_override=model_override,
            )
            reformatted_parsed = self._drop_redundant_framework_questions(
                self._try_extract_completion(reformatted),
                inferred_framework,
            )
            if reformatted_parsed.complete or len(reformatted_parsed.questions) > 0:
                return reformatted_parsed
            # If still empty after reformat, treat as implicit completion so the
            # session can proceed rather than hanging with no questions and no draft.
            summary = parsed.reply[:500] if parsed.reply else "Requirements gathered from conversation."
            return DiscoveryTurnResult(
                reply=parsed.reply or "Discovery complete — drafting plan now.",
                complete=True,
                summary=summary,
                questions=[],
            )
        return parsed

    async def _force_completion_or_questions(self, original_response: str, model_override: str | None = None) -> str:
        """Ask the model to reformat a prose-only response into either the completion JSON or questions."""
        prompt = (
            "Your previous response contained neither discovery questions nor the required completion JSON.\n"
            "You must do one of the following:\n\n"
            "Option A — If you have all the information needed, emit ONLY this JSON (no prose):\n"
            '{"discovery":"complete","summary":"<comprehensive requirements summary>"}\n\n'
            "Option B — If you need more information, emit 2-3 discovery questions in this exact format:\n"
            "Q1: ...\nA) ...\nB) ...\nC) ...\nD) Other — describe your preference\n\n"
            "Do not include any other text. Choose exactly one option.\n\n"
            f"Your previous response was:\n{original_response}"
        )
        return await self._llm_call(
            [
                {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model_override=model_override,
        )

    async def _expand_question_batch(self, original_response: str, model_override: str | None = None) -> str:
        """Rewrite a singleton question into a full 2-3 question discovery batch."""
        prompt = (
            "Rewrite the discovery questions as a single batched response with 2-3 questions.\n"
            "Return only question blocks in this exact format:\n"
            "Q1: ...\nA) ...\nB) ...\nC) ...\nD) Other — describe your preference\n\n"
            "Do not include any explanatory prose before or after questions.\n"
            "Do not ask one question at a time.\n\n"
            f"Original response:\n{original_response}"
        )
        return await self._llm_call(
            [
                {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model_override=model_override,
        )

    async def force_summary(
        self,
        conversation: list[dict[str, Any]],
        model_override: str | None = None,
    ) -> str:
        try:
            import litellm  # noqa: F401
        except ImportError:
            return "\n".join(m.get("content", "") for m in conversation if m.get("role") == "user")[:2000]

        messages = [
            {
                "role": "system",
                "content": "Summarize the discovered planning requirements in concise markdown.",
            },
            *conversation,
        ]
        return await self._llm_call(messages, model_override=model_override)
