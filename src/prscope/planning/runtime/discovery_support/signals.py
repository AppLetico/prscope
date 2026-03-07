from __future__ import annotations

import re
from typing import Any

from ..reasoning import FrameworkSignals
from .models import (
    CodeSignal,
    DiscoveryTurnResult,
    FeatureIntent,
    Framework,
    IndexedMatch,
    SignalIndex,
)

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

INTENT_STOP_WORDS = {"a", "an", "the", "new", "feature", "endpoint", "support", "system", "api"}
VENDOR_DIRS = {"node_modules", "venv", ".env", "dist", "build", "__pycache__", ".git", ".tox", "egg-info"}
BACKEND_DIR_NAMES = {"backend", "api", "server", "src", "services", "web", "app", "lib"}
LOW_SIGNAL_DIRS = {"benchmarks", "coverage", "docs", "examples", "fixtures", "plans", "public", "static"}
LOW_SIGNAL_FILE_NAMES = {"architecture.md", "changelog.md", "contributing.md", "readme", "readme.md"}
LOW_SIGNAL_EXTENSIONS = {".cfg", ".ini", ".json", ".jsonl", ".md", ".mdx", ".rst", ".toml", ".txt", ".yaml", ".yml"}
CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".ts",
    ".tsx",
}
CODE_NOUNS = {
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
MAX_TRUSTWORTHY_LINES = 5000


def _path_parts(path: str) -> list[str]:
    return [part for part in str(path or "").replace("\\", "/").lower().split("/") if part]


def is_low_signal_path(path: str) -> bool:
    parts = _path_parts(path)
    if not parts:
        return False
    if any(part in LOW_SIGNAL_DIRS for part in parts[:-1]):
        return True
    filename = parts[-1]
    if filename in LOW_SIGNAL_FILE_NAMES:
        return True
    return len(parts) == 1 and any(filename.endswith(ext) for ext in LOW_SIGNAL_EXTENSIONS)


def is_code_like_path(path: str) -> bool:
    parts = _path_parts(path)
    if not parts:
        return False
    filename = parts[-1]
    return any(filename.endswith(ext) for ext in CODE_EXTENSIONS)


def is_trustworthy_existing_feature_path(path: str) -> bool:
    lowered = str(path or "").replace("\\", "/").lower()
    if not lowered:
        return False
    if "test" in lowered or is_low_signal_path(lowered):
        return False
    if lowered.endswith((".tsx", ".jsx")):
        return False
    if route_file_score(lowered) > 0 or location_score(lowered) > 0:
        return True
    return False


def pattern_for_word(word: str) -> str:
    return rf"\b{re.escape(word)}\b"


def extract_feature_intent(user_message: str) -> FeatureIntent | None:
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
        if not normalized or normalized in INTENT_STOP_WORDS:
            continue
        keywords.append(normalized)
        if len(keywords) >= MAX_INTENT_KEYWORDS:
            break
    if not keywords:
        return None
    patterns: list[str] = []
    for keyword in keywords:
        patterns.append(pattern_for_word(keyword))
        patterns.append(pattern_for_word(keyword.replace("-", "_")))
        patterns.append(pattern_for_word(keyword.replace("_", "-")))
    return FeatureIntent(label=" ".join(keywords), keywords=keywords, patterns=list(dict.fromkeys(patterns)))


def should_bootstrap_scan(user_message: str) -> bool:
    text = str(user_message or "").strip().lower()
    if not text:
        return False
    if re.search(r"\b(endpoint|route|routing|api|http|webhook|handler)\b", text):
        return True
    intent = extract_feature_intent(text)
    if intent is None:
        return False
    return any(noun in text for noun in CODE_NOUNS)


def parse_existing_endpoint_followup_choice(user_message: str) -> str | None:
    normalized = " ".join((user_message or "").strip().split()).lower()
    if not normalized:
        return None
    normalized = re.sub(r"^\d+\.\s*", "", normalized)
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


def parse_enhancement_proposal_followup_choice(user_message: str) -> str | None:
    normalized = " ".join((user_message or "").strip().split()).lower()
    if not normalized:
        return None
    normalized = re.sub(r"^\d+\.\s*", "", normalized)
    if re.fullmatch(r"(q1[:\-\s]*)?(a|1)", normalized):
        return "A"
    if re.fullmatch(r"(q1[:\-\s]*)?(b|2)", normalized):
        return "B"
    if re.fullmatch(r"(q1[:\-\s]*)?(c|3)", normalized):
        return "C"
    if (
        "proceed with this proposal" in normalized
        or "proceed and draft the plan from this proposal" in normalized
        or "proceed to draft" in normalized
        or "go ahead and draft" in normalized
        or "draft it now" in normalized
    ):
        return "A"
    if (
        "revise the proposal" in normalized
        or "revise the proposal first" in normalized
        or "edit the proposal" in normalized
        or "change the proposal" in normalized
    ):
        return "B"
    if "cancel" in normalized or "leave it unchanged" in normalized:
        return "C"
    return None


def is_concrete_enhancement_request(user_message: str) -> bool:
    normalized = " ".join((user_message or "").strip().split()).lower()
    if not normalized:
        return False
    if normalized.endswith("?"):
        return False
    if normalized in {"yes", "yeah", "yep", "ok", "okay", "sure", "sounds good"}:
        return False
    if parse_existing_endpoint_followup_choice(normalized) is not None:
        return False
    generic_phrases = {
        "tell me which area to prioritize first",
        "propose targeted enhancements",
        "review current behavior",
    }
    if normalized in generic_phrases:
        return False
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    meaningful = [token for token in tokens if len(token) >= 4 and token not in INTENT_STOP_WORDS]
    return len(meaningful) >= 1


def route_file_score(path: str) -> int:
    normalized_path = str(path or "").replace("\\", "/")
    path_lower = normalized_path.lower()
    filename = path_lower.rsplit("/", 1)[-1]
    score = 0
    if is_low_signal_path(path_lower):
        score -= 6
    if any(token in path_lower for token in ("/api/", "/routes/", "/controllers/", "/handlers/")):
        score += 4
    if re.search(r"(routes?|router|controller|handler)\.", filename):
        score += 6
    if re.search(r"(^|[_\-])api\.", filename):
        score += 3
    if re.search(r"(server|app|main|urls|views)\.", filename):
        score += 2
    if "test" in path_lower:
        score -= 3
    return score


def location_score(path: str, file_line_count: int | None = None) -> int:
    score = 0
    lowered = str(path or "").replace("\\", "/").lower()
    if is_low_signal_path(lowered):
        score -= 5
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
    if file_line_count and file_line_count > MAX_TRUSTWORTHY_LINES:
        score -= 5
    return score


def detect_framework(index: SignalIndex) -> str | None:
    return build_framework_signals(index).inferred_framework


def build_framework_signals(index: SignalIndex) -> FrameworkSignals:
    candidates = index.get("framework", [])
    if not candidates:
        return FrameworkSignals(candidates={}, inferred_framework=None, evidence=[])
    best_name: str | None = None
    best_score = 0
    scores: dict[str, int] = {}
    evidence: list[str] = []
    for framework in FRAMEWORKS:
        score = 0
        for match in candidates:
            path_score = location_score(match.path, file_line_count=match.line_number)
            if path_score < 0:
                continue
            filename = str(match.path or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
            file_bonus = 2 if filename in {name.lower() for name in framework.file_patterns} else 0
            match_weight = 1 + max(path_score, 0) + file_bonus
            if any(pattern.search(match.line) for pattern in framework.route_patterns):
                score += match_weight
                if len(evidence) < 6:
                    evidence.append(f"{framework.name}:{match.path}:{match.line_number}")
        scores[framework.name] = score
        if score > best_score:
            best_name = framework.name
            best_score = score
    return FrameworkSignals(
        candidates={name: score for name, score in scores.items() if score > 0},
        inferred_framework=best_name if best_score > 0 else None,
        evidence=evidence,
    )


def is_framework_identification_question(text: str) -> bool:
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


def drop_redundant_framework_questions(
    parsed: DiscoveryTurnResult,
    inferred_framework: str | None,
) -> DiscoveryTurnResult:
    if parsed.complete or not parsed.questions or not inferred_framework:
        return parsed
    filtered_questions = [
        question for question in parsed.questions if not is_framework_identification_question(question.text)
    ]
    if len(filtered_questions) == len(parsed.questions):
        return parsed
    return DiscoveryTurnResult(
        reply=parsed.reply,
        complete=False,
        summary=parsed.summary,
        questions=filtered_questions,
    )


def detect_code_signals(index: SignalIndex) -> dict[str, int]:
    output: dict[str, int] = {}
    for signal in CODE_SIGNALS:
        count = len(index.get(signal.name, []))
        if count > 0:
            output[signal.name] = count
    return output


def detect_architecture(signal_scores: dict[str, int]) -> str | None:
    arch_scores = {
        "api_service": signal_scores.get("route", 0) * 3 + signal_scores.get("middleware", 0) * 2,
        "worker_service": signal_scores.get("worker", 0) * 3 + signal_scores.get("cron", 0) * 2,
        "cli_tool": signal_scores.get("cli", 0) * 3,
    }
    best = max(arch_scores, key=arch_scores.get)
    return best if arch_scores[best] > 0 else None


def build_signal_index(matches: list[dict[str, Any]]) -> SignalIndex:
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
        if lines_per_path[path] > MAX_TRUSTWORTHY_LINES:
            continue
        indexed = IndexedMatch(path=path, line_number=line_number, line=line)
        for signal_name, pattern in all_patterns:
            if pattern.search(line):
                index[signal_name].append(indexed)
    return index


def aggregate_evidence(evidence_list: list[Any]) -> list[Any]:
    if not evidence_list:
        return []
    by_path: dict[str, Any] = {}
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


def select_scan_directories(entries: list[dict[str, Any]]) -> list[str]:
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
        if name.startswith(".") or name_lower in VENDOR_DIRS or depth > 2:
            continue
        kind = str(entry.get("type", "")).lower()
        if kind == "dir":
            scores.setdefault(path, 0)
            if name_lower in LOW_SIGNAL_DIRS:
                scores[path] -= 4
            if name_lower in BACKEND_DIR_NAMES:
                scores[path] += 4
            elif name_lower not in {"tests", "test"}:
                scores[path] += 1
            continue
        parent = path.rsplit("/", 1)[0] if "/" in path else "."
        parent_lower = parent.rsplit("/", 1)[-1].lower()
        if parent_lower in VENDOR_DIRS or parent.startswith(".") or parent.count("/") > 2:
            continue
        scores.setdefault(parent, 0)
        if parent_lower in LOW_SIGNAL_DIRS:
            scores[parent] -= 4
        if re.search(r"\.(py|ts|tsx|js|jsx|go|rb|java|kt|cs)$", path, flags=re.IGNORECASE):
            scores[parent] += 3
        if route_file_score(path) > 0:
            scores[parent] += 5
        if parent_lower in BACKEND_DIR_NAMES:
            scores[parent] += 4
    if not scores:
        return ["."]
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    selected = [path for path, score in ranked if score > 0]
    if selected:
        return selected[:5]
    return [path for path, _ in ranked[:3]]


def format_grep_matches(matches: list[dict[str, Any]], limit: int = 14) -> str:
    if not matches:
        return "- (no matches)"
    preferred = sorted(
        matches,
        key=lambda item: (
            -route_file_score(str(item.get("path", ""))),
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
