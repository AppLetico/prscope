from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from typing import Any


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

    q_re = re.compile(
        r"^(?:Q\s*\d*|Question\s+\d+|\d+)\s*[:\.\)\-]\s*(.+)$",
        re.IGNORECASE,
    )
    opt_re = re.compile(r"^(?:[-*]\s*)?([A-D])[\)\.\-:]\s*(.+)$", re.IGNORECASE)

    def normalize_line(line: str) -> str:
        normalized = line.strip()
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
                    break
                elif stripped and not opt_match:
                    if options:
                        break
                    i += 1
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
    line: int = 0


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


def strip_json_comments(raw: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    no_line = re.sub(r"//[^\n\r]*", "", no_block)
    return no_line.strip()


def parse_discovery_complete_candidate(candidate: str) -> dict[str, Any] | None:
    for attempt in (candidate, strip_json_comments(candidate)):
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and str(parsed.get("discovery", "")).strip().lower() == "complete":
            return parsed
    return None


def extract_json_code_blocks(text: str) -> list[str]:
    blocks = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    return [block.strip() for block in blocks if block.strip()]


def extract_balanced_json_objects(text: str) -> list[str]:
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


def extract_completion_payload(text: str) -> tuple[dict[str, Any], str] | None:
    candidates = extract_json_code_blocks(text)
    candidates.extend(extract_balanced_json_objects(text))
    for candidate in candidates:
        parsed = parse_discovery_complete_candidate(candidate)
        if parsed is not None:
            return parsed, candidate
    return None


def try_extract_completion(text: str) -> DiscoveryTurnResult:
    payload = extract_completion_payload(text)
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
    prose = re.sub(r"```json[\s\S]*?```", "", text, flags=re.IGNORECASE)
    prose = prose.replace(payload_text, "", 1).strip()
    return DiscoveryTurnResult(
        reply=prose or "Discovery complete — drafting plan now.",
        complete=True,
        summary=summary or None,
    )
