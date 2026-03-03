"""
Semantic search for Prscope.

Uses LLM embeddings to find semantically similar code between
upstream PRs and the local codebase.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import get_prscope_dir

logger = logging.getLogger(__name__)

# Directories to skip when reading local code
SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".prscope",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "venv",
    ".venv",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    "target",
}

# File extensions to include for code analysis
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".rb",
    ".php",
    ".swift",
    ".c",
    ".cpp",
    ".h",
    ".cs",
    ".vue",
    ".svelte",
    ".md",
}

# Maximum file size to read (100KB)
MAX_FILE_SIZE = 100 * 1024

# Maximum content length for embedding
MAX_EMBED_CHARS = 8000


@dataclass
class CodeChunk:
    """A chunk of code from the local codebase."""

    path: str
    content: str
    start_line: int
    end_line: int

    def summary(self) -> str:
        """Get a summary for embedding."""
        self.content.split("\n")
        return f"File: {self.path}\n{self.content[:MAX_EMBED_CHARS]}"


@dataclass
class SimilarityResult:
    """Result of semantic similarity search."""

    local_path: str
    local_snippet: str
    similarity_score: float
    overlap_type: str  # "exact", "similar", "conceptual"


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float] | None:
    """
    Get embedding for text using LiteLLM.

    Returns None if embedding fails.
    """
    try:
        import litellm

        response = litellm.embedding(
            model=model,
            input=[text[:MAX_EMBED_CHARS]],
        )
        return response.data[0]["embedding"]
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def read_local_files(repo_root: Path) -> list[CodeChunk]:
    """
    Read all relevant code files from the local repository.

    Returns a list of CodeChunks.
    """
    chunks = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue

        # Skip ignored directories
        try:
            rel_path = path.relative_to(repo_root)
            if any(part in SKIP_DIRS for part in rel_path.parts):
                continue
        except ValueError:
            continue

        # Check extension
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue

        # Check file size
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                continue
        except OSError:
            continue

        # Read content
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                chunks.append(
                    CodeChunk(
                        path=str(rel_path),
                        content=content,
                        start_line=1,
                        end_line=len(content.split("\n")),
                    )
                )
        except Exception:
            continue

    return chunks


def extract_keywords_from_pr(title: str, body: str | None) -> list[str]:
    """
    Extract security/concept keywords from PR title and body.

    These keywords help find conceptually related code even when
    file paths don't match.
    """
    import re

    text = f"{title} {body or ''}".lower()

    # Third-party integrations to detect
    integration_keywords = [
        # Messaging platforms
        "telegram",
        "slack",
        "discord",
        "whatsapp",
        "teams",
        "signal",
        # Email services
        "gmail",
        "outlook",
        "sendgrid",
        "mailgun",
        "smtp",
        # Cloud providers
        "aws",
        "gcp",
        "azure",
        "s3",
        "lambda",
        "cloudflare",
        # Databases
        "mongodb",
        "postgresql",
        "mysql",
        "redis",
        "elasticsearch",
        # APIs/SDKs
        "stripe",
        "twilio",
        "github",
        "gitlab",
        "jira",
    ]

    # Security-related patterns
    security_keywords = [
        # Vulnerabilities
        "path traversal",
        "directory traversal",
        "lfi",
        "rfi",
        "injection",
        "xss",
        "csrf",
        "ssrf",
        "sqli",
        "sanitize",
        "validate",
        "escape",
        "encode",
        # Path security
        "isPathSafe",
        "sanitizePath",
        "path.resolve",
        "path.join",
        "allowlist",
        "blocklist",
        "whitelist",
        "blacklist",
        "../",
        "..\\",
        "absolute path",
        "relative path",
        # Auth/security
        "authentication",
        "authorization",
        "permission",
        "rbac",
        "token",
        "jwt",
        "session",
        "credential",
        # General patterns
        "security",
        "vulnerability",
        "exploit",
        "attack",
        "prevent",
        "restrict",
        "limit",
        "guard",
    ]

    found = []

    # Check for integrations first (important for filtering)
    for keyword in integration_keywords:
        if keyword.lower() in text:
            found.append(f"integration:{keyword}")

    for keyword in security_keywords:
        if keyword.lower() in text:
            found.append(keyword)

    # Also extract technical terms from the title
    # Look for function-like patterns: word(, isWord, validateWord, etc.
    technical_patterns = [
        r"\b(is[A-Z]\w+)",  # isPathSafe, isValid, etc.
        r"\b(validate\w*)",  # validate, validatePath, etc.
        r"\b(sanitize\w*)",  # sanitize, sanitizePath, etc.
        r"\b(check\w*)",  # checkPath, checkPermission, etc.
        r"\b(prevent\w*)",  # preventInjection, etc.
        r"\b(restrict\w*)",  # restrictAccess, etc.
    ]

    for pattern in technical_patterns:
        matches = re.findall(pattern, title + " " + (body or ""), re.IGNORECASE)
        found.extend(matches)

    return list(set(found))


def search_code_by_keywords(
    repo_root: Path,
    keywords: list[str],
    max_files: int = 10,
) -> list[CodeChunk]:
    """
    Search local codebase for files containing security/concept keywords.

    This finds conceptually related code even when file paths don't match.
    For example, if a PR fixes "path traversal", this will find local files
    that also deal with path traversal prevention.
    """
    import re

    all_chunks = read_local_files(repo_root)
    scored_chunks: list[tuple[CodeChunk, int]] = []

    for chunk in all_chunks:
        content_lower = chunk.content.lower()
        score = 0

        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Count occurrences
            if keyword_lower in content_lower:
                score += content_lower.count(keyword_lower)

            # Bonus for function/class definitions containing the keyword
            if re.search(rf"\b(function|def|class)\s+\w*{re.escape(keyword_lower)}\w*", content_lower):
                score += 5

            # Bonus for security-focused files
            if any(sec in chunk.path.lower() for sec in ["security", "auth", "validate", "sanitize"]):
                score += 3

        if score > 0:
            scored_chunks.append((chunk, score))

    # Sort by score descending, take top matches
    scored_chunks.sort(key=lambda x: -x[1])

    return [chunk for chunk, _ in scored_chunks[:max_files]]


def extract_matching_files(
    repo_root: Path,
    pr_files: list[str],
    feature_paths: list[str],
    pr_title: str = "",
    pr_body: str | None = None,
) -> list[CodeChunk]:
    """
    Extract local files that match PR patterns via multiple strategies:

    1. Path matching - same filename or path structure
    2. Feature glob matching - matches configured feature paths
    3. Keyword search - finds conceptually related code (NEW)

    This helps catch cases where the same functionality exists
    in different file paths (e.g., security fix in media/parse.ts
    vs security/index.ts).
    """
    import fnmatch
    import re

    all_chunks = read_local_files(repo_root)
    matched_paths = set()
    matched = []

    # Strategy 1 & 2: Path and glob matching
    for chunk in all_chunks:
        # Check if local file matches any PR file pattern
        for pr_file in pr_files:
            # Match by filename or similar path structure
            pr_name = Path(pr_file).name
            local_name = Path(chunk.path).name

            if pr_name == local_name:
                if chunk.path not in matched_paths:
                    matched.append(chunk)
                    matched_paths.add(chunk.path)
                break

            # Match by path suffix (e.g., "auth/handler.py" matches "src/auth/handler.py")
            if chunk.path.endswith(pr_file) or pr_file.endswith(chunk.path):
                if chunk.path not in matched_paths:
                    matched.append(chunk)
                    matched_paths.add(chunk.path)
                break
        else:
            # Check feature path globs
            for pattern in feature_paths:
                pattern = pattern.replace("\\", "/")
                if "**" in pattern:
                    regex_pattern = pattern.replace(".", r"\.")
                    regex_pattern = regex_pattern.replace("**", ".*")
                    regex_pattern = regex_pattern.replace("*", "[^/]*")
                    regex_pattern = f"^{regex_pattern}$"
                    if re.match(regex_pattern, chunk.path):
                        if chunk.path not in matched_paths:
                            matched.append(chunk)
                            matched_paths.add(chunk.path)
                        break
                elif fnmatch.fnmatch(chunk.path, pattern):
                    if chunk.path not in matched_paths:
                        matched.append(chunk)
                        matched_paths.add(chunk.path)
                    break

    # Strategy 3: Keyword-based search (NEW)
    # This finds conceptually related code even when paths don't match
    if pr_title or pr_body:
        keywords = extract_keywords_from_pr(pr_title, pr_body)
        if keywords:
            keyword_matches = search_code_by_keywords(repo_root, keywords, max_files=5)
            for chunk in keyword_matches:
                if chunk.path not in matched_paths:
                    matched.append(chunk)
                    matched_paths.add(chunk.path)

    return matched


def find_similar_implementations(
    pr_description: str,
    pr_files: list[str],
    local_chunks: list[CodeChunk],
    similarity_threshold: float = 0.75,
) -> list[SimilarityResult]:
    """
    Find local code that's semantically similar to PR changes.

    Uses embeddings to detect:
    - Exact matches (same functionality)
    - Similar implementations (related code)
    - Conceptual overlaps (same domain)
    """
    results = []

    # Build PR context for embedding
    pr_context = f"PR Description:\n{pr_description}\n\nFiles changed:\n" + "\n".join(pr_files)

    pr_embedding = get_embedding(pr_context)
    if not pr_embedding:
        return results

    for chunk in local_chunks:
        chunk_embedding = get_embedding(chunk.summary())
        if not chunk_embedding:
            continue

        similarity = cosine_similarity(pr_embedding, chunk_embedding)

        if similarity >= similarity_threshold:
            # Classify overlap type
            if similarity >= 0.9:
                overlap_type = "exact"
            elif similarity >= 0.8:
                overlap_type = "similar"
            else:
                overlap_type = "conceptual"

            results.append(
                SimilarityResult(
                    local_path=chunk.path,
                    local_snippet=chunk.content[:500],
                    similarity_score=similarity,
                    overlap_type=overlap_type,
                )
            )

    # Sort by similarity descending
    results.sort(key=lambda x: -x.similarity_score)

    return results[:10]  # Top 10 matches


class EmbeddingCache:
    """Cache for embeddings to avoid redundant API calls."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or get_prscope_dir() / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> list[float] | None:
        cache_file = self.cache_dir / f"{self._hash_text(text)}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        cache_file = self.cache_dir / f"{self._hash_text(text)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(embedding, f)
        except Exception:
            pass


def get_embedding_cached(text: str, cache: EmbeddingCache | None = None) -> list[float] | None:
    """Get embedding with caching."""
    if cache:
        cached = cache.get(text)
        if cached:
            return cached

    embedding = get_embedding(text)

    if embedding and cache:
        cache.set(text, embedding)

    return embedding
