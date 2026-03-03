"""
LLM interface for Prscope using LiteLLM.

Provides intelligent PR analysis with:
- Semantic relevance scoring
- "Already implemented" detection
- Implementation recommendations
- Confidence-based decisions

LiteLLM supports 100+ providers:
- OpenAI (gpt-4o, gpt-4-turbo)
- Anthropic (claude-3-opus, claude-3-sonnet)
- Google (gemini-pro, gemini-1.5-pro)
- Azure, AWS Bedrock, Ollama, and more

See: https://docs.litellm.ai/docs/providers
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .config import LLMConfig

logger = logging.getLogger(__name__)

# Suppress LiteLLM's verbose logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@dataclass
class RelevanceDecision:
    """LLM's decision on whether to implement a PR."""

    decision: str  # "implement", "skip", "partial"
    confidence: float  # 0.0 - 1.0
    reasoning: str
    already_implemented: bool
    implementation_overlap: str  # "none", "partial", "full"


@dataclass
class LLMAnalysisResult:
    """Complete result from LLM analysis of a PR."""

    # Relevance decision
    relevance: RelevanceDecision

    # Implementation details (only if decision == "implement")
    impacted_features: list[str] = field(default_factory=list)
    why_this_matters: str = ""
    implementation_plan: str = ""
    estimated_effort: str = ""  # "small", "medium", "large"
    tests: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    # Raw response for debugging
    raw_response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "relevance": {
                "decision": self.relevance.decision,
                "confidence": self.relevance.confidence,
                "reasoning": self.relevance.reasoning,
                "already_implemented": self.relevance.already_implemented,
                "implementation_overlap": self.relevance.implementation_overlap,
            },
            "impacted_features": self.impacted_features,
            "why_this_matters": self.why_this_matters,
            "implementation_plan": self.implementation_plan,
            "estimated_effort": self.estimated_effort,
            "tests": self.tests,
            "risks": self.risks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMAnalysisResult:
        """Create from dictionary."""
        rel_data = data.get("relevance", {})
        return cls(
            relevance=RelevanceDecision(
                decision=rel_data.get("decision", "skip"),
                confidence=rel_data.get("confidence", 0.0),
                reasoning=rel_data.get("reasoning", ""),
                already_implemented=rel_data.get("already_implemented", False),
                implementation_overlap=rel_data.get("implementation_overlap", "none"),
            ),
            impacted_features=data.get("impacted_features", []),
            why_this_matters=data.get("why_this_matters", ""),
            implementation_plan=data.get("implementation_plan", ""),
            estimated_effort=data.get("estimated_effort", ""),
            tests=data.get("tests", []),
            risks=data.get("risks", []),
        )

    @classmethod
    def skip(cls, reason: str = "LLM analysis unavailable") -> LLMAnalysisResult:
        """Create a skip result."""
        return cls(
            relevance=RelevanceDecision(
                decision="skip",
                confidence=0.0,
                reasoning=reason,
                already_implemented=False,
                implementation_overlap="none",
            ),
        )


# Enhanced system prompt for intelligent PR analysis
ANALYSIS_SYSTEM_PROMPT = """\
You are an expert software architect analyzing GitHub pull requests for relevance to a local codebase.

Your job is to determine:
1. Is this PR relevant to the local codebase?
2. Has something similar already been implemented locally?
3. Should the team implement these changes?

You will receive:
- PR title, description, and changed files
- Local codebase profile (structure, dependencies)
- Relevant local code snippets for comparison
- Semantic similarity results (if any overlaps detected)

## CRITICAL: Check for Existing Implementations

BEFORE recommending "implement", carefully examine the LOCAL CODE SNIPPETS provided.
Look for:
- Functions with similar names (e.g., if PR adds "isValidPath", check for "isPathSafe", "validatePath", "sanitizePath")
- Similar logic patterns (e.g., path traversal checks, input validation, sanitization)
- Comments referencing the same vulnerability or feature
- Security hardening that addresses the same attack vector

If the local code already has equivalent functionality (even with different names/paths):
- Set "already_implemented": true
- Set "implementation_overlap": "full" or "partial"
- Set "decision": "skip" (for full overlap) or "partial" (for partial overlap)

## CRITICAL: Check if Feature/Integration Exists

Before recommending "implement", ask: Does the LOCAL codebase even have this feature?

### Third-Party Integrations
If the PR mentions ANY of these, check if the local codebase actually uses them:
- **Messaging**: Telegram, Slack, Discord, WhatsApp, Teams, Signal
- **Email**: Gmail, Outlook, SendGrid, Mailgun
- **Cloud**: AWS, GCP, Azure (specific services)
- **Databases**: specific DB fixes (MongoDB, PostgreSQL, Redis)
- **APIs**: specific third-party API integrations

**How to check**: Look at the LOCAL CODE SNIPPETS and project description.
If you don't see imports, SDKs, or references to that integration → SKIP

Examples:
- PR fixes "Slack media downloads" but local code has NO Slack SDK/imports → SKIP
- PR fixes "Telegram bot errors" but local code is not a Telegram bot → SKIP
- PR fixes "MongoDB connection pooling" but local code uses PostgreSQL → SKIP

### Security PRs
For security fixes, first ask: Does the LOCAL codebase have the vulnerable feature?

- PR fixes "path traversal in media parser" but local has NO media parsing → SKIP
- PR fixes "LFI vulnerability" but local has no file reading from user input → SKIP

Then check if similar protection already exists in local code.

### General Rule
If the PR title mentions a specific technology/integration/feature that you don't
see evidence of in the local codebase (README, code snippets, dependencies) → SKIP

Respond ONLY with valid JSON in this exact format:
{
    "relevance": {
        "decision": "implement" | "skip" | "partial",
        "confidence": 0.0-1.0,
        "reasoning": "Brief explanation of your decision",
        "already_implemented": true | false,
        "implementation_overlap": "none" | "partial" | "full"
    },
    "impacted_features": ["feature1", "feature2"],
    "why_this_matters": "Why this PR is valuable (if implementing)",
    "implementation_plan": "Step-by-step implementation guidance (if implementing)",
    "estimated_effort": "small" | "medium" | "large",
    "tests": ["Test case 1", "Test case 2"],
    "risks": ["Risk 1", "Risk 2"]
}

Decision guidelines:
- "implement": PR introduces valuable, non-duplicate functionality. Confidence >= 0.7
- "partial": Some aspects are useful, others already exist. Confidence >= 0.5
- "skip": Not relevant, already implemented, or doesn't fit. Any confidence.
"""


def build_analysis_prompt(
    pr_title: str,
    pr_body: str | None,
    pr_files: list[dict[str, Any]],
    local_profile: dict[str, Any],
    matched_features: list[str],
    local_code_context: list[dict[str, str]] | None = None,
    similarity_results: list[dict[str, Any]] | None = None,
    project_name: str = "",
    project_description: str = "",
    integration_check: dict[str, Any] | None = None,
) -> str:
    """Build the user prompt for comprehensive PR analysis."""
    # PR summary
    files_summary = []
    for f in pr_files[:20]:
        files_summary.append(f"- {f.get('path', '')} (+{f.get('additions', 0)}/-{f.get('deletions', 0)})")

    # Local profile summary
    profile_summary = {
        "total_files": local_profile.get("file_tree", {}).get("total_files", 0),
        "top_extensions": dict(
            sorted(
                local_profile.get("file_tree", {}).get("extensions", {}).items(),
                key=lambda x: -x[1],
            )[:5]
        ),
        "dependencies": list(local_profile.get("dependencies", {}).keys()),
    }

    # Build project context section
    # Use README from profile if no explicit description provided
    readme_content = local_profile.get("readme", "") if local_profile else ""
    effective_description = project_description or readme_content

    project_section = ""
    if project_name or effective_description:
        # Truncate README for prompt efficiency
        if len(effective_description) > 3000:
            effective_description = effective_description[:3000] + "\n... (see full README)"

        project_section = f"""
## Local Project Context

**Project:** {project_name or "Unknown"}

**Description:**
{effective_description or "(No description provided)"}

---

"""

    prompt = f"""\
{project_section}## Pull Request

**Title:** {pr_title}

**Description:**
{pr_body or "(No description provided)"}

**Files Changed ({len(pr_files)} total):**
{chr(10).join(files_summary[:15])}
{f"... and {len(files_summary) - 15} more files" if len(files_summary) > 15 else ""}

## Local Codebase Profile

```json
{json.dumps(profile_summary, indent=2)}
```

## Pre-matched Features (by keyword/path rules)

{", ".join(matched_features) if matched_features else "None matched"}
"""

    # Add local code context if available
    if local_code_context:
        prompt += "\n## Relevant Local Code (for comparison)\n\n"
        prompt += (
            "**IMPORTANT: Carefully review this code for existing implementations of the same functionality.**\n\n"
        )
        for ctx in local_code_context[:5]:  # Limit to 5 files
            content = ctx.get("content", "")
            # Show more content for security-related files
            max_chars = 3500 if "security" in ctx.get("path", "").lower() else 2000
            prompt += f"### {ctx.get('path', 'unknown')}\n```\n{content[:max_chars]}\n```\n\n"

    # Add similarity results if available
    if similarity_results:
        prompt += "\n## Semantic Similarity Analysis\n\n"
        prompt += "The following local files were found to be semantically similar to this PR:\n\n"
        for sim in similarity_results[:5]:
            prompt += f"- **{sim.get('local_path')}** (similarity: {sim.get('similarity_score', 0):.2f}, type: {sim.get('overlap_type', 'unknown')})\n"
            if sim.get("local_snippet"):
                prompt += f"  ```\n  {sim.get('local_snippet', '')[:300]}...\n  ```\n"
        prompt += "\n**IMPORTANT**: If similarity is high (>0.8), carefully check if this is already implemented.\n"

    # Add integration check warning if applicable
    if integration_check and integration_check.get("missing_from_local"):
        missing = integration_check["missing_from_local"]
        prompt += f"""
## ⚠️ INTEGRATION WARNING

This PR mentions: **{", ".join(integration_check.get("mentioned_in_pr", []))}**

**NOT FOUND in local codebase:** {", ".join(missing)}

The local codebase does not appear to use {", ".join(missing)}.
If the PR is specifically about {", ".join(missing)} integration/features,
you should likely **SKIP** this PR as the integration doesn't exist locally.

"""

    prompt += """
## Your Task

Analyze this PR and determine:
1. Does the local codebase even have the feature/integration this PR is about?
2. Should the local codebase implement these changes?
3. Is any of this already implemented locally?

Be conservative - only recommend "implement" for genuinely valuable, non-duplicate changes.
If the PR is about a specific integration (Slack, Telegram, etc.) that doesn't exist locally, SKIP it.
"""

    return prompt


class LLMClient:
    """LiteLLM-based client for intelligent PR analysis."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.2,  # Lower for more deterministic decisions
        max_tokens: int = 3000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._litellm = None

    def _get_litellm(self):
        """Lazy import LiteLLM."""
        if self._litellm is None:
            try:
                import litellm

                self._litellm = litellm
            except ImportError:
                raise ImportError("LiteLLM is required for LLM features. Install with: pip install litellm")
        return self._litellm

    @property
    def enabled(self) -> bool:
        """Check if LLM is available (has required API key)."""
        try:
            self._get_litellm()
            model_lower = self.model.lower()

            import os

            if model_lower.startswith(("gpt-", "o1", "o3")):
                return bool(os.environ.get("OPENAI_API_KEY"))
            elif model_lower.startswith(("claude-",)):
                return bool(os.environ.get("ANTHROPIC_API_KEY"))
            elif model_lower.startswith(("gemini-",)):
                return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
            elif model_lower.startswith(("ollama",)):
                return True
            else:
                return True
        except Exception:
            return False

    def analyze_pr(
        self,
        pr_title: str,
        pr_body: str | None,
        pr_files: list[dict[str, Any]],
        local_profile: dict[str, Any],
        matched_features: list[str],
        local_code_context: list[dict[str, str]] | None = None,
        similarity_results: list[dict[str, Any]] | None = None,
        project_name: str = "",
        project_description: str = "",
        integration_check: dict[str, Any] | None = None,
    ) -> LLMAnalysisResult:
        """
        Analyze a PR with full context for intelligent decision-making.

        Args:
            pr_title: PR title
            pr_body: PR body/description
            pr_files: List of file changes
            local_profile: Local repo profile
            matched_features: Features matched by rules
            local_code_context: Relevant local code snippets
            similarity_results: Semantic similarity matches

        Returns:
            LLMAnalysisResult with decision and implementation details
        """
        if not self.enabled:
            return LLMAnalysisResult.skip("LLM not enabled or API key missing")

        litellm = self._get_litellm()

        user_prompt = build_analysis_prompt(
            pr_title=pr_title,
            pr_body=pr_body,
            pr_files=pr_files,
            local_profile=local_profile,
            matched_features=matched_features,
            local_code_context=local_code_context,
            similarity_results=similarity_results,
            project_name=project_name,
            project_description=project_description,
            integration_check=integration_check,
        )

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content

            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            # Parse relevance decision
            rel_data = data.get("relevance", {})
            relevance = RelevanceDecision(
                decision=rel_data.get("decision", "skip"),
                confidence=float(rel_data.get("confidence", 0.0)),
                reasoning=rel_data.get("reasoning", ""),
                already_implemented=rel_data.get("already_implemented", False),
                implementation_overlap=rel_data.get("implementation_overlap", "none"),
            )

            return LLMAnalysisResult(
                relevance=relevance,
                impacted_features=data.get("impacted_features", matched_features),
                why_this_matters=data.get("why_this_matters", ""),
                implementation_plan=data.get("implementation_plan", ""),
                estimated_effort=data.get("estimated_effort", "medium"),
                tests=data.get("tests", []),
                risks=data.get("risks", []),
                raw_response=data,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return LLMAnalysisResult.skip(f"Failed to parse LLM response: {e}")
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return LLMAnalysisResult.skip(f"LLM analysis failed: {e}")


class NoOpLLM:
    """No-op LLM for when LLM is disabled."""

    @property
    def enabled(self) -> bool:
        return False

    def analyze_pr(self, *args, **kwargs) -> LLMAnalysisResult:
        return LLMAnalysisResult.skip("LLM is disabled")


def get_llm_client(config: LLMConfig | None = None) -> LLMClient | NoOpLLM:
    """Get the configured LLM client."""
    if config is None or not config.enabled:
        return NoOpLLM()

    return LLMClient(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
