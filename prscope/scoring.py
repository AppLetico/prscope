"""
PR relevance scoring for Prscope.

Multi-stage evaluation:
1. Rule-based filtering (keywords + paths)
2. Semantic similarity search (detect already-implemented)
3. LLM analysis (final decision with local code context)

Relevant PRs are used as high-signal seed inputs for planning sessions.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Feature, PrscopeConfig
from .store import PullRequest, PRFile


@dataclass
class FeatureMatch:
    """Result of matching a PR against a single feature."""
    feature_name: str
    keyword_hits: list[str] = field(default_factory=list)
    path_hits: list[str] = field(default_factory=list)
    keyword_score: float = 0.0
    path_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class ScoringResult:
    """Complete scoring result for a PR."""
    pr_id: int
    
    # Rule-based scores
    rule_score: float
    matched_features: list[str]
    feature_matches: list[FeatureMatch]
    
    # Semantic analysis
    semantic_overlaps: list[dict[str, Any]] = field(default_factory=list)
    has_existing_implementation: bool = False
    
    # LLM decision (final authority)
    llm_decision: str = "pending"  # implement, skip, partial, pending
    llm_confidence: float = 0.0
    llm_reasoning: str = ""
    llm_analysis: dict[str, Any] | None = None
    
    # Final decision
    final_decision: str = "pending"  # relevant, skip, maybe, pending
    final_score: float = 0.0
    signals: dict[str, Any] = field(default_factory=dict)
    
    def should_seed_plan(self) -> bool:
        """Determine if this PR is a strong plan-seed candidate."""
        if self.llm_decision == "implement" and self.llm_confidence >= 0.7:
            return True
        if self.llm_decision == "partial" and self.llm_confidence >= 0.8:
            return True
        return False

    def should_generate_prd(self) -> bool:
        """Backward-compatible alias for legacy callers."""
        return self.should_seed_plan()


def tokenize(text: str) -> set[str]:
    """Extract lowercase tokens from text."""
    if not text:
        return set()
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return set(tokens)


def match_keyword(keyword: str, text_tokens: set[str]) -> bool:
    """Check if a keyword matches any token."""
    keyword_lower = keyword.lower()
    if keyword_lower in text_tokens:
        return True
    for token in text_tokens:
        if keyword_lower in token or token in keyword_lower:
            return True
    return False


def match_path_glob(pattern: str, file_path: str) -> bool:
    """Check if a file path matches a glob pattern."""
    pattern = pattern.replace("\\", "/")
    file_path = file_path.replace("\\", "/")
    
    if "**" in pattern:
        regex_pattern = pattern.replace(".", r"\.")
        regex_pattern = regex_pattern.replace("**", ".*")
        regex_pattern = regex_pattern.replace("*", "[^/]*")
        regex_pattern = f"^{regex_pattern}$"
        return bool(re.match(regex_pattern, file_path))
    
    return fnmatch.fnmatch(file_path, pattern)


def score_feature(
    feature: Feature,
    pr: PullRequest,
    files: list[PRFile],
    keyword_weight: float = 0.4,
    path_weight: float = 0.6,
) -> FeatureMatch:
    """Score a PR against a single feature."""
    match = FeatureMatch(feature_name=feature.name)
    
    pr_text = f"{pr.title or ''} {pr.body or ''}"
    text_tokens = tokenize(pr_text)
    
    if feature.keywords:
        for keyword in feature.keywords:
            if match_keyword(keyword, text_tokens):
                match.keyword_hits.append(keyword)
        match.keyword_score = len(match.keyword_hits) / len(feature.keywords)
    else:
        match.keyword_score = 0.0
    
    if feature.paths and files:
        matched_paths = set()
        for pattern in feature.paths:
            for f in files:
                if match_path_glob(pattern, f.path):
                    matched_paths.add(f.path)
        match.path_hits = list(matched_paths)
        match.path_score = len(matched_paths) / len(files) if files else 0.0
    else:
        match.path_score = 0.0
    
    if feature.keywords and feature.paths:
        match.combined_score = (
            keyword_weight * match.keyword_score +
            path_weight * match.path_score
        )
    elif feature.keywords:
        match.combined_score = match.keyword_score
    elif feature.paths:
        match.combined_score = match.path_score
    else:
        match.combined_score = 0.0
    
    return match


def score_pr_rules(
    pr: PullRequest,
    files: list[PRFile],
    config: PrscopeConfig,
) -> tuple[float, list[str], list[FeatureMatch]]:
    """
    Stage 1: Rule-based scoring.
    
    Returns (rule_score, matched_features, feature_matches)
    """
    feature_matches: list[FeatureMatch] = []
    matched_features: list[str] = []
    
    for feature in config.features:
        match = score_feature(
            feature,
            pr,
            files,
            keyword_weight=config.scoring.keyword_weight,
            path_weight=config.scoring.path_weight,
        )
        feature_matches.append(match)
        
        if match.combined_score > 0:
            matched_features.append(feature.name)
    
    if feature_matches:
        non_zero_scores = [m.combined_score for m in feature_matches if m.combined_score > 0]
        if non_zero_scores:
            rule_score = sum(non_zero_scores) / len(non_zero_scores)
        else:
            rule_score = 0.0
    else:
        rule_score = 0.0
    
    return rule_score, matched_features, feature_matches


def run_semantic_analysis(
    pr: PullRequest,
    files: list[PRFile],
    local_repo_path: Path,
    matched_features: list[str],
    config: PrscopeConfig,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Stage 2: Semantic similarity analysis.
    
    Returns (similarity_results, has_existing_implementation)
    """
    try:
        from .semantic import (
            extract_matching_files,
            find_similar_implementations,
            EmbeddingCache,
        )
    except ImportError:
        return [], False
    
    # Get feature paths for matching
    feature_paths = []
    for feature in config.features:
        if feature.name in matched_features:
            feature_paths.extend(feature.paths)
    
    # Extract matching local files (now includes keyword-based search)
    local_chunks = extract_matching_files(
        repo_root=local_repo_path,
        pr_files=[f.path for f in files],
        feature_paths=feature_paths,
        pr_title=pr.title,
        pr_body=pr.body,
    )
    
    if not local_chunks:
        return [], False
    
    # Build PR description
    pr_description = f"{pr.title}\n{pr.body or ''}"
    pr_file_paths = [f.path for f in files]
    
    # Find similar implementations
    try:
        cache = EmbeddingCache()
        similarities = find_similar_implementations(
            pr_description=pr_description,
            pr_files=pr_file_paths,
            local_chunks=local_chunks,
            similarity_threshold=0.7,
        )
        
        # Convert to dict format
        similarity_results = [
            {
                "local_path": s.local_path,
                "local_snippet": s.local_snippet,
                "similarity_score": s.similarity_score,
                "overlap_type": s.overlap_type,
            }
            for s in similarities
        ]
        
        # Check if any high-similarity matches suggest existing implementation
        has_existing = any(s.similarity_score >= 0.85 for s in similarities)
        
        return similarity_results, has_existing
        
    except Exception as e:
        # Semantic analysis is optional
        return [], False


def run_llm_analysis(
    pr: PullRequest,
    files: list[PRFile],
    local_profile: dict[str, Any],
    matched_features: list[str],
    local_repo_path: Path,
    similarity_results: list[dict[str, Any]],
    config: PrscopeConfig,
) -> dict[str, Any]:
    """
    Stage 3: LLM analysis with full context.
    
    Returns LLM analysis result as dict.
    """
    from .llm import get_llm_client, LLMAnalysisResult
    
    llm = get_llm_client(config.upstream_eval)
    if not llm.enabled:
        return LLMAnalysisResult.skip("LLM not enabled").to_dict()
    
    # Build file list for LLM
    files_data = [
        {"path": f.path, "additions": f.additions, "deletions": f.deletions}
        for f in files
    ]
    
    # Get local code context for matching files
    local_code_context = []
    integration_check = None
    try:
        from .semantic import extract_matching_files, extract_keywords_from_pr
        
        feature_paths = []
        for feature in config.features:
            if feature.name in matched_features:
                feature_paths.extend(feature.paths)
        
        local_chunks = extract_matching_files(
            repo_root=local_repo_path,
            pr_files=[f.path for f in files],
            feature_paths=feature_paths,
            pr_title=pr.title,
            pr_body=pr.body,
        )
        
        # Include more context for security-related files
        is_security_pr = any(kw in pr.title.lower() for kw in ['security', 'fix', 'vulnerability', 'lfi', 'xss', 'injection'])
        max_chars = 4000 if is_security_pr else 2000
        max_files = 8 if is_security_pr else 5
        
        local_code_context = [
            {"path": chunk.path, "content": chunk.content[:max_chars]}
            for chunk in local_chunks[:max_files]
        ]
        
        # Check for integrations mentioned in PR but missing from local code
        keywords = extract_keywords_from_pr(pr.title, pr.body)
        integrations = [k.replace("integration:", "") for k in keywords if k.startswith("integration:")]
        
        if integrations:
            # Search local codebase for integration references
            all_local_content = " ".join(c.content.lower() for c in local_chunks)
            readme = local_profile.get("readme", "").lower()
            deps = str(local_profile.get("dependencies", {})).lower()
            search_text = f"{all_local_content} {readme} {deps}"
            
            missing = [i for i in integrations if i.lower() not in search_text]
            found = [i for i in integrations if i.lower() in search_text]
            
            if missing:
                integration_check = {
                    "mentioned_in_pr": integrations,
                    "found_in_local": found,
                    "missing_from_local": missing,
                    "warning": f"PR mentions {', '.join(missing)} but local codebase has no references to it"
                }
    except Exception:
        pass
    
    # Run LLM analysis
    analysis = llm.analyze_pr(
        pr_title=pr.title,
        pr_body=pr.body,
        pr_files=files_data,
        local_profile=local_profile,
        matched_features=matched_features,
        local_code_context=local_code_context,
        similarity_results=similarity_results,
        project_name=config.project.name,
        project_description=config.project.description,
        integration_check=integration_check,
    )
    
    return analysis.to_dict()


def score_pr(
    pr: PullRequest,
    files: list[PRFile],
    config: PrscopeConfig,
    local_profile: dict[str, Any] | None = None,
    local_repo_path: Path | None = None,
    run_semantic: bool = True,
    run_llm: bool = True,
) -> ScoringResult:
    """
    Complete multi-stage PR scoring.
    
    Stage 1: Rule-based (keywords + paths)
    Stage 2: Semantic similarity (detect duplicates)
    Stage 3: LLM analysis (final decision)
    """
    # Stage 1: Rule-based scoring
    rule_score, matched_features, feature_matches = score_pr_rules(pr, files, config)
    
    result = ScoringResult(
        pr_id=pr.id,
        rule_score=rule_score,
        matched_features=matched_features,
        feature_matches=feature_matches,
        signals={
            "total_features": len(config.features),
            "matched_feature_count": len(matched_features),
            "file_count": len(files),
            "has_labels": bool(pr.labels),
            "label_count": len(pr.labels) if pr.labels else 0,
        },
    )
    
    # Early exit if rule score too low
    if rule_score < config.scoring.min_rule_score:
        result.final_decision = "skip"
        result.final_score = rule_score
        result.llm_decision = "skip"
        result.llm_reasoning = "Rule score below threshold"
        return result
    
    # Stage 2: Semantic similarity (if enabled and repo path available)
    if run_semantic and local_repo_path:
        similarity_results, has_existing = run_semantic_analysis(
            pr=pr,
            files=files,
            local_repo_path=local_repo_path,
            matched_features=matched_features,
            config=config,
        )
        result.semantic_overlaps = similarity_results
        result.has_existing_implementation = has_existing
    else:
        similarity_results = []
    
    # Stage 3: LLM analysis (if enabled)
    if run_llm and config.upstream_eval.enabled and local_profile:
        llm_result = run_llm_analysis(
            pr=pr,
            files=files,
            local_profile=local_profile,
            matched_features=matched_features,
            local_repo_path=local_repo_path or Path.cwd(),
            similarity_results=similarity_results,
            config=config,
        )
        
        result.llm_analysis = llm_result
        
        relevance = llm_result.get("relevance", {})
        result.llm_decision = relevance.get("decision", "skip")
        result.llm_confidence = relevance.get("confidence", 0.0)
        result.llm_reasoning = relevance.get("reasoning", "")
        
        # Final decision based on LLM
        if result.llm_decision == "implement" and result.llm_confidence >= 0.7:
            result.final_decision = "relevant"
            result.final_score = result.llm_confidence
        elif result.llm_decision == "partial" and result.llm_confidence >= 0.6:
            result.final_decision = "maybe"
            result.final_score = result.llm_confidence * 0.8
        else:
            result.final_decision = "skip"
            result.final_score = result.llm_confidence * 0.5
    else:
        # Fall back to rule-based decision
        if rule_score >= config.scoring.min_final_score:
            result.final_decision = "relevant"
        elif rule_score >= config.scoring.min_rule_score:
            result.final_decision = "maybe"
        else:
            result.final_decision = "skip"
        result.final_score = rule_score
        result.llm_decision = "pending"
        result.llm_reasoning = "LLM analysis not run"
    
    return result


def evaluate_pr(
    pr: PullRequest,
    files: list[PRFile],
    config: PrscopeConfig,
    local_profile_sha: str,
    store: "Store",  # type: ignore
    local_profile: dict[str, Any] | None = None,
    local_repo_path: Path | None = None,
) -> ScoringResult | None:
    """
    Evaluate a PR if not already evaluated.
    
    Returns ScoringResult if evaluated, None if skipped (already evaluated).
    """
    # Check if already evaluated
    if pr.head_sha and store.evaluation_exists(pr.id, local_profile_sha, pr.head_sha):
        return None
    
    # Run full scoring pipeline
    result = score_pr(
        pr=pr,
        files=files,
        config=config,
        local_profile=local_profile,
        local_repo_path=local_repo_path,
        run_semantic=config.upstream_eval.enabled,
        run_llm=config.upstream_eval.enabled,
    )
    
    # Save evaluation
    store.save_evaluation(
        pr_id=pr.id,
        local_profile_sha=local_profile_sha,
        pr_head_sha=pr.head_sha or "",
        rule_score=result.rule_score,
        final_score=result.final_score,
        matched_features=result.matched_features,
        signals=result.signals,
        llm_result=result.llm_analysis,
        decision=result.final_decision,
    )
    
    return result
