from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...config import PlanningConfig
from ...model_catalog import (
    model_has_elevated_json_contract_risk,
    model_prefers_compact_json,
    model_provider,
)

StageName = Literal["discovery", "initial_draft", "critic_review", "author_refine", "memory"]


@dataclass(frozen=True)
class StageModelSelection:
    stage: StageName
    primary_model: str
    fallback_models: tuple[str, ...]
    provider: str
    prefers_compact_json: bool
    elevated_json_contract_risk: bool

    @property
    def first_fallback_model(self) -> str | None:
        return self.fallback_models[0] if self.fallback_models else None


@dataclass(frozen=True)
class ResolvedModelPolicy:
    discovery: StageModelSelection
    initial_draft: StageModelSelection
    critic_review: StageModelSelection
    author_refine: StageModelSelection
    memory: StageModelSelection

    def for_stage(self, stage: StageName) -> StageModelSelection:
        return getattr(self, stage)


class RuntimeModelPolicyResolver:
    def __init__(self, config: PlanningConfig) -> None:
        self._config = config

    def resolve(
        self,
        *,
        session_author_model: str | None,
        session_critic_model: str | None,
        author_model_override: str | None,
        critic_model_override: str | None,
    ) -> ResolvedModelPolicy:
        author_base = author_model_override or session_author_model or self._config.author_model
        critic_base = critic_model_override or session_critic_model or self._config.critic_model
        structured_output_fallback = str(self._config.structured_output_fallback_model or "").strip()

        return ResolvedModelPolicy(
            discovery=self._selection(
                stage="discovery",
                primary_model=self._config.discovery_model or author_base,
            ),
            initial_draft=self._selection(
                stage="initial_draft",
                primary_model=self._config.initial_draft_model or author_base,
            ),
            critic_review=self._selection(
                stage="critic_review",
                primary_model=self._config.critic_review_model or critic_base,
                structured_output_fallback=structured_output_fallback,
            ),
            author_refine=self._selection(
                stage="author_refine",
                primary_model=self._config.author_refine_model or author_base,
                structured_output_fallback=structured_output_fallback,
            ),
            memory=self._selection(
                stage="memory",
                primary_model=self._config.memory_model or author_base,
            ),
        )

    @staticmethod
    def _selection(
        *,
        stage: StageName,
        primary_model: str,
        structured_output_fallback: str | None = None,
    ) -> StageModelSelection:
        normalized_primary = str(primary_model).strip()
        fallback_models: list[str] = []
        if stage in {"critic_review", "author_refine"}:
            fallback = str(structured_output_fallback or "").strip()
            if fallback and fallback != normalized_primary and model_has_elevated_json_contract_risk(normalized_primary):
                fallback_models.append(fallback)
        return StageModelSelection(
            stage=stage,
            primary_model=normalized_primary,
            fallback_models=tuple(fallback_models),
            provider=model_provider(normalized_primary),
            prefers_compact_json=model_prefers_compact_json(normalized_primary),
            elevated_json_contract_risk=model_has_elevated_json_contract_risk(normalized_primary),
        )
