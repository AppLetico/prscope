from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionCatalogEntry:
    id: str
    section: str
    concept: str
    description: str
    options: list[str] | None = None
    required: bool = True
    match_tokens: tuple[str, ...] = ()


DEFAULT_DECISION_CATALOG: tuple[DecisionCatalogEntry, ...] = (
    DecisionCatalogEntry(
        id="architecture.logging_strategy",
        section="architecture",
        concept="logging_strategy",
        description="What logging strategy should this feature use?",
        options=["log everything", "log only when issues occur", "configurable", "skip logging"],
        required=False,
        match_tokens=("log", "logging", "response time"),
    ),
    DecisionCatalogEntry(
        id="architecture.response_schema",
        section="architecture",
        concept="response_schema",
        description="What response schema should this feature use?",
        options=["keep current format", "add timing to responses", "bundle in a metrics structure"],
        required=False,
        match_tokens=("response format", "response structure", "schema", "client"),
    ),
    DecisionCatalogEntry(
        id="architecture.metrics_scope",
        section="architecture",
        concept="metrics_scope",
        description="What scope should metrics collection cover?",
        options=["health check only", "all routes", "specific routes"],
        required=False,
        match_tokens=("metrics", "endpoint", "scope"),
    ),
    DecisionCatalogEntry(
        id="architecture.cache_strategy",
        section="architecture",
        concept="cache_backend",
        description="What caching strategy should this feature use?",
        options=["shared cache", "local cache", "no caching"],
        required=False,
        match_tokens=("cache", "caching"),
    ),
    DecisionCatalogEntry(
        id="architecture.database",
        section="architecture",
        concept="primary_database",
        description="Which database should store the primary application data?",
        options=["PostgreSQL", "MySQL", "SQLite", "DynamoDB"],
        required=False,
        match_tokens=("database", "storage", "postgres", "mysql", "sqlite", "dynamodb"),
    ),
    DecisionCatalogEntry(
        id="architecture.api_protocol",
        section="architecture",
        concept="api_protocol",
        description="Which API protocol should this interface use?",
        options=["REST-style", "gRPC", "GraphQL"],
        required=False,
        match_tokens=("api", "protocol", "rest", "grpc", "graphql"),
    ),
)
