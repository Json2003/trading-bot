"""LangChain / LangGraph Agents API integration utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from langgraph_sdk import get_sync_client

logger = logging.getLogger(__name__)


def _coerce_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class LangChainAgentConfig:
    """Configuration for connecting to the LangChain Agents API."""

    api_key: str
    endpoint: str
    project: Optional[str]
    tracing_enabled: bool
    openai_api_key: Optional[str]

    @property
    def default_headers(self) -> Mapping[str, str]:
        return {
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }

    @classmethod
    def from_env(cls) -> Optional["LangChainAgentConfig"]:
        api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            return None

        endpoint = (
            os.getenv("LANGSMITH_ENDPOINT")
            or os.getenv("LANGCHAIN_ENDPOINT")
            or "https://api.smith.langchain.com"
        ).rstrip("/")

        project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
        tracing = _coerce_bool(
            os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
        )

        openai_api_key = os.getenv("OPENAI_API_KEY")

        return cls(
            api_key=api_key,
            endpoint=endpoint,
            project=project,
            tracing_enabled=tracing,
            openai_api_key=openai_api_key,
        )


class LangChainAgentService:
    """Thin wrapper around the LangGraph Agents API."""

    def __init__(self, config: LangChainAgentConfig):
        self.config = config
        self._client = None

    @classmethod
    def from_env(cls) -> Optional["LangChainAgentService"]:
        config = LangChainAgentConfig.from_env()
        if not config:
            logger.info("LangChain agent service disabled (missing API key).")
            return None

        service = cls(config)
        service._apply_tracing_environment()
        if not config.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY not set; hosted agents relying on OpenAI models may fail."
            )
        return service

    def _apply_tracing_environment(self) -> None:
        """Align legacy LangChain env vars with LangSmith configuration."""
        if self.config.tracing_enabled:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if self.config.project:
            os.environ.setdefault("LANGCHAIN_PROJECT", self.config.project)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", self.config.endpoint)
        os.environ.setdefault("LANGCHAIN_API_KEY", self.config.api_key)

    def _ensure_client(self):
        if self._client is None:
            self._client = get_sync_client(
                url=self.config.endpoint,
                api_key=self.config.api_key,
                headers=self.config.default_headers,
            )
        return self._client

    def is_enabled(self) -> bool:
        return True

    def start_run(
        self,
        assistant_id: str,
        payload: Mapping[str, Any],
        *,
        thread_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Kick off a LangChain agent run."""
        client = self._ensure_client()
        try:
            run = client.runs.create(
                thread_id,
                assistant_id=assistant_id,
                input=dict(payload),
                metadata=dict(metadata) if metadata else None,
            )
            if self.config.project:
                run.setdefault("project", self.config.project)
            return run
        except Exception:
            logger.exception("LangChain agent invocation failed")
            raise
