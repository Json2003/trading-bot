"""Thin client for interacting with MCP (Model Control Platform) servers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # prefer the real requests module (repo ships a stub in root)
    from feature_registry.vendor import import_requests as _import_requests

    requests = _import_requests()
except Exception:  # pragma: no cover - fallback for environments without vendor helper
    import requests  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class MCPConfig:
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 15

    @classmethod
    def from_env(cls) -> Optional["MCPConfig"]:
        base_url = os.getenv("MCP_BASE_URL")
        if not base_url:
            return None
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=os.getenv("MCP_API_KEY"),
            timeout=int(os.getenv("MCP_TIMEOUT", "15")),
        )


class MCPClient:
    """Simple HTTP wrapper for MCP server interactions."""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({"Authorization": f"Bearer {config.api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})

    @classmethod
    def from_env(cls) -> Optional["MCPClient"]:
        config = MCPConfig.from_env()
        if not config:
            return None
        return cls(config)

    def is_enabled(self) -> bool:
        return bool(self.config.base_url)

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        try:
            resp = self.session.request(method, url, timeout=self.config.timeout, **kwargs)
            resp.raise_for_status()
            if resp.content:
                return resp.json()
            return {}
        except requests.HTTPError as exc:
            logger.error("MCP request to %s failed: %s", url, exc)
            raise
        except Exception as exc:
            logger.exception("Unexpected MCP error for %s %s", method, url)
            raise

    def fetch_signal_batch(self) -> Dict[str, Any]:
        """Fetch the latest signal batch from MCP server."""
        return self._request("GET", "/signals/latest")

    def push_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Publish metrics or state updates back to MCP."""
        return self._request("POST", "/metrics", json=payload)

    def heartbeat(self) -> Dict[str, Any]:
        """Simple heartbeat check to validate connectivity."""
        return self._request("GET", "/health")
