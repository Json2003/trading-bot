"""Public FastAPI schemas re-exported from the core domain models."""

from __future__ import annotations

from models import OrderRequest as CoreOrderRequest

# Re-export the domain ``OrderRequest`` dataclass so the API layer can
# depend on the canonical representation without duplicating the schema.
OrderRequest = CoreOrderRequest

__all__ = ["OrderRequest"]
