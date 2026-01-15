"""
Tenant-Aware FAISS Index Registry

This module provides a registry pattern for managing per-tenant FAISS indexes.
Each wiki instance gets its own isolated vector index.

Thread Safety
-------------
- The registry is protected by an RLock
- Individual FaissIndex instances have their own locks
- Safe for concurrent use across async request handlers
"""

from __future__ import annotations

from threading import RLock
from typing import Dict

from .index import FaissIndex
from ..tenants import (
    ensure_tenant_directory,
    get_tenant_index_path,
    get_tenant_meta_path,
)


# ---------------------------------------------------------------------
# Global Index Registry
# ---------------------------------------------------------------------

_index_registry: Dict[str, FaissIndex] = {}
_registry_lock = RLock()


def get_tenant_index(wiki_id: str) -> FaissIndex:
    """
    Get or create the FAISS index for a tenant.

    This function is thread-safe and will:
    1. Validate the wiki_id
    2. Create the tenant's data directory if needed
    3. Load an existing index from disk, or create a new one

    Parameters
    ----------
    wiki_id : str
        The unique identifier for the wiki instance.

    Returns
    -------
    FaissIndex
        A tenant-scoped FAISS index instance.

    Raises
    ------
    InvalidTenantError
        If wiki_id is invalid.
    """
    with _registry_lock:
        if wiki_id in _index_registry:
            return _index_registry[wiki_id]

        # Ensure directory exists (validates wiki_id internally)
        ensure_tenant_directory(wiki_id)

        # Create index with tenant-scoped paths
        index = FaissIndex(
            index_path=get_tenant_index_path(wiki_id),
            meta_path=get_tenant_meta_path(wiki_id),
        )

        # Attempt to load existing data
        try:
            index.load()
        except Exception:
            # No existing index, start fresh
            pass

        _index_registry[wiki_id] = index
        return index


def save_tenant_index(wiki_id: str) -> bool:
    """
    Persist a tenant's index to disk.

    Returns True if saved, False if no index exists for this tenant.
    """
    with _registry_lock:
        index = _index_registry.get(wiki_id)
        if index is None:
            return False

        index.save()
        return True


def save_all_tenant_indexes() -> int:
    """
    Persist all loaded tenant indexes to disk.

    Returns the number of indexes saved.
    """
    with _registry_lock:
        count = 0
        for wiki_id, index in _index_registry.items():
            try:
                index.save()
                count += 1
            except Exception:
                # Log but continue saving others
                pass
        return count


def clear_tenant_index(wiki_id: str) -> bool:
    """
    Remove a tenant's index from the registry (does not delete files).
    """
    with _registry_lock:
        if wiki_id in _index_registry:
            del _index_registry[wiki_id]
            return True
        return False


def get_loaded_tenants() -> list[str]:
    """
    Return list of currently loaded tenant IDs.
    """
    with _registry_lock:
        return list(_index_registry.keys())
