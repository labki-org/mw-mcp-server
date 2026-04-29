"""
Pagination Metadata Helper

Listy tools must tell the LLM whether their result was capped at the
requested limit, so the LLM can choose to widen the search instead of
silently treating a truncated list as the full picture.

The LLM tool loop sees these structured dicts as the literal tool output,
so the `note` field is the primary nudge — phrased as a directive that an
LLM can act on rather than a dry stat line.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence


def paginated(
    results: Sequence[Any],
    *,
    limit: int,
    label: str = "results",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Wrap a list result with truncation metadata for the LLM.

    Parameters
    ----------
    results : Sequence[Any]
        The (already-limited) result rows.
    limit : int
        The cap applied to the underlying call.
    label : str
        Field name to use for the result list. Default ``"results"``.
    extra : Optional[Dict[str, Any]]
        Additional top-level fields to merge in (e.g. domain context).
    """
    count = len(results)
    truncated = count >= limit
    note = (
        f"Returned {count} of (≤{limit}) — the underlying call hit the limit, "
        "so MORE results may exist. Do NOT assume this is the full set: "
        "either re-run with a larger `limit`, narrow the query, or cross-check "
        "the answer with another tool before presenting it."
        if truncated
        else f"Returned {count} of {limit} — full result set (no truncation)."
    )
    payload: Dict[str, Any] = {
        # Callers always pass a freshly-built list; trust the input rather
        # than paying for a defensive O(n) copy on every tool call.
        label: results if isinstance(results, list) else list(results),
        "count": count,
        "limit": limit,
        "truncated": truncated,
        "note": note,
    }
    if extra:
        payload.update(extra)
    return payload
