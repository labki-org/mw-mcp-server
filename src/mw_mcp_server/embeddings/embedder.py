"""
Embedding Client

This module implements a robust, test-friendly embedding client that uses the
OpenAI embeddings API (or any compatible provider). It is responsible for:

- Efficient batching of text inputs
- Network and transport error isolation
- Strict response validation
- Deterministic output semantics for downstream tools (FAISS, search, etc.)

The class is stateless and safe to reuse across requests.
"""

from __future__ import annotations

from typing import List, Sequence, Optional
import logging
import httpx

from ..config import settings

logger = logging.getLogger("mcp.embedder")


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class Embedder:
    """
    Asynchronous embedding generator for batches of text.

    This class performs no caching and assumes the caller handles
    higher-level caching or persistent index management.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1/embeddings",
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize an Embedder.

        Parameters
        ----------
        api_key : Optional[str]
            Optional override for OpenAI API key. Defaults to settings.openai_api_key.

        model : Optional[str]
            Optional override for embedding model. Defaults to settings.embedding_model.

        base_url : str
            Base URL of the embeddings API endpoint.

        timeout : float
            HTTP timeout for each request.
        """
        self.api_key = api_key or settings.openai_api_key.get_secret_value()
        self.model = model or settings.embedding_model
        self.base_url = base_url
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(
        self,
        texts: Sequence[str],
        batch_size: int = 20,
    ) -> List[List[float]]:
        """
        Generate embeddings for a sequence of input texts.

        Parameters
        ----------
        texts : Sequence[str]
            List or sequence of input text strings.

        batch_size : int
            Maximum batch size per request. Helps avoid API token/size limits.

        Returns
        -------
        List[List[float]]
            A flat list of embeddings, each an array of floats.

        Raises
        ------
        EmbeddingError
            If any batch fails or the response is malformed.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                payload = {
                    "model": self.model,
                    "input": batch,
                }

                try:
                    response = await client.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.error(
                        "Embedding request failed (%s): batch size=%d, error=%s",
                        type(exc).__name__,
                        len(batch),
                        str(exc),
                    )
                    raise EmbeddingError(
                        f"Embedding generation failed: {type(exc).__name__}"
                    ) from exc

                embeddings = self._extract_embeddings(response.json())
                all_embeddings.extend(embeddings)

        return all_embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_embeddings(data: dict) -> List[List[float]]:
        """
        Parse and validate embedding output format.

        OpenAI returns:
            { "data": [ {"embedding": [...]}, ... ] }

        Raises
        ------
        EmbeddingError
            If the API returns unexpected structure.
        """
        if "data" not in data:
            raise EmbeddingError("Embedding response missing 'data' field.")

        records = data["data"]
        if not isinstance(records, list):
            raise EmbeddingError("'data' field must be a list.")

        embeddings: List[List[float]] = []

        for index, record in enumerate(records):
            if not isinstance(record, dict) or "embedding" not in record:
                raise EmbeddingError(
                    f"Malformed embedding record at index {index}: {record!r}"
                )

            emb = record["embedding"]
            if not isinstance(emb, list) or not all(
                isinstance(x, (float, int)) for x in emb
            ):
                raise EmbeddingError(
                    f"Invalid embedding vector at index {index}: must be float list."
                )

            embeddings.append([float(x) for x in emb])

        return embeddings
