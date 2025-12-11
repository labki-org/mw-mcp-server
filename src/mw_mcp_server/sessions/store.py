"""
Session Store

In-memory conversation history storage for chat sessions.

This module provides a simple, testable, and concurrency-safe session store
used by the MCP server to track multi-turn conversations with the LLM.

Design choices
--------------
- In-memory only (no persistence across process restarts).
- Per-session message lists with an optional maximum length.
- Thread-safe access using a re-entrant lock.
- Copy-on-read semantics (callers cannot mutate internal state).
- Global singleton `session_store` for typical application use, while still
  allowing custom instances to be created for tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from threading import RLock

from ..api.models import ChatMessage


class SessionStore:
    """
    In-memory store mapping session IDs to ordered lists of ChatMessage objects.

    This store is intended to be lightweight and fast for typical deployments
    where a single MCP server instance handles a moderate number of concurrent
    sessions. For horizontally scaled or highly persistent setups, this class
    can be replaced with a database-backed implementation exposing the same
    interface.
    """

    def __init__(self, max_messages_per_session: Optional[int] = None) -> None:
        """
        Initialize a new SessionStore.

        Parameters
        ----------
        max_messages_per_session : Optional[int]
            If provided, each session's message history is truncated to keep
            at most this many most recent messages. If None, history is
            unbounded (in-memory only).
        """
        self._store: Dict[str, List[ChatMessage]] = {}
        self._lock = RLock()
        self._max_messages_per_session = max_messages_per_session

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """
        Return the message history for a given session ID.

        The returned list is a shallow copy of the internal list to prevent
        callers from mutating internal state.

        Parameters
        ----------
        session_id : str
            Unique identifier for the chat session.

        Returns
        -------
        List[ChatMessage]
            A list of messages for the given session. Empty list if unknown.
        """
        with self._lock:
            messages = self._store.get(session_id, [])
            # Return a copy to avoid accidental external mutation
            return list(messages)

    def add_messages(self, session_id: str, new_messages: List[ChatMessage]) -> None:
        """
        Append new messages to the given session's history.

        If the session does not exist yet, it is created. If
        `max_messages_per_session` is set, the history is truncated to the most
        recent N messages after insertion.

        Parameters
        ----------
        session_id : str
            Unique identifier for the chat session.

        new_messages : List[ChatMessage]
            Messages to append in order.
        """
        if not new_messages:
            return

        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []

            self._store[session_id].extend(new_messages)

            if (
                self._max_messages_per_session is not None
                and self._max_messages_per_session > 0
            ):
                # Keep only the most recent N messages
                excess = len(self._store[session_id]) - self._max_messages_per_session
                if excess > 0:
                    self._store[session_id] = self._store[session_id][excess:]

    def clear(self, session_id: str) -> None:
        """
        Remove all history for a given session ID.

        Parameters
        ----------
        session_id : str
            Unique identifier for the chat session.
        """
        with self._lock:
            self._store.pop(session_id, None)

    # ------------------------------------------------------------------
    # Utility operations
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        """
        Remove all sessions and their histories from the store.

        Intended primarily for test setup/teardown or administrative resets.
        """
        with self._lock:
            self._store.clear()

    def has_session(self, session_id: str) -> bool:
        """
        Check whether a session exists in the store.

        Parameters
        ----------
        session_id : str

        Returns
        -------
        bool
            True if the session has any stored messages.
        """
        with self._lock:
            return session_id in self._store

    def __len__(self) -> int:
        """
        Return the number of active sessions in the store.
        """
        with self._lock:
            return len(self._store)


# Global singleton used by the application.
# The max_messages_per_session limit can be tuned based on memory and
# conversation needs. A modest default prevents unbounded growth.
session_store = SessionStore(max_messages_per_session=200)
