from typing import Dict, List
from ..api.models import ChatMessage

class SessionStore:
    def __init__(self):
        # Maps session_id -> List[ChatMessage]
        self._store: Dict[str, List[ChatMessage]] = {}

    def get_history(self, session_id: str) -> List[ChatMessage]:
        return self._store.get(session_id, [])

    def add_messages(self, session_id: str, new_messages: List[ChatMessage]):
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].extend(new_messages)

    def clear(self, session_id: str):
        if session_id in self._store:
            del self._store[session_id]

# Global singleton
session_store = SessionStore()
