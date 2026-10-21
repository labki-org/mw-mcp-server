"""
Context-window budgeting for chat requests.

Chat history is resent in full on every turn and tool results (e.g. a full
page's wikitext) are appended verbatim, so a long session eventually exceeds
the model's context window and every further request fails. These helpers
keep the message list under a token budget:

1. Each tool result is capped to a fixed size before it is appended.
2. Before each LLM call, older tool results from *this* turn are elided
   (oldest first), then the oldest history messages are dropped, until the
   estimate fits. The system prompt is sent separately and never trimmed,
   and the current turn's user messages are always kept.

Token counts use tiktoken when its encoding is available and fall back to a
~4 chars/token estimate otherwise (tiktoken downloads encodings on first use,
which can fail offline).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("mcp.llm.context")

# Per-message framing overhead in the chat format (role, separators).
_MESSAGE_OVERHEAD_TOKENS = 4

ELIDED_TOOL_RESULT = json.dumps(
    {"omitted": "Earlier tool result removed to stay within the context budget. "
                "Call the tool again if you still need it."}
)


@lru_cache(maxsize=8)
def _encoder_for(model: str) -> Optional[Callable[[str], List[int]]]:
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        # Wiki text can legitimately contain strings like "<|endoftext|>".
        return lambda text: enc.encode(text, disallowed_special=())
    except Exception:
        logger.warning("tiktoken encoding unavailable for %s; using char estimate", model)
        return None


def count_text_tokens(text: str, model: str = "gpt-4o") -> int:
    if not text:
        return 0
    encode = _encoder_for(model)
    if encode is None:
        return len(text) // 4 + 1
    return len(encode(text))


def count_message_tokens(message: Dict[str, Any], model: str = "gpt-4o") -> int:
    tokens = _MESSAGE_OVERHEAD_TOKENS + count_text_tokens(
        message.get("content") or "", model
    )
    tool_calls = message.get("tool_calls")
    if tool_calls:
        tokens += count_text_tokens(json.dumps(tool_calls, default=str), model)
    return tokens


def truncate_tool_content(content: str, max_chars: int) -> str:
    """Cap a serialized tool result, telling the model it was cut."""
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    omitted = len(content) - max_chars
    return (
        content[:max_chars]
        + f"\n...[truncated {omitted} characters to fit the context budget; "
        "request a narrower slice if you need more]"
    )


def fit_messages_to_budget(
    messages: List[Dict[str, Any]],
    budget_tokens: int,
    turn_start: int,
    model: str = "gpt-4o",
) -> int:
    """Trim *messages* in place to fit *budget_tokens*; return the new turn_start.

    ``messages[:turn_start]`` is prior-session history (user/assistant only);
    ``messages[turn_start:]`` is the current turn (new user messages, then
    assistant tool calls and tool results). Steps, until it fits:

    1. Replace the content of the oldest not-yet-elided tool results in the
       current turn, leaving the most recent one intact. The assistant
       tool_call / tool message pairing is preserved.
    2. Drop the oldest history message.

    If it still doesn't fit (a single huge user message), it is left as is and
    the provider's error is surfaced.
    """
    if budget_tokens <= 0:
        return turn_start

    sizes = [count_message_tokens(m, model) for m in messages]
    total = sum(sizes)
    if total <= budget_tokens:
        return turn_start

    tool_indices = [
        i for i in range(turn_start, len(messages)) if messages[i].get("role") == "tool"
    ]
    for i in tool_indices[:-1]:
        if total <= budget_tokens:
            break
        if messages[i].get("content") == ELIDED_TOOL_RESULT:
            continue
        messages[i] = {**messages[i], "content": ELIDED_TOOL_RESULT}
        new_size = count_message_tokens(messages[i], model)
        total += new_size - sizes[i]
        sizes[i] = new_size

    dropped = 0
    while total > budget_tokens and turn_start > 0:
        total -= sizes.pop(0)
        messages.pop(0)
        turn_start -= 1
        dropped += 1

    # Don't open the history with an orphaned assistant reply.
    while turn_start > 0 and messages[0].get("role") == "assistant":
        total -= sizes.pop(0)
        messages.pop(0)
        turn_start -= 1
        dropped += 1

    if dropped:
        logger.info("Dropped %d oldest history message(s) to fit context budget", dropped)
    if total > budget_tokens:
        logger.warning(
            "Messages still exceed context budget after trimming (%d > %d tokens)",
            total, budget_tokens,
        )
    return turn_start
