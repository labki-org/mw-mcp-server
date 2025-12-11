from mw_mcp_server.api.models import ChatRequest, ChatMessage

def test_chat_request_context_default():
    """Verify default context is 'chat'."""
    req = ChatRequest(messages=[ChatMessage(role="user", content="hello")])
    assert req.context == "chat"

def test_chat_request_context_editor():
    """Verify context can be set to 'editor'."""
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hello")], 
        context="editor"
    )
    assert req.context == "editor"

def test_chat_request_context_invalid():
    """Verify invalid context raises validation error."""
    try:
        ChatRequest(
            messages=[ChatMessage(role="user", content="hello")], 
            context="invalid"
        )
        assert False, "Should have raised validation error"
    except ValueError:
        pass
