import httpx
import jwt
import time
import os
from dotenv import load_dotenv

# 2. Activate it
# source .venv/bin/activate
# 3. Install deps
# pip install -r requirements.txt
# 4. Run the script
# export PYTHONPATH=src
# python3 test_connection.py

# Load env vars
load_dotenv()

SERVER_URL = "http://localhost:8000"
JWT_SECRET = os.getenv("JWT_SECRET", "some-long-random-secret")
JWT_ALGO = os.getenv("JWT_ALGO", "HS256")

def create_token():
    payload = {
        "sub": "TestUser",
        "roles": ["user"],
        "client_id": "test_script",
        "iat": int(time.time())
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def test_smw_query():
    token = create_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Simple query that should return something or empty, but not error if connected
    # Asking for the Main Page usually exists.
    payload = {
        "ask": "[[Main Page]]" 
    }
    
    print(f"Testing connectivity to {SERVER_URL}...")
    print(f"Using JWT with secret: {JWT_SECRET[:5]}...")
    
    try:
        resp = httpx.post(
            f"{SERVER_URL}/smw-query/",
            json=payload,
            headers=headers,
            timeout=10
        )
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print("Success! Response from MediaWiki via MCP:")
            print(resp.json())
        else:
            print("Error response:")
            print(resp.text)
            
    except httpx.ConnectError:
        print("Could not connect to MCP server. Is it running on port 8000?")
    except Exception as e:
        print(f"An error occurred: {e}")

def test_chat():
    token = create_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    payload = {
        "messages": [{"role": "user", "content": "You have access to a tool called 'mw_get_page'. Use it to fetch the content of 'Main Page'."}],
        "max_tokens": 512
    }
    
    print(f"\nTesting LLM /chat endpoint (expecting tool usage)...")
    try:
        resp = httpx.post(
            f"{SERVER_URL}/chat/",
            json=payload,
            headers=headers,
            timeout=60
        )
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print("Success! Response from OpenAI via MCP:")
            data = resp.json()
            print("Response Message:", data["messages"][-1]["content"])
            print("Used Tools:", data["used_tools"])
        else:
            print("Error response:")
            print(resp.text)
    except Exception as e:
        print(f"Chat test error: {e}")

def test_session_history():
    token = create_token()
    headers = {"Authorization": f"Bearer {token}"}
    session_id = "test-session-123"
    
    print(f"\nTesting Session History (Session ID: {session_id})...")
    
    # 1. Set context
    print("1. Telling LLM my name...")
    payload1 = {
        "messages": [{"role": "user", "content": "My name is Daniel."}],
        "session_id": session_id
    }
    resp1 = httpx.post(f"{SERVER_URL}/chat/", json=payload1, headers=headers, timeout=30)
    print(f"   Response 1: {resp1.json()['messages'][-1]['content']}")
    
    # 2. Ask for context
    print("2. Asking what my name is...")
    payload2 = {
        "messages": [{"role": "user", "content": "What is my name?"}],
        "session_id": session_id
    }
    resp2 = httpx.post(f"{SERVER_URL}/chat/", json=payload2, headers=headers, timeout=30)
    ans = resp2.json()['messages'][-1]['content']
    print(f"   Response 2: {ans}")
    
    if "Daniel" in ans:
        print("SUCCESS: LLM remembered context!")
    else:
        print("FAILURE: LLM did not remember context.")

if __name__ == "__main__":
    # test_smw_query()
    # test_chat()
    test_session_history()
