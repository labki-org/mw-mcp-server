import jwt
import os
from dotenv import load_dotenv

load_dotenv()

secret = os.getenv("JWT_MW_TO_MCP_SECRET")
print(f"Secret length: {len(secret)}")
print(f"Secret prefix: {secret[:10]}")

# Fixed timestamp and payload
now = 1234567890
payload = {
    "iss": "MWAssistant",
    "aud": "mw-mcp-server",
    "iat": now,
    "exp": now + 30,
    "user": "test_user",
    "roles": ["bureaucrat"],
    "scope": ["test_scope"]
}

encoded = jwt.encode(payload, secret, algorithm="HS256")
print(f"Token: {encoded}")
