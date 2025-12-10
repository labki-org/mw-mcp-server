import jwt

# This matches the PHP script's hardcoded secret
secret = "8n7yHEg3UttL-lEOKASg-dS_xkU0gTuqGLn7zvhg4Uh-x52rtA0Zh13WJmGd8ojDjxXJB7qR9U"

# Logic from security.py
def verify_token(token):
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="mw-mcp-server",
            issuer="MWAssistant",
            options={
                "require": ["iss", "aud", "iat", "exp", "user", "roles", "scope"],
                "verify_exp": False # debugging, though it should be valid
            }
        )
        print("SUCCESS: Token verified!")
        print(payload)
    except Exception as e:
        print(f"FAILURE: {str(e)}")
        
        # Debugging signature calculation manually
        parts = token.split('.')
        header_b64 = parts[0]
        payload_b64 = parts[1]
        sig_received = parts[2]
        
        import hmac
        import hashlib
        import base64
        
        msg = (header_b64 + "." + payload_b64).encode('utf-8')
        # Handle secret as bytes assuming utf-8 encoding (default for PyJWT string secrets)
        secret_bytes = secret.encode('utf-8')
        
        sig_calc_bytes = hmac.new(secret_bytes, msg, hashlib.sha256).digest()
        
        # standard base64 url encode without padding
        sig_calc_b64 = base64.urlsafe_b64encode(sig_calc_bytes).decode('utf-8').rstrip('=')
        
        print(f"\nDebug Info:")
        print(f"Header: {header_b64}")
        print(f"Payload: {payload_b64}")
        print(f"Secret (len {len(secret)}): {secret}")
        print(f"Sig Received: {sig_received}")
        print(f"Sig Calculated: {sig_calc_b64}")

# The PHP-generated token you provided
php_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJNV0Fzc2lzdGFudCIsImF1ZCI6Im13LW1jcC1zZXJ2ZXIiLCJpYXQiOjEyMzQ1Njc4OTAsImV4cCI6MTIzNDU2NzkyMCwidXNlciI6InRlc3RfdXNlciIsInJvbGVzIjpbImJ1cmVhdWNyYXQiXSwic2NvcGUiOlsidGVzdF9zY29wZSJdfQ.UD0JPOGPqy0i5vtoJF_N3hPiCzy_MzOy6lfTTetXcTU"

verify_token(php_token)
