"""
auth.py - checks the signed token that the e-commerce website gives the customer.

The chatbot NEVER creates tokens and never logs anyone in. It only VERIFIES:
  - the signature (made with the secret shared with the website's backend)
  - that the token has not expired
and then reads the customer id from the token's "sub" field.
The customer id is never taken from the request body or from the chat message.
"""

import jwt

MIN_SECRET_LENGTH = 32


class AuthError(Exception):
    """The token is missing, invalid, tampered with, or expired -> HTTP 401."""


def verify_token(token, secret):
    """Return the customer_unique_id ('sub') from a valid token, or raise AuthError."""
    if not secret or len(secret) < MIN_SECRET_LENGTH:
        # Server problem, not the caller's fault: never accept tokens with a weak secret.
        raise RuntimeError(f"HOST_SHARED_SECRET is missing or shorter than {MIN_SECRET_LENGTH} characters")
    if not token or not isinstance(token, str):
        raise AuthError("Missing token")
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],                   # fixed list: blocks the 'alg: none' trick
            options={"require": ["exp", "sub"]},    # a token without expiry or id is rejected
            leeway=5,                               # 5 seconds of clock difference is fine
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except jwt.PyJWTError:
        raise AuthError("Invalid token")

    customer_id = payload.get("sub")
    if not isinstance(customer_id, str) or not customer_id.strip() or len(customer_id) > 100:
        raise AuthError("Invalid token")
    return customer_id.strip()
