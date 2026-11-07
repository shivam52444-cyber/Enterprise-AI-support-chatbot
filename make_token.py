"""
make_token.py - plays the role of the E-COMMERCE WEBSITE'S BACKEND.

In real life the website logs the customer in, then creates this token for them.
Here we do it by hand so we can test the chatbot API.

Usage:
    python fake_host/make_token.py <customer_unique_id> [minutes_valid]

The chatbot API must never import this file. They share only the secret
(HOST_SHARED_SECRET in .env), just like the real website and the chatbot would.
"""

import os
import sys
import time

import jwt


def make_token(customer_unique_id, minutes=15, secret=None):
    secret = secret or os.environ.get("HOST_SHARED_SECRET")
    if not secret:
        raise RuntimeError("HOST_SHARED_SECRET is not set (put it in your .env file)")
    now = int(time.time())
    payload = {
        "sub": customer_unique_id,      # who the customer is
        "iat": now,                     # issued at
        "exp": now + minutes * 60,      # expires: a stolen token stops working soon
    }
    return jwt.encode(payload, secret, algorithm="HS256")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if len(sys.argv) < 2:
        sys.exit("Usage: python fake_host/make_token.py <customer_unique_id> [minutes_valid]")
    try:
        minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        print(make_token(sys.argv[1], minutes))
    except (ValueError, RuntimeError) as e:
        sys.exit(f"Error: {e}")
