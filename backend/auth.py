"""Authentication + RBAC. Stdlib crypto only (PBKDF2 + HMAC tokens).

Roles:
  admin    -> everything: user management, upload, reindex, chat, dashboard
  uploader -> upload documents, chat, dashboard
  viewer   -> chat and dashboard (read-only)

A default admin (admin / admin123) is created on first run — change it
immediately via the Admin page.
"""
import base64
import hashlib
import hmac
import json
import secrets
import time

import config
import store

ROLES = ("admin", "uploader", "viewer")
TOKEN_TTL = 60 * 60 * 12  # 12 hours


# ---------------------------------------------------------------- passwords

def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, salt), stored)


# ---------------------------------------------------------------- tokens

def _b64e(d: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")


def _b64d(s: str) -> dict:
    pad = "=" * (-len(s) % 4)
    return json.loads(base64.urlsafe_b64decode(s + pad))


def _sign(payload_b64: str) -> str:
    return hmac.new(config.SECRET_KEY.encode(), payload_b64.encode(),
                    hashlib.sha256).hexdigest()


def create_token(username: str, role: str) -> str:
    payload = _b64e({"u": username, "r": role, "exp": int(time.time()) + TOKEN_TTL})
    return f"{payload}.{_sign(payload)}"


def decode_token(token: str) -> dict | None:
    """Returns {'u','r','exp'} or None if invalid/expired."""
    try:
        payload_b64, sig = token.split(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(payload_b64)):
        return None
    try:
        payload = _b64d(payload_b64)
    except Exception:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload


# ---------------------------------------------------------------- users

def ensure_default_admin():
    if not store.list_users():
        store.create_user("admin", hash_password("admin123"), "admin")


def authenticate(username: str, password: str):
    u = store.get_user(username)
    if u and verify_password(password, u["password_hash"]):
        return u
    return None
