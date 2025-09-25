from __future__ import annotations
import os
import hmac
import hashlib
from dotenv import load_dotenv

load_dotenv()

ITERATIONS = int(os.getenv("PBKDF2_ITERATIONS", "1000"))
SALT_BYTES = int(os.getenv("PBKDF2_SALT_BYTES", "16"))
KEY_LEN = int(os.getenv("PBKDF2_KEY_LEN", "32"))


def _pbkdf2(password: str, salt_bytes: bytes):
    return hashlib.pbkdf2_hmac(
        "sha256",               # PRF (pseudo-random function)
        password.encode("utf-8"),  # input password → bytes
        salt_bytes,             # random salt (bytes)
        ITERATIONS,             # cost factor
        dklen=KEY_LEN           # derived key length
    )


def hash_password(password: str):
    salt = os.urandom(SALT_BYTES)
    dk = _pbkdf2(password, salt)
    return dk.hex(), salt.hex()


def verify_password(password: str, hash_hex: str, salt_hex: str):
    computed = _pbkdf2(password, bytes.fromhex(salt_hex)).hex()
    return hmac.compare_digest(computed, hash_hex)
