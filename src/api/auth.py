import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext


class AuthHandler:
    def __init__(self):
        self.secret_key = os.getenv(
            "JWT_SECRET", "default-secret-key-change-in-production"
        )
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        self.users_db = {
            "admin": self._get_password_hash("admin@123"),
            "auditor": self._get_password_hash("auditor@123"),
            "user": self._get_password_hash("user@123"),
        }

    def _get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        if username not in self.users_db:
            return None

        if not self.verify_password(password, self.users_db[username]):
            return None

        return username

    def create_access_token(self, username: str) -> str:
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        payload = {
            "sub": username,
            "exp": expire,
            "role": "admin" if username == "admin" else "user",
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> bool:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload.get("sub") in self.users_db
        except JWTError:
            return False

    def get_user_from_token(self, token: str) -> str:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload.get("sub", "anonymous")
        except JWTError:
            return "anonymous"
