"""
User model for authentication and profile.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class User(BaseModel):
    """User account model."""
    
    id: UUID = Field(default_factory=uuid4)
    email: str
    hashed_password: str
    display_name: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    preferences: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"<User {self.email}>"
