from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class User(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    password: str
    role: str = "CITIZEN"
    created_at: datetime = Field(default_factory=datetime.utcnow)