from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class WorkspaceCreate(BaseModel):
    """What the user sends when creating a workspace"""
    name: str
    description: Optional[str] = None

class Workspace(BaseModel):
    """What we return to the user"""
    id: str
    name: str
    description: Optional[str]
    created_at: datetime