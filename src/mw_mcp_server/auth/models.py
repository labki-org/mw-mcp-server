from pydantic import BaseModel
from typing import List

class UserContext(BaseModel):
    username: str
    roles: List[str]
    scopes: List[str]
    client_id: str

