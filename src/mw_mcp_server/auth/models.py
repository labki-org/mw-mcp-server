from pydantic import BaseModel
from typing import List

class UserContext(BaseModel):
    username: str
    roles: List[str]
    client_id: str
