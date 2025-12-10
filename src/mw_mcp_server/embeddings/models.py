from pydantic import BaseModel

class IndexedDocument(BaseModel):
    page_title: str
    section_id: str | None = None
    text: str
    namespace: int
    last_modified: str | None = None
