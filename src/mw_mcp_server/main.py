from fastapi import FastAPI
from .config import settings
from .api import chat_routes, search_routes, smw_routes, action_routes, health_routes, embedding_routes

app = FastAPI(title="mw-mcp-server")

app.include_router(chat_routes.router)
app.include_router(search_routes.router)
app.include_router(smw_routes.router)
app.include_router(action_routes.router)
app.include_router(health_routes.router)
app.include_router(embedding_routes.router)
