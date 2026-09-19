from fastapi import FastAPI

from backend.app.api import grading, health, rag

app = FastAPI(title="AI-Assisted Answer Sheet Grading System")

app.include_router(health.router)
app.include_router(grading.router)
app.include_router(rag.router)
