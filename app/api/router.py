from fastapi import APIRouter
from app.api.v1.endpoints import detect, health

api_router = APIRouter()

# Include version 1 routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(detect.router, prefix="/detect", tags=["Detection"])

