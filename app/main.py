from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.exceptions import VisionServiceException, vision_exception_handler
from app.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Ensure static directories exist
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="High-performance Computer Vision backend service for automated target hole detection, scoring, and dispersion analytics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 1. Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Register Global Exception Handlers
app.add_exception_handler(VisionServiceException, vision_exception_handler)

# 3. Mount Static Files (Uploads & Annotated Image Results)
app.mount("/static", StaticFiles(directory=str(settings.BASE_DIR / "static")), name="static")

# 4. Include API Routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root access directly to Interactive OpenAPI Docs."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

