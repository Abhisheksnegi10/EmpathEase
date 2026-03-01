"""
EmpathEase FastAPI Application Entry Point

This is the main application file that initializes FastAPI,
sets up middleware, routes, and event handlers.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.api.routes import emotion, privacy, vocal
from app.api.routes import ws as ws_route

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting EmpathEase backend...")
    
    yield
    
    # Shutdown
    logger.info("Shutting down EmpathEase backend...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Affective computing system for emotional support and therapeutic interaction",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js frontend
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:5174",  # Vite alternate port
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# Auth disabled until database is configured
# TODO: from app.api.routes import auth
# app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(emotion.router, prefix="/api/v1", tags=["Emotion"])
app.include_router(privacy.router, prefix="/api/v1", tags=["Privacy"])
app.include_router(vocal.router, prefix="/api/v1", tags=["Vocal"])
app.include_router(ws_route.router, tags=["WebSocket"])

# Serve frontend build in production (API-only in dev; Vite handles frontend)
frontend_build = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True), name="frontend")
else:
    logger.info("Frontend build not found at %s — API-only mode", frontend_build)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "status": "healthy",
        "environment": settings.app_env,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "api": "up",
        },
    }
