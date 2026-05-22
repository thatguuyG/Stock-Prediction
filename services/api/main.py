"""FastAPI app factory for the Phase 3.5 dashboard shim."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.routers import equity, health, orders, positions, signals


def create_app() -> FastAPI:
    application = FastAPI(
        title="Stock-Prediction Dashboard API",
        version="0.1.0",
        description="Read-only shim feeding the Next.js dashboard.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    application.include_router(health.router, tags=["health"])
    application.include_router(positions.router, tags=["positions"])
    application.include_router(signals.router, tags=["signals"])
    application.include_router(orders.router, tags=["orders"])
    application.include_router(equity.router, tags=["equity"])
    return application


app = create_app()
