"""FastAPI app factory for the Phase 3.5 dashboard shim."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.routers import equity, health, orders, positions, signals


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stock-Prediction Dashboard API",
        version="0.1.0",
        description="Read-only shim feeding the Next.js dashboard.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(health.router, tags=["health"])
    app.include_router(positions.router, tags=["positions"])
    app.include_router(signals.router, tags=["signals"])
    app.include_router(orders.router, tags=["orders"])
    app.include_router(equity.router, tags=["equity"])
    return app


app = create_app()
