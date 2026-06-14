from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    backtest,
    cross_section,
    evaluation,
    experiments,
    factors,
    features,
    health,
    jobs,
    market_data,
    prediction,
    rl,
    signals,
    strategy,
)


app = FastAPI(
    title="StockTrader API",
    description="FastAPI backend for the StockTrader quantitative trading workspace.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(market_data.router, prefix="/api")
app.include_router(features.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")
app.include_router(strategy.router, prefix="/api")
app.include_router(signals.router, prefix="/api")
app.include_router(evaluation.router, prefix="/api")
app.include_router(backtest.router, prefix="/api")
app.include_router(rl.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(cross_section.router, prefix="/api")
app.include_router(factors.router, prefix="/api")


@app.get("/")
def root() -> dict:
    return {
        "service": "stocktrader-api",
        "docs": "/docs",
        "health": "/api/health",
    }
