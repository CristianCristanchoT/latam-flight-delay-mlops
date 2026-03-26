import logging
import time

import fastapi
import pandas as pd
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Literal

from challenge.model import DelayModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

KNOWN_AIRLINES = {
    "Aerolineas Argentinas",
    "Aeromexico",
    "Air Canada",
    "Air France",
    "Alitalia",
    "American Airlines",
    "Austral",
    "Avianca",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Grupo LATAM",
    "Iberia",
    "JetSmart SPA",
    "K.L.M.",
    "Lacsa",
    "Latin American Wings",
    "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas",
    "Qantas Airways",
    "Sky Airline",
    "United Airlines",
}


class FlightInput(BaseModel):
    OPERA: str
    TIPOVUELO: Literal["N", "I"]
    MES: int

    @validator("MES")
    def mes_must_be_valid(cls, v: int) -> int:
        if not (1 <= v <= 12):
            raise ValueError("MES must be between 1 and 12")
        return v

    @validator("OPERA")
    def opera_must_be_known(cls, v: str) -> str:
        if v not in KNOWN_AIRLINES:
            raise ValueError(f"Unknown airline: '{v}'")
        return v


class PredictRequest(BaseModel):
    flights: List[FlightInput]


app = fastapi.FastAPI()
_model = DelayModel()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    flights_data = [flight.dict() for flight in request.flights]
    logging.info("Predict request received | flights=%s", flights_data)

    df = pd.DataFrame(flights_data)

    t0 = time.perf_counter()
    features = _model.preprocess(df)
    predictions = _model.predict(features)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logging.info("Predict result | predictions=%s | inference_time=%.2fms", predictions, elapsed_ms)
    return {"predict": predictions}