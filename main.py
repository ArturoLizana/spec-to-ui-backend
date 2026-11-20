import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agents_pipeline import run_three_agents

app = FastAPI(title="Spec-to-UI Multi-Agents API", version="1.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://projets.arlidev.fr,https://www.projets.arlidev.fr,http://localhost:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BuildRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1500)


class BuildResponse(BaseModel):
    plan: str
    html: str
    qaReport: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/build", response_model=BuildResponse)
def build_ui(payload: BuildRequest) -> BuildResponse:
    try:
        result = run_three_agents(payload.prompt.strip())
        return BuildResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur pipeline agents : {exc}") from exc