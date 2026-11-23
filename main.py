import json
import os
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel, Field # type: ignore
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

STATS_FILE = Path(os.getenv("STATS_FILE_PATH", "/app/data/stats.json"))
_stats_lock = Lock()


def _read_generations() -> int:
    if not STATS_FILE.exists():
        return 0
    try:
        data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        return int(data.get("generations", 0))
    except Exception:
        return 0


def _increment_generations() -> int:
    with _stats_lock:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        next_count = _read_generations() + 1
        STATS_FILE.write_text(
            json.dumps({"generations": next_count}),
            encoding="utf-8",
        )
        return next_count


class BuildRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1500)


class BuildResponse(BaseModel):
    plan: str
    html: str
    qaReport: str
    generations: int


class StatsResponse(BaseModel):
    generations: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    return StatsResponse(generations=_read_generations())


@app.post("/api/build", response_model=BuildResponse)
def build_ui(payload: BuildRequest) -> BuildResponse:
    try:
        result = run_three_agents(payload.prompt.strip())
        total_generations = _increment_generations()
        return BuildResponse(
            plan=result["plan"],
            html=result["html"],
            qaReport=result["qaReport"],
            generations=total_generations,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur pipeline agents : {exc}") from exc