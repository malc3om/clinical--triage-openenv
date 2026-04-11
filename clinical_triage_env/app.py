"""
app.py — FastAPI server for ClinicalTriageEnv.

Exposes the full OpenEnv HTTP API:
  GET  /           → Health check
  POST /reset      → Start new episode
  POST /step       → Execute one action
  GET  /state      → Current episode state
  GET  /tasks      → List all tasks
  POST /grade      → Grade an episode history
  GET  /health     → Health check (alias)

Runs on port 7860 for HF Spaces compatibility.
"""

from __future__ import annotations

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clinical_triage_env.models import (
    TriageAction,
    TriageObservation,
    TriageState,
    GradeResult,
)
from clinical_triage_env.server.environment import ClinicalTriageEnvironment

# ─── Create FastAPI app ─────────────────────────────────────────────────

app = FastAPI(
    title="ClinicalTriageEnv",
    description=(
        "An OpenEnv environment simulating Emergency Department triage. "
        "AI agent assesses patients, orders diagnostics, assigns ESI levels, "
        "and makes disposition decisions under time pressure and resource constraints."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Dashboard (if built)
dashboard_out_path = os.path.join(os.path.dirname(__file__), "..", "dashboard_out")
if os.path.exists(dashboard_out_path):
    app.mount("/dashboard", StaticFiles(directory=dashboard_out_path, html=True), name="dashboard")

# Session store for per-request environments
envs: dict[str, ClinicalTriageEnvironment] = {}
default_env = ClinicalTriageEnvironment()

# Keep track of websocket connected clients
active_websockets: list[WebSocket] = []


# ─── Request/Response Models ────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_stemi_code"
    episode_id: Optional[str] = None


class GradeRequest(BaseModel):
    task_id: str
    episode_history: list[dict]


class StepRequest(BaseModel):
    action_type: str
    patient_id: str
    parameter: str
    rationale: Optional[str] = None
    episode_id: Optional[str] = None


# ─── Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check — returns 200 with environment info."""
    return {
        "status": "ok",
        "env": "clinical-triage-env",
        "version": "1.0.0",
        "description": "Emergency Department Triage Environment for OpenEnv",
    }


@app.get("/health")
async def health():
    """Health check alias."""
    return {"status": "healthy"}



@app.post("/reset", response_model=TriageObservation)
async def reset(request: Optional[ResetRequest] = None):
    """Reset the environment for a new episode."""
    task_id = request.task_id if request else "task_stemi_code"
    episode_id = request.episode_id if request else None
    try:
        e = ClinicalTriageEnvironment()
        obs = e.reset(task_id=task_id, episode_id=episode_id)
        
        current_episode_id = e.state.episode_id
        if current_episode_id:
            envs[current_episode_id] = e
            
        global default_env
        default_env = e
        
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=TriageObservation)
async def step(request: StepRequest):
    """Execute one action in the environment."""
    e = envs.get(request.episode_id, default_env) if request.episode_id else default_env
    try:
        action = TriageAction(
            action_type=request.action_type,
            patient_id=request.patient_id,
            parameter=request.parameter,
            rationale=request.rationale,
        )
        result = e.step(action)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def get_state(episode_id: Optional[str] = None):
    """Return current episode state."""
    e = envs.get(episode_id, default_env) if episode_id else default_env
    return e.state.model_dump()



@app.get("/tasks")
async def list_tasks():
    """Return list of all available tasks with descriptions."""
    tasks = default_env.get_tasks()
    return {
        "tasks": [t.model_dump() for t in tasks],
        "action_schema": {
            "action_type": {
                "type": "enum",
                "values": [
                    "order_diagnostic",
                    "assign_esi_level",
                    "activate_pathway",
                    "disposition",
                    "request_consult",
                    "administer_medication",
                    "assign_bed",
                    "wait",
                ],
                "description": "Type of clinical action to take",
            },
            "patient_id": {
                "type": "string",
                "description": "ID of the patient this action targets (e.g. P1, P2)",
            },
            "parameter": {
                "type": "string",
                "description": "Action parameter: test name, ESI level (1-5), pathway type, disposition type (admit/discharge/transfer)",
            },
            "rationale": {
                "type": "string",
                "description": "Optional: agent's reasoning for this action",
            },
        },
    }


@app.post("/grade", response_model=GradeResult)
async def grade(request: GradeRequest):
    """Grade an episode history using the deterministic grader."""
    try:
        from clinical_triage_env.server.graders.stemi_grader import grade_stemi
        from clinical_triage_env.server.graders.chest_workup_grader import grade_chest_workup
        from clinical_triage_env.server.graders.mci_grader import grade_mci
        from clinical_triage_env.server.graders.sepsis_grader import grade_sepsis
        from clinical_triage_env.server.graders.stroke_grader import grade_stroke
        from clinical_triage_env.server.graders.pediatric_grader import grade_pediatric

        graders = {
            "task_stemi_code": grade_stemi,
            "task_chest_pain_workup": grade_chest_workup,
            "task_mci_surge": grade_mci,
            "task_sepsis_alert": grade_sepsis,
            "task_stroke_code": grade_stroke,
            "task_pediatric_resp": grade_pediatric,
        }

        grader = graders.get(request.task_id)
        if grader is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id: {request.task_id}",
            )

        result = grader(request.episode_history)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time dashboard updates."""
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("type")
            episode_id = data.get("episode_id")
            e = envs.get(episode_id, default_env) if episode_id else default_env
            
            if command == "reset":
                task_id = data.get("task_id", "task_stemi_code")
                obs = e.reset(task_id=task_id, episode_id=episode_id)
                await websocket.send_json({
                    "type": "observation",
                    "data": obs.model_dump()
                })
                
            elif command == "step":
                action_data = data.get("action", {})
                action = TriageAction(
                    action_type=action_data.get("action_type"),
                    patient_id=action_data.get("patient_id"),
                    parameter=action_data.get("parameter"),
                    rationale=action_data.get("rationale"),
                )
                obs = e.step(action)
                await websocket.send_json({
                    "type": "observation",
                    "data": obs.model_dump()
                })
                
            elif command == "agent_token":
                # This allow external inference scripts to stream tokens to the UI via the backend
                # Broadcast token to all OTHER clients (e.g. the dashboard)
                token = data.get("content")
                source = data.get("source", "thought")
                for client in active_websockets:
                    if client != websocket:
                        try:
                            await client.send_json({
                                "type": "token",
                                "content": token,
                                "source": source
                            })
                        except:
                            pass

    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        print(f"WS Error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


# ─── Run server ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
