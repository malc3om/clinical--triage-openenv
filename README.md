---
title: Clinical Triage Env
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

<div align="center">

# 🏥 ClinicalTriageEnv

### An OpenEnv Reinforcement Learning Environment for Emergency Department Triage

*AI agents step into the role of ED clinicians — assessing patients, ordering diagnostics, assigning acuity levels, and making disposition decisions under real-time pressure.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Team Unfazed** · Meta × Scaler OpenEnv Hackathon

---

</div>

## Why This Exists

Emergency triage is one of the highest-stakes sequential decision tasks in medicine:

- **Wrong triage kills** — a delayed STEMI costs heart muscle every minute past the 90-minute door-to-balloon window
- **Over-triage gridlocks the ED** — assigning ESI-1 to a stable patient blocks resuscitation bays
- **No standardized RL benchmark exists** for clinical decision-making under uncertainty

ClinicalTriageEnv is a **fully synthetic**, HIPAA-free simulation that benchmarks AI agents on **six clinical scenarios** of increasing difficulty — from a clear-cut STEMI to a five-patient mass casualty surge with only three beds, plus sepsis management, stroke code, and pediatric respiratory emergencies.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ClinicalTriageEnv                          │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │   Patient     │    │   Dynamic    │    │    Deterministic   │    │
│  │  Generator    │───▶│   Vitals     │───▶│     Graders        │    │
│  │  (6 tasks)    │    │   Engine     │    │  (6 per-task)      │    │
│  └──────────────┘    └──────────────┘    └────────────────────┘    │
│         │                    │                      │               │
│         ▼                    ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Environment Engine (step / reset / grade)       │   │
│  │     Dense Reward · Fatal Delay Detection · ESI Logic         │   │
│  │     Stochastic Patient Vitals · Resource Scarcity            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│        REST API         WebSocket       inference.py                │
│       (FastAPI)         (Real-time)    (LLM ReAct Agent)            │
│       :7860             :7860/ws      100% LLM decisions            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

### 🎲 Stochastic Patient Generation
Patient vitals vary between episodes via seeded randomness — heart rate, blood pressure, SpO2, temperature, and onset times all fluctuate within clinically realistic ranges. No two episodes are identical, preventing memorization.

### 🧬 Dynamic Vitals Engine
Patient vitals deteriorate over time based on their condition. A STEMI patient's blood pressure drops, heart rate climbs, and SpO2 falls unless the agent intervenes. Untreated anaphylaxis causes progressive hypotension. Each action costs simulated time — ordering bloodwork burns 30 minutes, administering epinephrine takes 1 minute.

### ⏱️ Fatal Delay Detection
Miss the 90-minute door-to-balloon window for a STEMI? **-10.0 penalty, episode terminated.** Fail to treat anaphylaxis within 15 minutes? Same. Sepsis hour-1 bundle deadline? Same. Clinical time windows are enforced with hard stops.

### 📊 Dense Reward Signal
Every step returns a reward composed of 5 weighted components:

| Component | Range | What It Rewards |
|-----------|-------|-----------------|
| Clinical Correctness | +0.05 to +0.20 | Ordering indicated tests, correct ESI assignment, medications |
| Efficiency | -0.05 | Penalty per unnecessary or redundant test |
| Time Pressure | -0.01 to -0.02/step | Critical patients bleed reward each step of delay |
| Sequence Bonus | +0.05 | Following evidence-based diagnostic ordering |
| Safety Guardrails | -0.50 to -10.0 | Discharging ESI-1, fatal window violations, loop detection |

### 🧠 ReAct Clinical Reasoning Agent
The `inference.py` agent uses structured Observation → Thought → Action reasoning with `<thought>` tags. Every action decision is made by the LLM — there are **zero hardcoded action sequences** and **zero deterministic fallbacks**. Parse failures result in a simple `wait` action (costing the agent a step).

---

## Tasks

### Task 1: `task_stemi_code` · Easy
> 58-year-old male. Crushing substernal chest pain radiating to left arm. ST-elevation in leads II, III, aVF. Hypotensive, diaphoretic, HR 102.

**The agent must**: Recognize STEMI → Assign ESI-1 → Activate cath lab → Administer aspirin → Admit. All within the 90-minute window.

**Max Steps**: 15 · **Baseline**: 0.72

### Task 2: `task_chest_pain_workup` · Medium
> 44-year-old female. Pleuritic chest pain, worse with inspiration. Recent long-haul flight. On oral contraceptives.

**Differential**: PE vs ACS vs MSK. **The agent must**: Navigate the diagnostic sequence — EKG first (rule out ACS), then D-dimer, then CT-PA if positive; order matters.

**Max Steps**: 20 · **Baseline**: 0.48

### Task 3: `task_sepsis_alert` · Medium
> 68-year-old male. Fever, chills, confusion, dark urine. HR 118, BP 82/48, SpO2 92%, Temp 39.2°C, GCS 13.

**The agent must**: Recognize sepsis → Order lactate and cultures → Administer IV antibiotics and fluids within 1-hour bundle → Assign ESI-2 → Admit to ICU.

**Max Steps**: 20 · **Baseline**: 0.60

### Task 4: `task_pediatric_resp` · Medium
> 4-year-old male. Severe wheezing with intercostal retractions, not eating. HR 145, SpO2 89%, RR 42.

**The agent must**: Administer nebulized albuterol/steroids → Allow reassessment time → Assign ESI-2 → Admit for persistent hypoxia.

**Max Steps**: 18 · **Baseline**: 0.65

### Task 5: `task_stroke_code` · Hard
> 72-year-old female. Sudden onset right-sided weakness, facial droop, slurred speech. BP 185/100, GCS 14.

**The agent must**: Activate stroke pathway → STAT CT Head non-contrast → Assign ESI-1 or 2 → Admit/transfer. Time is brain — every minute counts.

**Max Steps**: 18 · **Baseline**: 0.55

### Task 6: `task_mci_surge` · Hard
> Mass casualty incident. Five patients arrive simultaneously. Three beds available.

| Patient | Presentation | Expected ESI |
|---------|-------------|--------------|
| P1 · 72M | Unresponsive, GCS 6, HR 40, hypotensive | **ESI-1** |
| P2 · 28F | Deformed forearm fracture, stable vitals | ESI-3 |
| P3 · 15M | Anaphylaxis, stridor, BP 70/40, SpO2 88% | **ESI-1** |
| P4 · 60M | Rapid AFib at 148 bpm, dizzy | ESI-2 |
| P5 · 35F | Anxiety, hyperventilation, vitals normal | ESI-4 |

**The agent must**: Triage the sickest first (P1 and P3 before anyone else), allocate beds under scarcity, and manage five concurrent patients.

**Max Steps**: 25 · **Baseline**: 0.31

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/malc3om/clinical--triage-openenv.git
cd clinical--triage-openenv
pip install -r requirements.txt
```

### 2. Run the Environment Server

```bash
uvicorn clinical_triage_env.app:app --host 0.0.0.0 --port 7860
```

### 3. Run Inference (LLM Agent)

The agent uses the evaluator-injected `API_BASE_URL` and `API_KEY` to make real LLM API calls through a ReAct clinical reasoning loop.

```bash
export API_BASE_URL="https://api.openai.com/v1"
export API_KEY="your-api-key"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

### Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
```

---

## Programmatic API

```python
from clinical_triage_env.models import TriageAction
from clinical_triage_env.server.environment import ClinicalTriageEnvironment

env = ClinicalTriageEnvironment()
obs = env.reset(task_id="task_stemi_code")

action = TriageAction(
    action_type="assign_esi_level",
    patient_id="P1",
    parameter="1",
    rationale="Acute STEMI requires ESI-1"
)
result = env.step(action)
print(f"Reward: {result.reward:.3f} | Done: {result.done}")

# Get grader score at any time
score = env.get_task_grader_score()
print(f"Grader Score: {score.score:.3f}")
```

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root — health check or dashboard redirect |
| `GET` | `/ping` | Ping endpoint (HTTP 200) |
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start new episode `{"task_id": "task_stemi_code"}` |
| `POST` | `/step` | Execute action `{"action_type": "...", "patient_id": "P1", "parameter": "..."}` |
| `GET` | `/state` | Current episode state |
| `GET` | `/tasks` | List all available tasks |
| `POST` | `/grade` | Grade an episode history |
| `WS` | `/ws` | WebSocket for real-time dashboard streaming |

---

## Action Space

| `action_type` | `parameter` examples | Time Cost |
|---------------|---------------------|-----------|
| `order_diagnostic` | `EKG`, `d_dimer`, `CT_PA`, `troponin_I`, `cbc`, `lactate`, `CT_HEAD_NONCON`, `CTA_HEAD_NECK`, `vbg`, `urinalysis`, `procalcitonin`, `CXR` | 5–45 min |
| `assign_esi_level` | `1`, `2`, `3`, `4`, `5` | 1 min |
| `activate_pathway` | `cath_lab`, `stroke`, `trauma`, `sepsis` | 2 min |
| `disposition` | `admit`, `discharge`, `transfer`, `waiting_room` | 5 min |
| `request_consult` | `cardiology`, `pulmonology`, `neurology` | 10 min |
| `administer_medication` | `epinephrine`, `aspirin_325mg`, `ceftriaxone`, `IV_fluid_bolus`, `albuterol_nebulizer`, `dexamethasone` | 1–5 min |
| `assign_bed` | `resus_bay`, `monitored`, `hallway` | 2 min |
| `wait` | `""` | 15 min |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Current task identifier |
| `task_difficulty` | `str` | `easy`, `medium`, or `hard` |
| `step_number` / `max_steps` | `int` | Progress tracking |
| `elapsed_minutes` | `int` | Simulated time elapsed |
| `patients` | `List[Patient]` | All patients with vitals, labs, history |
| `available_beds` | `int` | Remaining beds |
| `reward` | `float` | Step reward |
| `reward_components` | `dict` | Breakdown of reward signal |
| `last_action_result` | `str` | Feedback from last action |
| `last_action_error` | `str` | Error message if action failed |
| `done` | `bool` | Episode termination flag |

Each patient includes: `vitals` (HR, BP, SpO2, RR, Temp, GCS), `vitals_trend` (↑/↓/→ per vital), `medical_history`, `current_medications`, `available_labs`, `pending_labs`, `imaging_available`, `pending_imaging`, and `resource_tokens_remaining`.

---

## Inference Output Format

```
[START] task_name=task_stemi_code task_id=task_stemi_code
[STEP] step=1 action={"action_type":"assign_esi_level",...} observation={...} reward=0.10
[STEP] step=2 action={"action_type":"activate_pathway",...} observation={...} reward=0.20
[STEP] step=3 action={"action_type":"order_diagnostic",...} observation={...} reward=0.15
[STEP] step=4 action={"action_type":"disposition",...} observation={...} reward=0.15
[END] task_name=task_stemi_code score=0.900
```

---

## Baseline Scores

| Task | Difficulty | Baseline Score |
|------|-----------|---------------|
| `task_stemi_code` | Easy | 0.72 |
| `task_chest_pain_workup` | Medium | 0.48 |
| `task_sepsis_alert` | Medium | 0.60 |
| `task_pediatric_resp` | Medium | 0.65 |
| `task_stroke_code` | Hard | 0.55 |
| `task_mci_surge` | Hard | 0.31 |
| **Average** | | **0.55** |

---

## Project Structure

```
clinical-triage-env/
├── inference.py                    # LLM-first ReAct clinical reasoning agent (zero hardcoding)
├── openenv.yaml                    # OpenEnv task manifest (6 tasks)
├── requirements.txt                # Python dependencies (single source of truth)
├── pyproject.toml                  # Package metadata
├── Dockerfile                      # Multi-stage build (non-root user)
├── .dockerignore                   # Excludes __pycache__, .git, lock files
├── validate_submission.py          # Pre-submission validation (41 checks)
├── run_demo.py                     # Full demo orchestrator
│
├── clinical_triage_env/            # Core Python package
│   ├── app.py                      # FastAPI server (REST + WebSocket + /ping)
│   ├── models.py                   # Pydantic v2 models (observation/action/state)
│   └── server/
│       ├── environment.py          # Main environment engine
│       ├── reward.py               # 5-component dense reward (all 6 tasks)
│       ├── patient_generator.py    # Stochastic patient generation (6 tasks)
│       ├── vitals_engine.py        # Dynamic vitals deterioration
│       ├── time_costs.py           # Action → simulated time mapping
│       └── graders/                # Deterministic graders (one per task)
│           ├── stemi_grader.py
│           ├── chest_workup_grader.py
│           ├── mci_grader.py
│           ├── sepsis_grader.py
│           ├── stroke_grader.py
│           └── pediatric_grader.py
│
└── clinical-triage-dashboard/      # Next.js brutalist dashboard
    └── src/app/
        ├── page.tsx                # 4-panel monitoring station
        ├── layout.tsx              # Root layout
        └── globals.css             # Terminal-green design system
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | **Yes** | `https://api.openai.com/v1` | LLM API endpoint (injected by evaluator) |
| `API_KEY` | **Yes** | — | API key for LLM (injected by evaluator) |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | For deploy | — | Hugging Face API token (fallback for API_KEY) |
| `TASK_NAME` | No | — | Run a single task only |
| `PORT` | No | `7860` | Server port |

---

## License

MIT

---

<div align="center">

*Built by **Team Unfazed** for the Meta × Scaler OpenEnv Hackathon*

</div>
