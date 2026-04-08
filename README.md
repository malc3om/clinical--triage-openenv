# ClinicalTriageEnv 🏥

> **An OpenEnv environment for Emergency Department triage simulation.**
> AI agents step into the role of ED triage clinicians — assessing patients, ordering diagnostics, assigning acuity levels, and making disposition decisions under time pressure and resource constraints.

**Team Unfazed** | Meta × Scaler OpenEnv Hackathon

---

## Why This Matters

Emergency triage is one of the highest-stakes sequential decision tasks humans perform:

- **Wrong triage kills**: Delayed MI treatment → myocardial death. Over-triage → ED gridlock.
- **No standardized RL benchmark exists** for clinical decision-making under uncertainty.
- **This environment is fully synthetic** — no HIPAA concerns, fully open-source.

The RL/agent community currently has no way to benchmark clinical AI agents. ClinicalTriageEnv closes that gap.

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Current task identifier |
| `task_difficulty` | `easy\|medium\|hard` | Difficulty tier |
| `step_number` | `int` | Current step in episode |
| `max_steps` | `int` | Maximum steps before timeout |
| `patients` | `List[PatientState]` | Patient(s) with vitals, labs, history |
| `available_beds` | `int` | Remaining beds (crucial for MCI) |
| `last_action_result` | `str` | Feedback from last action |
| `last_action_error` | `str\|null` | Error if last action was invalid |
| `reward_components` | `dict` | Breakdown of reward signals |

### PatientState Fields

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | `str` | Unique patient ID (P1, P2, ...) |
| `age` | `int` | Patient age |
| `sex` | `M\|F` | Biological sex |
| `chief_complaint` | `str` | Presenting complaint |
| `onset_minutes` | `int` | Minutes since symptom onset |
| `vitals` | `VitalSigns` | HR, BP, SpO2, RR, Temp, GCS |
| `medical_history` | `List[str]` | Past medical history |
| `available_labs` | `List[LabResult]` | Resulted lab values |
| `pending_labs` | `List[str]` | Ordered but not yet resulted |
| `imaging_available` | `List[str]` | Available imaging studies |
| `resource_tokens_remaining` | `int` | Budget for ordering tests |

---

## Action Space

| `action_type` | `parameter` examples | Description |
|---------------|---------------------|-------------|
| `order_diagnostic` | `EKG`, `d_dimer`, `CT_PA`, `troponin_I`, `aspirin_325mg` | Order a test or intervention |
| `assign_esi_level` | `1`, `2`, `3`, `4`, `5` | Assign Emergency Severity Index |
| `activate_pathway` | `cath_lab`, `stroke_code`, `trauma` | Activate clinical pathway |
| `disposition` | `admit`, `discharge`, `transfer`, `waiting_room` | Final patient disposition |
| `request_consult` | `cardiology`, `pulmonology` | Request specialist consult |
| `wait` | `""` | Wait for pending results |

---

## Reward Signal

The reward is **dense** — every step returns a signal composed of 5 weighted components:

| Component | Range | Description |
|-----------|-------|-------------|
| **Clinical correctness** | +0.1 to +0.3 | Ordering indicated tests, correct ESI |
| **Efficiency** | -0.05 | Penalty per unnecessary test |
| **Time pressure** | -0.02/step | ESI-1 patients bleed reward each step |
| **Sequence bonus** | +0.05 | Following evidence-based protocols |
| **Safety guardrails** | -0.5 | Discharging ESI-1, infinite loops |

Total per-step reward is clamped to [-1.0, +1.0].

---

## Tasks

### Task 1: `task_stemi_code` (Easy)

**Scenario**: 58yo male, crushing chest pain, ST-elevation on EKG. Hypotensive, diaphoretic.

**Agent must**: Assign ESI 1, activate cath lab, order aspirin, admit.

**Max steps**: 15 | **Baseline score**: 0.72

---

### Task 2: `task_chest_pain_workup` (Medium)

**Scenario**: 44yo female, pleuritic chest pain, recent long-haul flight, on oral contraceptives.

**Differential**: PE vs STEMI vs MSK vs anxiety. Agent must navigate the workup in the correct diagnostic sequence (EKG → D-dimer → CT-PA).

**Max steps**: 20 | **Baseline score**: 0.48

---

### Task 3: `task_mci_surge` (Hard)

**Scenario**: Mass casualty incident. 5 patients arrive simultaneously. Only 3 beds.

| Patient | Age/Sex | Presentation | Expected ESI |
|---------|---------|-------------|--------------|
| P1 | 72M | Unresponsive, GCS 6, HR 40 | 1 (immediate) |
| P2 | 28F | Broken arm, stable | 3 (delayed) |
| P3 | 15M | Anaphylaxis, stridor, BP 70/40 | 1 (immediate) |
| P4 | 60M | Rapid atrial fibrillation | 2 (urgent) |
| P5 | 35F | Anxiety, hyperventilation | 4 (non-urgent) |

**Max steps**: 25 | **Baseline score**: 0.31

---

## Setup

### Local Development

```bash
# Clone the repo
git clone <repo-url>
cd clinical_triage_env

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn clinical_triage_env.app:app --host 0.0.0.0 --port 7860 --reload

# In another terminal, validate the submission
python validate_submission.py
```

### Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
```

### Run Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-hf-token"

python inference.py
```

---

## Quickstart

```python
from clinical_triage_env.models import TriageAction
from clinical_triage_env.server.environment import ClinicalTriageEnvironment

env = ClinicalTriageEnvironment()
obs = env.reset(task_id="task_stemi_code")

# Take an action
action = TriageAction(
    action_type="assign_esi_level",
    patient_id="P1",
    parameter="1",
    rationale="Acute STEMI requires ESI 1"
)
result = env.step(action)

print(f"Reward: {result.reward}, Done: {result.done}")
print(f"Result: {result.last_action_result}")

# Get grader score
score = env.get_task_grader_score()
print(f"Grader: {score.score:.3f}")
```

---

## Baseline Scores

| Task | Baseline Score | Difficulty |
|------|---------------|------------|
| `task_stemi_code` | 0.720 | Easy |
| `task_chest_pain_workup` | 0.480 | Medium |
| `task_mci_surge` | 0.310 | Hard |
| **Average** | **0.503** | |

---

## License

MIT

---

*Built by Team Unfazed for the Meta × Scaler OpenEnv Hackathon.*
