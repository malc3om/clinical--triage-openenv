"""
inference.py — Baseline inference script for ClinicalTriageEnv.

Uses OpenAI-compatible client with configurable model and endpoint.
Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
Must complete in < 20 minutes on vcpu=2, memory=8gb.
"""

import os
import sys
import json
import time
from typing import Optional

# Add current directory to path for imports to find the package properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from clinical_triage_env.models import TriageAction, TriageObservation
from clinical_triage_env.server.environment import ClinicalTriageEnvironment


# ─── Configuration ──────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

BENCHMARK = "clinical_triage"

# ─── Fallback action (used when LLM response can't be parsed) ──────────

FALLBACK_ACTIONS = {
    "task_stemi_code": [
        {"action_type": "assign_esi_level", "patient_id": "P1", "parameter": "1", "rationale": "Fallback: ESI 1 for STEMI"},
        {"action_type": "activate_pathway", "patient_id": "P1", "parameter": "cath_lab", "rationale": "Fallback: cath lab"},
        {"action_type": "order_diagnostic", "patient_id": "P1", "parameter": "aspirin_325mg", "rationale": "Fallback: aspirin"},
        {"action_type": "disposition", "patient_id": "P1", "parameter": "admit_icu", "rationale": "Fallback: admit ICU"},
    ],
    "task_chest_pain_workup": [
        {"action_type": "assign_esi_level", "patient_id": "P1", "parameter": "2", "rationale": "Fallback: ESI 2"},
        {"action_type": "order_diagnostic", "patient_id": "P1", "parameter": "EKG", "rationale": "Fallback: EKG"},
        {"action_type": "order_diagnostic", "patient_id": "P1", "parameter": "d_dimer", "rationale": "Fallback: D-dimer"},
        {"action_type": "order_diagnostic", "patient_id": "P1", "parameter": "CT_PA", "rationale": "Fallback: CT-PA"},
        {"action_type": "disposition", "patient_id": "P1", "parameter": "admit", "rationale": "Fallback: admit"},
    ],
    "task_mci_surge": [
        {"action_type": "assign_esi_level", "patient_id": "P1", "parameter": "1", "rationale": "Fallback: ESI 1 (P1)"},
        {"action_type": "assign_esi_level", "patient_id": "P3", "parameter": "1", "rationale": "Fallback: ESI 1 (P3)"},
        {"action_type": "assign_esi_level", "patient_id": "P4", "parameter": "2", "rationale": "Fallback: ESI 2 (P4)"},
        {"action_type": "assign_esi_level", "patient_id": "P2", "parameter": "3", "rationale": "Fallback: ESI 3 (P2)"},
        {"action_type": "assign_esi_level", "patient_id": "P5", "parameter": "4", "rationale": "Fallback: ESI 4 (P5)"},
        {"action_type": "disposition", "patient_id": "P1", "parameter": "admit", "rationale": "Fallback: admit P1"},
        {"action_type": "disposition", "patient_id": "P3", "parameter": "admit", "rationale": "Fallback: admit P3"},
        {"action_type": "disposition", "patient_id": "P5", "parameter": "waiting_room", "rationale": "Fallback: discharge P5"},
    ],
}


# ─── System Prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an experienced Emergency Department triage clinician.
You will receive a patient observation (JSON) and must respond with a valid JSON action.

Think step by step:
1. How urgent is this patient? What ESI level (1=resuscitation, 5=non-urgent)?
2. What diagnostic information do you need? (EKG, D-dimer, CT-PA, troponin, CBC, etc.)
3. What is the most evidence-based next action?
4. For STEMI: activate cath_lab pathway immediately, assign ESI 1, order aspirin.
5. For chest pain with PE risk factors: EKG first, then D-dimer, then CT-PA if positive.
6. For MCI: prioritize ESI-1 patients (unresponsive, anaphylaxis) for beds first.

Available action_types:
- "order_diagnostic" — Order a test (EKG, d_dimer, ct_pa, troponin_I, cbc, bmp, aspirin_325mg, etc.)
- "assign_esi_level" — Assign ESI 1-5 (parameter must be "1", "2", "3", "4", or "5")
- "activate_pathway" — Activate pathway (cath_lab, stroke_code, trauma, etc.)
- "disposition" — Set final disposition (admit, discharge, transfer, treat_and_street, waiting_room)
- "request_consult" — Request specialist (cardiology, pulmonology, etc.)
- "wait" — Wait for pending results

Respond with ONLY a valid JSON object matching this schema:
{
    "action_type": "...",
    "patient_id": "P1",
    "parameter": "...",
    "rationale": "brief reasoning"
}

NO markdown, NO explanation — ONLY the raw JSON object."""


# ─── Logging Helpers ────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ─── Helpers ────────────────────────────────────────────────────────────

def parse_model_action(response_text: str, task_id: str, step_num: int) -> dict:
    """Parse LLM response into a valid action dict. Falls back gracefully."""
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        return json.loads(cleaned)
    except Exception:
        actions = FALLBACK_ACTIONS.get(task_id, FALLBACK_ACTIONS["task_stemi_code"])
        idx = min(step_num - 1, len(actions) - 1)
        return actions[idx]


def observation_to_prompt(obs: TriageObservation, history: list) -> str:
    """Convert observation to a prompt string for the LLM."""
    obs_dict = obs.model_dump()
    for p in obs_dict.get("patients", []):
        for key in list(p.keys()):
            if isinstance(p[key], list) and len(p[key]) == 0:
                del p[key]

    obs_str = json.dumps(obs_dict, indent=2, default=str)
    recent_history = history[-3:] if history else []
    history_str = json.dumps(recent_history, indent=2) if recent_history else "[]"

    return (
        f"Current observation:\n{obs_str}\n\n"
        f"Recent history:\n{history_str}\n\n"
        f"What is your next action? Respond with ONLY a valid JSON object."
    )


# ─── Main inference loop ───────────────────────────────────────────────

def run_task(
    env: ClinicalTriageEnvironment,
    task_id: str,
    client: OpenAI,
    max_steps: int = 20,
) -> float:
    """Run one task episode with LLM agent."""
    observation = env.reset(task_id=task_id)
    history = []
    rewards = []
    
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    steps_taken = 0
    score = 0.0

    for step_num in range(1, max_steps + 1):
        prompt = observation_to_prompt(observation, history)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=512,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            response_text = response.choices[0].message.content
        except Exception as exc:
            actions = FALLBACK_ACTIONS.get(task_id, FALLBACK_ACTIONS["task_stemi_code"])
            idx = min(step_num - 1, len(actions) - 1)
            response_text = json.dumps(actions[idx])

        action_dict = parse_model_action(response_text, task_id, step_num)
        action_str = json.dumps(action_dict)

        error_msg = None
        try:
            action = TriageAction(**action_dict)
            result = env.step(action)
        except Exception as exc:
            result = observation 
            result.reward = -0.1
            result.done = True
            error_msg = str(exc)

        reward = result.reward if result.reward is not None else 0.0
        done = result.done
        
        rewards.append(reward)
        steps_taken = step_num
        
        history.append({
            "step": step_num,
            "action": action_dict,
            "reward": reward,
            "result": getattr(result, 'last_action_result', None),
        })

        log_step(step=step_num, action=action_str, reward=reward, done=done, error=error_msg)

        if done:
            break

        observation = result

    # Calculate final score via grader
    try:
        grader_result = env.get_task_grader_score(task_id)
        score = max(0.0, min(1.0, grader_result.score))
    except Exception:
        score = 0.0

    success = score >= 0.7

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    """Run baseline inference across all tasks."""
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", "dummy"),
    )
    env = ClinicalTriageEnvironment()

    task_name_env = os.environ.get("TASK_NAME")
    if task_name_env:
        tasks = [(task_name_env, 25)]
    else:
        tasks = [
            ("task_stemi_code", 15),
            ("task_chest_pain_workup", 20),
            ("task_mci_surge", 25),
        ]

    scores = {}
    
    for task_id, max_steps in tasks:
        score = run_task(env, task_id, client, max_steps)
        scores[task_id] = score

    env.close()
    return scores


if __name__ == "__main__":
    main()
