"""
inference.py — LLM-powered clinical reasoning agent for ClinicalTriageEnv.

Uses an OpenAI-compatible LLM to make real-time triage decisions through
structured ReAct (Reasoning + Acting) prompting. The agent observes patient
state, reasons about clinical priorities, and selects evidence-based actions.

There are NO hardcoded action sequences. Every decision is made by the LLM
based on the current observation. The only fallback on parse failure is a
simple "wait" action that costs the agent a step.

Environment variables (injected by the evaluator):
  - API_BASE_URL: LLM API endpoint (required)
  - API_KEY:      API key for authentication (required)
  - MODEL_NAME:   Model identifier (default: gpt-4o-mini)
"""

import os
import sys
import json
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from clinical_triage_env.models import TriageAction, TriageObservation
from clinical_triage_env.server.environment import ClinicalTriageEnvironment


# ─── Configuration ──────────────────────────────────────────────────────

BENCHMARK = "clinical_triage"


def get_config():
    """Read configuration from environment variables at call time."""
    api_base_url = os.environ.get("API_BASE_URL") or "https://api.openai.com/v1"
    api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or "not-set"
    model_name = os.environ.get("MODEL_NAME") or "gpt-4o-mini"
    return api_base_url, api_key, model_name


# ─── Logging (OpenEnv-compatible format) ────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task_name={task} task_id={task}", flush=True)


def log_step(step: int, action: str, observation: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} observation={observation} reward={reward:.2f}", flush=True)


def log_end(task: str, score: float) -> None:
    print(f"[END] task_name={task} score={score:.3f}", flush=True)


# ─── System prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an experienced Emergency Department triage clinician.
You will receive a patient observation as JSON. You must respond with a single valid JSON action.

## Clinical Reasoning Protocol
For EVERY patient, follow this structured approach:
1. PRIMARY SURVEY: Assess airway, breathing, circulation (ABCs) from vitals.
2. RISK STRATIFICATION: Determine ESI level (1=resuscitation, 2=emergent, 3=urgent, 4=less urgent, 5=non-urgent).
3. DIFFERENTIAL DIAGNOSIS: Consider the chief complaint, history, and vitals together.
4. ACTION SELECTION: Choose the single most impactful next action.
5. TIME AWARENESS: Some conditions are time-critical (STEMI: 90 min to cath lab, Stroke: time is brain, Sepsis: 1hr antibiotics).

## Available Actions
- "order_diagnostic" — Order a test. parameter = test name (e.g. "EKG", "d_dimer", "CT_PA", "troponin_I", "cbc", "bmp", "lactate", "CT_HEAD_NONCON", "CTA_HEAD_NECK", "coags", "vbg", "urinalysis", "procalcitonin", "CXR")
- "assign_esi_level" — Assign Emergency Severity Index. parameter = "1", "2", "3", "4", or "5"
- "activate_pathway" — Activate a clinical pathway. parameter = "cath_lab", "stroke", "trauma", "sepsis"
- "disposition" — Set final patient disposition. parameter = "admit", "discharge", "transfer", "waiting_room"
- "request_consult" — Request a specialist. parameter = specialty name
- "administer_medication" — Give a medication. parameter = medication name (e.g. "aspirin_325mg", "ceftriaxone", "IV_fluid_bolus", "albuterol_nebulizer", "dexamethasone", "epinephrine")
- "assign_bed" — Assign a bed. parameter = bed identifier
- "wait" — Wait for pending results. parameter = ""

## Multi-Patient Triage (MCI)
When multiple patients are present, triage the MOST CRITICAL patient first.
Prioritize by: life threat > hemodynamic instability > respiratory distress > pain.

## Response Format
Enclose your reasoning in <thought> tags, then provide a single JSON object.
Example:
<thought>
Patient P1 presents with crushing chest pain, diaphoresis, and hypotension.
EKG shows ST-elevation in leads II, III, aVF — this is an inferior STEMI.
Time-critical: must activate cath lab pathway immediately.
</thought>
{
    "action_type": "activate_pathway",
    "patient_id": "P1",
    "parameter": "cath_lab",
    "rationale": "Inferior STEMI identified on EKG, activating cath lab within 90-minute window"
}

Respond ONLY with <thought> reasoning and a JSON object. No other text."""


# ─── Prompt construction ────────────────────────────────────────────────

def observation_to_prompt(obs: TriageObservation, history: list) -> str:
    """Convert the current observation and recent history into a prompt."""
    obs_dict = obs.model_dump()

    # Remove empty lists for readability
    for p in obs_dict.get("patients", []):
        for key in list(p.keys()):
            if isinstance(p[key], list) and len(p[key]) == 0:
                del p[key]
            if isinstance(p[key], dict) and len(p[key]) == 0:
                del p[key]

    obs_json = json.dumps(obs_dict, indent=2, default=str)

    # Show last 5 actions so the LLM has context
    recent = history[-5:] if history else []
    history_json = json.dumps(recent, indent=2) if recent else "[]"

    return (
        f"Current observation:\n{obs_json}\n\n"
        f"Actions taken so far in this episode:\n{history_json}\n\n"
        f"What is your next clinical action? Respond with <thought> reasoning then a JSON action."
    )


# ─── Response parsing ───────────────────────────────────────────────────

def parse_llm_response(response_text: str) -> Optional[dict]:
    """Extract a valid action dict from LLM output. Returns None on failure."""
    try:
        cleaned = response_text

        # Strip <thought> blocks
        if "<thought>" in cleaned and "</thought>" in cleaned:
            cleaned = cleaned.split("</thought>")[-1].strip()

        # Strip markdown code fences
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1].strip()

        cleaned = cleaned.strip()

        # Try to find JSON object
        if not cleaned.startswith("{"):
            # Look for first { in the text
            idx = cleaned.find("{")
            if idx >= 0:
                cleaned = cleaned[idx:]

        parsed = json.loads(cleaned)

        # Validate required fields
        if all(k in parsed for k in ("action_type", "patient_id", "parameter")):
            return parsed
        return None
    except Exception:
        return None


# ─── LLM Agent ──────────────────────────────────────────────────────────

def run_task_with_llm(
    env: ClinicalTriageEnvironment,
    task_id: str,
    max_steps: int = 20,
) -> float:
    """
    Run a single task episode using LLM-based clinical reasoning.

    Every action decision is made by the LLM. On parse failure, the agent
    simply waits (costing a step) — there is no hardcoded fallback logic.
    """
    api_base_url, api_key, model_name = get_config()

    print(f"[CONFIG] api_base_url={api_base_url} model={model_name}", flush=True)

    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key,
    )

    observation = env.reset(task_id=task_id)
    history: list = []
    llm_calls = 0
    parse_failures = 0

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    for step_num in range(1, max_steps + 1):
        prompt = observation_to_prompt(observation, history)

        # ── Call LLM ────────────────────────────────────────────────
        action_dict = None
        try:
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=1024,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            llm_calls += 1
            response_text = response.choices[0].message.content or ""
            action_dict = parse_llm_response(response_text)

            if action_dict is None:
                parse_failures += 1
                print(f"  [WARN] LLM response could not be parsed at step {step_num}", flush=True)

        except Exception as e:
            print(f"  [WARN] LLM API error at step {step_num}: {e}", flush=True)

        # If LLM failed to produce a valid action, wait (costs a step)
        if action_dict is None:
            pid = observation.patients[0].patient_id if observation.patients else "P1"
            action_dict = {
                "action_type": "wait",
                "patient_id": pid,
                "parameter": "",
                "rationale": "LLM parse failure — waiting",
            }

        # ── Execute action ──────────────────────────────────────────
        action_json = json.dumps(action_dict)
        error_msg = None

        try:
            action = TriageAction(**action_dict)
            result = env.step(action)
        except Exception as exc:
            error_msg = str(exc)
            result = observation
            result.reward = -0.1
            result.done = True

        reward = result.reward if result.reward is not None else 0.0
        done = result.done

        history.append({
            "step": step_num,
            "action": action_dict,
            "reward": reward,
            "result": getattr(result, "last_action_result", None),
        })

        obs_json = json.dumps(observation.model_dump(), default=str)
        log_step(
            step=step_num,
            action=action_json,
            observation=obs_json,
            reward=reward,
            done=done,
            error=error_msg,
        )

        if done:
            break
        observation = result

    # ── Get grader score ────────────────────────────────────────────
    try:
        grader_result = env.get_task_grader_score(task_id)
        score = max(0.0, min(1.0, grader_result.score))
    except Exception:
        score = 0.0

    print(f"  [INFO] LLM calls: {llm_calls}, parse failures: {parse_failures}", flush=True)
    log_end(task=task_id, score=score)
    return score


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    """Run inference across all registered tasks."""
    env = ClinicalTriageEnvironment()

    task_name_env = os.environ.get("TASK_NAME")

    if task_name_env:
        tasks = [(task_name_env, 25)]
    else:
        tasks = [
            ("task_stemi_code", 15),
            ("task_chest_pain_workup", 20),
            ("task_mci_surge", 25),
            ("task_sepsis_alert", 20),
            ("task_stroke_code", 18),
            ("task_pediatric_resp", 18),
        ]

    scores = {}

    for task_id, max_steps in tasks:
        score = run_task_with_llm(env, task_id, max_steps)
        scores[task_id] = score

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("FINAL SCORES", flush=True)
    print("=" * 60, flush=True)
    for task_id, score in scores.items():
        status = "PASS" if score >= 0.5 else "NEEDS_IMPROVEMENT"
        print(f"  {task_id}: {score:.3f}  [{status}]", flush=True)

    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  AVERAGE: {avg:.3f}", flush=True)
    print("=" * 60, flush=True)

    env.close()
    return scores


if __name__ == "__main__":
    main()
