"""
stemi_grader.py — Deterministic grader for Task 1: STEMI Code.

Scores 0.0–1.0 based on whether the agent:
  - Assigns ESI 1 (+0.25)
  - Activates cath_lab pathway (+0.30)
  - Correct disposition: admit (+0.25)
  - Time penalty for slow cath lab activation
  - Penalty for unnecessary diagnostics before cath lab
"""

from __future__ import annotations
from typing import List, Dict
from clinical_triage_env.models import GradeResult


def grade_stemi(episode_history: List[dict]) -> GradeResult:
    """Grade a completed STEMI Code episode. Returns 0.0–1.0 deterministically."""
    score = 0.0
    breakdown: Dict[str, float] = {}

    actions = [step["action"] for step in episode_history]

    # ── ESI assigned correctly → +0.25 ──────────────────────────────
    esi_actions = [a for a in actions if a.get("action_type") == "assign_esi_level"]
    if any(str(a.get("parameter", "")).strip() == "1" for a in esi_actions):
        breakdown["esi_correct"] = 0.25
        score += 0.25
    else:
        breakdown["esi_correct"] = 0.0

    # ── Cath lab activated → +0.30 ──────────────────────────────────
    pathway_actions = [a for a in actions if a.get("action_type") == "activate_pathway"]
    cath_activated = any("cath_lab" in str(a.get("parameter", "")).lower() for a in pathway_actions)
    if cath_activated:
        breakdown["cath_lab_activated"] = 0.30
        score += 0.30
    else:
        breakdown["cath_lab_activated"] = 0.0

    # ── Correct disposition: admit → +0.25 ──────────────────────────
    disp_actions = [a for a in actions if a.get("action_type") == "disposition"]
    if any("admit" in str(a.get("parameter", "")).lower() for a in disp_actions):
        breakdown["disposition_correct"] = 0.25
        score += 0.25
    else:
        breakdown["disposition_correct"] = 0.0

    # ── Time penalty: -0.05 for every 2 steps over 4 ───────────────
    steps_taken = len(episode_history)
    time_penalty = 0.0
    if steps_taken > 4:
        time_penalty = -0.05 * ((steps_taken - 4) // 2)
    breakdown["time_penalty"] = time_penalty
    score += time_penalty

    # ── Penalty: cath lab delayed past step 3 → -0.15 ──────────────
    cath_step = None
    for i, step in enumerate(episode_history):
        if "cath_lab" in str(step["action"].get("parameter", "")).lower():
            cath_step = i
            break

    delay_penalty = 0.0
    if cath_step is not None and cath_step > 3:
        delay_penalty = -0.15
    breakdown["cath_delay_penalty"] = delay_penalty
    score += delay_penalty

    # ── Bonus: aspirin ordered → +0.10 ─────────────────────────────
    diag_actions = [a for a in actions if a.get("action_type") == "order_diagnostic"]
    aspirin_ordered = any("aspirin" in str(a.get("parameter", "")).lower() for a in diag_actions)
    if aspirin_ordered:
        breakdown["aspirin_bonus"] = 0.10
        score += 0.10
    else:
        breakdown["aspirin_bonus"] = 0.0

    # ── Clamp to [0.0, 1.0] ────────────────────────────────────────
    final_score = max(0.0, min(1.0, score))

    explanation_parts = [f"{k}: {v:+.2f}" for k, v in breakdown.items() if v != 0.0]
    explanation = "STEMI grader: " + ", ".join(explanation_parts) if explanation_parts else "No scoring actions detected."

    return GradeResult(
        score=final_score,
        breakdown=breakdown,
        explanation=explanation,
    )
