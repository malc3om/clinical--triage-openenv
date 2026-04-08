"""
chest_workup_grader.py — Deterministic grader for Task 2: Chest Pain Workup.

Scores 0.0–1.0 based on whether the agent:
  - Ordered EKG (+0.20)
  - Ordered D-dimer (+0.15)
  - Ordered CT-PA (+0.20)
  - Correct sequence: EKG before CT-PA (+0.05 bonus)
  - Correct disposition: admit for PE (+0.25)
  - Penalty for CT-PA before EKG (-0.15)
  - Penalty for discharge without PE workup (-0.30)
  - Penalty for >6 diagnostics ordered
"""

from __future__ import annotations
from typing import List, Dict
from clinical_triage_env.models import GradeResult


def grade_chest_workup(episode_history: List[dict]) -> GradeResult:
    """Grade a completed Chest Pain Workup episode. Returns 0.0–1.0."""
    score = 0.0
    breakdown: Dict[str, float] = {}

    actions_in_order = [s["action"] for s in episode_history]
    action_params = [str(a.get("parameter", "")).lower() for a in actions_in_order]

    # ── Find key action indices ─────────────────────────────────────
    ekg_idx = None
    ctpa_idx = None
    for i, p in enumerate(action_params):
        if ekg_idx is None and ("ekg" in p or "ecg" in p):
            ekg_idx = i
        if ctpa_idx is None and ("ct" in p and "pa" in p) or ("ctpa" in p):
            ctpa_idx = i

    # ── EKG ordered → +0.20 ─────────────────────────────────────────
    if ekg_idx is not None:
        breakdown["ekg_ordered"] = 0.20
        score += 0.20
        # Bonus for correct sequence: EKG before CT-PA → +0.05
        if ctpa_idx is not None and ekg_idx < ctpa_idx:
            breakdown["correct_sequence_bonus"] = 0.05
            score += 0.05
        else:
            breakdown["correct_sequence_bonus"] = 0.0
    else:
        breakdown["ekg_ordered"] = 0.0
        breakdown["correct_sequence_bonus"] = 0.0

    # ── D-dimer ordered → +0.15 ──────────────────────────────────────
    d_dimer_ordered = any("d_dimer" in p or "d-dimer" in p or "ddimer" in p for p in action_params)
    if d_dimer_ordered:
        breakdown["d_dimer_ordered"] = 0.15
        score += 0.15
    else:
        breakdown["d_dimer_ordered"] = 0.0

    # ── CT-PA ordered → +0.20 ────────────────────────────────────────
    if ctpa_idx is not None:
        breakdown["ctpa_ordered"] = 0.20
        score += 0.20
    else:
        breakdown["ctpa_ordered"] = 0.0

    # ── Correct disposition: admit (PE confirmed) → +0.25 ────────────
    disp_actions = [a for a in actions_in_order if a.get("action_type") == "disposition"]
    admitted = any("admit" in str(a.get("parameter", "")).lower() for a in disp_actions)
    discharged = any("discharge" in str(a.get("parameter", "")).lower() for a in disp_actions)

    if admitted:
        breakdown["disposition_admit"] = 0.25
        score += 0.25
    else:
        breakdown["disposition_admit"] = 0.0

    # ── Penalty: CT-PA ordered before EKG → -0.15 ────────────────────
    if ctpa_idx is not None and (ekg_idx is None or ctpa_idx < ekg_idx):
        breakdown["ctpa_before_ekg_penalty"] = -0.15
        score -= 0.15
    else:
        breakdown["ctpa_before_ekg_penalty"] = 0.0

    # ── Penalty: discharge without PE workup → -0.30 ─────────────────
    if discharged:
        breakdown["discharge_penalty"] = -0.30
        score -= 0.30
    else:
        breakdown["discharge_penalty"] = 0.0

    # ── Resource efficiency: penalty if >6 diagnostics ────────────────
    diagnostic_count = sum(1 for a in actions_in_order if a.get("action_type") == "order_diagnostic")
    if diagnostic_count > 6:
        over_penalty = -0.05 * (diagnostic_count - 6)
        breakdown["resource_overuse_penalty"] = over_penalty
        score += over_penalty
    else:
        breakdown["resource_overuse_penalty"] = 0.0

    # ── ESI assignment bonus → +0.10 ─────────────────────────────────
    esi_actions = [a for a in actions_in_order if a.get("action_type") == "assign_esi_level"]
    esi_correct = any(str(a.get("parameter", "")).strip() in ("2", "3") for a in esi_actions)
    if esi_correct:
        breakdown["esi_correct"] = 0.10
        score += 0.10
    else:
        breakdown["esi_correct"] = 0.0

    # ── Clamp ────────────────────────────────────────────────────────
    final_score = max(0.0, min(1.0, score))

    explanation_parts = [f"{k}: {v:+.2f}" for k, v in breakdown.items() if v != 0.0]
    explanation = "Chest workup grader: " + ", ".join(explanation_parts) if explanation_parts else "No scoring actions detected."

    return GradeResult(
        score=final_score,
        breakdown=breakdown,
        explanation=explanation,
    )
