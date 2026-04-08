"""
mci_grader.py — Deterministic grader for Task 3: Mass Casualty Incident.

Scores 0.0–1.0 based on:
  - Correct ESI assignments for all 5 patients (0.10 each, max 0.50)
  - Correct bed allocation: P1 and P3 in immediate beds (+0.10 each)
  - Correctly NOT admitting P5 (+0.10)
  - Penalty for P5 getting a bed over P1/P3 (-0.20 each)
  - Bonus for handling ESI-1 patients in first 4 steps (+0.10)
"""

from __future__ import annotations
from typing import List, Dict
from clinical_triage_env.models import GradeResult

# Ground truth ESI assignments
CORRECT_ESI = {
    "P1": 1,  # 72yo, unresponsive, GCS 6 → immediate
    "P2": 3,  # 28yo, broken arm → delayed
    "P3": 1,  # 15yo, anaphylaxis → immediate
    "P4": 2,  # 60yo, rapid afib → urgent
    "P5": 4,  # 35yo, anxiety → non-urgent
}


def grade_mci(episode_history: List[dict]) -> GradeResult:
    """Grade a completed MCI Surge episode. Returns 0.0–1.0."""
    score = 0.0
    breakdown: Dict[str, float] = {}

    esi_assignments: Dict[str, int] = {}
    disposition_map: Dict[str, str] = {}

    for step in episode_history:
        a = step["action"]
        pid = str(a.get("patient_id", ""))
        action_type = a.get("action_type", "")

        if action_type == "assign_esi_level":
            try:
                esi_assignments[pid] = int(a.get("parameter", "0"))
            except (ValueError, TypeError):
                pass

        if action_type == "disposition":
            disposition_map[pid] = str(a.get("parameter", "")).lower()

    # ── Correct ESI assignments (0.10 each, max 0.50) ───────────────
    esi_score = 0.0
    for pid, expected in CORRECT_ESI.items():
        assigned = esi_assignments.get(pid)
        if assigned is not None:
            if assigned == expected:
                esi_score += 0.10
                breakdown[f"esi_{pid}_exact"] = 0.10
            elif abs(assigned - expected) == 1:
                esi_score += 0.05  # Partial credit for near-miss
                breakdown[f"esi_{pid}_near"] = 0.05
            else:
                breakdown[f"esi_{pid}_wrong"] = 0.0
        else:
            breakdown[f"esi_{pid}_missing"] = 0.0
    score += esi_score

    # ── Correct bed allocation: P1 and P3 admitted → +0.10 each ─────
    immediate_admits = [
        pid for pid, disp in disposition_map.items()
        if "admit" in disp and pid in ("P1", "P3")
    ]
    bed_score = 0.10 * len(immediate_admits)
    breakdown["immediate_bed_allocation"] = bed_score
    score += bed_score

    # ── Correctly NOT admitting P5 → +0.10 ──────────────────────────
    p5_disp = disposition_map.get("P5", "")
    p5_not_admitted = p5_disp in ("discharge", "treat_and_street", "waiting_room", "") or "discharge" in p5_disp
    if p5_disp and p5_not_admitted and "admit" not in p5_disp:
        breakdown["p5_correct_triage"] = 0.10
        score += 0.10
    else:
        breakdown["p5_correct_triage"] = 0.0

    # ── Penalty: P5 admitted over P1 or P3 → -0.20 each ─────────────
    p5_admitted = "P5" in [
        pid for pid, disp in disposition_map.items() if "admit" in disp
    ]
    if p5_admitted:
        p1_admitted = "P1" in [pid for pid, disp in disposition_map.items() if "admit" in disp]
        p3_admitted = "P3" in [pid for pid, disp in disposition_map.items() if "admit" in disp]
        if not p1_admitted:
            breakdown["p5_over_p1_penalty"] = -0.20
            score -= 0.20
        if not p3_admitted:
            breakdown["p5_over_p3_penalty"] = -0.20
            score -= 0.20

    # ── Bonus: ESI-1 patients addressed in first 4 steps → +0.10 ────
    first_4_pids = [
        str(s["action"].get("patient_id", ""))
        for s in episode_history[:4]
    ]
    if "P1" in first_4_pids and "P3" in first_4_pids:
        breakdown["priority_ordering_bonus"] = 0.10
        score += 0.10
    else:
        breakdown["priority_ordering_bonus"] = 0.0

    # ── P4 management bonus: ESI 2 + appropriate action → +0.05 ──────
    p4_esi = esi_assignments.get("P4")
    if p4_esi == 2:
        breakdown["p4_management_bonus"] = 0.05
        score += 0.05
    else:
        breakdown["p4_management_bonus"] = 0.0

    # ── Clamp ────────────────────────────────────────────────────────
    final_score = max(0.0, min(1.0, score))

    explanation_parts = [f"{k}: {v:+.2f}" for k, v in breakdown.items() if v != 0.0]
    explanation = "MCI grader: " + ", ".join(explanation_parts) if explanation_parts else "No scoring actions detected."

    return GradeResult(
        score=final_score,
        breakdown=breakdown,
        explanation=explanation,
    )
