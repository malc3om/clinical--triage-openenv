"""
reward.py — Dense, per-step reward computation for ClinicalTriageEnv.

The reward signal is NON-SPARSE: every step returns a float reward composed of
weighted sub-signals. This is critical for RL training and agent evaluation.

5 reward components:
  1. Clinical correctness signal (+0.1 to +0.3)
  2. Efficiency signal (-0.05 per unnecessary test)
  3. Time pressure signal (-0.02 per step for ESI-1/2 patients without care)
  4. Sequence correctness bonus (+0.05)
  5. Safety guardrails (-0.5 to -10.0 terminal penalty)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional

from clinical_triage_env.models import TriageAction, TriageState


# ─── Clinically indicated tests per task ────────────────────────────────

INDICATED_TESTS = {
    "task_stemi_code": {
        "troponin_i", "troponin", "cbc", "bmp", "bnp", "aspirin_325mg", "aspirin",
        "iv_access", "resuscitation", "ekg", "ecg",
    },
    "task_chest_pain_workup": {
        "ekg", "ecg", "d_dimer", "d-dimer", "ddimer", "ct_pa", "ctpa",
        "ct-pa", "troponin_i", "troponin", "cbc", "bmp", "cxr",
    },
    "task_mci_surge": {
        "ekg", "ecg", "cbc", "bmp", "epinephrine", "iv_access",
    },
    "task_sepsis_alert": {
        "lactate", "lactic_acid", "cbc", "bmp", "procalcitonin", "urinalysis",
        "blood_cultures", "cultures", "cxr",
    },
    "task_stroke_code": {
        "ct_head_noncon", "ct_head", "cta_head_neck", "cta", "cbc", "bmp",
        "coags", "inr", "glucose", "bmp_glucose",
    },
    "task_pediatric_resp": {
        "vbg", "abg", "cxr", "cbc", "bmp",
    },
}

# Correct ESI-1 patient IDs per task
ESI_1_PATIENTS = {
    "task_stemi_code": {"P1"},
    "task_chest_pain_workup": set(),
    "task_mci_surge": {"P1", "P3"},
    "task_sepsis_alert": set(),      # ESI-2 (not 1), but time-critical
    "task_stroke_code": set(),       # ESI-1 or 2 acceptable
    "task_pediatric_resp": set(),    # ESI-2 for pediatric respiratory
}

# Time-critical patients: any patient who should NOT wait without intervention
TIME_CRITICAL_PATIENTS = {
    "task_stemi_code": {"P1"},
    "task_mci_surge": {"P1", "P3"},
    "task_sepsis_alert": {"P1"},     # Hour-1 bundle deadline
    "task_stroke_code": {"P1"},      # "Time is brain"
    "task_pediatric_resp": {"P1"},   # Hypoxia needs rapid intervention
}

# Evidence-based ordering sequences
CORRECT_SEQUENCES = {
    "task_chest_pain_workup": [
        ("ekg", "ct_pa"),    # EKG before CT-PA
        ("ekg", "ctpa"),
        ("d_dimer", "ct_pa"),
        ("d_dimer", "ctpa"),  # D-dimer before CT-PA
    ],
    "task_stroke_code": [
        ("ct_head_noncon", "cta_head_neck"),  # Non-contrast CT before CTA
    ],
}


def compute_step_reward(
    action: TriageAction,
    state: TriageState,
    task_id: str,
) -> Tuple[float, Dict[str, float], str]:
    """
    Compute dense reward for a single step.

    Returns:
        (total_reward, components_dict, explanation_string)
    """
    components: Dict[str, float] = {}
    param_lower = action.parameter.lower().strip()
    action_type = action.action_type

    # ── 1. Clinical correctness signal ──────────────────────────────
    indicated = INDICATED_TESTS.get(task_id, set())
    if action_type == "order_diagnostic":
        if param_lower in indicated:
            components["clinical_correctness"] = 0.15
        else:
            components["clinical_correctness"] = -0.05  # Unnecessary test
    elif action_type == "assign_esi_level":
        components["clinical_correctness"] = 0.10  # Any ESI assignment is progress
    elif action_type == "activate_pathway":
        components["clinical_correctness"] = 0.20  # Pathway activation is high-value
    elif action_type == "disposition":
        components["clinical_correctness"] = 0.15  # Disposition decision
    elif action_type == "administer_medication":
        components["clinical_correctness"] = 0.10  # Medication is active treatment
    elif action_type == "request_consult":
        components["clinical_correctness"] = 0.05  # Consult is reasonable
    elif action_type == "wait":
        components["clinical_correctness"] = -0.02  # Waiting is mildly penalized
    else:
        components["clinical_correctness"] = 0.0

    # ── 2. Efficiency signal ────────────────────────────────────────
    diagnostics_so_far = len(state.diagnostics_ordered)
    if action_type == "order_diagnostic" and diagnostics_so_far > 5:
        components["efficiency_penalty"] = -0.05
    else:
        components["efficiency_penalty"] = 0.0

    # ── 3. Time pressure for critical patients ──────────────────────
    esi1_patients = ESI_1_PATIENTS.get(task_id, set())
    critical_patients = TIME_CRITICAL_PATIENTS.get(task_id, set())
    time_penalty = 0.0

    # ESI-1 patients bleed reward every step without disposition
    for pid in esi1_patients:
        if pid not in state.dispositions:
            time_penalty -= 0.02

    # Time-critical patients (broader set) bleed slightly less
    for pid in critical_patients - esi1_patients:
        if pid not in state.dispositions:
            time_penalty -= 0.01

    components["time_pressure"] = time_penalty

    # ── 4. Sequence correctness bonus ───────────────────────────────
    sequences = CORRECT_SEQUENCES.get(task_id, [])
    seq_bonus = 0.0
    if action_type == "order_diagnostic":
        for earlier, later in sequences:
            if param_lower == later or param_lower.replace("_", "") == later.replace("_", ""):
                # Check if earlier test was ordered before this one
                earlier_ordered = any(
                    earlier in d.lower() for d in state.diagnostics_ordered
                )
                if earlier_ordered:
                    seq_bonus += 0.05
    components["sequence_bonus"] = seq_bonus

    # ── 5. Safety guardrails (terminal penalties) ───────────────────
    safety_penalty = 0.0
    
    # Fatal delay thresholds
    elapsed = state.elapsed_minutes
    
    # STEMI: Door-to-balloon window is 90 mins
    if task_id == "task_stemi_code":
        cath_activated = any("cath" in p for p in state.pathways_activated)
        if elapsed > 90 and not cath_activated:
            safety_penalty = -10.0
            components["fatal_delay"] = -10.0
            
    # Sepsis: Hour-1 bundle — antibiotics and fluids within 60 mins
    if task_id == "task_sepsis_alert":
        if elapsed > 60:
            has_intervention = any(
                h.get("action", {}).get("action_type") in ("administer_medication", "activate_pathway")
                for h in state.episode_history
            )
            if not has_intervention:
                safety_penalty = -10.0
                components["fatal_delay"] = -10.0

    # Stroke: tPA window tight — imaging should be within 25 mins
    if task_id == "task_stroke_code":
        stroke_activated = any("stroke" in p for p in state.pathways_activated)
        ct_ordered = any("ct" in d.lower() for d in state.diagnostics_ordered)
        if elapsed > 60 and not (stroke_activated or ct_ordered):
            safety_penalty = -10.0
            components["fatal_delay"] = -10.0

    # Anaphylaxis (MCI Task Patient P3): Epinephrine window is tight
    if task_id == "task_mci_surge":
        p3_dispo = state.dispositions.get("P3")
        epi_given = any("epi" in d for d in state.diagnostics_ordered)
        if elapsed > 15 and not epi_given and not p3_dispo:
            safety_penalty = -10.0
            components["fatal_delay"] = -10.0

    # ESI-1 patients waiting > 60 mins without any major intervention
    for pid in esi1_patients:
        if pid not in state.dispositions and elapsed > 60:
            intervened = any(h.get("action", {}).get("patient_id") == pid 
                             and h.get("action", {}).get("action_type") in ["order_diagnostic", "activate_pathway"]
                             for h in state.episode_history)
            if not intervened:
                safety_penalty = -10.0
                components["fatal_delay"] = -10.0

    # Discharging a critically ill patient is catastrophic
    if action_type == "disposition" and "discharge" in param_lower:
        patient_esi = state.esi_assignments.get(action.patient_id)
        if patient_esi == 1:
            safety_penalty = -10.0
        elif action.patient_id in esi1_patients:
            safety_penalty = -10.0
        # Also check for task-specific discharge errors
        if task_id in ("task_sepsis_alert", "task_stroke_code", "task_stemi_code"):
            safety_penalty = min(safety_penalty, -5.0)
            
    # Infinite loop detection: same action 3 times in a row
    if len(state.episode_history) >= 3:
        last_3 = state.episode_history[-3:]
        if all(
            h.get("action", {}).get("action_type") == action_type
            and h.get("action", {}).get("parameter", "").lower() == param_lower
            for h in last_3
        ):
            safety_penalty = min(safety_penalty, -0.10)
    
    if safety_penalty != 0.0:
        components["safety_guardrail"] = safety_penalty

    # ── Total ───────────────────────────────────────────────────────
    total = sum(components.values())
    total = max(-10.5, min(1.0, total))

    explanation = f"Step reward: {total:+.3f} | " + ", ".join(
        f"{k}={v:+.3f}" for k, v in components.items() if v != 0.0
    )

    return total, components, explanation
