# Phase 2: Deterministic, seeded patient generator for all 3 tasks
"""
patient_generator.py — Synthetic patient data for ClinicalTriageEnv.

Every call with the same task_id produces IDENTICAL patients.
No external APIs needed. All data is synthetic.
"""

from __future__ import annotations

import random
from typing import List

from clinical_triage_env.models import (
    VitalSigns,
    LabResult,
    PatientState,
)


SEED_MAP = {
    "task_stemi_code": 42,
    "task_chest_pain_workup": 137,
    "task_mci_surge": 256,
}


# ─── Task 1: STEMI Code (Easy) ─────────────────────────────────────────

def generate_stemi_patient() -> List[PatientState]:
    """58yo male with classic STEMI. EKG shows ST-elevation. Troponin pending."""
    return [
        PatientState(
            patient_id="P1",
            age=58,
            sex="M",
            chief_complaint="Sudden crushing chest pain radiating to left arm, diaphoretic",
            onset_minutes=22,
            vitals=VitalSigns(
                heart_rate=102,
                systolic_bp=88,
                diastolic_bp=60,
                respiratory_rate=22,
                spo2=94.0,
                temperature=37.1,
                gcs=15,
            ),
            medical_history=["hypertension", "type_2_diabetes", "smoker_30_pack_years"],
            current_medications=["metformin_500mg", "lisinopril_10mg"],
            available_labs=[],
            pending_labs=["troponin_I", "cbc", "bmp"],
            imaging_available=["EKG"],  # EKG already done — shows ST-elevation
            pending_imaging=[],
            time_in_department_minutes=8,
            resource_tokens_remaining=10,
        )
    ]


# ─── Task 2: Chest Pain Workup (Medium) ────────────────────────────────

def generate_chest_pain_patient() -> List[PatientState]:
    """44yo female with pleuritic chest pain post-flight. Differential: PE vs STEMI vs MSK."""
    return [
        PatientState(
            patient_id="P1",
            age=44,
            sex="F",
            chief_complaint="Sharp pleuritic chest pain, worse with inspiration, onset 4 hours ago",
            onset_minutes=240,
            vitals=VitalSigns(
                heart_rate=98,
                systolic_bp=122,
                diastolic_bp=78,
                respiratory_rate=20,
                spo2=96.0,
                temperature=37.0,
                gcs=15,
            ),
            medical_history=["oral_contraceptives", "recent_long_haul_flight_3_days_ago"],
            current_medications=["combined_oral_contraceptive"],
            available_labs=[],
            pending_labs=[],
            imaging_available=[],
            pending_imaging=[],
            time_in_department_minutes=5,
            resource_tokens_remaining=12,
        )
    ]


# ─── Task 3: Mass Casualty Incident (Hard) ─────────────────────────────

def generate_mci_patients() -> List[PatientState]:
    """5 simultaneous patients. 3 beds. Resource scarcity triage."""
    return [
        # P1: 72yo M, unresponsive, HR 40, GCS 6 → ESI 1, immediate
        PatientState(
            patient_id="P1",
            age=72,
            sex="M",
            chief_complaint="Found unresponsive, bradycardic",
            onset_minutes=15,
            vitals=VitalSigns(
                heart_rate=40,
                systolic_bp=70,
                diastolic_bp=40,
                respiratory_rate=8,
                spo2=85.0,
                temperature=36.2,
                gcs=6,
            ),
            medical_history=["atrial_fibrillation", "CHF", "pacemaker"],
            current_medications=["warfarin", "digoxin", "furosemide"],
            available_labs=[],
            pending_labs=[],
            imaging_available=[],
            pending_imaging=[],
            time_in_department_minutes=0,
            resource_tokens_remaining=8,
        ),
        # P2: 28yo F, broken arm, pain 6/10, stable → ESI 3, delayed
        PatientState(
            patient_id="P2",
            age=28,
            sex="F",
            chief_complaint="Deformed left forearm after fall, pain 6/10",
            onset_minutes=45,
            vitals=VitalSigns(
                heart_rate=82,
                systolic_bp=118,
                diastolic_bp=72,
                respiratory_rate=16,
                spo2=99.0,
                temperature=36.8,
                gcs=15,
            ),
            medical_history=[],
            current_medications=[],
            available_labs=[],
            pending_labs=[],
            imaging_available=[],
            pending_imaging=[],
            time_in_department_minutes=0,
            resource_tokens_remaining=6,
        ),
        # P3: 15yo M, anaphylaxis, BP 70/40, stridor → ESI 1, immediate
        PatientState(
            patient_id="P3",
            age=15,
            sex="M",
            chief_complaint="Severe allergic reaction to peanuts, facial swelling, stridor, hypotensive",
            onset_minutes=10,
            vitals=VitalSigns(
                heart_rate=130,
                systolic_bp=70,
                diastolic_bp=40,
                respiratory_rate=28,
                spo2=88.0,
                temperature=37.4,
                gcs=14,
            ),
            medical_history=["peanut_allergy", "asthma"],
            current_medications=["albuterol_prn"],
            available_labs=[],
            pending_labs=[],
            imaging_available=[],
            pending_imaging=[],
            time_in_department_minutes=0,
            resource_tokens_remaining=8,
        ),
        # P4: 60yo M, stable atrial fibrillation, known history → ESI 2, urgent
        PatientState(
            patient_id="P4",
            age=60,
            sex="M",
            chief_complaint="Palpitations and dizziness, known atrial fibrillation, rate 148",
            onset_minutes=90,
            vitals=VitalSigns(
                heart_rate=148,
                systolic_bp=110,
                diastolic_bp=68,
                respiratory_rate=18,
                spo2=97.0,
                temperature=36.9,
                gcs=15,
            ),
            medical_history=["atrial_fibrillation", "hypertension"],
            current_medications=["metoprolol", "apixaban"],
            available_labs=[],
            pending_labs=[],
            imaging_available=[],
            pending_imaging=[],
            time_in_department_minutes=0,
            resource_tokens_remaining=8,
        ),
        # P5: 35yo F, anxiety/hyperventilation, vitals normal → ESI 4, non-urgent
        PatientState(
            patient_id="P5",
            age=35,
            sex="F",
            chief_complaint="Anxiety, hyperventilation, tingling in hands, no chest pain",
            onset_minutes=60,
            vitals=VitalSigns(
                heart_rate=90,
                systolic_bp=128,
                diastolic_bp=80,
                respiratory_rate=24,
                spo2=100.0,
                temperature=36.7,
                gcs=15,
            ),
            medical_history=["generalized_anxiety_disorder", "panic_disorder"],
            current_medications=["sertraline_50mg"],
            available_labs=[],
            pending_labs=[],
            imaging_available=[],
            pending_imaging=[],
            time_in_department_minutes=0,
            resource_tokens_remaining=4,
        ),
    ]


# ─── Lab result simulation (deterministic based on test name) ──────────

# These results are returned when the agent orders a diagnostic
LAB_RESULTS_STEMI = {
    "troponin_I": LabResult(name="troponin_I", value=2.8, unit="ng/mL", reference_range="<0.04", critical=True),
    "cbc_wbc": LabResult(name="cbc_wbc", value=11.2, unit="K/uL", reference_range="4.5-11.0", critical=False),
    "bmp_glucose": LabResult(name="bmp_glucose", value=210, unit="mg/dL", reference_range="70-100", critical=False),
    "bnp": LabResult(name="bnp", value=450, unit="pg/mL", reference_range="<100", critical=True),
    "cbc": LabResult(name="cbc", value=11.2, unit="K/uL", reference_range="4.5-11.0", critical=False),
    "bmp": LabResult(name="bmp", value=138, unit="mEq/L", reference_range="136-145", critical=False),
}

LAB_RESULTS_CHEST_PAIN = {
    "d_dimer": LabResult(name="d_dimer", value=1.8, unit="mg/L", reference_range="<0.5", critical=True),
    "d-dimer": LabResult(name="d_dimer", value=1.8, unit="mg/L", reference_range="<0.5", critical=True),
    "troponin_I": LabResult(name="troponin_I", value=0.02, unit="ng/mL", reference_range="<0.04", critical=False),
    "cbc_wbc": LabResult(name="cbc_wbc", value=8.5, unit="K/uL", reference_range="4.5-11.0", critical=False),
    "bmp": LabResult(name="bmp", value=140, unit="mEq/L", reference_range="136-145", critical=False),
    "cbc": LabResult(name="cbc", value=8.5, unit="K/uL", reference_range="4.5-11.0", critical=False),
}

# Imaging results (text-based, deterministic)
IMAGING_RESULTS_STEMI = {
    "EKG": "ST-elevation in leads II, III, aVF. Acute inferior STEMI.",
    "CXR": "No acute cardiopulmonary process. Heart size normal.",
}

IMAGING_RESULTS_CHEST_PAIN = {
    "EKG": "Normal sinus rhythm. No ST changes. No ischemic findings.",
    "ecg": "Normal sinus rhythm. No ST changes. No ischemic findings.",
    "CXR": "Clear lungs bilaterally. No pleural effusion.",
    "CT_PA": "Bilateral pulmonary emboli identified in segmental and subsegmental arteries. No right heart strain.",
    "ct_pa": "Bilateral pulmonary emboli identified in segmental and subsegmental arteries. No right heart strain.",
    "CTPA": "Bilateral pulmonary emboli identified in segmental and subsegmental arteries. No right heart strain.",
    "ctpa": "Bilateral pulmonary emboli identified in segmental and subsegmental arteries. No right heart strain.",
}


def get_lab_result(task_id: str, test_name: str) -> LabResult | None:
    """Return deterministic lab result for a given test in a given task."""
    test_key = test_name.lower().strip()
    if task_id == "task_stemi_code":
        return LAB_RESULTS_STEMI.get(test_key)
    elif task_id == "task_chest_pain_workup":
        return LAB_RESULTS_CHEST_PAIN.get(test_key)
    return None


def get_imaging_result(task_id: str, imaging_name: str) -> str | None:
    """Return deterministic imaging result text."""
    img_key = imaging_name.strip()
    if task_id == "task_stemi_code":
        return IMAGING_RESULTS_STEMI.get(img_key) or IMAGING_RESULTS_STEMI.get(img_key.upper())
    elif task_id == "task_chest_pain_workup":
        # Try multiple case variants
        result = IMAGING_RESULTS_CHEST_PAIN.get(img_key)
        if not result:
            result = IMAGING_RESULTS_CHEST_PAIN.get(img_key.upper())
        if not result:
            result = IMAGING_RESULTS_CHEST_PAIN.get(img_key.lower())
        return result
    return None


def generate_patients(task_id: str) -> List[PatientState]:
    """Generate deterministic patient(s) for a given task."""
    rng = random.Random(SEED_MAP.get(task_id, 0))
    if task_id == "task_stemi_code":
        return generate_stemi_patient()
    elif task_id == "task_chest_pain_workup":
        return generate_chest_pain_patient()
    elif task_id == "task_mci_surge":
        return generate_mci_patients()
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
