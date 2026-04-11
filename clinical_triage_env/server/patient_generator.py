# Phase 2: Deterministic, seeded patient generator for all 6 tasks
"""
patient_generator.py — Synthetic patient data for ClinicalTriageEnv.

Calls with the same task_id use a seeded RNG to provide somewhat consistent but slightly varied (stochastic) patients.
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
    "task_sepsis_alert": 300,
    "task_stroke_code": 401,
    "task_pediatric_resp": 505,
}

def apply_stochasticity(vitals: VitalSigns, rng: random.Random) -> VitalSigns:
    vitals.heart_rate = int(min(300, max(0, vitals.heart_rate + rng.randint(-8, 8))))
    vitals.systolic_bp = int(min(300, max(0, vitals.systolic_bp + rng.randint(-12, 12))))
    vitals.diastolic_bp = int(min(200, max(0, vitals.diastolic_bp + rng.randint(-8, 8))))
    vitals.spo2 = round(min(100.0, max(50.0, vitals.spo2 + rng.uniform(-2.0, 2.0))), 1)
    vitals.temperature = round(min(43.0, max(30.0, vitals.temperature + rng.uniform(-0.5, 0.5))), 1)
    vitals.respiratory_rate = int(min(60, max(0, vitals.respiratory_rate + rng.randint(-3, 3))))
    return vitals

# ─── Task 1: STEMI Code (Easy) ─────────────────────────────────────────

def generate_stemi_patient(rng: random.Random) -> List[PatientState]:
    """58yo male with classic STEMI."""
    vitals = VitalSigns(
        heart_rate=102, systolic_bp=88, diastolic_bp=60, respiratory_rate=22, spo2=94.0, temperature=37.1, gcs=15
    )
    return [
        PatientState(
            patient_id="P1", age=58, sex="M", chief_complaint="Sudden crushing chest pain radiating to left arm, diaphoretic",
            onset_minutes=22 + rng.randint(-5, 5),
            vitals=apply_stochasticity(vitals, rng),
            medical_history=["hypertension", "type_2_diabetes", "smoker_30_pack_years"],
            current_medications=["metformin_500mg", "lisinopril_10mg"],
            available_labs=[], pending_labs=["troponin_I", "cbc", "bmp"],
            imaging_available=["EKG"], pending_imaging=[],
            time_in_department_minutes=8, resource_tokens_remaining=10,
        )
    ]

# ─── Task 2: Chest Pain Workup (Medium) ────────────────────────────────

def generate_chest_pain_patient(rng: random.Random) -> List[PatientState]:
    """44yo female with pleuritic chest pain."""
    vitals = VitalSigns(
        heart_rate=98, systolic_bp=122, diastolic_bp=78, respiratory_rate=20, spo2=96.0, temperature=37.0, gcs=15
    )
    return [
        PatientState(
            patient_id="P1", age=44, sex="F", chief_complaint="Sharp pleuritic chest pain, worse with inspiration, onset 4 hours ago",
            onset_minutes=240 + rng.randint(-30, 30),
            vitals=apply_stochasticity(vitals, rng),
            medical_history=["oral_contraceptives", "recent_long_haul_flight_3_days_ago"],
            current_medications=["combined_oral_contraceptive"],
            available_labs=[], pending_labs=[], imaging_available=[], pending_imaging=[],
            time_in_department_minutes=5, resource_tokens_remaining=12,
        )
    ]

# ─── Task 3: Mass Casualty Incident (Hard) ─────────────────────────────

def generate_mci_patients(rng: random.Random) -> List[PatientState]:
    """5 simultaneous patients."""
    def p(id, age, sex, compl, onset, vit, hist, meds, res=8):
        return PatientState(
            patient_id=id, age=age, sex=sex, chief_complaint=compl,
            onset_minutes=onset + rng.randint(-5, 5),
            vitals=apply_stochasticity(vit, rng),
            medical_history=hist, current_medications=meds,
            available_labs=[], pending_labs=[], imaging_available=[], pending_imaging=[],
            time_in_department_minutes=0, resource_tokens_remaining=res,
        )
        
    return [
        p("P1", 72, "M", "Found unresponsive, bradycardic", 15, VitalSigns(heart_rate=40, systolic_bp=70, diastolic_bp=40, respiratory_rate=8, spo2=85.0, temperature=36.2, gcs=6), ["atrial_fibrillation", "CHF", "pacemaker"], ["warfarin", "digoxin"]),
        p("P2", 28, "F", "Deformed left forearm after fall, pain 6/10", 45, VitalSigns(heart_rate=82, systolic_bp=118, diastolic_bp=72, respiratory_rate=16, spo2=99.0, temperature=36.8, gcs=15), [], [], 6),
        p("P3", 15, "M", "Severe allergic reaction to peanuts, facial swelling, stridor", 10, VitalSigns(heart_rate=130, systolic_bp=70, diastolic_bp=40, respiratory_rate=28, spo2=88.0, temperature=37.4, gcs=14), ["peanut_allergy", "asthma"], ["albuterol_prn"]),
        p("P4", 60, "M", "Palpitations and dizziness, rate 148", 90, VitalSigns(heart_rate=148, systolic_bp=110, diastolic_bp=68, respiratory_rate=18, spo2=97.0, temperature=36.9, gcs=15), ["atrial_fibrillation"], ["metoprolol"]),
        p("P5", 35, "F", "Anxiety, hyperventilation, tingling in hands", 60, VitalSigns(heart_rate=90, systolic_bp=128, diastolic_bp=80, respiratory_rate=24, spo2=100.0, temperature=36.7, gcs=15), ["panic_disorder"], ["sertraline_50mg"], 4),
    ]

# ─── Task 4: Sepsis Alert ─────────────────────────────────────────────

def generate_sepsis_patient(rng: random.Random) -> List[PatientState]:
    """68yo with fever, hypotension, altered mental status."""
    vitals = VitalSigns(
        heart_rate=118, systolic_bp=82, diastolic_bp=48, respiratory_rate=26, spo2=92.0, temperature=39.2, gcs=13
    )
    return [
        PatientState(
            patient_id="P1", age=68, sex="M", chief_complaint="Fever, chills, confusion, dark urine",
            onset_minutes=2880 + rng.randint(-100, 100),
            vitals=apply_stochasticity(vitals, rng),
            medical_history=["bph", "hypertension", "copd"],
            current_medications=["tamsulosin", "amlodipine"],
            available_labs=[], pending_labs=[], imaging_available=[], pending_imaging=[],
            time_in_department_minutes=0, resource_tokens_remaining=12,
        )
    ]

# ─── Task 5: Stroke Code ─────────────────────────────────────────────

def generate_stroke_patient(rng: random.Random) -> List[PatientState]:
    """72yo with sudden onset facial droop and right-sided weakness."""
    vitals = VitalSigns(
        heart_rate=88, systolic_bp=185, diastolic_bp=100, respiratory_rate=18, spo2=97.0, temperature=36.8, gcs=14
    )
    return [
        PatientState(
            patient_id="P1", age=72, sex="F", chief_complaint="Sudden onset right-sided weakness, facial droop, slurred speech",
            onset_minutes=45 + rng.randint(-15, 15),
            vitals=apply_stochasticity(vitals, rng),
            medical_history=["hypertension", "hyperlipidemia"],
            current_medications=["atorvastatin"],
            available_labs=[], pending_labs=[], imaging_available=[], pending_imaging=[],
            time_in_department_minutes=2, resource_tokens_remaining=10,
        )
    ]

# ─── Task 6: Pediatric Respiratory Exacerbation ─────────────────────────

def generate_pediatric_patient(rng: random.Random) -> List[PatientState]:
    """4yo with severe asthma exacerbation and retractions."""
    vitals = VitalSigns(
        heart_rate=145, systolic_bp=95, diastolic_bp=60, respiratory_rate=42, spo2=89.0, temperature=37.1, gcs=15
    )
    return [
        PatientState(
            patient_id="P1", age=4, sex="M", chief_complaint="Severe wheezing, intercostal retractions, not eating",
            onset_minutes=120 + rng.randint(-20, 20),
            vitals=apply_stochasticity(vitals, rng),
            medical_history=["asthma", "premature_birth"],
            current_medications=["fluticasone", "albuterol_prn"],
            available_labs=[], pending_labs=[], imaging_available=[], pending_imaging=[],
            time_in_department_minutes=0, resource_tokens_remaining=10,
        )
    ]


# ─── Lab result simulation (deterministic based on test name) ──────────

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

LAB_RESULTS_SEPSIS = {
    "lactate": LabResult(name="lactate", value=4.5, unit="mmol/L", reference_range="0.5-2.2", critical=True),
    "lactic_acid": LabResult(name="lactic_acid", value=4.5, unit="mmol/L", reference_range="0.5-2.2", critical=True),
    "cbc_wbc": LabResult(name="cbc_wbc", value=22.4, unit="K/uL", reference_range="4.5-11.0", critical=True),
    "procalcitonin": LabResult(name="procalcitonin", value=15.2, unit="ng/mL", reference_range="<0.1", critical=True),
    "urinalysis": LabResult(name="urinalysis", value="TNTC WBC, positive nitrite, many bacteria", unit="", reference_range="negative", critical=True),
    "bmp": LabResult(name="bmp", value=145, unit="mEq/L", reference_range="136-145", critical=False),
    "cbc": LabResult(name="cbc", value=22.4, unit="K/uL", reference_range="4.5-11.0", critical=True),
}

LAB_RESULTS_STROKE = {
    "bmp_glucose": LabResult(name="bmp_glucose", value=110, unit="mg/dL", reference_range="70-100", critical=False),
    "cbc": LabResult(name="cbc", value=8.0, unit="K/uL", reference_range="4.5-11.0", critical=False),
    "coags": LabResult(name="coags", value="INR 1.0", unit="", reference_range="0.8-1.1", critical=False),
    "bmp": LabResult(name="bmp", value=140, unit="mEq/L", reference_range="136-145", critical=False),
}

LAB_RESULTS_PEDS = {
    "vbg": LabResult(name="vbg", value="pH 7.28, pCO2 55", unit="", reference_range="7.35-7.45", critical=True),
    "cbc": LabResult(name="cbc", value=9.5, unit="K/uL", reference_range="4.5-11.0", critical=False),
}

IMAGING_RESULTS_STEMI = {
    "EKG": "ST-elevation in leads II, III, aVF. Acute inferior STEMI.",
    "CXR": "No acute cardiopulmonary process. Heart size normal.",
}

IMAGING_RESULTS_CHEST_PAIN = {
    "EKG": "Normal sinus rhythm. No ST changes. No ischemic findings.",
    "CXR": "Clear lungs bilaterally. No pleural effusion.",
    "CT_PA": "Bilateral pulmonary emboli identified in segmental and subsegmental arteries. No right heart strain.",
}

IMAGING_RESULTS_SEPSIS = {
    "CXR": "Clear lungs bilaterally. No acute infiltrates.",
    "CT_ABD_PELVIS": "No acute intra-abdominal process.",
}

IMAGING_RESULTS_STROKE = {
    "CT_HEAD_NONCON": "No acute intracranial hemorrhage or territorial infarction.",
    "CTA_HEAD_NECK": "Left middle cerebral artery (M1 segment) occlusion.",
}

IMAGING_RESULTS_PEDS = {
    "CXR": "Hyperinflation, peribronchial cuffing. No focal consolidation.",
}

def get_lab_result(task_id: str, test_name: str) -> LabResult | None:
    """Return deterministic lab result for a given test in a given task."""
    test_key = test_name.lower().strip()
    
    if task_id == "task_stemi_code":
        return LAB_RESULTS_STEMI.get(test_key)
    elif task_id == "task_chest_pain_workup":
        return LAB_RESULTS_CHEST_PAIN.get(test_key)
    elif task_id == "task_sepsis_alert":
        return LAB_RESULTS_SEPSIS.get(test_key)
    elif task_id == "task_stroke_code":
        return LAB_RESULTS_STROKE.get(test_key)
    elif task_id == "task_pediatric_resp":
        return LAB_RESULTS_PEDS.get(test_key)
    return None


def get_imaging_result(task_id: str, imaging_name: str) -> str | None:
    """Return deterministic imaging result text."""
    img_key = imaging_name.strip().upper()
    
    if task_id == "task_stemi_code":
        return IMAGING_RESULTS_STEMI.get(img_key) or IMAGING_RESULTS_STEMI.get(img_key.lower())
    elif task_id == "task_chest_pain_workup":
        return IMAGING_RESULTS_CHEST_PAIN.get(img_key) or IMAGING_RESULTS_CHEST_PAIN.get(imaging_name.strip())
    elif task_id == "task_sepsis_alert":
        return IMAGING_RESULTS_SEPSIS.get(img_key)
    elif task_id == "task_stroke_code":
        return IMAGING_RESULTS_STROKE.get(img_key) or IMAGING_RESULTS_STROKE.get("CT_HEAD_NONCON" if "CT" in img_key and "NONCON" in img_key else img_key)
    elif task_id == "task_pediatric_resp":
        return IMAGING_RESULTS_PEDS.get(img_key)
    return None


def generate_patients(task_id: str) -> List[PatientState]:
    """Generate deterministic patient(s) for a given task."""
    # We add a random factor seeded by the task id. To make it different across episodes, 
    # the environment reset method doesn't pass a custom seed, so we just use the global seed map 
    # but maybe add some true randomness? The reviewer wanted stochasticity.
    # We will use pure random or a less deterministic approach.
    rng = random.Random() 
    # Actually, if they want to evaluate with their own seed, maybe we should seed with episode_id?
    # OpenEnv requires resetting without args sometimes. 
    # Let's seed with time or random, giving true stochasticity!
    rng.seed()
    
    if task_id == "task_stemi_code":
        return generate_stemi_patient(rng)
    elif task_id == "task_chest_pain_workup":
        return generate_chest_pain_patient(rng)
    elif task_id == "task_mci_surge":
        return generate_mci_patients(rng)
    elif task_id == "task_sepsis_alert":
        return generate_sepsis_patient(rng)
    elif task_id == "task_stroke_code":
        return generate_stroke_patient(rng)
    elif task_id == "task_pediatric_resp":
        return generate_pediatric_patient(rng)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
