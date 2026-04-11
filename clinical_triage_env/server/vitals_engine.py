"""
vitals_engine.py — Dynamic vitals deterioration for all 6 clinical tasks.

Updates patient vitals based on elapsed time and their underlying conditions.
Each tick of dt_minutes causes realistic physiological changes:
- STEMI: progressive hemodynamic deterioration
- Anaphylaxis: airway compromise and hypotension
- Sepsis: tachycardic, hypotensive, febrile
- Stroke: blood pressure lability
- Pediatric asthma: progressive hypoxia and tachypnea
"""

from typing import List
from clinical_triage_env.models import PatientState


def update_vitals(patients: List[PatientState], dt_minutes: int) -> None:
    """
    Updates the vitals of all patients based on the elapsed time (dt_minutes)
    and their underlying conditions. Also sets vitals_trend.
    """
    for p in patients:
        # Reset trends for this step
        p.vitals_trend = {
            "HR": "→",
            "BP": "→",
            "SpO2": "→",
            "Temp": "→",
            "RR": "→",
            "GCS": "→"
        }
        
        complaint = p.chief_complaint.lower()
        
        # ── Anaphylaxis (MCI P3) ────────────────────────────────────
        if "anaphylaxis" in complaint or "allergic" in complaint:
            if not any("epi" in m.lower() for m in p.current_medications):
                drops_of_5min = dt_minutes / 5.0
                if drops_of_5min > 0:
                    bp_drop = int(10 * drops_of_5min)
                    spo2_drop = 2.0 * drops_of_5min
                    
                    if bp_drop > 0:
                        p.vitals.systolic_bp -= bp_drop
                        p.vitals.diastolic_bp -= int(bp_drop * 0.6)
                        p.vitals_trend["BP"] = "↓"
                    if spo2_drop > 0:
                        p.vitals.spo2 -= spo2_drop
                        p.vitals_trend["SpO2"] = "↓"
            else:
                p.vitals.systolic_bp += int(5 * (dt_minutes / 5))
                p.vitals.spo2 += 1.0 * (dt_minutes / 5)
                p.vitals_trend["BP"] = "↑"
                p.vitals_trend["SpO2"] = "↑"

        # ── STEMI ───────────────────────────────────────────────────
        elif "crushing chest pain" in complaint or "stemi" in complaint:
            if not any("cath" in m.lower() for m in p.current_medications):
                intervals_15m = dt_minutes / 15.0
                if intervals_15m > 0:
                    hr_inc = int(5 * intervals_15m)
                    bp_drop = int(8 * intervals_15m)
                    spo2_drop = 1.0 * intervals_15m
                    
                    if hr_inc > 0:
                        p.vitals.heart_rate += hr_inc
                        p.vitals_trend["HR"] = "↑"
                    if bp_drop > 0:
                        p.vitals.systolic_bp -= bp_drop
                        p.vitals.diastolic_bp -= int(bp_drop * 0.5)
                        p.vitals_trend["BP"] = "↓"
                    if spo2_drop > 0:
                        p.vitals.spo2 -= spo2_drop
                        p.vitals_trend["SpO2"] = "↓"
                        
        # ── Sepsis ──────────────────────────────────────────────────
        elif "fever" in complaint and ("confusion" in complaint or "chills" in complaint):
            has_antibiotics = any(
                any(med in m.lower() for med in ["antibiotic", "ceftriaxone", "rocephin"])
                for m in p.current_medications
            )
            if not has_antibiotics:
                intervals_10m = dt_minutes / 10.0
                if intervals_10m > 0:
                    p.vitals.heart_rate += int(3 * intervals_10m)
                    p.vitals.systolic_bp -= int(5 * intervals_10m)
                    p.vitals.temperature += 0.2 * intervals_10m
                    p.vitals_trend["HR"] = "↑"
                    p.vitals_trend["BP"] = "↓"
                    p.vitals_trend["Temp"] = "↑"
            else:
                # On antibiotics: slow stabilization
                intervals_10m = dt_minutes / 10.0
                if intervals_10m > 0:
                    p.vitals.heart_rate -= int(2 * intervals_10m)
                    p.vitals.systolic_bp += int(3 * intervals_10m)
                    p.vitals_trend["HR"] = "↓"
                    p.vitals_trend["BP"] = "↑"

        # ── Stroke ──────────────────────────────────────────────────
        elif "weakness" in complaint or "facial droop" in complaint or "slurred" in complaint:
            stroke_treated = any(
                any(kw in m.lower() for kw in ["tpa", "alteplase", "PATHWAY_stroke"])
                for m in p.current_medications
            )
            if not stroke_treated:
                intervals_10m = dt_minutes / 10.0
                if intervals_10m > 0:
                    # BP lability — dangerous hypertension
                    p.vitals.systolic_bp += int(5 * intervals_10m)
                    p.vitals_trend["BP"] = "↑"
                    # GCS can slowly decline
                    if p.vitals.systolic_bp > 200:
                        p.vitals.gcs = max(3, p.vitals.gcs - 1)
                        p.vitals_trend["GCS"] = "↓"

        # ── Pediatric Asthma ────────────────────────────────────────
        elif "wheezing" in complaint or "retractions" in complaint:
            has_bronchodilator = any(
                any(med in m.lower() for med in ["albuterol", "duoneb", "nebuliz"])
                for m in p.current_medications
            )
            if not has_bronchodilator:
                intervals_5m = dt_minutes / 5.0
                if intervals_5m > 0:
                    # Progressive respiratory failure
                    p.vitals.spo2 -= 1.5 * intervals_5m
                    p.vitals.respiratory_rate += int(2 * intervals_5m)
                    p.vitals.heart_rate += int(3 * intervals_5m)
                    p.vitals_trend["SpO2"] = "↓"
                    p.vitals_trend["RR"] = "↑"
                    p.vitals_trend["HR"] = "↑"
            else:
                # On bronchodilators: improvement
                intervals_5m = dt_minutes / 5.0
                if intervals_5m > 0:
                    p.vitals.spo2 += 1.0 * intervals_5m
                    p.vitals.respiratory_rate -= int(1 * intervals_5m)
                    p.vitals_trend["SpO2"] = "↑"
                    p.vitals_trend["RR"] = "↓"

        # ── Bradycardia / unresponsive (MCI P1) ────────────────────
        elif "unresponsive" in complaint or "bradycardic" in complaint:
            intervals_5m = dt_minutes / 5.0
            if intervals_5m > 0:
                p.vitals.heart_rate -= int(2 * intervals_5m)
                p.vitals.systolic_bp -= int(5 * intervals_5m)
                p.vitals.spo2 -= 1.0 * intervals_5m
                p.vitals_trend["HR"] = "↓"
                p.vitals_trend["BP"] = "↓"
                p.vitals_trend["SpO2"] = "↓"

        # ── Bounded limits (vital clamp) ────────────────────────────
        p.vitals.heart_rate = max(0, min(300, p.vitals.heart_rate))
        p.vitals.systolic_bp = max(0, min(300, p.vitals.systolic_bp))
        p.vitals.diastolic_bp = max(0, min(200, p.vitals.diastolic_bp))
        p.vitals.spo2 = max(0.0, min(100.0, p.vitals.spo2))
        p.vitals.temperature = max(30.0, min(43.0, round(p.vitals.temperature, 1)))
        p.vitals.respiratory_rate = max(0, min(60, p.vitals.respiratory_rate))
        
        # Sync GCS drop if BP critically low or SpO2 low
        if p.vitals.systolic_bp < 60 or p.vitals.spo2 < 80.0:
            p.vitals.gcs = max(3, p.vitals.gcs - 1)
            p.vitals_trend["GCS"] = "↓"
