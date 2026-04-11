from clinical_triage_env.models import GradeResult

def grade_sepsis(history: list[dict]) -> GradeResult:
    """
    Grader for task_sepsis_alert (Medium)
    Required actions:
    1. Order lactate/lactic_acid (0.2)
    2. Read critical lactate, order antibiotics/fluids (0.3)
    3. Proper ESI level 2 (0.2)
    4. Disposition to admit/ICU (0.3)
    """
    score = 0.0
    reasons = []
    
    ordered_lactate = False
    meds_given = False
    esi_correct = False
    disposition_admit = False
    
    for step in history:
        action = step.get("action", {})
        atype = action.get("action_type")
        param = action.get("parameter", "").lower()
        
        if atype == "order_diagnostic" and ("lactate" in param or "lactic" in param or "cultures" in param):
            if not ordered_lactate:
                ordered_lactate = True
                score += 0.2
                reasons.append("Ordered sepsis workup (lactate/cultures) (+0.2)")
                
        if atype == "administer_medication" and ("fluid" in param or "antibiotic" in param or "rocephin" in param or "ceftriaxone" in param):
            if not meds_given:
                meds_given = True
                score += 0.3
                reasons.append("Administered broad-spectrum IV antibiotics/fluids (+0.3)")
                
        if atype == "assign_esi_level":
            try:
                if int(param) == 2:
                    if not esi_correct:
                        esi_correct = True
                        score += 0.2
                        reasons.append("Assigned ESI level 2 (high risk/sepsis alert) (+0.2)")
                elif int(param) > 2:
                    reasons.append("Penalty: ESI under-triaged for severe sepsis (-0.2)")
                    score -= 0.2
            except:
                pass
                
        if atype == "disposition" and ("admit" in param or "icu" in param):
            if not disposition_admit:
                disposition_admit = True
                score += 0.3
                reasons.append("Correct disposition: admission to inpatient/ICU (+0.3)")
                
        if atype == "disposition" and "discharge" in param:
            score -= 0.5
            reasons.append("Penalty: Fatal error, discharged septic patient (-0.5)")

    score = max(0.0, min(1.0, score))
    
    if score == 0.0:
        reasons.append("Agent failed to recognize and treat severe sepsis.")
        
    return GradeResult(score=score, explanation="; ".join(reasons))
