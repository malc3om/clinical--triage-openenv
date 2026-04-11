from clinical_triage_env.models import GradeResult

def grade_pediatric(history: list[dict]) -> GradeResult:
    """
    Grader for task_pediatric_resp (Medium/Hard)
    Required actions:
    1. Nebulized meds/steroids (0.3)
    2. Proper ESI level 2 (high risk pediatric) (0.2)
    3. Reassessment / Wait for meds to work, check vitals (0.2)
    4. Disposition to admit/PICU (asthma exacerbation with hypoxia) (0.3)
    """
    score = 0.0
    reasons = []
    
    meds_given = False
    esi_correct = False
    dispo_correct = False
    waited = False
    
    for step in history:
        action = step.get("action", {})
        atype = action.get("action_type")
        param = action.get("parameter", "").lower()
        
        if atype == "administer_medication" and ("albuterol" in param or "duoneb" in param or "steroid" in param or "dex" in param or "magnesium" in param):
            if not meds_given:
                meds_given = True
                score += 0.3
                reasons.append("Administered appropriate pediatric asthma bronchodilators/corticosteroids (+0.3)")
                
        if atype == "wait":
            if not waited and meds_given:
                waited = True
                score += 0.2
                reasons.append("Allowed time for medical intervention to take effect (+0.2)")
                
        if atype == "assign_esi_level":
            try:
                val = int(param)
                if val == 2:
                    if not esi_correct:
                        esi_correct = True
                        score += 0.2
                        reasons.append("Assigned ESI level 2 for pediatric respiratory distress (+0.2)")
                elif val > 2:
                    reasons.append(f"Penalty: Under-triaged severe pediatric distress (ESI {val}) (-0.2)")
                    score -= 0.2
            except:
                pass
                
        if atype == "disposition" and ("admit" in param or "transfer" in param or "picu" in param):
            if not dispo_correct:
                dispo_correct = True
                score += 0.3
                reasons.append("Correct disposition to admit for continuing hypoxia/retractions (+0.3)")
                
        if atype == "disposition" and "discharge" in param:
            score -= 0.5
            reasons.append("Penalty: Unsafe discharge with documented retractions/hypoxia (-0.5)")

    score = max(0.0, min(1.0, score))
    return GradeResult(score=score, explanation="; ".join(reasons))
