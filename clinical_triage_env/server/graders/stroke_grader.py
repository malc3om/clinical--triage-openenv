from clinical_triage_env.models import GradeResult

def grade_stroke(history: list[dict]) -> GradeResult:
    """
    Grader for task_stroke_code (Hard)
    Required actions:
    1. Activate stroke code/pathway (0.3)
    2. Order CT Head non-contrast (0.3)
    3. Proper ESI level 2 or 1 (0.2)
    4. Disposition to admit/stroke center (0.2)
    """
    score = 0.0
    reasons = []
    
    code_activated = False
    ct_ordered = False
    esi_correct = False
    dispo_correct = False
    
    for step in history:
        action = step.get("action", {})
        atype = action.get("action_type")
        param = action.get("parameter", "").lower()
        
        if atype == "activate_pathway" and "stroke" in param:
            if not code_activated:
                code_activated = True
                score += 0.3
                reasons.append("Activated Stroke Pathway (+0.3)")
                
        if atype == "order_diagnostic" and "ct" in param and ("head" in param or "brain" in param):
            if not ct_ordered:
                ct_ordered = True
                score += 0.3
                reasons.append("Ordered appropriate STAT neuroimaging (CT Head) (+0.3)")
                
        if atype == "assign_esi_level":
            try:
                val = int(param)
                if val in [1, 2]:
                    if not esi_correct:
                        esi_correct = True
                        score += 0.2
                        reasons.append(f"Assigned ESI level {val} (high risk) (+0.2)")
                else:
                    reasons.append(f"Penalty: Under-triaged stroke (ESI {val}) (-0.2)")
                    score -= 0.2
            except:
                pass
                
        if atype == "disposition" and ("admit" in param or "transfer" in param):
            if not dispo_correct:
                dispo_correct = True
                score += 0.2
                reasons.append("Correct disposition (+0.2)")
                
        if atype == "disposition" and "discharge" in param:
            score -= 1.0
            reasons.append("Penalty: Fatal error, discharged acute stroke patient (-1.0)")

    score = max(0.0, min(1.0, score))
    return GradeResult(score=score, explanation="; ".join(reasons))
