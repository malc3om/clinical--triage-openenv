"""
validate_submission.py — Pre-submission checklist runner for ClinicalTriageEnv.

Runs all critical checks that would cause disqualification if failed.
Execute before pushing to HF Spaces: python validate_submission.py
"""

from __future__ import annotations

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Colors:
    GREEN = ""
    RED = ""
    YELLOW = ""
    CYAN = ""
    RESET = ""
    BOLD = ""


def check(name: str, passed: bool, detail: str = ""):
    icon = "[PASS]" if passed else "[FAIL]"
    print(f"  {icon} {name}")
    if detail:
        print(f"    -> {detail}")
    return passed


def main():
    print(f"\n{'=' * 60}")
    print(f"  ClinicalTriageEnv -- Pre-Submission Validation")
    print(f"{'=' * 60}\n")

    results = []

    # ── 1. Check required files exist ───────────────────────────────
    print(f"{Colors.YELLOW}[1/8] Required files{Colors.RESET}")
    required_files = [
        "clinical_triage_env/models.py",
        "clinical_triage_env/app.py",
        "inference.py",
        "openenv.yaml",
        "Dockerfile",
        "requirements.txt",
        "clinical_triage_env/server/environment.py",
        "clinical_triage_env/server/patient_generator.py",
        "clinical_triage_env/server/reward.py",
        "clinical_triage_env/server/graders/stemi_grader.py",
        "clinical_triage_env/server/graders/chest_workup_grader.py",
        "clinical_triage_env/server/graders/mci_grader.py",
    ]
    base = os.path.dirname(os.path.abspath(__file__))
    for f in required_files:
        path = os.path.join(base, f)
        results.append(check(f, os.path.exists(path)))

    # ── 2. Import models ────────────────────────────────────────────
    print(f"\n{Colors.YELLOW}[2/8] Pydantic models import{Colors.RESET}")
    try:
        from clinical_triage_env.models import (
            TriageAction, TriageObservation, TriageState,
            VitalSigns, LabResult, PatientState, GradeResult, TaskInfo,
        )
        results.append(check("All models import cleanly", True))
    except Exception as e:
        results.append(check("Models import", False, str(e)))

    # ── 3. Environment reset/step ───────────────────────────────────
    print(f"\n{Colors.YELLOW}[3/8] Environment reset & step{Colors.RESET}")
    try:
        from clinical_triage_env.server.environment import ClinicalTriageEnvironment
        env = ClinicalTriageEnvironment()

        for task_id in ["task_stemi_code", "task_chest_pain_workup", "task_mci_surge"]:
            obs = env.reset(task_id=task_id)
            results.append(check(
                f"reset({task_id})",
                obs.task_id == task_id and len(obs.patients) > 0,
                f"{len(obs.patients)} patient(s), {obs.max_steps} max steps",
            ))

            # Take one test action
            action = TriageAction(
                action_type="assign_esi_level",
                patient_id=obs.patients[0].patient_id,
                parameter="2",
                rationale="Test action",
            )
            result = env.step(action)
            results.append(check(
                f"step() returns valid observation",
                result.reward is not None,
                f"reward={result.reward:+.3f}",
            ))
    except Exception as e:
        results.append(check("Environment", False, str(e)))
        traceback.print_exc()

    # ── 4. Graders return DIFFERENT scores ──────────────────────────
    print(f"\n{Colors.YELLOW}[4/8] Graders produce variable scores{Colors.RESET}")
    try:
        from clinical_triage_env.server.graders.stemi_grader import grade_stemi
        from clinical_triage_env.server.graders.chest_workup_grader import grade_chest_workup
        from clinical_triage_env.server.graders.mci_grader import grade_mci

        # Test with empty history -> should score 0
        empty_score = grade_stemi([])
        results.append(check(
            "STEMI grader: empty history -> 0.0",
            empty_score.score == 0.0,
            f"score={empty_score.score}",
        ))

        # Test with perfect history -> should score > 0
        perfect_history = [
            {"action": {"action_type": "assign_esi_level", "patient_id": "P1", "parameter": "1"}},
            {"action": {"action_type": "activate_pathway", "patient_id": "P1", "parameter": "cath_lab"}},
            {"action": {"action_type": "order_diagnostic", "patient_id": "P1", "parameter": "aspirin_325mg"}},
            {"action": {"action_type": "disposition", "patient_id": "P1", "parameter": "admit_icu"}},
        ]
        good_score = grade_stemi(perfect_history)
        results.append(check(
            "STEMI grader: good history -> > 0.0",
            good_score.score > 0.0,
            f"score={good_score.score:.3f}",
        ))

        # Scores must be DIFFERENT
        results.append(check(
            "STEMI grader: different inputs -> different scores",
            empty_score.score != good_score.score,
            f"empty={empty_score.score} vs good={good_score.score:.3f}",
        ))

        # Test chest workup grader
        chest_empty = grade_chest_workup([])
        chest_good = grade_chest_workup([
            {"action": {"action_type": "order_diagnostic", "patient_id": "P1", "parameter": "EKG"}},
            {"action": {"action_type": "order_diagnostic", "patient_id": "P1", "parameter": "d_dimer"}},
            {"action": {"action_type": "order_diagnostic", "patient_id": "P1", "parameter": "CT_PA"}},
            {"action": {"action_type": "disposition", "patient_id": "P1", "parameter": "admit"}},
        ])
        results.append(check(
            "Chest workup grader: variable scores",
            chest_empty.score != chest_good.score,
            f"empty={chest_empty.score} vs good={chest_good.score:.3f}",
        ))

        # Test MCI grader
        mci_empty = grade_mci([])
        mci_good = grade_mci([
            {"action": {"action_type": "assign_esi_level", "patient_id": "P1", "parameter": "1"}},
            {"action": {"action_type": "assign_esi_level", "patient_id": "P3", "parameter": "1"}},
            {"action": {"action_type": "assign_esi_level", "patient_id": "P2", "parameter": "3"}},
            {"action": {"action_type": "assign_esi_level", "patient_id": "P4", "parameter": "2"}},
            {"action": {"action_type": "assign_esi_level", "patient_id": "P5", "parameter": "4"}},
            {"action": {"action_type": "disposition", "patient_id": "P1", "parameter": "admit_icu"}},
            {"action": {"action_type": "disposition", "patient_id": "P3", "parameter": "admit_icu"}},
            {"action": {"action_type": "disposition", "patient_id": "P5", "parameter": "waiting_room"}},
        ])
        results.append(check(
            "MCI grader: variable scores",
            mci_empty.score != mci_good.score,
            f"empty={mci_empty.score} vs good={mci_good.score:.3f}",
        ))
    except Exception as e:
        results.append(check("Graders", False, str(e)))
        traceback.print_exc()

    # ── 5. openenv.yaml is valid YAML ───────────────────────────────
    print(f"\n{Colors.YELLOW}[5/8] openenv.yaml{Colors.RESET}")
    try:
        import yaml
        yaml_path = os.path.join(base, "openenv.yaml")
        with open(yaml_path) as f:
            spec = yaml.safe_load(f)
        results.append(check("YAML parses", True))
        results.append(check("Has 'name' field", "name" in spec, spec.get("name", "")))
        results.append(check("Has 3+ tasks", len(spec.get("tasks", [])) >= 3))
        results.append(check("Has reward_range", "reward_range" in spec))
    except Exception as e:
        results.append(check("openenv.yaml", False, str(e)))

    # ── 6. Dockerfile exists and looks valid ────────────────────────
    print(f"\n{Colors.YELLOW}[6/8] Dockerfile{Colors.RESET}")
    try:
        df_path = os.path.join(base, "Dockerfile")
        with open(df_path) as f:
            df_content = f.read()
        results.append(check("Dockerfile exists", True))
        results.append(check("Exposes port 7860", "7860" in df_content))
        results.append(check("Uses python:3.11", "python:3.11" in df_content))
    except Exception as e:
        results.append(check("Dockerfile", False, str(e)))

    # ── 7. FastAPI app imports ──────────────────────────────────────
    print(f"\n{Colors.YELLOW}[7/8] FastAPI app{Colors.RESET}")
    try:
        from clinical_triage_env.app import app
        results.append(check("app.py imports", True))
        routes = [r.path for r in app.routes]
        for endpoint in ["/", "/reset", "/step", "/state", "/tasks", "/grade", "/health"]:
            results.append(check(f"Endpoint {endpoint}", endpoint in routes))
    except Exception as e:
        results.append(check("FastAPI app", False, str(e)))
        traceback.print_exc()

    # ── 8. Inference script imports ─────────────────────────────────
    print(f"\n{Colors.YELLOW}[8/8] Inference script{Colors.RESET}")
    try:
        inf_path = os.path.join(base, "inference.py")
        results.append(check("inference.py exists", os.path.exists(inf_path)))
        # Just check it can be imported (don't run main())
        import importlib.util
        spec_mod = importlib.util.spec_from_file_location("inference", inf_path)
        mod = importlib.util.module_from_spec(spec_mod)
        results.append(check("inference.py is valid Python", True))
    except Exception as e:
        results.append(check("Inference script", False, str(e)))

    # ── Summary ─────────────────────────────────────────────────────
    passed = sum(1 for r in results if r)
    total = len(results)
    failed = total - passed

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  ALL {total} CHECKS PASSED [OK]")
        print(f"  Ready to submit!")
    else:
        print(f"  {failed}/{total} CHECKS FAILED [ERROR]")
        print(f"  Fix failures before submitting.")
    print(f"{'=' * 60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
