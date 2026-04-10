import pytest
from clinical_triage_env.server.graders.stemi_grader import grade_stemi
from clinical_triage_env.server.graders.chest_workup_grader import grade_chest_workup
from clinical_triage_env.server.graders.mci_grader import grade_mci

def test_stemi_grader_bounds():
    result = grade_stemi([])
    assert 0.0 <= result.score <= 1.0

def test_chest_workup_grader_bounds():
    result = grade_chest_workup([])
    assert 0.0 <= result.score <= 1.0

def test_mci_grader_bounds():
    result = grade_mci([])
    assert 0.0 <= result.score <= 1.0
