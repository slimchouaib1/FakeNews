# backend/CounterArg_Service/tests/test_prompt.py
from backend.CounterArg_Service.generator import build_prompt

def test_build_prompt():
    p = build_prompt("Some claim")
    assert "CLAIM:" in p
    assert "COUNTER-ARGUMENT" in p
