from generator import build_prompt

def test_prompt_contains_claim():
    p = build_prompt("hello")
    assert "hello" in p
