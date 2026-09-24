"""
Test the parse_requirements function from structured_screening.py

The default test uses a fake Anthropic client with a canned JSON reply, so it
costs nothing and needs no config.json. The live variant calls the real
Claude Haiku API and only runs with SOURCINGX_LIVE_API=1.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

from structured_screening import parse_requirements, RequirementType


JOB_DESCRIPTION = """Looking for a fullstack team lead with 5+ years experience,
    must have React and Node.js, 2+ years leading a team,
    reject candidates from consultancies or banks,
    bonus if from Wiz or 8200"""

# Shaped like PARSER_PROMPT's example output; wrapped in a code fence the way
# Haiku often answers, so _parse_json_response's fence stripping is exercised.
CANNED_RESPONSE = "```json\n" + json.dumps({
    "must_have": [
        {"type": "skill_frontend", "description": "Has React", "values": ["React"]},
        {"type": "skill_backend", "description": "Has Node.js", "values": ["Node.js"]},
        {"type": "experience_years", "description": "5+ years fullstack", "min_value": 5},
        {"type": "leadership_years", "description": "2+ years team lead", "min_value": 2},
    ],
    "nice_to_have": [
        {"type": "custom", "description": "Wiz or 8200 background", "values": ["Wiz", "8200"], "boost": 2},
    ],
    "reject_if": [
        {"type": "company_type", "description": "Reject consultancies", "values": ["consulting", "outsourcing"]},
        {"type": "company_type", "description": "Reject banks", "values": ["bank"]},
    ],
}, indent=2) + "\n```"


class FakeAnthropicClient:
    """Stands in for anthropic.Anthropic: records calls, returns canned text."""

    def __init__(self, text):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._text = text

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


def _check_requirements(requirements):
    """Assert the JD's key requirements were extracted."""
    must = requirements.get("must_have", [])
    reject = requirements.get("reject_if", [])
    nice = requirements.get("nice_to_have", [])

    def of_type(reqs, t):
        return [r for r in reqs if r.type == t]

    frontend = of_type(must, RequirementType.SKILL_FRONTEND)
    assert any("react" in str(r.values).lower() for r in frontend), frontend

    backend = of_type(must, RequirementType.SKILL_BACKEND)
    assert any("node" in str(r.values).lower() for r in backend), backend

    exp = of_type(must, RequirementType.EXPERIENCE_YEARS)
    assert any(r.min_value == 5 for r in exp), exp

    lead = of_type(must, RequirementType.LEADERSHIP_YEARS)
    assert any(r.min_value == 2 for r in lead), lead

    reject_company = of_type(reject, RequirementType.COMPANY_TYPE)
    assert any("consult" in str(r.values).lower() for r in reject_company), reject_company
    assert any("bank" in str(r.values).lower() for r in reject_company), reject_company

    nice_custom = of_type(nice, RequirementType.CUSTOM)
    assert any("wiz" in str(r.values).lower() for r in nice_custom), nice_custom
    assert any("8200" in str(r.values) for r in nice_custom), nice_custom


def test_parse_requirements():
    """parse_requirements turns the model's JSON into Requirement objects."""
    client = FakeAnthropicClient(CANNED_RESPONSE)

    requirements = parse_requirements(JOB_DESCRIPTION, client)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5-20251001"
    assert JOB_DESCRIPTION in call["messages"][0]["content"]

    _check_requirements(requirements)

    # Flags parse_requirements sets on top of the model's JSON.
    assert all(r.is_must_have for r in requirements["must_have"])
    assert all(r.is_must_have for r in requirements["reject_if"])
    assert all(not r.is_must_have for r in requirements["nice_to_have"])
    assert requirements["nice_to_have"][0].boost_points == 2


@pytest.mark.live_api
def test_parse_requirements_live():
    """Same checks against the real Claude Haiku API (costs money)."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        with open("config.json", "r") as f:
            api_key = json.load(f)["anthropic_api_key"]
    client = anthropic.Anthropic(api_key=api_key)

    _check_requirements(parse_requirements(JOB_DESCRIPTION, client))


if __name__ == "__main__":
    # Runs the free, mocked test. For the live one:
    #   SOURCINGX_LIVE_API=1 pytest test_structured_screening.py -m live_api
    test_parse_requirements()
    print("test_parse_requirements: PASS")
    sys.exit(0)
