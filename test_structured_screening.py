"""
Test the parse_requirements function from structured_screening.py

This test verifies that parse_requirements correctly extracts structured
requirements from a natural language job description using Claude Haiku.
"""

import json
import sys
import anthropic
from structured_screening import parse_requirements, RequirementType


def load_config():
    """Load API key from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)


def test_parse_requirements():
    """Test that parse_requirements correctly extracts requirements from JD."""

    config = load_config()
    client = anthropic.Anthropic(api_key=config["anthropic_api_key"])

    job_description = """Looking for a fullstack team lead with 5+ years experience,
    must have React and Node.js, 2+ years leading a team,
    reject candidates from consultancies or banks,
    bonus if from Wiz or 8200"""

    print("=" * 60)
    print("Testing parse_requirements()")
    print("=" * 60)
    print(f"\nJob Description:\n{job_description}\n")

    # Parse the requirements
    requirements = parse_requirements(job_description, client)

    print("=" * 60)
    print("PARSED REQUIREMENTS")
    print("=" * 60)

    # Print must_have requirements
    print("\n--- MUST HAVE ---")
    for req in requirements.get("must_have", []):
        print(f"  Type: {req.type.value}")
        print(f"  Description: {req.description}")
        if req.values:
            print(f"  Values: {req.values}")
        if req.min_value is not None:
            print(f"  Min Value: {req.min_value}")
        if req.max_value is not None:
            print(f"  Max Value: {req.max_value}")
        print()

    # Print reject_if requirements
    print("\n--- REJECT IF ---")
    for req in requirements.get("reject_if", []):
        print(f"  Type: {req.type.value}")
        print(f"  Description: {req.description}")
        if req.values:
            print(f"  Values: {req.values}")
        print()

    # Print nice_to_have requirements
    print("\n--- NICE TO HAVE ---")
    for req in requirements.get("nice_to_have", []):
        print(f"  Type: {req.type.value}")
        print(f"  Description: {req.description}")
        if req.values:
            print(f"  Values: {req.values}")
        print(f"  Boost Points: {req.boost_points}")
        print()

    # Verify expected extractions
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    checks = []

    # Check for skill_frontend (React)
    frontend_reqs = [r for r in requirements.get("must_have", []) if r.type == RequirementType.SKILL_FRONTEND]
    has_react = any("react" in str(r.values).lower() for r in frontend_reqs)
    checks.append(("must_have: skill_frontend (React)", has_react, frontend_reqs))

    # Check for skill_backend (Node.js)
    backend_reqs = [r for r in requirements.get("must_have", []) if r.type == RequirementType.SKILL_BACKEND]
    has_node = any("node" in str(r.values).lower() for r in backend_reqs)
    checks.append(("must_have: skill_backend (Node.js)", has_node, backend_reqs))

    # Check for experience_years (5)
    exp_reqs = [r for r in requirements.get("must_have", []) if r.type == RequirementType.EXPERIENCE_YEARS]
    has_5_years = any(r.min_value == 5 for r in exp_reqs)
    checks.append(("must_have: experience_years (5)", has_5_years, exp_reqs))

    # Check for leadership_years (2)
    lead_reqs = [r for r in requirements.get("must_have", []) if r.type == RequirementType.LEADERSHIP_YEARS]
    has_2_years_lead = any(r.min_value == 2 for r in lead_reqs)
    checks.append(("must_have: leadership_years (2)", has_2_years_lead, lead_reqs))

    # Check reject_if for company_type (consultancies, banks)
    reject_company_reqs = [r for r in requirements.get("reject_if", []) if r.type == RequirementType.COMPANY_TYPE]
    has_consultancy = any("consult" in str(r.values).lower() for r in reject_company_reqs)
    has_bank = any("bank" in str(r.values).lower() for r in reject_company_reqs)
    checks.append(("reject_if: company_type (consultancies)", has_consultancy, reject_company_reqs))
    checks.append(("reject_if: company_type (banks)", has_bank, reject_company_reqs))

    # Check nice_to_have for custom (Wiz, 8200)
    nice_custom_reqs = [r for r in requirements.get("nice_to_have", []) if r.type == RequirementType.CUSTOM]
    has_wiz = any("wiz" in str(r.values).lower() for r in nice_custom_reqs)
    has_8200 = any("8200" in str(r.values) for r in nice_custom_reqs)
    checks.append(("nice_to_have: custom (Wiz)", has_wiz, nice_custom_reqs))
    checks.append(("nice_to_have: custom (8200)", has_8200, nice_custom_reqs))

    # Print verification results
    print("\nExpected Extractions:")
    all_passed = True
    for check_name, passed, reqs in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {check_name}")
        if not passed and reqs:
            print(f"         Found: {[(r.type.value, r.values, r.min_value) for r in reqs]}")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL VERIFICATIONS PASSED!")
    else:
        print("SOME VERIFICATIONS FAILED - see details above")
    print("=" * 60)

    return requirements, all_passed


if __name__ == "__main__":
    # Ensure output is not buffered
    sys.stdout.flush()
    requirements, passed = test_parse_requirements()
    sys.exit(0 if passed else 1)
