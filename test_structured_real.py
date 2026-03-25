"""
Test structured_screening.py with real profiles from the database.

Tests:
- Amit Levin (should FAIL - no React/Vue/Angular)
- Uri Goldberg (should PASS - has React, Angular, Node.js)
- Eran Shoham (should FAIL - no frontend framework)
"""

import json
from pathlib import Path

# Load config
config_path = Path(__file__).parent / 'config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# Initialize Supabase client
from db import SupabaseClient, get_profile

client = SupabaseClient(config['supabase_url'], config['supabase_key'])

# Test profiles
TEST_URLS = [
    "https://www.linkedin.com/in/amit-levin-08680b103",  # Should FAIL - no React/Vue/Angular
    "https://www.linkedin.com/in/uri-goldberg-63b278147",  # Should PASS - has React, Angular, Node.js
    "https://www.linkedin.com/in/eransho",  # Should FAIL - no frontend framework
]

print("="*80)
print("STRUCTURED SCREENING TEST - Real Profiles")
print("="*80)

# Fetch profiles
profiles = []
for url in TEST_URLS:
    profile = get_profile(client, url)
    if profile:
        print(f"\nLoaded: {profile.get('name')} ({url})")
        print(f"  Skills: {profile.get('skills', [])[:10]}...")
        raw = profile.get('raw_data', {})
        profiles.append(raw)
    else:
        print(f"\nNOT FOUND: {url}")

if not profiles:
    print("\nNo profiles found in database!")
    exit(1)

# Import structured screening
from structured_screening import (
    create_default_fullstack_requirements,
    screen_profile_structured,
    RequirementType
)

# Create requirements
requirements = create_default_fullstack_requirements()

print("\n" + "="*80)
print("REQUIREMENTS:")
print("="*80)
for req in requirements.get('must_have', []):
    print(f"  MUST HAVE: {req.description} - {req.values or req.min_value or req.max_value}")
for req in requirements.get('reject_if', []):
    print(f"  REJECT IF: {req.description} - {req.values}")
for req in requirements.get('nice_to_have', []):
    print(f"  NICE TO HAVE: {req.description} (+{req.boost_points})")

# Screen each profile
print("\n" + "="*80)
print("SCREENING RESULTS:")
print("="*80)

for profile in profiles:
    name = profile.get('name') or 'Unknown'

    # Run structured screening
    result = screen_profile_structured(profile, requirements)

    print(f"\n{'='*60}")
    print(f"PROFILE: {name}")
    print(f"{'='*60}")
    print(f"  SCORE: {result.score}")
    print(f"  FIT: {result.fit}")
    print(f"  SUMMARY: {result.summary}")

    print(f"\n  CHECK DETAILS:")
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        must_have = "[MUST]" if check.requirement.is_must_have else "[NICE]"
        print(f"    {status} {must_have} {check.reason}")

    # Extract skills for reference
    skills = profile.get('skills', [])
    frontend_skills = [s for s in skills if any(fw.lower() in s.lower() for fw in ['react', 'vue', 'angular', 'next', 'svelte'])]
    backend_skills = [s for s in skills if any(bk.lower() in s.lower() for bk in ['node', 'python', 'java', 'go', 'c#', 'php', 'ruby'])]

    print(f"\n  RAW DATA:")
    print(f"    Frontend skills found: {frontend_skills if frontend_skills else 'NONE'}")
    print(f"    Backend skills found: {backend_skills[:5] if backend_skills else 'NONE'}")

    # Current title
    current_employers = profile.get('current_employers', [])
    if current_employers:
        emp = current_employers[0]
        print(f"    Current: {emp.get('employee_title')} at {emp.get('employer_name')}")

print("\n" + "="*80)
print("COMPARISON WITH OLD SCREENING:")
print("="*80)
print("Old screening results (for reference):")
print("  - Amit Levin: Score 8 (SHOULD HAVE FAILED - no React/Vue/Angular)")
print("  - Uri Goldberg: Score 9 (CORRECT - has React, Angular, Node.js)")
print("  - Eran Shoham: Score 6 (SHOULD HAVE FAILED - no frontend framework)")
print("\nStructured screening enforces hard requirements - no trade-offs allowed!")
