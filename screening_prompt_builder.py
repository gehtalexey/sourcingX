"""
Screening Prompt Builder

Builds a single API call from structured text boxes.
User defines must-haves, nice-to-haves, reject-ifs in UI.
This module assembles them into a user prompt for the generic system prompt.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# GENERIC SYSTEM PROMPT - Teaches AI HOW to evaluate
# ============================================================================

STRUCTURED_SYSTEM_PROMPT = """You are a technical recruiter screening candidates for Israeli tech companies.

# YOUR TASK
1. Read the REQUIREMENTS provided by the user
2. Check EACH requirement against the candidate's profile
3. Output PASS or FAIL for each requirement
4. Apply stability caps
5. Calculate final score and category

# GATE CHECKING RULES

## Must-Have Requirements
- Each must-have MUST be checked individually
- Output "GATE X: [requirement] → PASS/FAIL (evidence)"
- If ANY must-have fails → Category "No Fit", Score 1-2
- You CANNOT trade off one requirement against another
- Missing information = FAIL (don't assume skills not listed)

## Reject-If Rules
- These are instant disqualifiers
- If ANY reject-if matches → Category "No Fit", Score 1-2
- Check these alongside must-haves

## Nice-to-Have Requirements
- Only check these if ALL must-haves passed
- Each adds bonus points to base score of 6

# STABILITY RULES (CRITICAL - caps final score)

Use the pre-calculated STABILITY SUMMARY provided. Apply these caps:
- 3+ short-stint companies (<12 months each) → MAX SCORE 4
- Current role <6 months → MAX SCORE 5
- Both conditions → MAX SCORE 4

Output: "STABILITY: [X] short stints, current=[Y] months → PASS/CAPPED AT [Z]"

# EXPERIENCE CALCULATION RULES

Role durations are PRE-CALCULATED. Use those numbers, do NOT recalculate from dates.

What counts as relevant experience:
- Count FULL: Software Engineer, Developer, Frontend/Backend/Fullstack Engineer, Tech Lead, Team Lead, Staff Engineer, Engineering Manager
- Count HALF: DevOps/SRE/Platform Engineer (only if skills show coding: APIs, services)
- Count HALF: Military service (IDF, 8200, Mamram, Talpiot, C4I, IAF)
- DO NOT count: QA, Project Manager, Product Manager, Customer Success, Support, IT, Data Analyst, Business roles

Leadership experience (for team lead roles):
- Can be from ANY role in history (current OR past)
- Look for: Team Lead, Tech Lead, Engineering Manager, CTO, VP Engineering, Director
- Military officer roles count as leadership

# SKILLS EVALUATION

## Frontend Skills
- PASS: React, Vue, Angular (2+), Next.js, Svelte
- FAIL: JavaScript/TypeScript alone (no framework)
- FAIL: jQuery only (outdated)
- FAIL: AngularJS (v1, outdated - different from Angular 2+)
- FAIL: React Native alone (mobile, not web frontend)

## Backend Skills
- PASS: Node.js, Python, Java, Go, C#/.NET, PHP, Ruby, Scala, Kotlin
- FAIL: JavaScript alone (could be frontend only)
- FAIL: SQL alone (database, not backend programming)
- Note: C/C++ at security companies = valid backend

## Fullstack
- Must have BOTH frontend framework AND backend technology
- "Software Engineer" at Israeli startups often = fullstack (verify with skills)

# COMPANY EVALUATION

Read `employer_description` carefully. Don't guess from company name.

## Good Companies (software product focus):
- SaaS / B2B software products
- Cybersecurity companies
- DevTools / Developer platforms
- Fintech (software-focused, not traditional banking)
- Top Israeli tech: Wiz, Monday, Snyk, Wix, AppsFlyer, Fiverr, JFrog, Taboola, ironSource

## Bad Companies (reject for product company roles):
- IT consulting / outsourcing / body shops: Ness, Matrix, Malam Team, Accenture
- E-commerce / retail (selling products, not building software)
- Ticketing / travel / events
- Traditional banking / insurance / telecom
- Marketing / creative agencies
- Hardware companies (unless software role)

# ISRAELI MARKET CONTEXT

## Top Signal Boosters:
+2 points: Wiz, Monday, Snyk, Wix, AppsFlyer, Fiverr
+2 points: 8200, Mamram, Talpiot (elite military units)
+1 point: Microsoft IL, Google IL, Meta IL, Amazon IL
+1 point: Check Point, CyberArk, JFrog, Taboola
+1 point: Technion, TAU, Hebrew University (CS degrees)
+1 point: 3+ years at current company (stability)

## Military Service:
- Israeli military is mandatory (ages 18-21)
- Exclude from "max years experience" calculations
- 8200 = signals intelligence (technical)
- Mamram = IT corps (technical)
- Talpiot = elite tech program (very strong signal)

# TITLE EVALUATION (for team lead roles)

## Accept:
- Team Lead, Tech Lead, Engineering Team Lead
- Software Team Lead, Development Team Lead
- Engineering Manager (if hands-on)

## Reject:
- "Backend Team Lead", "Backend Infra Lead" → backend-only
- "Frontend Team Lead" → frontend-only
- "DevOps Lead", "Platform Lead", "SRE Lead" → infra-only
- "Data Team Lead", "ML Lead", "AI Lead" → data/ML-only
- "VP", "Director", "CTO", "Chief", "Head of" → overqualified

# SCORING

Base score after all gates pass: 6

Apply boosters (nice-to-haves): +1 to +2 each
Apply stability cap: may reduce max score

Final categories:
- Score 9-10 (Category A): All must-haves + most nice-to-haves + top signals
- Score 7-8 (Category B): All must-haves + some nice-to-haves
- Score 6 (Category C): All must-haves only, minimal extras
- Score 1-2 (No Fit): Failed any must-have or matched reject-if

# OUTPUT FORMAT (required)

```
GATE 1: [requirement text] → PASS/FAIL (evidence)
GATE 2: [requirement text] → PASS/FAIL (evidence)
...

REJECT CHECK: [rules checked] → CLEAR/REJECTED ([reason])

STABILITY: [X] short stints, current=[Y] months → PASS/CAPPED AT [Z]

NICE-TO-HAVE:
+X: [requirement] → MET/NOT MET
...

SUMMARY:
Gates Passed: X/Y
Stability: PASS/CAPPED
Category: [A/B/C/No Fit]
Score: [1-10]
Reason: [1-2 sentences explaining the score]
```"""


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RequirementInput:
    """A single requirement from the UI text box."""
    text: str
    boost_points: int = 1  # For nice-to-haves


# ============================================================================
# USER PROMPT BUILDER
# ============================================================================

def build_user_prompt(
    must_haves: List[str],
    nice_to_haves: List[Dict[str, any]],  # {"text": str, "points": int}
    reject_ifs: List[str],
    profile_data: Dict,
    stability_summary: str = "",
    experience_summary: str = "",
) -> str:
    """
    Build the user prompt from structured inputs.

    Args:
        must_haves: List of must-have requirement strings
        nice_to_haves: List of {"text": str, "points": int} dicts
        reject_ifs: List of reject-if condition strings
        profile_data: The candidate profile data
        stability_summary: Pre-calculated stability info
        experience_summary: Pre-calculated experience info

    Returns:
        Complete user prompt string
    """

    sections = []

    # Header
    sections.append("# REQUIREMENTS TO CHECK")

    # Must-Haves
    sections.append("\n## MUST-HAVE (all required, any fail = No Fit)")
    for i, req in enumerate(must_haves, 1):
        if req.strip():
            sections.append(f"{i}. {req.strip()}")

    if not must_haves or not any(r.strip() for r in must_haves):
        sections.append("(No must-have requirements specified)")

    # Reject-Ifs
    if reject_ifs and any(r.strip() for r in reject_ifs):
        sections.append("\n## REJECT IF (any match = No Fit)")
        for i, req in enumerate(reject_ifs, 1):
            if req.strip():
                sections.append(f"- {req.strip()}")

    # Nice-to-Haves
    if nice_to_haves and any(n.get("text", "").strip() for n in nice_to_haves):
        sections.append("\n## NICE-TO-HAVE (bonus points if met)")
        for n in nice_to_haves:
            text = n.get("text", "").strip()
            points = n.get("points", 1)
            if text:
                sections.append(f"+{points}: {text}")

    # Pre-calculated summaries
    if stability_summary:
        sections.append(f"\n## STABILITY SUMMARY (pre-calculated)\n{stability_summary}")

    if experience_summary:
        sections.append(f"\n## EXPERIENCE SUMMARY (pre-calculated)\n{experience_summary}")

    # Profile data
    sections.append("\n# CANDIDATE PROFILE")
    sections.append(format_profile_for_prompt(profile_data))

    # Instruction
    sections.append("\n---\nScreen this candidate against the requirements above.")

    return "\n".join(sections)


def format_profile_for_prompt(profile: Dict) -> str:
    """Format profile data for inclusion in prompt."""

    lines = []

    # Basic info
    name = profile.get("name", "Unknown")
    lines.append(f"Name: {name}")

    # Skills
    skills = profile.get("skills", [])
    if skills:
        lines.append(f"Skills: {', '.join(skills[:20])}")

    # Current role
    current_employers = profile.get("current_employers", [])
    if current_employers:
        curr = current_employers[0]
        title = curr.get("employee_title", "N/A")
        company = curr.get("employer_name", "N/A")
        description = curr.get("employer_linkedin_description", "")
        start = curr.get("start_date", "")

        lines.append(f"\nCurrent Role: {title} at {company}")
        if start:
            lines.append(f"Start Date: {start}")
        if description:
            lines.append(f"employer_description: {description[:300]}")

    # Past roles
    past_employers = profile.get("past_employers", [])
    if past_employers:
        lines.append(f"\nPast Roles ({len(past_employers)} positions):")
        for emp in past_employers[:8]:  # Limit to 8 past roles
            title = emp.get("employee_title", "N/A")
            company = emp.get("employer_name", "N/A")
            start = emp.get("start_date", "")
            end = emp.get("end_date", "")
            description = emp.get("employer_linkedin_description", "")

            date_range = f"{start} - {end}" if start else ""
            lines.append(f"  - {title} at {company} ({date_range})")
            if description:
                lines.append(f"    employer_description: {description[:150]}")

    # Education
    education = profile.get("education", []) or profile.get("all_schools", [])
    if education:
        if isinstance(education, list) and education:
            if isinstance(education[0], dict):
                schools = [e.get("school_name", e.get("name", "")) for e in education]
            else:
                schools = education
            lines.append(f"\nEducation: {', '.join(schools[:3])}")

    return "\n".join(lines)


# ============================================================================
# SCREENING FUNCTION
# ============================================================================

def screen_with_structured_prompt(
    profile: Dict,
    must_haves: List[str],
    nice_to_haves: List[Dict],
    reject_ifs: List[str],
    client,  # Anthropic client
    stability_summary: str = "",
    experience_summary: str = "",
    model: str = "claude-haiku-4-5-20251001",
) -> Dict:
    """
    Screen a single profile using structured prompt.

    Returns:
        Dict with keys: score, category, reasoning, raw_response
    """

    user_prompt = build_user_prompt(
        must_haves=must_haves,
        nice_to_haves=nice_to_haves,
        reject_ifs=reject_ifs,
        profile_data=profile,
        stability_summary=stability_summary,
        experience_summary=experience_summary,
    )

    response = client.messages.create(
        model=model,
        max_tokens=800,
        system=STRUCTURED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0,
    )

    raw_response = response.content[0].text

    # Parse score and category from response
    score = 0
    category = "Unknown"

    # Extract score
    import re
    score_match = re.search(r'Score:\s*(\d+)', raw_response)
    if score_match:
        score = int(score_match.group(1))

    # Extract category
    cat_match = re.search(r'Category:\s*(A|B|C|No Fit)', raw_response)
    if cat_match:
        category = cat_match.group(1)
        if category == "A":
            category = "Strong Fit"
        elif category == "B":
            category = "Good Fit"
        elif category == "C":
            category = "Fit"
        else:
            category = "No Fit"

    return {
        "name": profile.get("name", "Unknown"),
        "score": score,
        "category": category,
        "reasoning": raw_response,
    }


def screen_batch_structured(
    profiles: List[Dict],
    must_haves: List[str],
    nice_to_haves: List[Dict],
    reject_ifs: List[str],
    client,
    stability_summaries: Dict[str, str] = None,  # keyed by profile name or URL
    experience_summaries: Dict[str, str] = None,
    model: str = "claude-haiku-4-5-20251001",
    max_workers: int = 5,
) -> List[Dict]:
    """
    Screen multiple profiles in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    stability_summaries = stability_summaries or {}
    experience_summaries = experience_summaries or {}

    def screen_one(profile):
        name = profile.get("name", "")
        url = profile.get("linkedin_url", "")
        key = url or name

        return screen_with_structured_prompt(
            profile=profile,
            must_haves=must_haves,
            nice_to_haves=nice_to_haves,
            reject_ifs=reject_ifs,
            client=client,
            stability_summary=stability_summaries.get(key, ""),
            experience_summary=experience_summaries.get(key, ""),
            model=model,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(screen_one, p): p for p in profiles}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                profile = futures[future]
                results.append({
                    "name": profile.get("name", "Unknown"),
                    "score": 0,
                    "category": "Error",
                    "reasoning": str(e),
                })

    return results
