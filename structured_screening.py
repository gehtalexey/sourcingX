"""
Structured Screening Module

Instead of one long prompt, this module:
1. Parses natural language into structured requirements
2. Evaluates each requirement independently
3. Aggregates results with clear pass/fail logic

No requirement can be "traded off" against another.
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import anthropic


class RequirementType(Enum):
    SKILL_FRONTEND = "skill_frontend"
    SKILL_BACKEND = "skill_backend"
    EXPERIENCE_YEARS = "experience_years"
    LEADERSHIP_YEARS = "leadership_years"
    COMPANY_TYPE = "company_type"
    TITLE_REJECT = "title_reject"
    EXPERIENCE_MAX = "experience_max"
    CUSTOM = "custom"


@dataclass
class Requirement:
    type: RequirementType
    description: str
    values: List[str] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_must_have: bool = True
    boost_points: int = 0


@dataclass
class CheckResult:
    requirement: Requirement
    passed: bool
    reason: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class ScreeningResult:
    profile_name: str
    score: int
    fit: str
    checks: List[CheckResult] = field(default_factory=list)
    summary: str = ""

    @property
    def must_have_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.requirement.is_must_have)


# ============================================================================
# REQUIREMENT PARSER - Translates natural language to structured requirements
# ============================================================================

PARSER_PROMPT = """Extract screening requirements from this job description.

Return JSON with this exact structure:
{{
  "must_have": [
    {{"type": "skill_frontend", "description": "Has React/Vue/Angular", "values": ["React", "Vue", "Angular"]}},
    {{"type": "skill_backend", "description": "Has Node.js/Python/Java", "values": ["Node.js", "Python", "Java"]}},
    {{"type": "experience_years", "description": "5+ years fullstack", "min_value": 5}},
    {{"type": "leadership_years", "description": "2+ years team lead", "min_value": 2}},
    {{"type": "company_type", "description": "Software product company", "values": ["SaaS", "startup", "tech"]}},
    {{"type": "title_reject", "description": "Not backend/frontend only", "values": ["Backend", "Frontend", "DevOps", "Data", "ML", "AI"]}},
    {{"type": "experience_max", "description": "Max 20 years", "max_value": 20}}
  ],
  "nice_to_have": [
    {{"type": "custom", "description": "Top company background", "values": ["Wiz", "Monday", "Wix"], "boost": 2}},
    {{"type": "custom", "description": "8200/Mamram", "values": ["8200", "Mamram"], "boost": 1}}
  ],
  "reject_if": [
    {{"type": "company_type", "description": "Reject consultancies", "values": ["consulting", "outsourcing", "body shop"]}},
    {{"type": "company_type", "description": "Reject non-tech", "values": ["bank", "insurance", "telecom", "retail"]}}
  ]
}}

Rules:
- skill_frontend: Must have at least ONE of the frontend frameworks
- skill_backend: Must have at least ONE of the backend technologies
- experience_years: Total relevant experience minimum
- leadership_years: Team lead/management experience minimum
- company_type: Current company must match (for must_have) or not match (for reject_if)
- title_reject: Current title must NOT contain these keywords
- experience_max: Total career must be under this

Job Description:
{job_description}

Return ONLY valid JSON, no other text."""


def parse_requirements(job_description: str, client: anthropic.Anthropic) -> Dict[str, List[Requirement]]:
    """Parse natural language job description into structured requirements."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": PARSER_PROMPT.format(job_description=job_description)
        }],
        temperature=0.1
    )

    text = response.content[0].text

    # Extract JSON from response
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        raise ValueError(f"Could not parse requirements from response: {text[:200]}")

    data = json.loads(json_match.group())

    result = {
        "must_have": [],
        "nice_to_have": [],
        "reject_if": []
    }

    for req_data in data.get("must_have", []):
        result["must_have"].append(Requirement(
            type=RequirementType(req_data.get("type", "custom")),
            description=req_data.get("description", ""),
            values=req_data.get("values", []),
            min_value=req_data.get("min_value"),
            max_value=req_data.get("max_value"),
            is_must_have=True
        ))

    for req_data in data.get("nice_to_have", []):
        result["nice_to_have"].append(Requirement(
            type=RequirementType(req_data.get("type", "custom")),
            description=req_data.get("description", ""),
            values=req_data.get("values", []),
            min_value=req_data.get("min_value"),
            max_value=req_data.get("max_value"),
            is_must_have=False,
            boost_points=req_data.get("boost", 1)
        ))

    for req_data in data.get("reject_if", []):
        result["reject_if"].append(Requirement(
            type=RequirementType(req_data.get("type", "custom")),
            description=req_data.get("description", ""),
            values=req_data.get("values", []),
            is_must_have=True  # reject_if are treated as must-not-have
        ))

    return result


# ============================================================================
# INDIVIDUAL CHECKERS - Each requirement type has its own checker
# ============================================================================

def check_skill_frontend(profile: Dict, requirement: Requirement) -> CheckResult:
    """Check if profile has required frontend frameworks."""
    skills = [s.lower() for s in profile.get("skills", [])]

    # Frontend frameworks to look for
    frontend_frameworks = ["react", "vue", "angular", "next.js", "svelte", "nextjs"]
    if requirement.values:
        frontend_frameworks = [v.lower() for v in requirement.values]

    found = []
    for skill in skills:
        for framework in frontend_frameworks:
            if framework in skill:
                found.append(skill)
                break

    passed = len(found) > 0
    return CheckResult(
        requirement=requirement,
        passed=passed,
        reason=f"Frontend frameworks: {found}" if found else "No frontend framework (React/Vue/Angular) found",
        evidence=found
    )


def check_skill_backend(profile: Dict, requirement: Requirement) -> CheckResult:
    """Check if profile has required backend technologies."""
    skills = [s.lower() for s in profile.get("skills", [])]

    # Backend technologies to look for
    backend_tech = ["node", "python", "java", "go", "c#", "php", "ruby", "laravel", ".net"]
    if requirement.values:
        backend_tech = [v.lower() for v in requirement.values]

    found = []
    for skill in skills:
        for tech in backend_tech:
            if tech in skill and "javascript" not in skill:  # Avoid matching JavaScript as backend
                found.append(skill)
                break

    passed = len(found) > 0
    return CheckResult(
        requirement=requirement,
        passed=passed,
        reason=f"Backend tech: {found}" if found else "No backend tech (Node/Python/Java) found",
        evidence=found
    )


def check_experience_years(profile: Dict, requirement: Requirement) -> CheckResult:
    """Check if profile has minimum years of experience."""
    # Use pre-calculated experience from profile
    total_years = profile.get("total_experience_years", 0)

    # Try to extract from durations_text if available
    durations = profile.get("durations_text", "")
    if "INDUSTRY EXPERIENCE:" in durations:
        match = re.search(r'INDUSTRY EXPERIENCE:\s*(\d+)y', durations)
        if match:
            total_years = int(match.group(1))

    min_years = requirement.min_value or 0
    passed = total_years >= min_years

    return CheckResult(
        requirement=requirement,
        passed=passed,
        reason=f"Experience: {total_years}y (need {min_years}y)",
        evidence=[f"{total_years} years"]
    )


def check_leadership_years(profile: Dict, requirement: Requirement) -> CheckResult:
    """Check if profile has minimum years of leadership experience."""
    # Look for leadership titles in past employers
    leadership_keywords = ["lead", "manager", "director", "head", "cto", "vp", "chief"]
    leadership_months = 0

    for emp in profile.get("current_employers", []) + profile.get("past_employers", []):
        title = (emp.get("employee_title") or "").lower()
        if any(kw in title for kw in leadership_keywords):
            # Estimate months from dates
            start = emp.get("start_date", "")
            end = emp.get("end_date", "") or "2026-03"
            if start:
                try:
                    start_parts = start.split("-")
                    end_parts = end.split("-")
                    months = (int(end_parts[0]) - int(start_parts[0])) * 12
                    if len(start_parts) > 1 and len(end_parts) > 1:
                        months += int(end_parts[1]) - int(start_parts[1])
                    leadership_months += max(0, months)
                except:
                    pass

    leadership_years = leadership_months / 12
    min_years = requirement.min_value or 0
    passed = leadership_years >= min_years

    return CheckResult(
        requirement=requirement,
        passed=passed,
        reason=f"Leadership: {leadership_years:.1f}y (need {min_years}y)",
        evidence=[f"{leadership_years:.1f} years leadership"]
    )


def check_company_type(profile: Dict, requirement: Requirement, is_reject: bool = False) -> CheckResult:
    """Check if current company matches required type."""
    current_employers = profile.get("current_employers", [])
    if not current_employers:
        return CheckResult(
            requirement=requirement,
            passed=not is_reject,  # If no company info, pass for must-have, fail for reject
            reason="No current employer info",
            evidence=[]
        )

    company = current_employers[0].get("employer_name", "")
    description = (current_employers[0].get("employer_linkedin_description") or "").lower()

    # Keywords to check
    check_values = [v.lower() for v in requirement.values]

    found_keywords = []
    for keyword in check_values:
        if keyword in description or keyword in company.lower():
            found_keywords.append(keyword)

    if is_reject:
        # For reject_if: FAIL if keywords found
        passed = len(found_keywords) == 0
        reason = f"Company '{company}' contains reject keywords: {found_keywords}" if found_keywords else f"Company '{company}' OK"
    else:
        # For must_have: PASS if keywords found
        passed = len(found_keywords) > 0
        reason = f"Company '{company}' matches: {found_keywords}" if found_keywords else f"Company '{company}' doesn't match required type"

    return CheckResult(
        requirement=requirement,
        passed=passed,
        reason=reason,
        evidence=[company, description[:100]]
    )


def check_title_reject(profile: Dict, requirement: Requirement) -> CheckResult:
    """Check if current title contains rejected keywords."""
    current_employers = profile.get("current_employers", [])
    if not current_employers:
        return CheckResult(
            requirement=requirement,
            passed=True,
            reason="No current title info",
            evidence=[]
        )

    title = (current_employers[0].get("employee_title") or "").lower()

    reject_keywords = [v.lower() for v in requirement.values]
    found = [kw for kw in reject_keywords if kw in title]

    passed = len(found) == 0
    return CheckResult(
        requirement=requirement,
        passed=passed,
        reason=f"Title '{title}' contains rejected keywords: {found}" if found else f"Title '{title}' OK",
        evidence=[title]
    )


def check_experience_max(profile: Dict, requirement: Requirement) -> CheckResult:
    """Check if profile is under maximum experience threshold."""
    total_years = profile.get("total_experience_years", 0)

    durations = profile.get("durations_text", "")
    if "TOTAL CAREER SPAN:" in durations:
        match = re.search(r'TOTAL CAREER SPAN:\s*(\d+)y', durations)
        if match:
            total_years = int(match.group(1))

    max_years = requirement.max_value or 99
    passed = total_years <= max_years

    return CheckResult(
        requirement=requirement,
        passed=passed,
        reason=f"Experience: {total_years}y (max {max_years}y)",
        evidence=[f"{total_years} years"]
    )


def check_custom(profile: Dict, requirement: Requirement, client: anthropic.Anthropic = None) -> CheckResult:
    """Check custom requirement using AI."""
    if not client:
        return CheckResult(
            requirement=requirement,
            passed=True,  # Default pass if no AI available
            reason="Custom check skipped (no AI client)",
            evidence=[]
        )

    # Use AI to check custom requirement
    prompt = f"""Check if this profile meets this requirement: "{requirement.description}"
Values to look for: {requirement.values}

Profile skills: {profile.get('skills', [])}
Current title: {profile.get('current_employers', [{}])[0].get('employee_title', 'N/A')}
Current company: {profile.get('current_employers', [{}])[0].get('employer_name', 'N/A')}
Past companies: {[e.get('employer_name') for e in profile.get('past_employers', [])[:5]]}

Return JSON: {{"passed": true/false, "reason": "brief explanation"}}"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        result = json.loads(re.search(r'\{.*\}', response.content[0].text, re.DOTALL).group())
        return CheckResult(
            requirement=requirement,
            passed=result.get("passed", False),
            reason=result.get("reason", ""),
            evidence=[]
        )
    except:
        return CheckResult(
            requirement=requirement,
            passed=False,
            reason="Custom check failed",
            evidence=[]
        )


# ============================================================================
# ORCHESTRATOR - Runs all checks and aggregates results
# ============================================================================

CHECKER_MAP = {
    RequirementType.SKILL_FRONTEND: check_skill_frontend,
    RequirementType.SKILL_BACKEND: check_skill_backend,
    RequirementType.EXPERIENCE_YEARS: check_experience_years,
    RequirementType.LEADERSHIP_YEARS: check_leadership_years,
    RequirementType.COMPANY_TYPE: check_company_type,
    RequirementType.TITLE_REJECT: check_title_reject,
    RequirementType.EXPERIENCE_MAX: check_experience_max,
    RequirementType.CUSTOM: check_custom,
}


def screen_profile_structured(
    profile: Dict,
    requirements: Dict[str, List[Requirement]],
    client: anthropic.Anthropic = None
) -> ScreeningResult:
    """Screen a profile against structured requirements."""

    name = profile.get("name") or "Unknown"
    checks = []
    boost_points = 0

    # 1. Check must-have requirements
    for req in requirements.get("must_have", []):
        checker = CHECKER_MAP.get(req.type, check_custom)
        if req.type == RequirementType.CUSTOM and checker == check_custom:
            result = checker(profile, req, client)
        else:
            result = checker(profile, req)
        checks.append(result)

    # 2. Check reject-if requirements (inverted logic)
    for req in requirements.get("reject_if", []):
        if req.type == RequirementType.COMPANY_TYPE:
            result = check_company_type(profile, req, is_reject=True)
        else:
            checker = CHECKER_MAP.get(req.type, check_custom)
            result = checker(profile, req)
            # Invert for reject_if
            result = CheckResult(
                requirement=result.requirement,
                passed=not result.passed,
                reason=result.reason,
                evidence=result.evidence
            )
        checks.append(result)

    # 3. Check nice-to-have requirements (for boost points)
    for req in requirements.get("nice_to_have", []):
        checker = CHECKER_MAP.get(req.type, check_custom)
        if req.type == RequirementType.CUSTOM and checker == check_custom:
            result = checker(profile, req, client)
        else:
            result = checker(profile, req)

        if result.passed:
            boost_points += req.boost_points

        # Mark as non-must-have so it doesn't affect pass/fail
        result.requirement.is_must_have = False
        checks.append(result)

    # 4. Calculate final score
    must_have_checks = [c for c in checks if c.requirement.is_must_have]
    failed_must_haves = [c for c in must_have_checks if not c.passed]

    if failed_must_haves:
        # Any must-have failure = score 1-2
        score = 2 if len(failed_must_haves) == 1 else 1
        fit = "Not a Fit"
    else:
        # All must-haves passed, base score 6 + boosts
        base_score = 6
        score = min(10, base_score + boost_points)

        if score >= 8:
            fit = "Strong Fit"
        elif score >= 6:
            fit = "Good Fit"
        else:
            fit = "Partial Fit"

    # 5. Build summary
    failed_reasons = [f"FAIL: {c.reason}" for c in failed_must_haves]
    passed_reasons = [f"PASS: {c.reason}" for c in must_have_checks if c.passed]
    boost_reasons = [f"+{c.requirement.boost_points}: {c.reason}" for c in checks if not c.requirement.is_must_have and c.passed]

    summary_parts = failed_reasons + passed_reasons[:3] + boost_reasons
    summary = " | ".join(summary_parts[:5])

    return ScreeningResult(
        profile_name=name,
        score=score,
        fit=fit,
        checks=checks,
        summary=summary
    )


# ============================================================================
# BATCH SCREENING WITH PARALLEL EXECUTION
# ============================================================================

from concurrent.futures import ThreadPoolExecutor, as_completed


def screen_profiles_batch_structured(
    profiles: List[Dict],
    requirements: Dict[str, List[Requirement]],
    client: anthropic.Anthropic = None,
    max_workers: int = 5
) -> List[ScreeningResult]:
    """Screen multiple profiles in parallel."""

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_profile = {
            executor.submit(screen_profile_structured, profile, requirements, client): profile
            for profile in profiles
        }

        for future in as_completed(future_to_profile):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                profile = future_to_profile[future]
                results.append(ScreeningResult(
                    profile_name=profile.get("name", "Unknown"),
                    score=0,
                    fit="Error",
                    summary=str(e)
                ))

    return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_default_fullstack_requirements() -> Dict[str, List[Requirement]]:
    """Create default requirements for fullstack team lead screening."""
    return {
        "must_have": [
            Requirement(
                type=RequirementType.SKILL_FRONTEND,
                description="Has frontend framework (React/Vue/Angular)",
                values=["React", "Vue", "Angular", "Next.js", "Svelte"],
                is_must_have=True
            ),
            Requirement(
                type=RequirementType.SKILL_BACKEND,
                description="Has backend tech (Node/Python/Java)",
                values=["Node.js", "Python", "Java", "Go", "C#", "PHP", "Ruby"],
                is_must_have=True
            ),
            Requirement(
                type=RequirementType.LEADERSHIP_YEARS,
                description="2+ years team lead",
                min_value=2,
                is_must_have=True
            ),
            Requirement(
                type=RequirementType.TITLE_REJECT,
                description="Not backend/frontend/devops only",
                values=["Backend", "Frontend", "DevOps", "Platform", "Infra", "Data", "ML", "AI"],
                is_must_have=True
            ),
            Requirement(
                type=RequirementType.EXPERIENCE_MAX,
                description="Max 20 years experience",
                max_value=20,
                is_must_have=True
            ),
        ],
        "nice_to_have": [
            Requirement(
                type=RequirementType.CUSTOM,
                description="Top company (Wiz, Monday, Wix)",
                values=["Wiz", "Monday", "Wix", "Snyk", "AppsFlyer"],
                is_must_have=False,
                boost_points=2
            ),
            Requirement(
                type=RequirementType.CUSTOM,
                description="Elite unit (8200, Mamram)",
                values=["8200", "Mamram", "Talpiot"],
                is_must_have=False,
                boost_points=1
            ),
        ],
        "reject_if": [
            Requirement(
                type=RequirementType.COMPANY_TYPE,
                description="Reject non-software companies",
                values=["bank", "insurance", "telecom", "consulting", "outsourcing", "ticket", "travel", "retail"],
                is_must_have=True
            ),
        ]
    }
