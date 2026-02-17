"""
Test script for AI screening logic.
Run: python test_screening.py

This helps debug screening results before deploying.
"""

import json
import os
from pathlib import Path

# Load config for OpenAI key
config_path = Path(__file__).parent / 'config.json'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
    OPENAI_KEY = config.get('openai_api_key')
else:
    OPENAI_KEY = os.environ.get('OPENAI_API_KEY')

if not OPENAI_KEY:
    print("ERROR: No OpenAI API key found. Set in config.json or OPENAI_API_KEY env var")
    exit(1)

from openai import OpenAI

# ============================================================================
# TEST DATA - Modify these to test different scenarios
# ============================================================================

JOB_DESCRIPTION = """
DevOps Team Lead
Tel Aviv, Israel

We're looking for a DevOps Team Lead to play a key role in our platform group.

Requirements:
- 6+ years in DevOps, including 2+ years in a leadership/Tech Lead role
- Deep expertise in Kubernetes and cloud-native infrastructure
- Expertise in Infrastructure as Code (IaC), particularly Terraform
- Strong background in CI/CD systems
- In-depth knowledge of cloud infrastructure (GCP preferred)
"""

EXTRA_REQUIREMENTS = """
Must have 5 years of DevOps experience.
Must have 2 years of team leader experience in the recent 5 years.
Kubernetes is a must to be mentioned somewhere in the profile.

Reject junior, students, freelancers.
Reject overqualified like VP, Director, CTO, Head of etc.
Reject project companies (consulting/outsourcing).
"""

# Test profiles - add profiles that were misclassified
TEST_PROFILES = [
    {
        "name": "Royee Tager",
        "current_title": "DevOps Engineer",  # From DB - stale data
        "current_company": "Luminate Security acquired by Symantec",
        "headline": "DevOps Engineer at Luminate Security acquired by Symantec / Broadcom",
        "all_employers": "Dell EMC XtremIO, eXelate, Hewlett Packard Enterprise, Jungo, Mavenir, Verint",
        "all_titles": "DevOps & IT Engineer, DevOps Technical Lead, IT Specialist, System Integration, System Integration & Deployment, Unix & Linux System Administrator",
        "skills": "Linux, Bash, Solaris, Apache, Red Hat Linux, Unix, Virtualization, Cloud Computing, VMware, RedHat, Scripting, Databases, NetApp, Storage, Windows Server, Puppet, CentOS, MySQL, Shell Scripting, IIS, Netbackup, Perl, CGI, NFS, DNS, Ubuntu, Xen, Subversion, Amazon Web Services (AWS), Operating Systems, Python, DevOps, Git, Jenkins, System Deployment, Docker, Ansible, Amazon EC2, Configuration Management, Github, Amazon S3, Vagrant, Tomcat, docker, Continuous Integration, Microservices, Kubernetes, Agile Methodologies, Apache Mesos, Big Data",
        "summary": "Specialties: Linux & Unix OS (Red Hat, CentOS, Debain, Ubuntu, Solaris), Windows...",
        "past_positions": """
- DevOps Technical Lead at eXelate (Dec 2015 - Aug 2017)
- DevOps & IT Engineer at Dell EMC XtremIO (Dec 2013 - Nov 2015)
- IT Specialist at Jungo (Dec 2011 - Nov 2013)
- Unix & Linux System Administrator at Hewlett Packard Enterprise (May 2009 - Nov 2011)
- System Integration at Mavenir (Aug 2007 - Apr 2009)
- System Integration & Deployment at Verint (Apr 2004 - Jul 2007)
        """,
        "expected_result": "PARTIAL FIT (5-6) - Has Kubernetes in skills, has DevOps Technical Lead experience, but NO Team Leader title. Data is stale/incomplete from Crustdata."
    },
    {
        "name": "Boaz Elgat",
        "current_title": "Head of DevOps",  # Should be REJECTED (overqualified)
        "current_company": "AI21 Labs",
        "headline": "Head of DevOps at AI21 Labs",
        "all_employers": "AI21 Labs, JFrog, Transmit Security, Payoneer",
        "all_titles": "Head of DevOps, DevOps Team Lead, Senior DevOps Engineer",
        "skills": "Kubernetes, Terraform, AWS, GCP, Docker, CI/CD",
        "summary": "",
        "past_positions": """
- Head of DevOps at AI21 Labs (2022 - Present)
- DevOps Team Lead at JFrog (2019 - 2022)
- Senior DevOps Engineer at Transmit Security (2017 - 2019)
        """,
        "expected_result": "NOT A FIT - Title is 'Head of DevOps' which should be rejected as overqualified"
    },
]

# ============================================================================
# SCREENING LOGIC (from dashboard.py)
# ============================================================================

def get_screening_prompt():
    """Get the system prompt for screening."""
    return """You are an expert technical recruiter for top Israeli startups. You screen DevOps/Platform engineers and Team Leads.

## Scoring Rubric
- **9-10**: Top-tier company + strong cloud/K8s + leadership + 6+ years. Rare.
- **7-8**: Good company background + solid DevOps skills + team lead experience.
- **5-6**: Decent experience but gaps in leadership or key skills.
- **3-4**: Limited experience or missing must-haves.
- **1-2**: Not a fit - missing critical requirements or should be rejected.

## CRITICAL: Rejection Rules
- If title contains "Head of", "VP", "Director", "CTO", "Chief" → Score 1-2, Not a Fit (overqualified)
- If title is "Junior", "Intern", "Student", "Freelance" → Score 1-2, Not a Fit
- If company is consulting/outsourcing (Matrix, Tikal, Ness, Sela) → Score 1-2, Not a Fit

## CRITICAL: Must-Have Rules
- If "Kubernetes" not mentioned ANYWHERE in profile → Score ≤3, cannot be Strong/Good Fit
- If no team lead/manager experience in past 5 years → Score ≤5

## Output
Be direct and calibrated. Reference specific signals from the profile."""


def screen_profile_test(profile: dict, job_description: str, extra_requirements: str) -> dict:
    """Screen a single profile and return the result."""
    client = OpenAI(api_key=OPENAI_KEY)

    # Build profile summary (matching dashboard.py format)
    profile_summary = f"""## Key Profile Fields (READ THESE CAREFULLY):
- **CURRENT TITLE**: {profile.get('current_title', 'N/A')}
- **CURRENT COMPANY**: {profile.get('current_company', 'N/A')}
- **Headline**: {profile.get('headline', 'N/A')}
- **All Titles (career history)**: {profile.get('all_titles', 'N/A')}
- **All Employers (career history)**: {profile.get('all_employers', 'N/A')}
- **Skills**: {profile.get('skills', 'N/A')}
- **Summary/About**: {profile.get('summary', 'N/A')}

## Work History (with calculated durations):
{profile.get('past_positions', 'N/A')}
"""

    # Build the user prompt with extra requirements
    user_prompt = f"""Evaluate this candidate against the job description.

## Job Description:
{job_description}

## CRITICAL - Extra Requirements (HARD RULES - MUST enforce):
{extra_requirements}

ENFORCEMENT RULES:
- If a requirement says "must have X" or "X is a must" → Candidate MUST show X in their profile or score ≤3
- If a requirement says "reject Y" → If candidate matches Y → Score 1-2, mark as "Not a Fit"
- These are DISQUALIFIERS, not preferences

## HARD RULES - Rejection & Must-Have Criteria:

### REJECTION RULES (Score 1-2 if matched):
1. "Reject overqualified/VP/Director/CTO/Head of" → Check CURRENT TITLE. If title contains VP, Director, CTO, Chief, Head of → Score 1-2, "Not a Fit"
2. "Reject junior/students/freelancers" → If current role is Junior, Intern, Student, or Freelance → Score 1-2
3. "Reject project companies" → If current company is consulting/outsourcing → Score 1-2

### MUST-HAVE RULES (Score ≤3 if missing):
1. "Kubernetes is a must" → Search the ENTIRE profile for "Kubernetes" or "K8s". If NOT found → Score ≤3
2. "Must have N years experience" → Calculate from work history. If not met → Score ≤3
3. "Must have team lead experience" → Check if ANY role had "lead", "manager" in title

### HOW TO CHECK (use ALL available fields):
- **CURRENT TITLE**: The candidate's current job title
- **CURRENT COMPANY**: The candidate's current employer
- **Headline**: Often contains current title and company
- **All Titles (career history)**: Array of ALL job titles - check for leadership roles
- **All Employers (career history)**: Array of ALL companies worked at
- **Skills**: Array of skills - search for required technologies here
- **Summary/About**: LinkedIn "About" section - may contain skills, experience details
- **Work History**: Detailed positions with dates

### CRITICAL:
- A candidate missing a MUST-HAVE can NEVER be "Strong Fit" or "Good Fit"
- A candidate matching a REJECTION criterion is ALWAYS "Not a Fit" (Score 1-2)

## Candidate Profile:
{profile_summary}

Respond with ONLY valid JSON:
{{"score": <1-10>, "fit": "<Strong Fit|Good Fit|Partial Fit|Not a Fit>", "summary": "<2-3 sentences>", "why": "<explain the score based on rules>", "rule_checks": {{"kubernetes_found": <true/false>, "is_overqualified": <true/false>, "has_lead_experience": <true/false>}}}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": get_screening_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.1  # Low temperature for consistent results
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 80)
    print("SCREENING TEST")
    print("=" * 80)
    print(f"\nTesting {len(TEST_PROFILES)} profiles...\n")

    for profile in TEST_PROFILES:
        print("-" * 80)
        print(f"PROFILE: {profile['name']}")
        print(f"Current: {profile['current_title']} @ {profile['current_company']}")
        print(f"Skills: {profile.get('skills', 'N/A')}")
        print(f"\nEXPECTED: {profile['expected_result']}")
        print("\nScreening...")

        result = screen_profile_test(profile, JOB_DESCRIPTION, EXTRA_REQUIREMENTS)

        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"\nRESULT:")
            print(f"  Score: {result.get('score')}")
            print(f"  Fit: {result.get('fit')}")
            print(f"  Summary: {result.get('summary')}")
            print(f"  Why: {result.get('why')}")
            if 'rule_checks' in result:
                print(f"  Rule Checks: {result.get('rule_checks')}")

        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
