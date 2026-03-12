"""
Email Generator Module - Generate personalized email subject lines and openers using OpenAI.

Uses full Crustdata profile data to find the most interesting angle for each candidate.
Subject line and opener MUST use different angles for variety.
"""

import json
import time
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import OpenAI

# Try to import usage tracker (optional)
try:
    from usage_tracker import get_usage_tracker
except ImportError:
    def get_usage_tracker():
        return None


# ===== Company Name Normalization =====

# Suffixes to strip from company names
COMPANY_SUFFIXES = [
    r'\s*,?\s*Inc\.?$', r'\s*,?\s*Corp\.?$', r'\s*,?\s*Ltd\.?$',
    r'\s*,?\s*LLC\.?$', r'\s*,?\s*GmbH$', r'\s*,?\s*S\.A\.?$',
    r'\s*,?\s*PLC$', r'\s*,?\s*LTD$', r'\s*Technologies$',
    r'\s*Technology$', r'\s*Labs$', r'\s*Group$', r'\s*Solutions$',
    r'\s*Software$', r'\s*\(Israel\)$', r'\s*\(IL\)$',
]

# Known company name mappings
COMPANY_NAME_MAP = {
    'check point software technologies': 'Check Point',
    'palo alto networks': 'Palo Alto Networks',
    'meta platforms': 'Meta',
    'alphabet': 'Google',
    'microsoft corporation': 'Microsoft',
    'amazon.com': 'Amazon',
    'apple inc': 'Apple',
    'fiverr international': 'Fiverr',
    'monday.com': 'Monday',
    'wix.com': 'Wix',
    'ai21 labs': 'AI21',
    'nso group': 'NSO',
    'healthy.io': 'Healthy',
}


def normalize_company_name(name: str) -> str:
    """Clean company name by stripping suffixes and normalizing known names."""
    if not name:
        return ''

    # Check known mappings first
    name_lower = name.lower().strip()
    if name_lower in COMPANY_NAME_MAP:
        return COMPANY_NAME_MAP[name_lower]

    # Strip suffixes
    cleaned = name.strip()
    for pattern in COMPANY_SUFFIXES:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Remove trailing .io for domain-style names but keep the brand
    if cleaned.endswith('.io'):
        cleaned = cleaned[:-3]

    return cleaned.strip()


# ===== Prompt Templates =====

SENDER_PERSONAS = {
    'recruiter': 'a technical recruiter',
    'engineering_manager': 'an Engineering Manager',
    'vp_rd': 'a VP R&D',
    'cto': 'a CTO',
    'founder': 'a Founder/CEO',
    'hr_director': 'an HR Director',
}

TONE_DESCRIPTIONS = {
    'casual': 'casual and conversational, like texting a colleague',
    'professional': 'professional but warm, not stiff or corporate',
    'friendly': 'friendly and approachable, with a touch of humor',
    'direct': 'direct and to-the-point, no fluff',
}

LENGTH_DESCRIPTIONS = {
    'short': '1 sentence only',
    'medium': '1-2 sentences',
    'long': '2-3 sentences',
}


def build_email_prompt(sender: str, tone: str, length: str, custom_instruction: str = None) -> str:
    """Build the system prompt for email generation."""

    sender_desc = SENDER_PERSONAS.get(sender, 'a recruiter')
    tone_desc = TONE_DESCRIPTIONS.get(tone, TONE_DESCRIPTIONS['professional'])
    length_desc = LENGTH_DESCRIPTIONS.get(length, LENGTH_DESCRIPTIONS['medium'])

    custom_section = f"\n\nADDITIONAL INSTRUCTIONS FROM USER:\n{custom_instruction}" if custom_instruction else ""

    return f"""You are {sender_desc} at an Israeli tech company writing personalized cold outreach emails to engineering candidates.

## OUTPUT FORMAT
Return ONLY valid JSON:
{{
  "subject_line": "short punchy subject (<10 words)",
  "subject_angle": "education|title|skills|career|company|domain",
  "email_opener": "personalized opener",
  "opener_angle": "education|title|skills|career|company|domain"
}}

## CRITICAL RULES

1. **DIFFERENT ANGLES**: subject_angle and opener_angle MUST be different
   - If subject uses education -> opener must use skills/title/company/career/domain
   - If subject uses skills -> opener must use education/title/company/career/domain
   - NEVER use the same angle for both

2. **SUBJECT LINE - MUST BE PERSONALIZED & VARIED**:
   - Under 10 words, under 60 characters
   - MUST mention something specific from THEIR profile
   - Use DIFFERENT formats - don't repeat the same pattern
   - NO generic questions like "Ready to...", "Interested in...", "Looking for..." - FORBIDDEN

   SUBJECT FORMAT VARIETY (rotate through these):
   - "[School] grad at [Company]?" - "Technion grad at Wiz?"
   - "[Skill] at [Company]?" - "Go at Monday?"
   - "[Skill] in production?" - "Rust in production?"
   - "[Prev] to [Current]?" - "Google to Snyk?" (tier-1 only)
   - "Still [activity] at [Company]?" - "Still building infra at Orca?"
   - "[Title] at [Company]?" - "Staff engineer at Snyk?"
   - "[Domain] at [Company]?" - "Security backend at Wiz?"
   - "After [X years] at [Company]?" - "After 5 years at Check Point?"
   - "[Skill1] and [Skill2]?" - "Kafka and gRPC?"
   - "From [old role] to [new]?" - "From lead back to IC?"
   - "[Field] background at [Company]?" - "Physics background at AI21?"
   - "Building [what] at [Company]?" - "Building cloud security at Wiz?"

   BAD (generic - NEVER use):
   - "Ready to...", "Interested in...", "Looking for...", "Excited about..."
   - Any question that doesn't mention something specific about THEM

3. **OPENER - MUST BE PERSONALIZED & VARIED**:
   - {length_desc}. Tone: {tone_desc}.
   - Must reference a DIFFERENT aspect than subject
   - Use the FULL profile context (summary, role descriptions, company info, education details)

   OPENER FORMAT VARIETY (rotate through these):
   - Ask about their work: "What's the trickiest part of [their domain] at [company]?"
   - Reference career path: "Lead to IC is bold - what drove that?"
   - Education hook: "[School] [field] is solid prep. How does it help at [company]?"
   - Skill curiosity: "[Skill] at scale gets complex. What's your approach?"
   - Company transition: "Big move from [prev]. What pulled you to [current]?"
   - Tenure observation: "[X] years at [company] is rare. What keeps you engaged?"
   - Domain interest: "[Company]'s work on [domain from description] sounds interesting."
   - Role curiosity: "Your focus on [something from role description] caught my eye."
   - Background uniqueness: "[non-CS field] to engineering is unique. How does that help?"
   - Team/culture: "How's the [backend/infra/platform] team culture at [company]?"

4. **FORBIDDEN** (instant fail):
   - Em dashes (use regular dash - or comma)
   - "I noticed", "I came across", "I was impressed"
   - "I hope this finds you well", "Hope you're doing well"
   - "Reaching out because", "Just following up"
   - "Ready to..." subject lines
   - Generic phrases that could apply to anyone
   - Mentioning the same company twice
   - Mentioning companies from before 2018

5. **COMPANY NAMES**: Clean them up
   - Strip: Ltd, Inc, Corp, Technologies, Labs, Group, Solutions, .io
   - "Check Point Software Technologies, Ltd." -> "Check Point"
   - "AI21 Labs" -> "AI21"

6. **ANGLE SELECTION** (pick what's MOST distinctive - DON'T default to skills):

   CHECK IN THIS ORDER and pick the FIRST that's notable:
   1. Career pattern? Manager->IC, Founder, 5+ years tenure, acquisition → use CAREER angle
   2. Tier-1 company? Google, Meta, Apple, Amazon, Microsoft, Waze → use COMPANY angle
   3. Notable school? Technion, TAU, Hebrew U, Weizmann, BGU → use EDUCATION angle
   4. Interesting title? Staff, Principal, Founding, Architect → use TITLE angle
   5. Unique background? Physics, Math, non-CS field → use BACKGROUND angle
   6. Rare skills? Rust, Go, Scala, K8s, Kafka, gRPC, Spark → use SKILLS angle (LAST resort)

   IMPORTANT: Skills angle is OVERUSED. Only use it when nothing else is distinctive.

7. **COMPANY TRANSITIONS**: Only mention for tier-1 companies (Google, Meta, Apple, Amazon, Microsoft, Waze). Most people have previous jobs - that's not distinctive!

8. **SAME COMPANY CHECK**: If previous company = current company (or parent/child like Waze/Google), do NOT use company transition angle. Focus on role growth, skills, or education instead.
{custom_section}

Today's date: {datetime.now().strftime("%Y-%m-%d")}"""


# ===== Profile Trimming (similar to dashboard.py) =====

def _first_sentence(text: str) -> str:
    """Extract first sentence from text."""
    if not text:
        return None
    # Find first period, question mark, or exclamation followed by space or end
    match = re.search(r'^[^.!?]*[.!?]', text)
    if match:
        return match.group(0).strip()
    # No sentence ending found, return first 150 chars
    return text[:150].strip() if len(text) > 150 else text


def trim_profile_for_email(raw: dict) -> dict:
    """Create trimmed copy of raw Crustdata JSON for email generation.

    Includes rich context: summary, role descriptions, company info, education details.
    """
    if not isinstance(raw, dict):
        return raw

    trimmed = {}

    # Basic info - include summary for personal context
    for key in ['name', 'first_name', 'headline', 'location', 'skills',
                'all_titles', 'all_employers', 'all_schools', 'languages']:
        if key in raw:
            trimmed[key] = raw[key]

    # Include summary/about section (truncated) - good for personal angles
    if raw.get('summary'):
        summary = raw['summary']
        # Take first 300 chars or 2 sentences
        if len(summary) > 300:
            summary = summary[:300] + '...'
        trimmed['summary'] = summary

    # Current employers - include role description
    if 'current_employers' in raw:
        trimmed['current_employers'] = []
        for emp in (raw['current_employers'] or [])[:2]:
            title = (emp.get('employee_title') or '').strip()
            if not title:
                continue
            entry = {
                'title': title,
                'company': normalize_company_name(emp.get('employer_name', '')),
                'start_date': emp.get('start_date'),
                'company_description': _first_sentence(emp.get('employer_linkedin_description')),
            }
            # Include what they do in the role (if available)
            role_desc = emp.get('employee_description') or emp.get('description')
            if role_desc:
                entry['role_description'] = _first_sentence(role_desc)
            trimmed['current_employers'].append(entry)

    # Past employers - include company descriptions for context
    if 'past_employers' in raw:
        trimmed['past_employers'] = []
        for emp in (raw['past_employers'] or [])[:5]:  # Up to 5 for career path
            title = (emp.get('employee_title') or '').strip()
            if not title:
                continue
            end_date = emp.get('end_date')
            # Skip if ended before 2016
            if end_date and end_date < '2016':
                continue
            entry = {
                'title': title,
                'company': normalize_company_name(emp.get('employer_name', '')),
                'start_date': emp.get('start_date'),
                'end_date': end_date,
            }
            # Company description helps understand industry/domain
            company_desc = emp.get('employer_linkedin_description')
            if company_desc:
                entry['company_description'] = _first_sentence(company_desc)
            trimmed['past_employers'].append(entry)

    # Education - include more details
    if 'education_background' in raw:
        trimmed['education'] = []
        for edu in (raw['education_background'] or [])[:3]:
            school = edu.get('institute_name') or edu.get('school_name') or ''
            if not school:
                continue
            entry = {
                'school': school,
                'degree': edu.get('degree_name') or edu.get('degree'),
                'field': edu.get('field_of_study'),
            }
            # Include honors, activities, dates for context
            if edu.get('activities'):
                entry['activities'] = edu['activities'][:100]  # Truncate
            if edu.get('grade') or edu.get('gpa'):
                entry['honors'] = edu.get('grade') or edu.get('gpa')
            if edu.get('start_date') and edu.get('end_date'):
                entry['years'] = f"{edu['start_date'][:4]}-{edu['end_date'][:4]}"
            trimmed['education'].append(entry)

    # Calculate tenure at current company (useful for angles)
    if trimmed.get('current_employers'):
        start = trimmed['current_employers'][0].get('start_date')
        if start:
            try:
                from datetime import datetime
                start_date = datetime.strptime(start[:7], '%Y-%m')
                months = (datetime.now() - start_date).days // 30
                years = months // 12
                remaining_months = months % 12
                if years > 0:
                    trimmed['tenure_at_current'] = f"{years}y {remaining_months}m"
                else:
                    trimmed['tenure_at_current'] = f"{months}m"
            except:
                pass

    # Detect interesting career patterns
    patterns = []
    titles = [e.get('title', '').lower() for e in trimmed.get('past_employers', [])]
    current_title = trimmed.get('current_employers', [{}])[0].get('title', '').lower()

    # Manager/Lead to IC transition
    if any('lead' in t or 'manager' in t or 'head' in t for t in titles) and \
       'lead' not in current_title and 'manager' not in current_title:
        patterns.append('was_manager_now_ic')

    # Long tenure (5+ years at one company)
    if trimmed.get('tenure_at_current'):
        try:
            years = int(trimmed['tenure_at_current'].split('y')[0])
            if years >= 5:
                patterns.append('long_tenure_5plus_years')
        except:
            pass

    # Founding/early employee
    if any('founding' in t or 'first' in t or '#1' in t for t in titles + [current_title]):
        patterns.append('founding_or_early_employee')

    if patterns:
        trimmed['career_patterns'] = patterns

    return trimmed


# ===== Single Profile Generation =====

def generate_email_for_profile(
    profile: dict,
    client: OpenAI,
    sender: str = 'recruiter',
    tone: str = 'professional',
    length: str = 'medium',
    custom_instruction: str = None,
    ai_model: str = 'gpt-4o-mini',
    tracker=None
) -> dict:
    """Generate email subject line and opener for a single profile.

    Args:
        profile: Profile dict with raw_data or raw_crustdata
        client: OpenAI client instance
        sender: Sender persona key
        tone: Tone key
        length: Length key
        custom_instruction: Optional user instruction
        ai_model: Model to use (gpt-4o-mini or gpt-4o)
        tracker: Optional usage tracker

    Returns:
        Dict with subject_line, subject_angle, email_opener, opener_angle
    """
    start_time = time.time()

    # Get raw data
    raw = profile.get('raw_crustdata') or profile.get('raw_data') or profile
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raw = profile

    # Validate we have data
    name = raw.get('name') or raw.get('first_name') or profile.get('name') or 'Unknown'
    if not raw.get('current_employers') and not raw.get('past_employers') and not raw.get('skills'):
        return {
            "subject_line": "",
            "email_opener": "",
            "error": "Insufficient profile data"
        }

    # Trim profile for email context
    trimmed = trim_profile_for_email(raw)

    # Build prompts
    system_prompt = build_email_prompt(sender, tone, length, custom_instruction)

    user_prompt = f"""## Candidate Profile:
```json
{json.dumps(trimmed, indent=2, default=str)}
```

Generate a personalized subject line and email opener. Remember: subject and opener MUST use DIFFERENT angles."""

    try:
        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,  # Slightly higher for creativity/variety
            max_tokens=200,
            response_format={"type": "json_object"}
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Log usage
        if tracker and hasattr(response, 'usage') and response.usage:
            tracker.log_openai(
                tokens_input=response.usage.prompt_tokens,
                tokens_output=response.usage.completion_tokens,
                model=ai_model,
                profiles_screened=0,  # Not screening
                status='success',
                response_time_ms=elapsed_ms
            )

        result = json.loads(response.choices[0].message.content)

        # Validate different angles
        if result.get('subject_angle') == result.get('opener_angle'):
            result['warning'] = 'Same angle used for both (AI error)'

        return result

    except json.JSONDecodeError as e:
        return {
            "subject_line": "",
            "email_opener": "",
            "error": f"JSON parse error: {str(e)[:80]}"
        }
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        if tracker:
            tracker.log_openai(
                tokens_input=0, tokens_output=0, model=ai_model,
                profiles_screened=0, status='error',
                error_message=str(e)[:200], response_time_ms=elapsed_ms
            )
        return {
            "subject_line": "",
            "email_opener": "",
            "error": f"API error: {str(e)[:80]}"
        }


# ===== Batch Generation =====

def generate_emails_batch(
    profiles: list,
    openai_api_key: str,
    sender: str = 'recruiter',
    tone: str = 'professional',
    length: str = 'medium',
    custom_instruction: str = None,
    ai_model: str = 'gpt-4o-mini',
    max_workers: int = 10,
    progress_callback=None,
    cancel_flag=None
) -> list:
    """Generate emails for multiple profiles in parallel.

    Args:
        profiles: List of profile dicts
        openai_api_key: OpenAI API key
        sender: Sender persona
        tone: Tone setting
        length: Length setting
        custom_instruction: Optional user instruction
        ai_model: Model to use
        max_workers: Concurrent threads
        progress_callback: Function(completed, total, result) called after each
        cancel_flag: Dict with 'cancelled' key to check

    Returns:
        List of results with profile info + generated email content
    """
    results = []
    completed_count = [0]
    lock = threading.Lock()
    total = len(profiles)

    tracker = get_usage_tracker()

    def generate_single(profile, index):
        # Check cancellation
        if cancel_flag and cancel_flag.get('cancelled'):
            return None

        # Create client per thread
        client = OpenAI(api_key=openai_api_key)

        try:
            result = generate_email_for_profile(
                profile, client,
                sender=sender, tone=tone, length=length,
                custom_instruction=custom_instruction,
                ai_model=ai_model, tracker=tracker
            )

            # Add profile info
            raw = profile.get('raw_crustdata') or profile.get('raw_data') or profile
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except:
                    raw = {}

            name = raw.get('name') or profile.get('name') or f"Profile {index}"

            # Get current role info
            current_employers = raw.get('current_employers', [])
            current_title = ''
            current_company = ''
            if current_employers:
                current_title = current_employers[0].get('employee_title', '')
                current_company = normalize_company_name(current_employers[0].get('employer_name', ''))

            result['name'] = name
            result['current_title'] = current_title or profile.get('current_title', '')
            result['current_company'] = current_company or profile.get('current_company', '')
            result['linkedin_url'] = profile.get('linkedin_url', '') or raw.get('linkedin_flagship_url', '')
            result['email'] = profile.get('email', '') or profile.get('salesql_email', '')
            result['index'] = index

        except Exception as e:
            result = {
                "subject_line": "",
                "email_opener": "",
                "error": f"Generation error: {str(e)[:80]}",
                "name": profile.get('name', f"Profile {index}"),
                "current_title": profile.get('current_title', ''),
                "current_company": profile.get('current_company', ''),
                "linkedin_url": profile.get('linkedin_url', ''),
                "email": profile.get('email', ''),
                "index": index
            }

        with lock:
            completed_count[0] += 1

        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(generate_single, profile, i): i
            for i, profile in enumerate(profiles)
        }

        for future in as_completed(future_to_index):
            if cancel_flag and cancel_flag.get('cancelled'):
                executor.shutdown(wait=False, cancel_futures=True)
                break

            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    if progress_callback:
                        progress_callback(len(results), total, result)
            except Exception as e:
                idx = future_to_index[future]
                error_result = {
                    "subject_line": "",
                    "email_opener": "",
                    "error": f"Thread error: {str(e)[:80]}",
                    "name": f"Profile {idx}",
                    "index": idx
                }
                results.append(error_result)
                if progress_callback:
                    progress_callback(len(results), total, error_result)

    # Sort by original index
    results.sort(key=lambda x: x.get('index', 0))
    return results


# ===== Angle Distribution Tracking =====

def get_angle_distribution(results: list) -> dict:
    """Analyze angle distribution in generated emails."""
    subject_angles = {}
    opener_angles = {}

    for r in results:
        sa = r.get('subject_angle', 'unknown')
        oa = r.get('opener_angle', 'unknown')
        subject_angles[sa] = subject_angles.get(sa, 0) + 1
        opener_angles[oa] = opener_angles.get(oa, 0) + 1

    return {
        'subject_angles': subject_angles,
        'opener_angles': opener_angles,
        'total': len(results)
    }


# ===== Cost Estimation =====

def estimate_cost(profile_count: int, ai_model: str = 'gpt-4o-mini') -> float:
    """Estimate cost for generating emails.

    Assumes ~1500 input tokens and ~100 output tokens per profile.
    """
    input_tokens = profile_count * 1500
    output_tokens = profile_count * 100

    if ai_model == 'gpt-4o':
        # gpt-4o: $2.50/1M input, $10/1M output
        cost = (input_tokens * 2.50 / 1_000_000) + (output_tokens * 10.00 / 1_000_000)
    else:
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

    return cost
