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

# Try to import anthropic (optional)
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

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

# Israeli school name mappings (long names -> short)
SCHOOL_NAME_MAP = {
    'hamikhlalah ha\'academit lehandassah sami shamoon': 'SCE',
    'sami shamoon college of engineering': 'SCE',
    'hamikhlalah haacademit lehandassah sami shamoon': 'SCE',
    'the academic college of tel aviv-yaffo': 'MTA',
    'tel aviv-yaffo academic college': 'MTA',
    'the college of management academic studies': 'COMAS',
    'college of management academic studies': 'COMAS',
    'the interdisciplinary center': 'IDC Herzliya',
    'interdisciplinary center herzliya': 'IDC Herzliya',
    'reichman university': 'Reichman',
    'the open university of israel': 'Open University',
    'open university of israel': 'Open University',
    'afeka tel aviv academic college of engineering': 'Afeka',
    'afeka - tel aviv academic college of engineering': 'Afeka',
    'holon institute of technology': 'HIT',
    'ort braude college of engineering': 'Braude',
    'braude college of engineering': 'Braude',
    'ruppin academic center': 'Ruppin',
    'sapir academic college': 'Sapir',
    'the technion - israel institute of technology': 'Technion',
    'technion - israel institute of technology': 'Technion',
    'tel aviv university': 'TAU',
    'the hebrew university of jerusalem': 'Hebrew U',
    'hebrew university of jerusalem': 'Hebrew U',
    'ben-gurion university of the negev': 'BGU',
    'ben gurion university of the negev': 'BGU',
    'bar-ilan university': 'Bar-Ilan',
    'bar ilan university': 'Bar-Ilan',
    'weizmann institute of science': 'Weizmann',
    'university of haifa': 'Haifa U',
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


def normalize_school_name(name: str) -> str:
    """Clean school name by mapping long Israeli school names to short versions."""
    if not name:
        return ''

    name_lower = name.lower().strip()
    if name_lower in SCHOOL_NAME_MAP:
        return SCHOOL_NAME_MAP[name_lower]

    return name.strip()


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


def build_email_prompt(sender: str, tone: str, length: str, custom_instruction: str = None, position: str = None, generate_type: str = 'both') -> str:
    """Build the system prompt for email generation."""

    sender_desc = SENDER_PERSONAS.get(sender, 'a recruiter')
    tone_desc = TONE_DESCRIPTIONS.get(tone, TONE_DESCRIPTIONS['professional'])
    length_desc = LENGTH_DESCRIPTIONS.get(length, LENGTH_DESCRIPTIONS['medium'])

    custom_section = f"\n\nADDITIONAL INSTRUCTIONS FROM USER:\n{custom_instruction}" if custom_instruction else ""
    position_section = f" for a **{position}** position" if position else ""

    # Output format based on generate_type
    if generate_type == 'subject_only':
        json_format = """{
  "subject_line": "short punchy subject (<10 words)",
  "subject_angle": "education|title|skills|career|company|domain"
}"""
        generate_desc = "a personalized subject line"
    elif generate_type == 'opener_only':
        json_format = """{
  "email_opener": "personalized opener",
  "opener_angle": "education|title|skills|career|company|domain"
}"""
        generate_desc = "a personalized email opener"
    else:
        json_format = """{
  "subject_line": "short punchy subject (<10 words)",
  "subject_angle": "education|title|skills|career|company|domain",
  "email_opener": "personalized opener",
  "opener_angle": "education|title|skills|career|company|domain"
}"""
        generate_desc = "a personalized subject line and email opener"

    return f"""You are {sender_desc} at an Israeli tech company writing {generate_desc} for candidates{position_section}.

## OUTPUT FORMAT
Return ONLY valid JSON:
{json_format}

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
   - "[Skill/expertise] at [Company]?" - "Go at Monday?" / "Enterprise sales at Gong?"
   - "[Prev] to [Current]?" - "Google to Snyk?" (tier-1 only)
   - "Still [activity] at [Company]?" - "Still building infra at Orca?" / "Still closing enterprise at Salesforce?"
   - "[Title] at [Company]?" - "Staff engineer at Snyk?" / "Head of Marketing at Wiz?"
   - "[Domain] at [Company]?" - "Security backend at Wiz?" / "PLG growth at Monday?"
   - "After [X years] at [Company]?" - "After 5 years at Check Point?"
   - "From [old role] to [new]?" - "From lead back to IC?" / "From SDR to AE?"
   - "[Field] background at [Company]?" - "Physics background at AI21?"
   - "Building [what] at [Company]?" - "Building cloud security at Wiz?" / "Building EMEA pipeline at Snyk?"

   BAD (generic - NEVER use):
   - "Ready to...", "Interested in...", "Looking for...", "Excited about..."
   - Any question that doesn't mention something specific about THEM

3. **OPENER - MUST BE PERSONALIZED & SPECIFIC**:
   - {length_desc}. Tone: {tone_desc}.
   - Must reference a DIFFERENT aspect than subject
   - MUST mention something SPECIFIC from their profile (a tech, project, domain, company product)
   - Write like a real human recruiter, not a bot. Make STATEMENTS about their background, not questions.
   - The opener should explain WHY you're reaching out to THIS person specifically - what in their background fits.

   STYLE: Write a direct statement connecting their experience to the opportunity. NO questions, NO flattery, NO exclamation marks.

   GOOD OPENER EXAMPLES (follow this exact tone):
   - "Your recent experience at ironSource and Unity with Kubernetes and AWS infrastructure fits exactly what we're building."
   - "After leading DevOps at CyberArk with a focus on cloud automation, you'd bring the right background for our platform team."
   - "Your move from backend at Waze to infra at a startup shows you like building from scratch - that's what this role is about."
   - "With your Golang and microservices work at Monday, plus the Technion CS background, you'd be a strong fit for our backend team."
   - "Your 4 years scaling CI/CD pipelines at WalkMe is directly relevant to what we need on our DevOps side."
   - "The combination of team leadership at JFrog and hands-on Terraform work is hard to find - that's why I'm reaching out."
   - "Your track record closing enterprise deals at Gong, especially in the EMEA market, aligns well with what we're looking for."
   - "Going from SDR at Outreach to closing mid-market at Monday shows fast growth - we need that energy on our sales team."
   - "Your product work at Wix scaling the editor for millions of users is the kind of experience that fits our PM role."
   - "Leading demand gen at Similarweb with a focus on PLG and content - that's the exact playbook we're building here."

   BAD OPENER EXAMPLES (NEVER use these patterns):
   - "Leading DevOps at X sounds exciting/intense/dynamic!" - fake flattery, sounds like a bot
   - "Must be a thrilling/wild ride!" - cliché, nobody talks like this
   - "How's the transition treating you?" - generic question
   - "What's been your favorite project?" - generic question
   - "What keeps your team motivated?" - generic, no specifics
   - "How has X shaped your approach?" - generic corporate-speak
   - Any sentence ending with "!" - too enthusiastic, sounds fake
   - Any question about how they "feel" about their work
   - Starting with "Handling..." or "Leading..." followed by flattery

4. **FORBIDDEN WORDS & PHRASES** (instant fail):
   - Em dashes (use regular dash - or comma)
   - "I noticed", "I came across", "I was impressed", "caught my eye"
   - "I hope this finds you well", "Hope you're doing well"
   - "Reaching out because", "Just following up"
   - "Ready to..." subject lines
   - Generic phrases that could apply to anyone
   - Mentioning the same company twice
   - Mentioning companies from before 2018

   FORBIDDEN FLATTERY (sounds fake - instant reject):
   - "impressive", "intriguing", "fascinating", "exciting", "dynamic", "thrilling"
   - "great milestone", "quite the challenge", "quite the dance"
   - "sounds intense", "sounds amazing", "sounds incredible", "sounds dynamic"
   - "must be a wild ride", "must be a thrilling ride", "must be exciting"
   - "leading the charge", "at the forefront", "cutting edge"
   - Any superlative praise about their career
   - Any sentence with "!" (exclamation marks sound fake in cold outreach)

   FORBIDDEN GENERIC QUESTIONS:
   - "What keeps you motivated?"
   - "What inspired you to..."
   - "How has X shaped your approach?"
   - "What's been the most exciting/challenging part?"
   - "How do you keep the team motivated/innovative?"
   - Any question without a specific tech/domain/project mentioned

5. **SCHOOL NAMES**: Use short versions
   - "Hamikhlalah Ha'academit Lehandassah Sami Shamoon" -> "SCE"
   - "The Academic College of Tel Aviv-Yaffo" -> "MTA"
   - "Tel Aviv University" -> "TAU"
   - "Ben-Gurion University of the Negev" -> "BGU"
   - "The Technion - Israel Institute of Technology" -> "Technion"
   - If school name is long and not well-known, skip education angle entirely

6. **COMPANY NAMES**: Clean them up
   - Strip: Ltd, Inc, Corp, Technologies, Labs, Group, Solutions, .io
   - "Check Point Software Technologies, Ltd." -> "Check Point"
   - "AI21 Labs" -> "AI21"

7. **ANGLE SELECTION** (pick what's MOST distinctive - DON'T default to skills):

   CHECK IN THIS ORDER and pick the FIRST that's notable:
   1. Career pattern? Manager->IC, Founder, 5+ years tenure, acquisition, fast promotion → use CAREER angle
   2. Tier-1 company? Google, Meta, Apple, Amazon, Microsoft, Salesforce, Gong → use COMPANY angle
   3. Notable school? Technion, TAU, Hebrew U, Weizmann, BGU, top MBA → use EDUCATION angle
   4. Interesting title? Staff, Principal, Founding, Architect, VP, Head of, Director → use TITLE angle
   5. Unique background? Physics, Math, non-CS field, military intelligence → use BACKGROUND angle
   6. Domain expertise? Specific market, vertical, or functional depth → use DOMAIN angle
   7. Specific skills or tools? Only when nothing else is distinctive → use SKILLS angle (LAST resort)

   IMPORTANT: Skills angle is OVERUSED. Only use it when nothing else is distinctive.

8. **COMPANY TRANSITIONS**: Only mention for tier-1 companies (Google, Meta, Apple, Amazon, Microsoft, Waze). Most people have previous jobs - that's not distinctive!

9. **SAME COMPANY CHECK**: If previous company = current company (or parent/child like Waze/Google), do NOT use company transition angle. Focus on role growth, skills, or education instead.
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
            # Normalize school name to short version
            school = normalize_school_name(school)
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
    client,
    sender: str = 'recruiter',
    tone: str = 'professional',
    length: str = 'medium',
    custom_instruction: str = None,
    ai_model: str = 'gpt-4o-mini',
    tracker=None,
    generate_type: str = 'both',
    position: str = None,
    ai_provider: str = 'openai'
) -> dict:
    """Generate email subject line and/or opener for a single profile.

    Args:
        profile: Profile dict with raw_data or raw_crustdata
        client: OpenAI or Anthropic client instance
        sender: Sender persona key
        tone: Tone key
        length: Length key
        custom_instruction: Optional user instruction
        ai_model: Model to use (gpt-4o-mini, gpt-4o, or claude-haiku-4-5-20251001)
        tracker: Optional usage tracker
        generate_type: What to generate - 'both', 'subject_only', or 'opener_only'
        position: Optional position/role being recruited for (e.g., 'DevOps Engineer', 'Sales Manager')
        ai_provider: "openai" or "anthropic"

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
    system_prompt = build_email_prompt(sender, tone, length, custom_instruction, position, generate_type)

    # Customize user prompt based on generate_type
    if generate_type == 'subject_only':
        user_prompt = f"""## Candidate Profile:
```json
{json.dumps(trimmed, indent=2, default=str)}
```

Generate ONLY a personalized subject line. Return JSON with: subject_line, subject_angle."""
    elif generate_type == 'opener_only':
        user_prompt = f"""## Candidate Profile:
```json
{json.dumps(trimmed, indent=2, default=str)}
```

Generate ONLY a personalized email opener (first 2-3 sentences). Return JSON with: email_opener, opener_angle."""
    else:  # both
        user_prompt = f"""## Candidate Profile:
```json
{json.dumps(trimmed, indent=2, default=str)}
```

Generate a personalized subject line and email opener. Remember: subject and opener MUST use DIFFERENT angles."""

    try:
        if ai_provider == 'anthropic':
            # Anthropic API call
            response = client.messages.create(
                model=ai_model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt + "\n\nReturn ONLY valid JSON, no other text."}
                ],
                temperature=0.7,
                max_tokens=200,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Log usage
            if tracker and hasattr(response, 'usage') and response.usage:
                tracker.log_openai(
                    tokens_input=response.usage.input_tokens,
                    tokens_output=response.usage.output_tokens,
                    model=ai_model,
                    profiles_screened=0,
                    status='success',
                    response_time_ms=elapsed_ms
                )

            raw_content = response.content[0].text
            # Strip markdown code fences if present
            if raw_content.strip().startswith('```'):
                raw_content = re.sub(r'^```(?:json)?\s*', '', raw_content.strip())
                raw_content = re.sub(r'\s*```$', '', raw_content.strip())
            result = json.loads(raw_content)
        else:
            # OpenAI API call
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

        # Fill in empty values for fields not generated based on generate_type
        if generate_type == 'subject_only':
            result.setdefault('email_opener', '')
            result.setdefault('opener_angle', '')
        elif generate_type == 'opener_only':
            result.setdefault('subject_line', '')
            result.setdefault('subject_angle', '')

        # Validate different angles only if generating both
        if generate_type == 'both' and result.get('subject_angle') == result.get('opener_angle'):
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
    api_key: str,
    sender: str = 'recruiter',
    tone: str = 'professional',
    length: str = 'medium',
    custom_instruction: str = None,
    ai_model: str = 'gpt-4o-mini',
    max_workers: int = 10,
    progress_callback=None,
    cancel_flag=None,
    generate_type: str = 'both',
    position: str = None,
    ai_provider: str = 'openai'
) -> list:
    """Generate emails for multiple profiles in parallel.

    Args:
        profiles: List of profile dicts
        api_key: OpenAI or Anthropic API key
        sender: Sender persona
        tone: Tone setting
        length: Length setting
        custom_instruction: Optional user instruction
        ai_model: Model to use
        max_workers: Concurrent threads
        progress_callback: Function(completed, total, result) called after each
        cancel_flag: Dict with 'cancelled' key to check
        generate_type: What to generate - 'both', 'subject_only', or 'opener_only'
        position: Optional position/role being recruited for
        ai_provider: "openai" or "anthropic"

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
        if ai_provider == 'anthropic':
            client = anthropic.Anthropic(api_key=api_key)
        else:
            client = OpenAI(api_key=api_key)

        try:
            result = generate_email_for_profile(
                profile, client,
                sender=sender, tone=tone, length=length,
                custom_instruction=custom_instruction,
                ai_model=ai_model, tracker=tracker,
                generate_type=generate_type,
                position=position,
                ai_provider=ai_provider
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

            # Additional fields for export
            result['first_name'] = raw.get('first_name', '') or profile.get('first_name', '')
            result['last_name'] = raw.get('last_name', '') or profile.get('last_name', '')
            # If no first/last name, try to split from full name
            if not result['first_name'] and name and ' ' in name:
                parts = name.split(' ', 1)
                result['first_name'] = parts[0]
                result['last_name'] = parts[1] if len(parts) > 1 else ''
            result['location'] = raw.get('location', '') or profile.get('location', '')

            # Get university from education
            university = ''
            edu_background = raw.get('education_background', []) or []
            if edu_background:
                # Get first school (most recent)
                first_edu = edu_background[0] if edu_background else {}
                university = first_edu.get('institute_name') or first_edu.get('school_name') or ''
                university = normalize_school_name(university)
            result['university'] = university or profile.get('university', '')

        except Exception as e:
            fallback_name = profile.get('name', f"Profile {index}")
            first_name = profile.get('first_name', '')
            last_name = profile.get('last_name', '')
            if not first_name and fallback_name and ' ' in fallback_name:
                parts = fallback_name.split(' ', 1)
                first_name = parts[0]
                last_name = parts[1] if len(parts) > 1 else ''
            result = {
                "subject_line": "",
                "email_opener": "",
                "error": f"Generation error: {str(e)[:80]}",
                "name": fallback_name,
                "first_name": first_name,
                "last_name": last_name,
                "current_title": profile.get('current_title', ''),
                "current_company": profile.get('current_company', ''),
                "linkedin_url": profile.get('linkedin_url', ''),
                "email": profile.get('email', ''),
                "location": profile.get('location', ''),
                "university": profile.get('university', ''),
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
    elif 'haiku' in ai_model:
        # Claude Haiku: $0.80/1M input, $4.00/1M output
        cost = (input_tokens * 0.80 / 1_000_000) + (output_tokens * 4.00 / 1_000_000)
    else:
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

    return cost
