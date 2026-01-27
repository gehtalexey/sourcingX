---
name: pre-filter-candidates
description: Pre-filter candidates from enriched CSV before AI screening. Removes past candidates, applies blacklist, and optional filters.
argument-hint: [input-csv-path] [client-folder-path]
allowed-tools: Read, Bash(python:*), Write
---

# Pre-Filter Candidates Before AI Screening

Filter enriched LinkedIn profiles using CSV-based rules BEFORE running expensive AI screening.

## Input

- **Profiles CSV**: `$ARGUMENTS[0]` or `filtered_profiles.csv`
- **Client Folder**: `$ARGUMENTS[1]` - folder containing `past_candidates.csv`
- **Filter CSVs** (stored in Downloads folder):
  - `filters for Cloude Enricher - Blacklist.csv` - Companies to ALWAYS exclude
  - `filters for Cloude Enricher - Universities.csv` - Top universities (optional bonus flag)
  - `filters for Cloude Enricher - NotRelevant Companies.csv` - Large corps/banks to exclude (optional)
  - `Nexite -  FS Mobile Tech Lead -Filterly - Target Companies (1).csv` - Target companies (optional)

## Filtering Rules

### 0. PAST CANDIDATES (Always Applied FIRST)
Exclude any candidate already processed for this client.
- File: `past_candidates.csv` in the client folder
- Match on `first_name` + `last_name` (case-insensitive)
- If file doesn't exist, skip this filter
- **Applied before all other filters**

### 1. BLACKLIST (Always Applied)
Exclude any candidate where `current_company` matches a blacklisted company.
- 63 companies: lakefs, teragen, island, moovit, snappy, trigo, autofleet, apono, dropit, etc.
- Match is case-insensitive and partial (e.g., "Lemonade" matches "Lemonade Inc.")

### 2. NOT RELEVANT COMPANIES - CURRENT (Optional - Ask User)
Exclude candidates currently at large corporations, banks, consulting firms.
- ~650 companies: Deloitte, McKinsey, KPMG, Bank of America, etc.
- Ask: "Filter out candidates currently at large corporations/banks/consulting?"

### 2b. NOT RELEVANT COMPANIES - PAST (Optional - Ask User)
Exclude candidates who worked at not-relevant companies in their past positions.
- Checks the `past_positions` column for any mention of not-relevant companies
- Ask: "Also filter out candidates who worked at large corps/banks in the past?"

### 3. TARGET COMPANIES (Optional - Ask User)
Keep ONLY candidates from target companies (Israel + Global lists).
- ~600 companies: Wiz, Monday.com, Gong, AppsFlyer, Anthropic, OpenAI, etc.
- Ask: "Keep only candidates from target startup/tech companies?"

### 4. TOP UNIVERSITIES (Optional - Flag Only)
Add `top_university` column = True if education contains a top university.
- ~80 universities: Technion, Tel Aviv University, Hebrew University, MIT, Stanford, Harvard, etc.
- Ask: "Flag candidates from top universities?"

### 5. JOB HOPPERS (Optional - Ask User)
Exclude candidates with multiple short stints (less than 1 year at 2+ companies).
- Parses `past_positions` for duration patterns like "[X.X yrs]"
- Counts positions with < 1 year tenure
- Excludes if 2 or more short stints found
- Ask: "Filter out job hoppers (2+ roles under 1 year)?"

### 6. PROJECT/CONSULTING COMPANIES (Optional - Ask User)
Exclude candidates currently at consulting/outsourcing/project-based companies.
- Companies like: Tikal, Matrix, Ness, Sela, Malam Team, Bynet, SQLink, John Bryce, etc.
- These are staff augmentation firms, not product companies
- Ask: "Filter out candidates from consulting/project companies?"

### 7. LONG TENURE SINGLE COMPANY (Optional - Ask User)
Exclude candidates with 8+ years at a single company.
- May indicate comfort zone, limited exposure to different environments
- Check `current_years_in_role` for current position
- Ask: "Filter out candidates with 8+ years at one company?"

### 8. NON-HANDS-ON TITLES (Optional - Ask User)
Exclude candidates with senior management titles that are typically not hands-on.
- Titles to exclude: Director, Head of, VP, Vice President, CTO, CEO, COO, Chief, Group Manager, Engineering Manager
- Titles to KEEP: Team Lead, Tech Lead, Senior Engineer, Staff Engineer, Principal Engineer
- Ask: "Filter out Director/VP/Head level titles?"

## Output

- **Filtered CSV**: `filtered_candidates.csv` (after applying exclusions)
- **Summary**: Count of excluded candidates per rule
- **New Column**: `top_university` (True/False) if university flagging enabled

## Python Implementation

```python
import pandas as pd
import re
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

# Load profiles
df = pd.read_csv(input_file)
original_count = len(df)

# Helper: check if company matches any in list (case-insensitive, partial match)
def matches_list(company, company_list):
    if pd.isna(company) or not str(company).strip():
        return False
    company_lower = str(company).lower().strip()
    for c in company_list:
        if c in company_lower or company_lower in c:
            return True
    return False

# 0. ALWAYS filter past candidates FIRST
past_candidates_file = os.path.join(client_folder, "past_candidates.csv")
if os.path.exists(past_candidates_file):
    past_df = pd.read_csv(past_candidates_file)
    # Match on first_name + last_name (case-insensitive)
    if 'first_name' in past_df.columns and 'last_name' in past_df.columns:
        past_names = set(
            (str(row['first_name']).lower().strip() + ' ' + str(row['last_name']).lower().strip())
            for _, row in past_df.iterrows()
            if pd.notna(row['first_name']) and pd.notna(row['last_name'])
        )
        df['_full_name'] = df['first_name'].fillna('').str.lower().str.strip() + ' ' + df['last_name'].fillna('').str.lower().str.strip()
        df['_is_past'] = df['_full_name'].isin(past_names)
        past_count = df['_is_past'].sum()
        df = df[~df['_is_past']].drop(columns=['_is_past', '_full_name'])
        print(f"Past Candidates: Removed {past_count} already processed")
    else:
        print("Warning: past_candidates.csv missing first_name/last_name columns, skipping")
else:
    print(f"No past_candidates.csv found in {client_folder}, skipping")

# 1. Load blacklist
blacklist_df = pd.read_csv(r"C:\Users\gehta\Downloads\filters for Cloude Enricher - Blacklist.csv")
blacklist = [c.lower().strip() for c in blacklist_df['Company'].dropna().tolist() if c.strip()]

# 1. ALWAYS apply blacklist
df['_blacklisted'] = df['current_company'].apply(lambda x: matches_list(x, blacklist))
blacklist_count = df['_blacklisted'].sum()
df = df[~df['_blacklisted']].drop(columns=['_blacklisted'])

print(f"Blacklist: Removed {blacklist_count} candidates")

# 2. Optional: Not Relevant Companies (current)
if apply_not_relevant_current:
    not_relevant_df = pd.read_csv(r"C:\Users\gehta\Downloads\filters for Cloude Enricher - NotRelevant Companies.csv")
    not_relevant = [c.lower().strip() for c in not_relevant_df['Company'].dropna().tolist() if c.strip()]
    df['_not_relevant'] = df['current_company'].apply(lambda x: matches_list(x, not_relevant))
    not_relevant_count = df['_not_relevant'].sum()
    df = df[~df['_not_relevant']].drop(columns=['_not_relevant'])
    print(f"Not Relevant (current): Removed {not_relevant_count} candidates")

# 2b. Optional: Not Relevant Companies (past positions)
if apply_not_relevant_past:
    if 'not_relevant' not in dir():
        not_relevant_df = pd.read_csv(r"C:\Users\gehta\Downloads\filters for Cloude Enricher - NotRelevant Companies.csv")
        not_relevant = [c.lower().strip() for c in not_relevant_df['Company'].dropna().tolist() if c.strip()]

    def matches_list_in_text(text, company_list):
        if pd.isna(text) or not str(text).strip():
            return False
        text_lower = str(text).lower()
        for c in company_list:
            if c in text_lower:
                return True
        return False

    df['_past_not_relevant'] = df['past_positions'].apply(lambda x: matches_list_in_text(x, not_relevant))
    past_not_relevant_count = df['_past_not_relevant'].sum()
    df = df[~df['_past_not_relevant']].drop(columns=['_past_not_relevant'])
    print(f"Not Relevant (past): Removed {past_not_relevant_count} candidates")

# 3. Optional: Target Companies Only
if apply_target_only:
    target_df = pd.read_csv(r"C:\Users\gehta\Downloads\Nexite -  FS Mobile Tech Lead -Filterly - Target Companies (1).csv")
    # Combine Israel and Global columns
    israel_col = target_df.columns[0]  # "Company Name Israel"
    global_col = target_df.columns[2]  # "Company Name Global"
    target_israel = [c.lower().strip() for c in target_df[israel_col].dropna().tolist() if str(c).strip()]
    target_global = [c.lower().strip() for c in target_df[global_col].dropna().tolist() if str(c).strip()]
    target_companies = list(set(target_israel + target_global))

    df['_is_target'] = df['current_company'].apply(lambda x: matches_list(x, target_companies))
    non_target_count = (~df['_is_target']).sum()
    df = df[df['_is_target']].drop(columns=['_is_target'])
    print(f"Target Filter: Removed {non_target_count} non-target company candidates")

# 4. Optional: Flag top universities
if flag_universities:
    uni_df = pd.read_csv(r"C:\Users\gehta\Downloads\filters for Cloude Enricher - Universities.csv")
    universities = [u.lower().strip() for u in uni_df.iloc[:, 0].dropna().tolist()
                   if u.strip() and not u.startswith('🇮🇱') and not u.startswith('🇪🇺') and not u.startswith('🇺🇸')]

    def has_top_university(education):
        if pd.isna(education) or not str(education).strip():
            return False
        edu_lower = str(education).lower()
        for uni in universities:
            if uni in edu_lower:
                return True
        return False

    df['top_university'] = df['education'].apply(has_top_university)
    top_uni_count = df['top_university'].sum()
    print(f"Top University: Flagged {top_uni_count} candidates")

# 5. Optional: Filter job hoppers
if filter_job_hoppers:
    import re

    def count_short_stints(past_positions):
        if pd.isna(past_positions) or not str(past_positions).strip():
            return 0
        # Find all year patterns like "[0.5 yrs]", "[0.8 yrs]"
        years_pattern = r'\[(\d+\.?\d*)\s*yrs?\]'
        matches = re.findall(years_pattern, str(past_positions))
        short_stints = sum(1 for y in matches if float(y) < 1.0)
        return short_stints

    df['_short_stints'] = df['past_positions'].apply(count_short_stints)
    df['_is_job_hopper'] = df['_short_stints'] >= 2
    job_hopper_count = df['_is_job_hopper'].sum()
    df = df[~df['_is_job_hopper']].drop(columns=['_short_stints', '_is_job_hopper'])
    print(f"Job Hoppers: Removed {job_hopper_count} candidates with 2+ short stints")

# 6. Optional: Filter consulting/project companies
if filter_consulting:
    consulting_companies = [
        'tikal', 'matrix', 'ness', 'sela', 'malam', 'bynet', 'sqlink', 'john bryce',
        'experis', 'manpower', 'comblack', 'aman', 'outsource', 'consulting',
        'services ltd', 'solutions ltd', 'technologies ltd', 'software house',
        'ness technologies', 'matrix it', 'malam team', 'hpe', 'dxc', 'infosys',
        'tata', 'wipro', 'cognizant', 'accenture', 'capgemini', 'atos'
    ]

    def is_consulting_company(company):
        if pd.isna(company) or not str(company).strip():
            return False
        company_lower = str(company).lower()
        for c in consulting_companies:
            if c in company_lower:
                return True
        return False

    df['_is_consulting'] = df['current_company'].apply(is_consulting_company)
    consulting_count = df['_is_consulting'].sum()
    df = df[~df['_is_consulting']].drop(columns=['_is_consulting'])
    print(f"Consulting Companies: Removed {consulting_count} candidates")

# 7. Optional: Filter long tenure (8+ years at one company)
if filter_long_tenure:
    df['_long_tenure'] = df['current_years_in_role'].apply(lambda x: x >= 8 if pd.notna(x) else False)
    long_tenure_count = df['_long_tenure'].sum()
    df = df[~df['_long_tenure']].drop(columns=['_long_tenure'])
    print(f"Long Tenure: Removed {long_tenure_count} candidates with 8+ years at one company")

# 8. Optional: Filter non-hands-on titles
if filter_management_titles:
    exclude_titles = [
        'director', 'head of', 'vp ', 'vice president', 'cto', 'ceo', 'coo', 'cfo',
        'chief ', 'group manager', 'engineering manager', 'r&d manager', 'dev manager',
        'development manager', 'general manager', 'president', 'founder'
    ]
    # Titles to explicitly keep (override exclusions)
    keep_titles = ['team lead', 'tech lead', 'staff', 'principal', 'senior', 'architect']

    def is_management_title(title):
        if pd.isna(title) or not str(title).strip():
            return False
        title_lower = str(title).lower()
        # First check if it's a title we want to keep
        for keep in keep_titles:
            if keep in title_lower:
                return False
        # Then check if it's an excluded title
        for excl in exclude_titles:
            if excl in title_lower:
                return True
        return False

    df['_is_management'] = df['current_title'].apply(is_management_title)
    management_count = df['_is_management'].sum()
    df = df[~df['_is_management']].drop(columns=['_is_management'])
    print(f"Management Titles: Removed {management_count} Director/VP/Head level candidates")

# Save filtered results
df.to_csv(output_file, index=False)
print(f"\nFinal: {len(df)} candidates (from {original_count} original)")
```

## Workflow

1. Ask user for client folder path (contains `past_candidates.csv`)
2. Load profiles CSV
3. **Filter past candidates FIRST** (always, if file exists)
4. Apply blacklist (always)
5. Ask user which optional filters to apply
6. Apply optional filters based on user choice
7. Save filtered CSV
8. Report summary statistics

## Past Candidates File Format

The `past_candidates.csv` in the client folder must have `first_name` and `last_name` columns. Matching is case-insensitive. Example:

```csv
first_name,last_name,status,date_screened
John,Doe,rejected,2024-01-15
Jane,Smith,interviewed,2024-01-20
```
