# Screen Profiles with Claude (Terminal)

Use this guide to screen enriched LinkedIn profiles against a job description using Claude in the terminal.

## Prerequisites

1. Enriched profiles (JSON or CSV) from the dashboard
2. A job description

## How to Screen

### Step 1: Share Your Files

Paste the path to your enriched profiles file:
```
C:\Users\gehta\linkedin-enricher\enriched_20240115_143022.json
```

### Step 2: Share the Job Description

Paste the full job description text directly in the chat.

### Step 3: Request Screening

Say something like:
- "Screen these profiles against the JD"
- "Rank the top 10 candidates"
- "Find the best matches for this role"

## What You'll Get

For each candidate:
- **Score** (1-10)
- **Fit Level** (Strong Fit / Good Fit / Partial Fit / Not a Fit)
- **Summary** - Quick overview of the candidate
- **Strengths** - What makes them a good match
- **Gaps** - What's missing or concerning
- **Recommendation** - Final verdict

## Example Prompts

### Basic Screening
"Screen these 20 profiles against the job description and rank them"

### Filtered Screening
"Show me only Strong Fit and Good Fit candidates"

### Specific Focus
"Screen these profiles, prioritizing candidates with Python and AWS experience"

### Comparison
"Compare the top 3 candidates and tell me which one to interview first"

## Tips

- Start with a small batch (5-10 profiles) to test
- Be specific about must-have vs nice-to-have requirements
- Ask follow-up questions about specific candidates
- Request a summary table for quick comparison

## Output Formats

Ask for results in your preferred format:
- "Give me a summary table"
- "Export as a ranked list"
- "Show detailed analysis for each"
