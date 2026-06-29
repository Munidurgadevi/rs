# Submission Specification

## Overview

This document defines the submission requirements for the Redrob AI Candidate Discovery & Ranking Challenge. Submissions must produce a ranked list of top 100 candidates based on the Senior AI Engineer job description.

## Input Format

### candidates.jsonl Structure

The input file is a JSONL (JSON Lines) file where each line contains a candidate object:

```json
{
  "candidate_id": "CAND_0000001",
  "profile": {
    "anonymized_name": "Name",
    "headline": "Professional headline",
    "summary": "Multi-sentence summary",
    "location": "City, State",
    "country": "Country",
    "years_of_experience": 6.9,
    "current_title": "Job Title",
    "current_company": "Company",
    "current_company_size": "10001+",
    "current_industry": "Industry"
  },
  "career_history": [
    {
      "company": "Company",
      "title": "Title",
      "start_date": "YYYY-MM-DD",
      "end_date": null,
      "duration_months": 27,
      "is_current": true,
      "industry": "Industry",
      "company_size": "10001+",
      "description": "Role description"
    }
  ],
  "education": [...],
  "skills": [...],
  "certifications": [...],
  "languages": [...],
  "redrob_signals": { ... }
}
```

See `candidate_schema.json` for the complete schema specification.

**Key Fields:**
- `candidate_id`: Must match pattern `CAND_XXXXXXX` (7 digits)
- `profile`: Contains basic candidate info including location and experience
- `career_history`: Array of past roles with company, title, duration
- `skills`: Array of skill objects with name, proficiency, endorsements
- `redrob_signals`: Behavioral signals including response rate, activity metrics

## Output Format

### submission.csv

Row 1 must be the header row:
```csv
candidate_id,rank,score,reasoning
```

Rows 2-101 must contain exactly 100 data rows with:
- `candidate_id`: CAND_XXXXXXX format
- `rank`: Integer from 1 to 100
- `score`: Float score (non-increasing with rank)
- `reasoning`: String explaining why candidate was ranked

**Requirements:**
- Scores must be non-increasing by rank (higher rank = lower or equal score)
- Tie-breaking: For equal scores, candidate_id must be ascending
- All 100 ranks (1-100) must appear exactly once

## Running the Ranker

```bash
py fast_rank.py --candidates candidates.jsonl.gz --out submission.csv
```

Or using the uncompressed file:
```bash
py fast_rank.py --candidates candidates.jsonl --out submission.csv
```

## Compute Constraints

| Constraint | Requirement |
|------------|-------------|
| Python | 3.11+ |
| Compute | CPU only (no GPU) |
| Network | No external API calls during ranking |
| Runtime | ~32 seconds for 100K candidates on 8-core CPU, 16GB RAM |
| Pre-computation | None required |
| Dependencies | Pure Python / standard library |

## Metadata Template Requirements

The `submission_metadata.yaml` file in your repo must contain:

### Required Fields

| Field | Description |
|-------|-------------|
| `team_name` | Team identifier |
| `primary_contact.name` | Full name of primary contact |
| `primary_contact.email` | Email for organizer communication |
| `primary_contact.phone` | Phone number for outreach |
| `team_members` | List of team members with emails |
| `github_repo` | Repository URL |
| `sandbox_link` | Hosted demo environment URL |
| `reproduce_command` | Single command to produce submission.csv |
| `compute.platform` | Computer platform description |
| `compute.cpu_cores` | Number of CPU cores |
| `compute.ram_gb` | Available RAM in GB |
| `compute.python_version` | Python version |
| `compute.os` | Operating system |
| `compute.uses_gpu_for_inference` | Must be `false` |
| `compute.has_network_during_ranking` | Must be `false` |
| `compute.pre_computation_required` | Must be `false` |
| `ai_tools_used` | List of AI tools used |
| `ai_usage_summary` | Description of AI usage |
| `methodology_summary` | Approach summary (≤200 words) |
| `declarations.read_submission_spec` | Must be `true` |
| `declarations.code_is_original_work` | Must be `true` |
| `declarations.no_collusion` | Must be `true` |
| `declarations.reproduction_tested` | Must be `true` |

## Validation

Validate your submission with:
```bash
py validate_submission.py submission.csv
```

This validates:
- Correct header format
- Exactly 100 data rows
- Valid candidate_id format
- Valid rank and score values
- Non-increasing scores with proper tie-breaking