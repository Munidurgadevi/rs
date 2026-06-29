# Redrob Signals Documentation

## Overview

Redrob signals are behavioral and platform engagement metrics that provide insights into candidate availability, engagement, profile quality, and location preference. These 23 signals are used in the AI Candidate Discovery & Ranking system to assess candidate fit beyond technical qualifications.

## Signal Categories

### Availability Signals (4 signals)

| Signal | Type | Description | Scoring Impact |
|--------|------|-------------|--------------|
| `open_to_work_flag` | boolean | Indicates if candidate is actively seeking opportunities | +0.2 if true |
| `last_active_date` | date | Date of last platform activity | +0.2 if <=30 days, +0.1 if <=90 days, -0.3 if >180 days |
| `willing_to_relocate` | boolean | Willingness to relocate for role | +0.05 if true |
| `notice_period_days` | integer | Days before candidate can start | +0.15 if <=30, +0.1 if <=60, +0.05 if <=90, -0.1 if >120 |

### Engagement Signals (4 signals)

| Signal | Type | Description | Scoring Impact |
|--------|------|-------------|--------------|
| `recruiter_response_rate` | float (0-1) | Fraction of recruiter messages responded to | Direct multiplier (0.3 weight) |
| `saved_by_recruiters_30d` | integer | Times profile saved by recruiters in 30 days | +0.15 if >=10, +0.1 if >=5, +0.05 if >=2 |
| `search_appearance_30d` | integer | Times appeared in recruiter searches (30 days) | +0.1 if >=100, +0.05 if >=50 |
| `interview_completion_rate` | float (0-1) | Fraction of scheduled interviews attended | +0.1 if >=0.8, +0.05 if >=0.6, -0.15 if <0.4 |

### Profile Quality Signals (5 signals)

| Signal | Type | Description | Scoring Impact |
|--------|------|-------------|--------------|
| `profile_completeness_score` | integer (0-100) | Profile completion percentage | 0.5 weight in quality score |
| `verified_email` | boolean | Email verification status | +0.1 |
| `verified_phone` | boolean | Phone verification status | +0.1 |
| `linkedin_connected` | boolean | LinkedIn profile linked | +0.05 |
| `github_activity_score` | integer (0-100) or -1 | GitHub activity score | +0.15 if >=50, +0.1 if >=20, +0.05 if >=10 |

### Location Signals (2 signals)

| Signal | Type | Description | Scoring Impact |
|--------|------|-------------|--------------|
| `location` | string | City/region from profile | Used in location_score() |
| `country` | string | Country from profile | Returns 1.0 for India+preferred loc |

### Additional Behavioral Signals (10 signals)

These signals are used in `fast_rank.py` for additional bonus/penalty calculations:

| Signal | Type | Scoring Impact |
|--------|------|----------------|
| `signup_date` | date | -0.05 penalty if account <1 year old |
| `profile_views_received_30d` | integer | +0.08 if >=30, +0.04 if >=10 |
| `applications_submitted_30d` | integer | +0.1 if >=10, +0.05 if >=3 |
| `avg_response_time_hours` | integer | +0.1 if <=4, +0.05 if <=12, -0.1 if >48 |
| `skill_assessment_scores` | object | +0.15 if avg top 3 >=80, +0.08 if >=60 |
| `connection_count` | integer | +0.08 if >=500, +0.04 if >=100 |
| `endorsements_received` | integer | +0.1 if >=30, +0.05 if >=10 |
| `expected_salary_range_inr_lpa.max` | integer | -0.1 penalty if >50 LPA |
| `preferred_work_mode` | string | +0.05 if remote/hybrid/flexible |
| `offer_acceptance_rate` | float | +0.1 if >=0.8, +0.05 if >=0.6, -0.15 if <0.4 |

## Scoring Formula

The behavioral score is computed as a weighted combination:

```python
availability = 0.3 * availability_score + 0.3 * engagement_score + 0.25 * quality_score + 0.15 * location_score
```

Where each component uses the signals above to produce a score between 0.0 and 1.0.

## Preferred India Locations

The following locations receive location preference bonuses:
- Pune, Noida, Hyderabad, Mumbai, Delhi, NCR, Bangalore, Gurgaon, Chennai, Kolkata, Vizag, Trivandrum

## Implementation Reference

See `ai_ranking_system/behavioral_scorer.py` for the complete scoring implementation.
See `fast_rank.py` for additional signal bonuses applied to the final score.