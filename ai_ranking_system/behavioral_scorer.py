"""
Behavioral Signal Scoring Module.
Analyzes recruiter engagement and candidate availability signals.
Implements stronger signal weighting aligned with JD requirements.
"""

from datetime import datetime
from typing import Dict

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'mindtree', 'tech mahindra', 'hcl', 'lti', 'persistent'
}

class BehavioralSignalScorer:
    def __init__(self, reference_date: str = "2026-06-26"):
        self.reference_date = datetime.strptime(reference_date, "%Y-%m-%d")

    def compute_availability_score(self, signals: Dict) -> float:
        score = 0.5
        
        open_to_work = signals.get('open_to_work_flag', False)
        if open_to_work:
            score += 0.2
        
        last_active = signals.get('last_active_date', '')
        if last_active:
            try:
                last_active_date = datetime.strptime(last_active, "%Y-%m-%d")
                days_since_active = (self.reference_date - last_active_date).days
                
                if days_since_active <= 30:
                    score += 0.2
                elif days_since_active <= 90:
                    score += 0.1
                elif days_since_active <= 180:
                    score += 0.0
                else:
                    score -= 0.3
            except:
                pass
        
        willing_to_relocate = signals.get('willing_to_relocate', False)
        if willing_to_relocate:
            score += 0.05
        
        notice_period = signals.get('notice_period_days', 90)
        if notice_period <= 30:
            score += 0.15
        elif notice_period <= 60:
            score += 0.1
        elif notice_period <= 90:
            score += 0.05
        elif notice_period > 120:
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def compute_engagement_score(self, signals: Dict) -> float:
        score = 0.4
        
        recruiter_response_rate = signals.get('recruiter_response_rate', 0.5)
        # Stronger weighting for response rate - JD explicitly mentions down-weighting low response
        score += min(recruiter_response_rate, 1.0) * 0.30
        
        saved_by_recruiters = signals.get('saved_by_recruiters_30d', 0)
        if saved_by_recruiters >= 10:
            score += 0.15
        elif saved_by_recruiters >= 5:
            score += 0.1
        elif saved_by_recruiters >= 2:
            score += 0.05
        
        search_appearance = signals.get('search_appearance_30d', 0)
        if search_appearance >= 100:
            score += 0.1
        elif search_appearance >= 50:
            score += 0.05
        
        interview_completion = signals.get('interview_completion_rate', 0.7)
        if interview_completion >= 0.8:
            score += 0.1
        elif interview_completion >= 0.6:
            score += 0.05
        elif interview_completion < 0.4:
            score -= 0.15
        
        return max(0.0, min(1.0, score))

    def compute_profile_quality_score(self, signals: Dict) -> float:
        profile_completeness = signals.get('profile_completeness_score', 0) / 100.0
        
        verified_email = signals.get('verified_email', False)
        verified_phone = signals.get('verified_phone', False)
        linkedin_connected = signals.get('linkedin_connected', False)
        
        verification_score = 0.0
        if verified_email:
            verification_score += 0.1
        if verified_phone:
            verification_score += 0.1
        if linkedin_connected:
            verification_score += 0.05
        
        github_score = signals.get('github_activity_score', -1)
        if github_score == -1:
            github_bonus = 0.0
        elif github_score >= 50:
            github_bonus = 0.15
        elif github_score >= 20:
            github_bonus = 0.1
        elif github_score >= 10:
            github_bonus = 0.05
        else:
            github_bonus = 0.0
        
        return min(1.0, profile_completeness * 0.5 + verification_score + github_bonus)

    def compute_location_score(self, profile: Dict) -> float:
        location = profile.get('location', '').lower()
        country = profile.get('country', '').lower()
        
        preferred_locations = ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'ncr', 'bangalore', 'gurgaon']
        
        for loc in preferred_locations:
            if loc in location or loc in country:
                if country == 'india':
                    return 1.0
                return 0.8
        
        if country == 'india':
            return 0.7
        
        return 0.3

    def compute_total_behavioral_score(self, candidate: Dict) -> float:
        signals = candidate.get('redrob_signals', {})
        profile = candidate.get('profile', {})
        
        availability = self.compute_availability_score(signals)
        engagement = self.compute_engagement_score(signals)
        quality = self.compute_profile_quality_score(signals)
        location = self.compute_location_score(profile)
        
        return 0.3 * availability + 0.3 * engagement + 0.25 * quality + 0.15 * location