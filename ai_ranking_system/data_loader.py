"""
Candidate Data Loader.
Handles loading and preprocessing of candidate JSONL files.
"""

import json
from typing import List, Dict, Iterator

class CandidateLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_all(self) -> Iterator[Dict]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def count_candidates(self) -> int:
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def get_candidate_by_id(self, candidate_id: str) -> Dict:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidate = json.loads(line)
                    if candidate.get('candidate_id') == candidate_id:
                        return candidate
        return {}