#!/usr/bin/env python3
"""
Entry point for: python -m ai_ranking_system.rank --candidates ./candidates.jsonl --out ./submission.csv
"""
if __name__ == '__main__':
    from .rank import main
    main()