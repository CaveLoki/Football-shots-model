"""
CLI entrypoint for the scheduled workflow. Also runnable by hand:
    python scripts/run_weekly_scrape.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline

if __name__ == "__main__":
    summary = pipeline.scrape_and_store_completed()
    print(json.dumps(summary, indent=2, default=str))
    if summary["failures"]:
        print(f"WARNING: {len(summary['failures'])} match(es) failed to parse:")
        for match_id, err in summary["failures"]:
            print(f"  {match_id}: {err}")
