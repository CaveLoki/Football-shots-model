"""
CLI entrypoint for the scheduled workflow. Also runnable by hand:
    python scripts/run_predict.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline

if __name__ == "__main__":
    preds = pipeline.generate_weekly_predictions()
    if preds.empty:
        print("No predictions generated (no upcoming fixtures found, or no trained model yet).")
    else:
        print(f"Generated {len(preds)} prediction rows.")
        print(preds.sort_values("pred_shots", ascending=False).head(15).to_string(index=False))
