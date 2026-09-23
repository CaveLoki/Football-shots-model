"""
Not part of the delivered app — a one-off sanity check that the feature
engineering / training / drift / prediction logic all run end-to-end
without errors, using synthetic data (since FBref can't be reached from
this build environment). Safe to delete or keep as a regression check.
"""
import os
import sys
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import db, features, train, drift, pipeline

random.seed(42)
np.random.seed(42)

TEAMS = {
    "Premier League": ["Arsenal", "Chelsea", "Liverpool", "Man City", "Spurs", "Newcastle"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico", "Sevilla"],
}
PLAYERS_PER_TEAM = 5

print("Building synthetic season...")
rows = []
match_id_counter = 0
start = datetime.strptime(config.SEASON_START_DATE, "%Y-%m-%d")

team_players = {}
for comp, teams in TEAMS.items():
    for t in teams:
        team_players[t] = [f"{t} Player {i}" for i in range(PLAYERS_PER_TEAM)]

for comp, teams in TEAMS.items():
    for week in range(10):  # 10 matchweeks
        date = (start + timedelta(weeks=week)).strftime("%Y-%m-%d")
        shuffled = teams[:]
        random.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            home, away = shuffled[i], shuffled[i + 1]
            match_id = f"m{match_id_counter:06d}"
            match_id_counter += 1
            for team, opponent, is_home in [(home, away, 1), (away, home, 0)]:
                # Give teams a fixed "defensive strength" so drift/opponent-rating logic has signal
                base_shots = 2.0 + (hash(team) % 3)
                for p in team_players[team]:
                    minutes = random.choice([90, 90, 90, 60, 30])
                    shots = np.random.poisson(base_shots / PLAYERS_PER_TEAM * 2) if minutes >= 45 else 0
                    sot = min(shots, np.random.poisson(max(shots * 0.4, 0.1)))
                    rows.append({
                        "match_id": match_id, "player_name": p, "team": team, "opponent": opponent,
                        "is_home": is_home, "date": date, "competition": comp, "season": config.SEASON_LABEL,
                        "minutes": minutes, "shots": int(shots), "shots_on_target": int(sot), "position": "FW",
                    })
            match_rows_meta = {
                "match_id": match_id, "date": date, "competition": comp, "season": config.SEASON_LABEL,
                "home_team": home, "away_team": away, "home_score": random.randint(0, 3),
                "away_score": random.randint(0, 3), "status": "completed",
                "scraped_at": datetime.utcnow().isoformat(),
            }
            rows.append(("__match__", match_rows_meta))

player_rows = [r for r in rows if not isinstance(r, tuple)]
match_rows = [r[1] for r in rows if isinstance(r, tuple)]

raw_df = pd.DataFrame(player_rows)
matches_df = pd.DataFrame(match_rows)

# reset DB for a clean test
if os.path.exists(config.DB_PATH):
    os.remove(config.DB_PATH)
db.init_db()
db.upsert_matches(matches_df)
db.upsert_player_match_stats(raw_df)
print(f"Stored {len(raw_df)} player-match rows across {matches_df.match_id.nunique()} matches.")

print("\nBuilding features...")
feat = features.build_feature_table(raw_df, config.RECENT_N_OVERALL, config.RECENT_N_OPPONENT)
print(f"Feature table shape: {feat.shape}")
assert not feat.empty, "feature table should not be empty"
for col in features.FEATURE_COLUMNS:
    assert col in feat.columns, f"missing feature column {col}"
print("Feature columns present. Sample:")
print(feat[["player_name", "team", "opponent"] + features.FEATURE_COLUMNS + ["target_shots", "target_sot"]].head(3))

print("\nTraining...")
metrics = train.train_and_save(feat)
print(metrics)
assert os.path.exists(config.MODEL_SHOTS_PATH)
assert os.path.exists(config.MODEL_SOT_PATH)

print("\nRunning drift check...")
drift_report = drift.run_drift_check(feat)
print(drift_report)

print("\nBuilding upcoming fixture rows + predicting...")
upcoming = pd.DataFrame([{
    "match_id": "future001", "date": (start + timedelta(weeks=11)).strftime("%Y-%m-%d"),
    "competition": "Premier League", "season": config.SEASON_LABEL,
    "home_team": "Arsenal", "away_team": "Chelsea", "home_score": None, "away_score": None,
    "status": "scheduled", "scraped_at": datetime.utcnow().isoformat(),
}])
pred_rows = features.build_upcoming_feature_rows(raw_df, upcoming, config.RECENT_N_OVERALL, config.RECENT_N_OPPONENT)
print(f"Upcoming feature rows: {len(pred_rows)}")
assert not pred_rows.empty, "should generate at least one upcoming prediction row"

scored = train.predict(pred_rows, config.DEFAULT_SHOT_THRESHOLDS, config.DEFAULT_SOT_THRESHOLDS,
                        config.DEFAULT_MIN_PERCENTAGE)
print(scored[["player_name", "team", "opponent", "pred_shots", "pred_sot", "prob_json"]].head(10).to_string())

print("\nAll smoke-test assertions passed.")
