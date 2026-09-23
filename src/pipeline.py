"""
Orchestration layer. Every entry point here is what app.py and the GitHub
Actions workflow actually call — scraper.py and features.py stay "dumb"
(no knowledge of the DB, no knowledge of what's already been fetched).
"""

from datetime import datetime, timedelta

import pandas as pd

import config
from src import db, scraper, features, train, drift


def scrape_and_store_completed(date_from: str = None, date_to: str = None) -> dict:
    """
    Pull newly completed matches in [date_from, date_to] (defaults to the
    last 8 days) and store any that aren't already in the database.
    Never pulls before config.SEASON_START_DATE, no matter what date_from
    is passed.

    Returns a small summary dict for logging / the Streamlit "scrape now" panel.
    """
    db.init_db()
    if date_from is None:
        date_from, date_to = scraper.last_n_days_range(8)
    date_from = max(date_from, config.SEASON_START_DATE)

    session = scraper._session()
    discovered = scraper.discover_matches(date_from, date_to, session=session)
    completed = discovered[discovered.status == "completed"]

    already_known = db.known_match_ids()
    new_matches = completed[~completed.match_id.isin(already_known)]

    all_player_rows = []
    failures = []
    for _, m in new_matches.iterrows():
        try:
            pdf = scraper.scrape_match_player_stats(
                m.match_id, m.competition, m.date, m.home_team, m.away_team, session=session
            )
            if not pdf.empty:
                all_player_rows.append(pdf)
        except Exception as e:
            failures.append((m.match_id, str(e)))

    if all_player_rows:
        player_df = pd.concat(all_player_rows, ignore_index=True)
        player_df = player_df[player_df.minutes.fillna(0) >= config.MIN_MINUTES_TO_COUNT]
        db.upsert_player_match_stats(player_df)

    db.upsert_matches(discovered)  # store both completed and scheduled rows seen

    return {
        "date_range": (date_from, date_to),
        "matches_discovered": len(discovered),
        "new_completed_matches": len(new_matches),
        "player_rows_added": sum(len(d) for d in all_player_rows),
        "failures": failures,
    }


def scrape_upcoming(days_ahead: int = 7) -> pd.DataFrame:
    """Pull the upcoming fixture list (scheduled matches only) for the next N days."""
    db.init_db()
    date_from, date_to = scraper.next_n_days_range(days_ahead)
    session = scraper._session()
    discovered = scraper.discover_matches(date_from, date_to, session=session)
    upcoming = discovered[discovered.status == "scheduled"]
    db.upsert_matches(discovered)
    return upcoming


def weekly_retrain(recent_n_overall: int = None, recent_n_opponent: int = None) -> dict:
    """
    Full retrain on everything in the database: rebuild features for all
    completed matches, time-based train/test split, fit both models, run
    drift detection against the season baseline, persist model + metrics.
    """
    db.init_db()
    recent_n_overall = recent_n_overall or config.RECENT_N_OVERALL
    recent_n_opponent = recent_n_opponent or config.RECENT_N_OPPONENT

    raw = db.load_player_match_stats()
    if raw.empty:
        return {"status": "no_data"}

    feat_df = features.build_feature_table(raw, recent_n_overall, recent_n_opponent)
    metrics = train.train_and_save(feat_df)
    drift_report = drift.run_drift_check(feat_df)

    db.save_model_metrics(metrics)
    db.save_drift_metrics(drift_report)

    return {"status": "ok", "metrics": metrics, "drift": drift_report.to_dict("records")}


def generate_weekly_predictions(min_percentage: float = None,
                                 shot_thresholds=None, sot_thresholds=None,
                                 recent_n_overall: int = None,
                                 recent_n_opponent: int = None) -> pd.DataFrame:
    """
    For every player likely to feature in the next 7 days' fixtures, build
    their current feature row (using the opponent from their next match)
    and score it with the saved model. Returns the predictions DataFrame
    and also persists it (timestamped) to the `predictions` table.
    """
    db.init_db()
    upcoming = scrape_upcoming(7)
    if upcoming.empty:
        return pd.DataFrame()

    raw = db.load_player_match_stats()
    if raw.empty:
        return pd.DataFrame()

    recent_n_overall = recent_n_overall or config.RECENT_N_OVERALL
    recent_n_opponent = recent_n_opponent or config.RECENT_N_OPPONENT
    min_percentage = min_percentage if min_percentage is not None else config.DEFAULT_MIN_PERCENTAGE
    shot_thresholds = shot_thresholds or config.DEFAULT_SHOT_THRESHOLDS
    sot_thresholds = sot_thresholds or config.DEFAULT_SOT_THRESHOLDS

    pred_rows = features.build_upcoming_feature_rows(
        raw, upcoming, recent_n_overall, recent_n_opponent
    )
    if pred_rows.empty:
        return pd.DataFrame()

    scored = train.predict(pred_rows, shot_thresholds, sot_thresholds, min_percentage)
    scored["generated_at"] = datetime.utcnow().isoformat()

    # The `predictions` table has a fixed schema (prob_json carries every
    # threshold probability); the dynamic eligible_* columns are a
    # display-time convenience recomputed from prob_json + the slider, so
    # only the fixed columns get persisted here.
    db_cols = ["generated_at", "match_id", "player_name", "team", "opponent", "date",
               "competition", "pred_shots", "pred_sot", "sample_size_overall",
               "sample_size_vs_opp", "prob_json"]
    db.save_predictions(scored[db_cols])
    return scored
