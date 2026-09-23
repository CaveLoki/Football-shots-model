"""
All persistence goes through this module. SQLite is enough at this scale
(one season, ~8 competitions, weekly appends) and keeps the whole project
file-based, which is what lets a GitHub Actions job and a Streamlit app
share state just by sharing the repo.
"""

import sqlite3
import contextlib
import pandas as pd

import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    date             TEXT NOT NULL,
    competition      TEXT NOT NULL,
    season           TEXT NOT NULL,
    home_team        TEXT NOT NULL,
    away_team        TEXT NOT NULL,
    home_score       INTEGER,
    away_score       INTEGER,
    status           TEXT NOT NULL,        -- 'completed' | 'scheduled'
    scraped_at       TEXT
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    match_id         TEXT NOT NULL,
    player_name      TEXT NOT NULL,
    team             TEXT NOT NULL,
    opponent         TEXT NOT NULL,
    is_home          INTEGER NOT NULL,
    date             TEXT NOT NULL,
    competition      TEXT NOT NULL,
    season           TEXT NOT NULL,
    minutes          INTEGER,
    shots            INTEGER,
    shots_on_target  INTEGER,
    position         TEXT,
    PRIMARY KEY (match_id, player_name, team)
);

CREATE TABLE IF NOT EXISTS predictions (
    generated_at     TEXT NOT NULL,
    match_id         TEXT NOT NULL,
    player_name      TEXT NOT NULL,
    team             TEXT NOT NULL,
    opponent         TEXT NOT NULL,
    date             TEXT NOT NULL,
    competition      TEXT NOT NULL,
    pred_shots       REAL,
    pred_sot         REAL,
    sample_size_overall  INTEGER,
    sample_size_vs_opp   INTEGER,
    prob_json        TEXT,     -- JSON: {"shots>=1": 0.82, "sot>=2": 0.31, ...}
    PRIMARY KEY (generated_at, match_id, player_name, team)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    retrain_ts       TEXT PRIMARY KEY,
    mae_shots        REAL,
    mae_sot          REAL,
    n_train_rows     INTEGER,
    n_test_rows      INTEGER,
    holdout_from     TEXT,
    holdout_to       TEXT
);

CREATE TABLE IF NOT EXISTS drift_metrics (
    retrain_ts       TEXT NOT NULL,
    feature_name     TEXT NOT NULL,
    psi_score        REAL,
    flag             TEXT,     -- 'ok' | 'warning' | 'alert'
    PRIMARY KEY (retrain_ts, feature_name)
);
"""


@contextlib.contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create every table if it doesn't exist yet. Safe to call every run."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def upsert_matches(df: pd.DataFrame):
    """df columns must match the `matches` table."""
    if df.empty:
        return
    with get_conn() as conn:
        df.to_sql("_tmp_matches", conn, if_exists="replace", index=False)
        conn.execute("""
            INSERT INTO matches (match_id, date, competition, season, home_team,
                                  away_team, home_score, away_score, status, scraped_at)
            SELECT match_id, date, competition, season, home_team, away_team,
                   home_score, away_score, status, scraped_at FROM _tmp_matches
            WHERE true
            ON CONFLICT(match_id) DO UPDATE SET
                home_score=excluded.home_score,
                away_score=excluded.away_score,
                status=excluded.status,
                scraped_at=excluded.scraped_at
        """)
        conn.execute("DROP TABLE _tmp_matches")


def upsert_player_match_stats(df: pd.DataFrame):
    if df.empty:
        return
    with get_conn() as conn:
        df.to_sql("_tmp_pms", conn, if_exists="replace", index=False)
        conn.execute("""
            INSERT OR REPLACE INTO player_match_stats
            SELECT * FROM _tmp_pms
        """)
        conn.execute("DROP TABLE _tmp_pms")


def load_player_match_stats(since: str = None) -> pd.DataFrame:
    q = "SELECT * FROM player_match_stats"
    params = ()
    if since:
        q += " WHERE date >= ?"
        params = (since,)
    with get_conn() as conn:
        return pd.read_sql(q, conn, params=params)


def load_matches(status: str = None) -> pd.DataFrame:
    q = "SELECT * FROM matches"
    params = ()
    if status:
        q += " WHERE status = ?"
        params = (status,)
    with get_conn() as conn:
        return pd.read_sql(q, conn, params=params)


def known_match_ids() -> set:
    with get_conn() as conn:
        rows = conn.execute("SELECT match_id FROM matches WHERE status='completed'").fetchall()
    return {r[0] for r in rows}


def save_predictions(df: pd.DataFrame):
    if df.empty:
        return
    with get_conn() as conn:
        df.to_sql("predictions", conn, if_exists="append", index=False)


def latest_predictions() -> pd.DataFrame:
    with get_conn() as conn:
        latest_ts = conn.execute("SELECT MAX(generated_at) FROM predictions").fetchone()[0]
        if latest_ts is None:
            return pd.DataFrame()
        return pd.read_sql(
            "SELECT * FROM predictions WHERE generated_at = ?", conn, params=(latest_ts,)
        )


def save_model_metrics(row: dict):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO model_metrics
            (retrain_ts, mae_shots, mae_sot, n_train_rows, n_test_rows, holdout_from, holdout_to)
            VALUES (:retrain_ts, :mae_shots, :mae_sot, :n_train_rows, :n_test_rows, :holdout_from, :holdout_to)
        """, row)


def load_model_metrics() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM model_metrics ORDER BY retrain_ts", conn)


def save_drift_metrics(df: pd.DataFrame):
    if df.empty:
        return
    with get_conn() as conn:
        df.to_sql("drift_metrics", conn, if_exists="append", index=False)


def load_drift_metrics() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM drift_metrics ORDER BY retrain_ts", conn)
