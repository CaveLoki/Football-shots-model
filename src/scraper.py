"""
FBref scraper.

Design choices, and why:

- One request per DAY (not per competition, not per team) via
  fbref.com/en/matches/{date}, which lists every match across every
  competition on that date, completed or scheduled. This is what keeps
  weekly request volume low — a week is ~7 requests to discover matches,
  regardless of how many of our 8 competitions played that week.
- Every match report page is fetched exactly once, ever: completed matches
  are cached to disk and never re-fetched (see `_cached_get`), and the
  pipeline only asks for matches not already in the database.
- "Politeness" here means: realistic delays between requests, a normal
  browser User-Agent, and retry-with-backoff on 429/403 — not identity
  spoofing or IP rotation. If FBref temporarily blocks a run, the right
  response is to back off and let the next scheduled run pick it up, not
  to route around the block.
- FBref renders some tables (notably match-report player stat tables)
  inside HTML comments — this is a known quirk of their page templating,
  not obfuscation aimed at scrapers. `_extract_commented_tables` unwraps it.
"""

import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

import config

BASE_URL = "https://fbref.com"


# ---------------------------------------------------------------------------
# HTTP layer: session, caching, retry/backoff
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": config.USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def _cache_path(url: str) -> str:
    key = hashlib.sha256(url.encode()).hexdigest()
    return os.path.join(config.REQUEST_CACHE_DIR, f"{key}.html")


def _cached_get(session: requests.Session, url: str, allow_cache: bool = True) -> str:
    """
    GET a URL with disk caching, randomized delay, and retry/backoff.
    Cached pages never re-hit the network — this is the main defence
    against rate limiting, since a re-run (or a retrain covering
    already-scraped weeks) costs zero extra requests.
    """
    cache_file = _cache_path(url)
    if allow_cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    delay = random.uniform(config.MIN_DELAY_SECONDS, config.MAX_DELAY_SECONDS)
    time.sleep(delay)

    for attempt in range(1, config.MAX_RETRIES + 1):
        resp = session.get(url, timeout=30)
        if resp.status_code == 200:
            if allow_cache:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(resp.text)
            return resp.text
        if resp.status_code in (429, 403, 503):
            wait = config.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            wait += random.uniform(0, 5)
            print(f"[scraper] {resp.status_code} on {url} — backing off {wait:.0f}s "
                  f"(attempt {attempt}/{config.MAX_RETRIES})")
            time.sleep(wait)
            continue
        resp.raise_for_status()

    raise RuntimeError(f"Failed to fetch {url} after {config.MAX_RETRIES} attempts")


def _extract_commented_tables(html: str) -> BeautifulSoup:
    """
    FBref wraps several tables inside HTML comments. Pull every comment's
    content back into the DOM so pandas/bs4 can see those tables normally.
    """
    soup = BeautifulSoup(html, "lxml")
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            comment.replace_with(BeautifulSoup(comment, "lxml"))
    return soup


# ---------------------------------------------------------------------------
# Step 1: discover matches for a date range (completed or scheduled)
# ---------------------------------------------------------------------------

def discover_matches(date_from: str, date_to: str, session: requests.Session = None) -> pd.DataFrame:
    """
    Walk /en/matches/{date} for every day in [date_from, date_to] and return
    a DataFrame of matches in our target competitions, with columns:
    match_id, date, competition, season, home_team, away_team,
    home_score, away_score, status.

    One request per calendar day, regardless of how many of our 8
    competitions play that day.
    """
    session = session or _session()
    d0 = datetime.strptime(date_from, "%Y-%m-%d")
    d1 = datetime.strptime(date_to, "%Y-%m-%d")
    comp_names = set(config.COMPETITIONS.keys())

    rows = []
    day = d0
    while day <= d1:
        date_str = day.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/en/matches/{date_str}"
        # Don't cache "today" or future dates — schedules/scores for them
        # can still change between runs.
        allow_cache = day.date() < datetime.utcnow().date()
        try:
            html = _cached_get(session, url, allow_cache=allow_cache)
        except RuntimeError as e:
            print(f"[scraper] skipping {date_str}: {e}")
            day += timedelta(days=1)
            continue

        soup = _extract_commented_tables(html)
        for table in soup.find_all("table"):
            comp_caption = table.find("caption")
            if comp_caption is None:
                continue
            comp_text = comp_caption.get_text(strip=True)
            matched_comp = next((c for c in comp_names if c in comp_text), None)
            if matched_comp is None:
                continue

            for tr in table.select("tbody tr"):
                if tr.get("class") and "thead" in tr.get("class"):
                    continue
                link = tr.select_one('td[data-stat="score"] a, td[data-stat="score"]')
                report_link = None
                for a in tr.find_all("a"):
                    if "/en/matches/" in a.get("href", "") and a.get("href", "").count("/") >= 3:
                        href = a["href"]
                        if re.search(r"/en/matches/[0-9a-f]{8}/", href):
                            report_link = href
                home = tr.select_one('td[data-stat="home_team"]')
                away = tr.select_one('td[data-stat="away_team"]')
                score = tr.select_one('td[data-stat="score"]')
                if home is None or away is None:
                    continue

                match_id = None
                if report_link:
                    m = re.search(r"/en/matches/([0-9a-f]{8})/", report_link)
                    if m:
                        match_id = m.group(1)
                if match_id is None:
                    continue

                score_text = score.get_text(strip=True) if score else ""
                if re.match(r"^\d+\s*[–-]\s*\d+$", score_text):
                    hs, as_ = re.split(r"[–-]", score_text)
                    status = "completed"
                    home_score, away_score = int(hs), int(as_)
                else:
                    status = "scheduled"
                    home_score = away_score = None

                rows.append({
                    "match_id": match_id,
                    "date": date_str,
                    "competition": matched_comp,
                    "season": config.SEASON_LABEL,
                    "home_team": home.get_text(strip=True),
                    "away_team": away.get_text(strip=True),
                    "home_score": home_score,
                    "away_score": away_score,
                    "status": status,
                    "scraped_at": datetime.utcnow().isoformat(),
                })

        day += timedelta(days=1)

    return pd.DataFrame(rows).drop_duplicates(subset="match_id")


# ---------------------------------------------------------------------------
# Step 2: per-match player shot stats (completed matches only)
# ---------------------------------------------------------------------------

def scrape_match_player_stats(match_id: str, competition: str, date: str,
                               home_team: str, away_team: str,
                               session: requests.Session = None) -> pd.DataFrame:
    """
    Fetch one match report and return per-player rows:
    player_name, team, opponent, is_home, minutes, shots, shots_on_target, position.
    """
    session = session or _session()
    url = f"{BASE_URL}/en/matches/{match_id}/"
    html = _cached_get(session, url, allow_cache=True)
    soup = _extract_commented_tables(html)

    rows = []
    for team, opponent, is_home in [(home_team, away_team, 1), (away_team, home_team, 0)]:
        summary_table = None
        for table in soup.find_all("table"):
            tid = table.get("id", "")
            if tid.startswith("stats_") and tid.endswith("_summary"):
                caption = table.find("caption")
                cap_text = caption.get_text(strip=True) if caption else ""
                if team.split()[0].lower() in cap_text.lower() or tid in cap_text.lower():
                    summary_table = table
                    break
        if summary_table is None:
            continue

        try:
            df = pd.read_html(str(summary_table))[0]
        except ValueError:
            continue

        # FBref summary tables have multi-level headers; flatten them.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] for c in df.columns]

        # Drop the trailing "N Players" totals row and header-repeat rows.
        df = df[~df.iloc[:, 0].astype(str).str.contains("Players", na=False)]

        col_map = {c: c for c in df.columns}
        name_col = next((c for c in df.columns if c.lower() == "player"), None)
        min_col = next((c for c in df.columns if c.lower() == "min"), None)
        sh_col = next((c for c in df.columns if c.lower() == "sh"), None)
        sot_col = next((c for c in df.columns if c.lower() == "sot"), None)
        pos_col = next((c for c in df.columns if c.lower() == "pos"), None)

        if name_col is None:
            continue

        for _, r in df.iterrows():
            name = r.get(name_col)
            if pd.isna(name) or str(name).strip() == "":
                continue
            rows.append({
                "match_id": match_id,
                "player_name": str(name).strip(),
                "team": team,
                "opponent": opponent,
                "is_home": is_home,
                "date": date,
                "competition": competition,
                "season": config.SEASON_LABEL,
                "minutes": _safe_int(r.get(min_col)) if min_col else None,
                "shots": _safe_int(r.get(sh_col)) if sh_col else None,
                "shots_on_target": _safe_int(r.get(sot_col)) if sot_col else None,
                "position": r.get(pos_col) if pos_col else None,
            })

    return pd.DataFrame(rows)


def _safe_int(v):
    try:
        if pd.isna(v):
            return None
        return int(v)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Convenience: this week's date range helper
# ---------------------------------------------------------------------------

def last_n_days_range(n: int = 7, end_date: str = None) -> tuple:
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()
    start = end - timedelta(days=n)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def next_n_days_range(n: int = 7, start_date: str = None) -> tuple:
    start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.utcnow()
    end = start + timedelta(days=n)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
