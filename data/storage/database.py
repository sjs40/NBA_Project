import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

class NBADatabase:
  "SQLite database for storing NBA data."

  def init_db(self, db_path: str = "data/nba_analytics.db"):
    self.db_path = Path(db_path)
    self.db_path.parent.mkdir(parents=True, exist_ok=True)
    self._initialize_db()

  @contextmanager
  def get_connection(self):
    """Connect manager for database connections."""
    conn = sqlite3.connect(self.db_path)
    try:
      yield conn
    finally:
      conn.close()

  def _initialize_db(self):
    """Create tables if they don't exist"""
    with self.get_connection() as conn:
      conn.executescript("""
        CREATE TABLE IF NOT EXISTS teams (
          team_id INTEGER PRIMARY KEY,
          abbreviation TEXT,
          full_name TEXT,
                    city TEXT,
                    nickname TEXT,
                    conference TEXT,
                    division TEXT
                );
                
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    game_date DATE,
                    season TEXT,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    home_pts INTEGER,
                    away_pts INTEGER,
                    home_win INTEGER,
                    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
                    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
                );
                
                CREATE TABLE IF NOT EXISTS team_game_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT,
                    team_id INTEGER,
                    is_home INTEGER,
                    pts INTEGER,
                    fgm INTEGER,
                    fga INTEGER,
                    fg3m INTEGER,
                    fg3a INTEGER,
                    ftm INTEGER,
                    fta INTEGER,
                    oreb INTEGER,
                    dreb INTEGER,
                    reb INTEGER,
                    ast INTEGER,
                    stl INTEGER,
                    blk INTEGER,
                    tov INTEGER,
                    pf INTEGER,
                    plus_minus INTEGER,
                    FOREIGN KEY (game_id) REFERENCES games(game_id),
                    FOREIGN KEY (team_id) REFERENCES teams(team_id)
                );
                
                CREATE TABLE IF NOT EXISTS elo_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER,
                    rating REAL,
                    rating_date DATE,
                    season TEXT,
                    FOREIGN KEY (team_id) REFERENCES teams(team_id)
                );
                
                CREATE TABLE IF NOT EXISTS power_rankings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER,
                    rank INTEGER,
                    composite_score REAL,
                    elo_score REAL,
                    net_rating REAL,
                    record_score REAL,
                    ranking_date DATE,
                    season TEXT,
                    FOREIGN KEY (team_id) REFERENCES teams(team_id)
                );
                
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT,
                    prediction_date TIMESTAMP,
                    model_name TEXT,
                    home_win_prob REAL,
                    predicted_winner_id INTEGER,
                    predicted_spread REAL,
                    predicted_total REAL,
                    actual_winner_id INTEGER,
                    was_correct INTEGER,
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
                CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
                CREATE INDEX IF NOT EXISTS idx_elo_date ON elo_ratings(rating_date);
                CREATE INDEX IF NOT EXISTS idx_rankings_date ON power_rankings(ranking_date);
                """)

  def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
    """Save a DataFrame to the database."""
    with self.get_connection() as conn:
      df.to_sql(table_name, conn, if_exists=if_exists, index=False)

  def query(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """Execute a query and return results as DataFrame."""
    with self.get_connection() as conn:
      return pd.read_sql_query(sql, conn, params=params)

  def get_games(self, season: Optional[str] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> pd.DataFrame:
    """Get games with optional filters."""
    sql = "SELECT * FROM games WHERE 1=1"
    params = []

    if season:
      sql += " AND season = ?"
      params.append(season)
    if start_date:
      sql += " AND game_date >= ?"
      params.append(start_date)
    if end_date:
      sql += " AND game_date <= ?"
      params.append(end_date)

    sql += " ORDER BY game_date"
    return self.query(sql, tuple(params) if params else None)

  def get_latest_elo_ratings(self) -> pd.DataFrame:
    """Get the most recent Elo ratings for each team."""
    sql = """
      SELECT e.*
      FROM elo_ratings e
      INNER JOIN (
        SELECT team_id, MAX(rating_date), as max_date
        FROM elo_ratings
        GROUP BY team_id
      ) latest ON e.team_id = latest.team_id AND e.rating_date = latest.max_date
    """
    return self.query(sql)

