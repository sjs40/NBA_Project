import pandas as pd
import numpy as np
from typing import Optional, List, Dict

class PaceEfficiency:
  """
  Calculate pace and efficiency metrics.

  These are the most important metrics for comparing teams:
  - Pace: how fast does a team play?
  - Offensive Rating: points per 100 possessions
  - Defensive Rating: points allowed per 100 possessions
  - Net Rating: Offensive Rating - Defensive Rating
  """

  @staticmethod
  def estimate_possessions(fga: float, fta: float, oreb: float, tov: float) -> float:
    """
    Estimate possessions using the standard formula:

    POSS = FGA + 0.44 * FTA - OREB + TOV

    The 0.44 factor accounts for and-ones, technical FTs, etc.
    """
    return fga + 0.44 * fta - oreb + tov

  @classmethod
  def pace(cls, team_poss: float, opp_poss: float, minutes: float = 48.0) -> float:
    """
    Calculate pace (possessions per 48 minutes).

    Pace = 48 * ((Team Poss + Opp Poss) / 2 * Minutes))
    """
    if minutes == 0:
      return 0.0
    return 48 * ((team_poss + opp_poss) / (2 * minutes))

  @staticmethod
  def offensive_rating(points: float, possessions: float) -> float:
    """
    Calculate Offensive Rating (points per 100 possessions).

    ORtg = (Points / Possessions) * 100
    """
    if possessions == 0:
      return 0.0
    return (points / possessions) * 100

  @staticmethod
  def defensive_rating(opp_points: float, possessions: float) -> float:
    """
    Calcualte Defensive Rating (points allowed per 100 possessions).

    DRtg = (Opp Points / Possessions) * 100
    """
    if possessions == 0:
      return 0.0
    return (opp_points / possessions) * 100

  @classmethod
  def net_rating(cls, off_rating: float, def_rating: float) -> float:
    """
    Calculate Net Rating (Offensive Rating - Defensive Rating).

    NetRtg = OffRtg - DefRtg

    Positive = outscoring opponents per 100 possessions
    This is the single best measure of team quality
    """
    return off_rating - def_rating

  @classmethod
  def calculate_game_ratings(cls, team_stats: dict, opp_stats: dict, minutes: float = 48.0) -> dict:
    """Calculate all pace/efficiency metrics for a game."""

    team_poss = cls.estimate_possessions(
      team_stats['fga'], team_stats['fta'], team_stats['oreb'], team_stats['tov']
    )
    opp_poss = cls.estimate_possessions(
      opp_stats['fga'], opp_stats['fta'], opp_stats['oreb'], opp_stats['tov']
    )

    # Avg possessions (should be similar for both teams)
    avg_poss = (team_poss + opp_poss) / 2

    off_rtg = cls.offensive_rating(team_stats['pts'], avg_poss)
    def_rtg = cls.defensive_rating(opp_stats['pts'], avg_poss)

    return {
      'team_possessions': team_poss ,
      'opp_possessions': opp_poss,
      'avg_possessions': avg_poss,
      'pace': cls.pace(team_poss, opp_poss, minutes),
      'off_rating': off_rtg,
      'def_rating': def_rtg,
      'net_rating': cls.net_rating(off_rtg, def_rtg)
    }

  @classmethod
  def calculate_team_ratings(cls, games_df: pd.DataFrame, game_stats_df: pd.DataFrame, team_id: int) -> Dict[str, float]:
    """
    Calculate season ratings using ACTUAL possession data from box scores
    """
    team_games = games_df[
      (games_df['HOME_TEAM_ID'] == team_id) | (games_df['AWAY_TEAM_ID'] == team_id)
    ].copy()

    if len(team_games) == 0:
      return cls._empty_season_ratings()

    total_pts_scored = 0
    total_pts_allowed = 0
    total_team_poss = 0
    total_opp_poss = 0
    games_processed = 0

    for _, game in team_games.iterrows():
      game_id = game['GAME_ID']
      game_stats = game_stats_df[game_stats_df['GAME_ID'] == game_id]

      if len(game_stats) < 2:
        continue

      is_home = game['HOME_TEAM_ID'] == team_id
      opp_team_id = game['AWAY_TEAM_ID'] if is_home else game['HOME_TEAM_ID']

      team_stats_row = game_stats[game_stats['TEAM_ID'] == team_id]
      opp_stats_row = game_stats[game_stats['TEAM_ID'] == opp_team_id]

      if len(team_stats_row) == 0 or len(opp_stats_row) == 0:
        continue

      team_stats = team_stats_row.iloc[0]
      opp_stats = opp_stats_row.iloc[0]

      # Calculate actual possessions from box score data
      team_poss = cls.estimate_possessions(team_stats['fga'], team_stats['fta'], team_stats['oreb'], team_stats['tov'])
      opp_poss = cls.estimate_possessions(opp_stats['fga'], opp_stats['fta'], opp_stats['oreb'], opp_stats['tov'])

      total_pts_scored += team_stats['pts']
      total_pts_allowed += opp_stats['pts']
      total_team_poss += team_poss
      total_opp_poss += opp_poss
      games_processed += 1

    if games_processed == 0:
      return cls._empty_season_ratings()

    # Avg. possessions
    total_poss = (total_team_poss + total_opp_poss) / 2

    # Calculate per-100-possession ratings
    off_rtg = cls.offensive_rating(total_pts_scored, total_poss) if total_poss > 0 else 0
    def_rtg = cls.defensive_rating(total_pts_allowed, total_poss) if total_poss > 0 else 0

    avg_poss_per_game = total_poss / games_processed
    pace = avg_poss_per_game * 2

    return {
      'TEAM_ID': team_id,
      'games_played': games_processed,
      'total_pts': total_pts_scored,
      'total_pts_allowed': total_pts_allowed,
      'total_possessions': total_poss,
      'off_rating': off_rtg,
      'def_rating': def_rtg,
      'net_rating': cls.net_rating(off_rtg, def_rtg),
      'pace': pace
    }


  @classmethod
  def calculate_season_ratings(cls, games_df: pd.DataFrame, game_stats_df: pd.DataFrame, team_id: int) -> dict:
    """
    Calculate season-long ratings for a team.

    Args:
      - games_df: DataFrame with game-level stats
      - game_stats_df: DataFrame with detailed game stats per team containing:
        - game_id: Game ID
        - team_id: Team ID
        - pts: Points scored
        - fga: Field goals attempted
        - fta: Free throws attempted
        - oreb: Offensive rebounds
        - tov: Turnovers
        - minutes: Minutes played (optional, defaults to 48)
      - team_id: Team to calculate for

    Returns:
      - Dict with season averages including:
        - games_played, wins, losses, win_pct
        - total_pts, total_pts_allowed
        - ppg, opp_ppg,
        - total_possessions
        - off_rating, def_rating, net_rating
        - pace
    """
    team_games = games_df[
      (games_df['HOME_TEAM_ID'] == team_id) | (games_df['AWAY_TEAM_ID'] == team_id)
    ].copy()

    if len(team_games) == 0:
      return cls._empty_season_ratings()

    # Get game IDs for this team
    team_game_ids = team_games['GAME_ID'].unique()

    # Filter stats to relevant games
    relevant_stats = game_stats_df[game_stats_df['GAME_ID'].isin(team_game_ids)].copy()

    if len(relevant_stats) == 0:
      return cls._empty_season_ratings()

    # Initialize accumulators
    total_pts_scored = 0
    total_pts_allowed = 0
    total_team_possessions = 0
    total_opp_possessions = 0
    total_minutes = 0
    wins = 0
    games_processed = 0

    # Process each game
    for game_id in team_game_ids:
      game_info = team_games[team_games['GAME_ID'] == game_id].iloc[0]
      game_stats = relevant_stats[relevant_stats['GAME_ID'] == game_id]

      # Determine if team is home or away
      is_home = game_info['HOME_TEAM_ID'] == team_id
      opp_team_id = game_info['AWAY_TEAM_ID'] if is_home else game_info['HOME_TEAM_ID']

      # Get team and opponent stats for this game
      team_stats_row = game_stats[game_stats['TEAM_ID'] == team_id]
      opp_stats_row = game_stats[game_stats['TEAM_ID'] == opp_team_id]

      # Skip if we don't have stats for both teams
      if len(team_stats_row) == 0 or len(opp_stats_row) == 0:
        continue

      team_stats_row = team_stats_row.iloc[0]
      opp_stats_row = opp_stats_row.iloc[0]

      # Extract stats
      pts_scored = team_stats_row['pts']
      pts_allowed = opp_stats_row['pts']

      # Calculate possessions for this game
      team_poss = cls.estimate_possessions(
        fga=team_stats_row['fga'], fta=team_stats_row['fta'], oreb=team_stats_row['oreb'], tov=team_stats_row['tov']
      )
      opp_poss = cls.estimate_possessions(
        fga=opp_stats_row['fga'], fta=opp_stats_row['fta'], oreb=opp_stats_row['oreb'], tov=opp_stats_row['tov']
      )

      # Get minutes (default to 48 for regulation)
      minutes = team_stats_row.get('minutes', 48)
      if pd.isna(minutes) or minutes == 0:
        minutes = 48

      # Accumulate totals
      total_pts_scored += pts_scored
      total_pts_allowed += pts_allowed
      total_team_possessions += team_poss
      total_opp_possessions += opp_poss
      total_minutes += minutes
      games_processed += 1

      # Track wins
      if pts_scored > pts_allowed:
        wins += 1

    if games_processed == 0:
      return cls._empty_season_ratings()

    # Calculate averages
    losses = games_processed - wins
    win_pct = wins / games_processed
    ppg = total_pts_scored / games_processed
    opp_ppg = total_pts_allowed / games_processed

    # Use average of team and opponent possessions for rating calculations
    total_possessions = (total_team_possessions + total_opp_possessions) / 2

    # Calculate efficiency ratings
    off_rating = cls.offensive_rating(total_pts_scored, total_possessions)
    def_rating = cls.defensive_rating(total_pts_allowed, total_possessions)
    net_rtg = cls.net_rating(off_rating, def_rating)

    # Calculate pace
    avg_pace = cls.pace(
      total_team_possessions / games_processed,
      total_opp_possessions / games_processed,
      total_minutes / games_processed
    )

    return {
      'TEAM_ID': team_id,
      'games_played': games_processed,
      'wins': wins,
      'losses': losses,
      'win_pct': win_pct,
      'total_pts': total_pts_scored,
      'total_pts_allowed': total_pts_allowed,
      'ppg': ppg,
      'opp_ppg': opp_ppg,
      'total_possessions': total_possessions,
      'possessions_per_game': total_possessions / games_processed,
      'off_rating': off_rating,
      'def_rating': def_rating,
      'net_rating': net_rtg,
      'pace': avg_pace
    }

  @staticmethod
  def _empty_season_ratings() -> Dict[str, float]:
    return {
      'TEAM_ID': None,
      'games_played': 0,
      'wins': 0,
      'losses': 0,
      'win_pct': 0.0,
      'total_pts': 0,
      'total_pts_allowed': 0,
      'ppg': 0.0,
      'opp_ppg': 0.0,
      'total_possessions': 0,
      'possessions_per_game': 0.0,
      'off_rating': 100.0,
      'def_rating': 100.0,
      'net_rating': 0.0,
      'pace': 100.0
    }

  @classmethod
  def calculate_all_team_net_ratings(cls, games_df: pd.DataFrame, game_stats_df: pd.DataFrame) -> Dict[int, float]:
    """Calculate production net ratings for all teams"""
    home_teams = set(games_df['HOME_TEAM_ID'].unique())
    away_teams = set(games_df['AWAY_TEAM_ID'].unique())
    all_team_ids = home_teams | away_teams

    net_ratings = {}
    for team_id in all_team_ids:
      ratings = cls.calculate_team_ratings(games_df, game_stats_df, team_id)
      net_ratings[team_id] = ratings['net_rating']

    return net_ratings

  @classmethod
  def get_full_ratings_dataframe(cls, games_df: pd.DataFrame, game_stats_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Get complete ratings DataFrame for all teams"""
    home_teams = set(games_df['HOME_TEAM_ID'].unique())
    away_teams = set(games_df['AWAY_TEAM_ID'].unique())
    all_team_ids = home_teams | away_teams

    ratings_list = []
    for team_id in all_team_ids:
      ratings = cls.calculate_team_ratings(games_df, game_stats_df, team_id)

      team_info = teams_df[teams_df['id'] == team_id]
      if len(team_info) > 0:
        team_name = f"{team_info.iloc[0]['city']} {team_info.iloc[0]['nickname']}"
      else:
        team_name = f"Team {team_id}"

      ratings['team_name'] = team_name
      ratings_list.append(ratings)

    df = pd.DataFrame(ratings_list)
    return df.sort_values('net_rating', ascending=False).reset_index(drop=True)

  @classmethod
  def calculate_all_team_ratings(cls, games_df: pd.DataFrame, game_stats_df: pd.DataFrame, team_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Calculate season ratings for all teams

    Args:
      games_df: DataFrame with game info
      game_stats_df: DataFrame with detailed game stats
      team_ids: Optional list of team IDs to calculate for. If None, calculates for all teams found in data

    Returns:
      DataFrame with one row per team containing all rating metrics
    """
    if team_ids is None:
      home_teams = games_df['HOME_TEAM_ID'].unique()
      away_teams = games_df['AWAY_TEAM_ID'].unique()
      team_ids = list(set(home_teams) | set(away_teams))

    ratings_list = []

    for team_id in team_ids:
      team_ratings = cls.calculate_season_ratings(
        games_df=games_df,
        game_stats_df=game_stats_df,
        team_id=team_id
      )
      ratings_list.append(team_ratings)

    df = pd.DataFrame(ratings_list)

    # Sort by net rating (best teams first)
    df = df.sort_values('net_rating', ascending=False).reset_index(drop=True)

    return df

  @classmethod
  def calculate_rolling_ratings(cls, games_df: pd.DataFrame, game_stats_df: pd.DataFrame, team_id: int, n_games: int = 10, as_of_date: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate rolling ratings over the last N games.

    Useful for tracking recent performance and trend.

    Args:
      games_df: DataFrame with game info
      game_stats_df: DataFrame with detailed game stats
      team_id: Team ID to calculate for
      n_games: Number of recent games to include (default: 10)
      as_of_date: Calculate as of this date (for backtesting)

    Returns:
      Dict with rolling ratings
    """
    # Filter games for this team
    team_games = games_df[(games_df['HOME_TEAM_ID'] == team_id) | (games_df['AWAY_TEAM_ID'] == team_id)].copy()

    # Filter by date if specified
    if as_of_date is not None:
      team_games = team_games[team_games['GAME_DATE'] < as_of_date]

    # Sort by date and take last N games
    team_games = team_games.sort_values('GAME_DATE', ascending=False).head(n_games)

    if len(team_games) == 0:
      return cls._empty_season_ratings()

    # Filter stats to these games
    rolling_stats = game_stats_df[
      game_stats_df['GAME_ID'].isin(team_games['GAME_ID'])
    ]

    # Calculate ratings using the same method
    return cls.calculate_season_ratings(games_df, rolling_stats, team_id)



