import pandas as pd
import numpy as np
from typing import Optional

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
      'possessions': avg_poss,
      'pace': cls.pace(team_poss, opp_poss, minutes),
      'off_rating': off_rtg,
      'def_rating': def_rtg,
      'net_rating': cls.net_rating(off_rtg, def_rtg)
    }

  @classmethod
  def calculate_season_ratings(cls, games_df: pd.DataFrame, team_id: int) -> dict:
    """
    Calculate season-long ratings for a team.

    Args:
      - games_df: DataFrame with game-level stats
      - team_id: Team to calculate for

    Returns:
      - Dict with season averages
    """
    team_games = games_df[
      (games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)
    ]

    total_pts = 0
    total_opp_pts = 0
    total_poss = 0

    for _, game in team_games.iterrows():
      is_home = game['home_team_id'] == team_id

      if is_home:
        pts = game['home_pts']
        opp_pts = game['away_pts']
      else:
        pts = game['away_pts']
        opp_pts = game['home_pts']

      # Simplified
      total_pts += pts
      total_opp_pts += opp_pts

    # Simplified - in practice, you'd sum actual possessions
    games_played = len(team_games)
    avg_poss_per_game = 100 # League average approximation
    total_poss = games_played * avg_poss_per_game

    off_rtg = cls.offensive_rating(total_pts, total_poss)
    def_rtg = cls.defensive_rating(total_opp_pts, total_poss)

    return {
      'games_played': games_played,
      'ppg': total_pts / games_played if games_played > 0 else 0,
      'opp_ppg': total_opp_pts / games_played if games_played > 0 else 0,
      'off_rating': off_rtg,
      'def_rating': def_rtg,
      'net_rating': cls.net_rating(off_rtg, def_rtg)
    }

