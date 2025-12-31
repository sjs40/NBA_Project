import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class CompositeRankings:
  """
  Composite power ranking system that combines multiple metrics.

  Components:
  1. Elo Rating
  2. Net Rating
  3. Win Percentage
  4. Strength of Schedule adjustment
  5. Recent form (last 10 games)
  """

  # Default weights for each component
  DEFAULT_WEIGHTS = {
    'elo': 0.30,
    'net_rating': 0.30,
    'win_pct': 0.15,
    'sos_adjusted_wins': 0.15,
    'recent_form': 0.10
  }

  def __init__(self, weights: Optional[Dict[str, float]] = None):
    self.weights = weights or self.DEFAULT_WEIGHTS
    self._validate_weights()

  def _validate_weights(self):
    """Ensure weights sum to 1."""
    total = sum(self.weights.values())
    if not np.isclose(total, 1.0):
      # Normalize weights
      self.weights = {k: v/total for k, v in self.weights.items()}

  @staticmethod
  def normalize_metric(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Normalize a metric to 0-100 scale. Uses min-max normalization with league context
    """
    min_val = values.min()
    max_val = values.max()

    if max_val == min_val:
      return pd.Series([50.0] * len(values), index=values.index)

    normalized = (values - min_val) / (max_val - min_val) * 100

    if not higher_is_better:
      normalized = 100 - normalized

    return normalized

  def calculate_strength_of_schedule(self, games_df: pd.DataFrame, team_id: int, opponent_ratings: Dict[int, float]) -> float:
    """
    Calculate SoS based on opponent ratings.

    SOS = Average opponent rating
    """
    team_games = games_df[
      (games_df['HOME_TEAM_ID'] == team_id) | (games_df['AWAY_TEAM_ID'] == team_id)
    ]

    opponent_ids = []
    for _, game in team_games.iterrows():
      opp_id = (game['AWAY_TEAM_ID'] if game['HOME_TEAM_ID'] == team_id else game['HOME_TEAM_ID'])
      opponent_ids.append(opp_id)

    if not opponent_ids:
      return 1500.0 # Default

    opp_ratings = [opponent_ratings.get(opp_id, 1500.0) for opp_id in opponent_ids]
    return np.mean(opp_ratings)

  def calculate_recent_form(self, games_df: pd.DataFrame, team_id: int, n_games: int = 10) -> float:
    """
    Calculate recent form (win percentage in last N games)
    """
    team_games = games_df[(games_df['HOME_TEAM_ID'] == team_id) | (games_df['AWAY_TEAM_ID'] == team_id)].sort_values('GAME_DATE', ascending=False).head(n_games)

    if len(team_games) == 0:
      return 0.5

    wins = 0
    for _, game in team_games.iterrows():
      is_home = game['HOME_TEAM_ID'] == team_id
      if is_home:
        if game['HOME_PTS'] > game['AWAY_PTS']:
          wins += 1
      else:
        if game['AWAY_PTS'] > game['HOME_PTS']:
          wins += 1

    return wins / len(team_games)

  def calculate_rankings(self, teams_df: pd.DataFrame, games_df: pd.DataFrame, elo_ratings: Dict[int, float], net_ratings: Dict[int, float]) -> pd.DataFrame:
    """
    Calculate composite power rankings for all teams.

    Args:
      teams_df: DataFrame with team info
      games_df: DataFrame with game results
      elo_ratings: Dict of team_id -> Elo rating
      net_ratings: Dict of team_id -> Net rating

    Returns:
      DataFrame with rankings
    """
    rankings_data = []

    for _, team in teams_df.iterrows():
      team_id = team['id']

      # Get team's games
      team_games = games_df[(games_df['HOME_TEAM_ID'] == team_id) | (games_df['AWAY_TEAM_ID'] == team_id)]

      # Calculate win percentage
      if len(team_games) > 0:
        wins = sum(
          1 for _, g in team_games.iterrows()
          if (g['HOME_TEAM_ID'] == team_id and g['HOME_PTS'] > g['AWAY_PTS']) or (g['AWAY_TEAM_ID'] == team_id and g['AWAY_PTS'] > g['HOME_PTS'])
        )
        win_pct = wins / len(team_games)
      else:
        win_pct = 0.5

      # Get metrics
      elo = elo_ratings.get(team_id, 1500.0)
      net_rtg = net_ratings.get(team_id, 0.0)
      sos = self.calculate_strength_of_schedule(games_df, team_id, elo_ratings)
      recent = self.calculate_recent_form(games_df, team_id)

      # SOS-adjusted wins (wins above/below expected based on schedule)
      expected_win_pct = 0.5 + (sos - 1500) / 400 * 0.1 # Rough approx
      sos_adjusted = win_pct - expected_win_pct + 0.5

      rankings_data.append({
        'team_id': team_id,
        'team_name': f"{team.get('city', '')} {team.get('nickname', '')}".strip(),
        'abbreviation': team.get('abbreviation', ''),
        'elo': elo,
        'net_rating': net_rtg,
        'win_pct': win_pct,
        'sos': sos,
        'sos_adjusted_wins': sos_adjusted,
        'recent_form': recent,
        'games_played': len(team_games)
      })

    df = pd.DataFrame(rankings_data)

    # Normalize all metrics to 0-100 scale
    df['elo_norm'] = self.normalize_metric(df['elo'])
    df['net_rating_norm'] = self.normalize_metric(df['net_rating'])
    df['win_pct_norm'] = self.normalize_metric(df['win_pct'])
    df['sos_adjusted_norm'] = self.normalize_metric(df['sos_adjusted_wins'])
    df['recent_form_norm'] = self.normalize_metric(df['recent_form'])

    # Calculate composite score
    df['composite_score'] = (
      df['elo_norm'] * self.weights['elo'] +
      df['net_rating_norm'] * self.weights['net_rating'] +
      df['win_pct_norm'] * self.weights['win_pct'] +
      df['sos_adjusted_norm'] * self.weights['sos_adjusted_wins'] +
      df['recent_form_norm'] * self.weights['recent_form']
    )

    # Sort and rank
    df = df.sort_values('composite_score', ascending=False)
    df['rank'] = range(1, len(df) + 1)

    return df

  def format_rankings_output(self, rankings_df: pd.DataFrame) -> str:
    """Format rankings for display."""
    output = ["="*70]
    output.append("NBA POWER RANKINGS")
    output.append("="*70)
    output.append(f"{'Rank':<6}{'Team':<25}{'Score':<10}{'Record':<10}{'Net Rtg':<10}")
    output.append("-"*70)

    for _, row in rankings_df.iterrows():
      games = row['games_played']
      wins = int(row['win_pct'] * games) if games > 0 else 0
      losses = games - wins

      output.append(
        f"{row['rank']:<6}"
        f"{row['team_name']:<25}"
        f"{row['composite_score']:.1f}{'':>5}"
        f"{wins}-{losses}{'':>5}"
        f"{row['net_rating']:+.1f}"
      )

    return "\n".join(output)