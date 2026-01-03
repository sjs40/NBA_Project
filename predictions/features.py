import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class FeatureEngineer:
  """
  Create features for NBA game prediction.

  Features fall into several categories:
  1. Team strength metrics (Elo, Net Rating)
  2. Recent performance (last N games)
  3. Head-to-head history
  4. Rest/schedule factors
  5. Home/away splits
  6. Situational factors
  """

  def init_features(self, lookback_games: int = 10):
    self.lookback_games = lookback_games

  def calculate_rolling_stats(self, games_df: pd.DataFrame, team_id: int, as_of_date: datetime, n_games: int = 10) -> Dict[str, float]:
    """Calculate rolling stats for a team as of a specific date"""
    # Get team's games before the specified date
    team_games = games_df[
      ((games_df['HOME_TEAM_ID'] == team_id) | (games_df['AWAY_TEAM_ID'] == team_id)) &
      (games_df['GAME_DATE'] < as_of_date)
    ].sort_values('GAME_DATE', ascending=False).head(n_games)

    if len(team_games) == 0:
      return self._empty_rolling_stats()

    # Calculate stats
    pts_scored = []
    pts_allowed = []
    margins = []
    wins = 0

    for _, game in team_games.iterrows():
      is_home = game['HOME_TEAM_ID'] == team_id

      if is_home:
        scored = game['HOME_PTS']
        allowed = game['AWAY_PTS']
      else:
        scored = game['AWAY_PTS']
        allowed = game['HOME_PTS']

      pts_scored.append(scored)
      pts_allowed.append(allowed)
      margins.append(scored - allowed)

      if scored > allowed:
        wins += 1

    return {
      'rolling_ppg': np.mean(pts_scored),
      'rolling_papg': np.mean(pts_allowed),
      'rolling_margin': np.mean(margins),
      'rolling_win_pct': wins / len(team_games),
      'rolling_std_margin': np.std(margins) if len(margins) > 1 else 0,
      'games_played': len(team_games)
    }

  def _empty_rolling_stats(self) -> Dict[str, float]:
    """Return empty rolling stats for teams with no history"""
    return {
      'rolling_ppg': 110.0,
      'rolling_papg': 110.0,
      'rolling_margin': 0.0,
      'rolling_win_pct': 0.5,
      'rolling_std_margin': 10.0,
      'games_played': 0
    }

  def calculate_rest_days(self, games_df: pd.DataFrame, team_id: int, game_date: datetime) -> int:
    """Calculate days of rest before a game"""
    previous_games = games_df[
      ((games_df['HOME_TEAM_ID'] == team_id) | (games_df['AWAY_TEAM_ID'] == team_id)) &
      (games_df['GAME_DATE'] < game_date)
    ].sort_values('GAME_DATE', ascending=False)

    if len(previous_games) == 0:
      return 3 # Assume normal rest

    last_game_date = previous_games.iloc[0]['GAME_DATE']
    if isinstance(last_game_date, str):
      last_game_date = pd.to_datetime(last_game_date)
    if isinstance(game_date, str):
      game_date = pd.to_datetime(game_date)

    return (game_date - last_game_date).days

  def is_back_to_back(self, rest_days: int) -> bool:
    """Check if team is on a back-to-back"""
    return rest_days <= 1

  def calculate_head_to_head(self, games_df: pd.DataFrame, team1_id: int, team2_id: int, as_of_date: datetime, n_games: int = 5) -> Dict[str, float]:
    """Calculate head-to-head history between two teams"""
    h2h_games = games_df[
      (((games_df['HOME_TEAM_ID'] == team1_id)
      & (games_df['AWAY_TEAM_ID'] == team2_id)) |
       ((games_df['HOME_TEAM_ID'] == team2_id) &
        (games_df['AWAY_TEAM_ID'] == team1_id))) &
      (games_df['GAME_DATE'] < as_of_date)
    ].sort_values('GAME_DATE', ascending=False).head(n_games)

    if len(h2h_games) == 0:
      return {'h2h_win_pct': 0.5, 'h2h_avg_margin': 0, 'h2h_games': 0}

    team1_wins = 0
    margins = []

    for _, game in h2h_games.iterrows():
      is_team1_home = game['HOME_TEAM_ID'] == team1_id

      if is_team1_home:
        margin = game['HOME_PTS'] - game['AWAY_PTS']
      else:
        margin = game['AWAY_PTS'] - game['HOME_PTS']

      margins.append(margin)
      if margin > 0:
        team1_wins += 1

    return {
      'h2h_win_pct': team1_wins / len(h2h_games),
      'h2h_avg_margin': np.mean(margins),
      'h2h_games': len(h2h_games)
    }

  def calculate_home_away_splits(self, games_df: pd.DataFrame, team_id: int, as_of_date: datetime) -> Dict[str, float]:
    """Calculate home vs away splits"""
    team_games = games_df[(games_df['HOME_TEAM_ID'] == team_id) | (games_df['AWAY_TEAM_ID'] == team_id)]

    home_games = team_games[team_games['HOME_TEAM_ID'] == team_id]
    away_games = team_games[team_games['AWAY_TEAM_ID'] == team_id]

    def calc_split_stats(games: pd.DataFrame, is_home: bool):
      if len(games) == 0:
        return 0.5, 0.0

      wins = 0
      margins = []

      for _, g in games.iterrows():
        if is_home:
          margin = g['HOME_PTS'] - g['AWAY_PTS']
        else:
          margin = g['AWAY_PTS'] - g['HOME_PTS']

        margins.append(margin)
        if margin > 0:
          wins += 1

        return wins / len(games), np.mean(margins)

    home_win_pct, home_margin = calc_split_stats(home_games, True)
    away_win_pct, away_margin = calc_split_stats(away_games, False)

    return {
      'home_win_pct': home_win_pct,
      'home_margin': home_margin,
      'away_win_pct': away_win_pct,
      'away_margin': away_margin,
      'home_games': len(home_games),
      'away_games': len(away_games)
    }

  def create_game_features(self, games_df: pd.DataFrame, home_team_id: int, away_team_id: int, game_date: datetime, elo_ratings: Dict[int, float], net_ratings: Dict[int, float]) -> Dict[str, float]:
    """
    Create all features for a single game prediction

    Returns:
      Dict of feature name -> value
    """
    features = {}

    # === Team Strength Features ===
    home_elo = elo_ratings.get(home_team_id, 1500.0)
    away_elo = elo_ratings.get(away_team_id, 1500.0)
    features['elo_diff'] = home_elo - away_elo
    features['home_elo'] = home_elo
    features['away_elo'] = away_elo

    home_net = net_ratings.get(home_team_id, 0)
    away_net = net_ratings.get(away_team_id, 0)
    features['net_rating_diff'] = home_net - away_net

    # === Rolling Stats ===
    home_rolling = self.calculate_rolling_stats(
      games_df, home_team_id, game_date, self.lookback_games
    )
    away_rolling = self.calculate_rolling_stats(
      games_df, away_team_id, game_date, self.lookback_games
    )

    features['home_rolling_margin'] = home_rolling['rolling_margin']
    features['away_rolling_margin'] = away_rolling['rolling_margin']
    features['rolling_margin_diff'] = home_rolling['rolling_margin'] - away_rolling['rolling_margin']
    features['home_rolling_win_pct'] = home_rolling['rolling_win_pct']
    features['away_rolling_win_pct'] = away_rolling['rolling_win_pct']

    # === Rest/Schedule ===
    home_rest = self.calculate_rest_days(games_df, home_team_id, game_date)
    away_rest = self.calculate_rest_days(games_df, away_team_id, game_date)
    features['home_rest_days'] = home_rest
    features['away_rest_days'] = away_rest
    features['rest_advantage'] = home_rest - away_rest
    features['home_b2b'] = int(self.is_back_to_back(home_rest))
    features['away_b2b'] = int(self.is_back_to_back(away_rest))

    # === Head-to-Head ===
    h2h = self.calculate_head_to_head(games_df, home_team_id, away_team_id, game_date)
    features['h2h_home_win_pct'] = h2h['h2h_win_pct']
    features['h2h_avg_margin'] = h2h['h2h_avg_margin']

    # === Home/Away Splits ===
    home_splits = self.calculate_home_away_splits(games_df, home_team_id, game_date)
    away_splits = self.calculate_home_away_splits(games_df, away_team_id, game_date)
    features['home_team_home_margin'] = home_splits['home_margin']
    features['away_team_away_margin'] = away_splits['away_margin']

    return features

  def create_training_dataset(self, games_df: pd.DataFrame, elo_ratings_history: pd.DataFrame, min_games_threshold: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create training dataset from historical games.

    Returns:
      X: Feature DataFrame
      y: Target Series (1 = home win, 0 = away win)
    """
    features_list = []
    targets = []

    games_df = games_df.sort_values('GAME_DATE')

    for idx, game in games_df.iterrows():
      game_date = game['GAME_DATE']

      # Skip early season games
      home_games = len(games_df[
                         ((games_df['HOME_TEAM_ID'] == game['HOME_TEAM_ID']) |
                          (games_df['AWAY_TEAM_ID'] == game['HOME_TEAM_ID'])) &
                         (games_df['GAME_DATE'] < game_date)
                       ])
      if home_games < min_games_threshold:
        continue

      # Get Elo ratings as of game date
      elo_as_of = elo_ratings_history[
        elo_ratings_history['game_date'] <= game_date
      ].groupby('TEAM_ID')['rating_after'].last().to_dict()

      # Calculate net ratings
      net_ratings = {}

      # Create features
      features = self.create_game_features(
        games_df=games_df,
        home_team_id=game['HOME_TEAM_ID'],
        away_team_id=game['AWAY_TEAM_ID'],
        game_date=game_date,
        elo_ratings=elo_as_of,
        net_ratings=net_ratings
      )

      features['game_id'] = game.get('GAME_ID', idx)
      features_list.append(features)

      # Target: 1 if home team won
      targets.append(1 if game['HOME_PTS'] > game['AWAY_PTS'] else 0)

    X = pd.DataFrame(features_list)
    y = pd.Series(targets, name='home_win')

    return X, y


