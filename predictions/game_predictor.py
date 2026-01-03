"""
NBA Game Prediction Module

Preducts game outcomes including:
- Win probability
- Point spread
- Total points (over/under)

Uses a combination of:
- Elo ratings (win probability, base spread)
- Pace and efficiency (total points prediction)
- Recent form adjustments
- Home court advantage
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GamePrediction:
  """Container for game prediction results"""
  home_team_id: int
  away_team_id: int
  home_team_name: str
  away_team_name: str

  # Win probability
  home_win_prob: float
  away_win_prob: float

  # Spread predicitons (negative = home favored)
  predicted_spread: float
  spread_confidence: float

  # Total points prediction
  predicted_total: float
  total_confidence: float

  # Component breakdowns
  home_predicted_score: float
  away_predicted_score: float

  # Factors used
  home_elo: float
  away_elo: float
  home_net_rating: float
  away_net_rating: float
  home_pace: float
  away_pace: float
  home_off_rating: float
  away_off_rating: float
  home_def_rating: float
  away_def_rating: float

  # Adjustments applied
  home_court_adjustment: float
  rest_adjustment: float
  recent_form_adjustment: float

class GamePredictor:
  """
  Predicts NBA game outcomes using multiple factors

  Methodology:
  1. Win Probability: Elo-based logistic model
  2. Spread: Combination of Elo difference and net rating difference
  3. Total: Pace-adjusted efficiency projection
  """

  # Constants
  HOME_COURT_ELO = 100 # Elo points for home court
  HOME_COURT_TOTAL_POINTS = 3.5 # Point advantage for home team
  HOME_COURT_BOOST = 1.5 # Home team typically scores ~1.5 more
  AWAY_PENALTY = -1.0 # Away teams typically scores ~1.0 less
  ELO_POINTS_FACTOR = 25 # Elo points per 1 point of spread
  LEAGUE_AVG_PACE = 100 # Baseline pace
  LEAGUE_AVG_RATING = 110 # Approximate league average ORtg
  LEAGUE_AVG_TOTAL = 225 # Approximate league average total
  RECENT_GAME_DAMPENER = 0.1 #Dampens effect of recent form

  def __init__(self, elo_ratings: Dict[int, float], team_ratings: pd.DataFrame, games_df: Optional[pd.DataFrame] = None):
    """
    Initialize predictor with team data.

    Args:
      elo_ratings: Dict of team_id -> Elo rating
      team_ratings: DataFrame with columns:
        - team_id, off_rating, def_rating, net_rating, pace
      games_df: Optional DataFrame of games for recent form calculation
    """
    self.elo_ratings = elo_ratings
    self.team_ratings = team_ratings
    self.games_df = games_df

    # Create lookup dict for team ratings
    self.ratings_lookup = {}
    for _, row in team_ratings.iterrows():
      self.ratings_lookup[row['TEAM_ID']] = {
        'off_rating': row.get('off_rating', 110),
        'def_rating': row.get('def_rating', 110),
        'net_rating': row.get('net_rating', 0),
        'pace': row.get('pace', 100)
      }

  def get_elo(self, team_id: int) -> float:
    """Get Elo rating for a team"""
    return self.elo_ratings.get(team_id, 1500.0)

  def get_team_ratings(self, team_id: int) -> Dict[str, float]:
    """Get efficiency ratings for a team"""
    return self.ratings_lookup.get(team_id, {
      'off_rating': 110,
      'def_rating': 110,
      'net_rating': 0,
      'pace': 100
    })

  def calculate_win_probability(self, home_elo: float, away_elo: float, home_advantage: float = None) -> float:
    """
    Calcuate home team win probability using Elo.

    Uses logistic function:
    P(home wins) = 1 / (1 + 10^((away_elo - home_elo - HCA) / 400))
    """
    if home_advantage is None:
      home_advantage = self.HOME_COURT_ELO
    exponent = (away_elo - home_elo - home_advantage) / 400
    return 1 / (1 + 10 ** exponent)

  def calculate_spread(self, home_elo: float, away_elo: float, home_net_rating: float, away_net_rating: float) -> Tuple[float, float]:
    """
    Calculate predicted spread.

    Combines:
    - Elo-based spread (25 Elo points ~ 1 point)
    - Net rating differential
    - Home court advantage

    Returns:
      Tuple of (spread, confidence)
      Negative spread = home team favored
    """
    # Elo-based spread component
    elo_diff = home_elo - away_elo
    elo_spread = elo_diff / self.ELO_POINTS_FACTOR

    # Net rating component
    net_rating_diff = home_net_rating - away_net_rating

    # Combine with weights (elo is more stable, net rating more current)
    # Weight: 40% Elo, 40% net rating, 20% home court
    combined_spread = (
      elo_spread * 0.40 +
      net_rating_diff * 0.40 +
      self.HOME_COURT_TOTAL_POINTS * 0.20
    )

    # Add remaining home court advantage
    final_spread = combined_spread + (self.HOME_COURT_TOTAL_POINTS * 0.80)

    # Confidence based on agreement between Elo and Net Rating
    agreement = 1 - abs(elo_spread - net_rating_diff) / 20
    confidence = max(0.3, min(0.95, agreement))

    return -final_spread, confidence

  def calculate_total(self, home_off_rating: float, home_def_rating: float, away_off_rating: float, away_def_rating: float, home_pace: float, away_pace: float) -> Tuple[float, float, float]:
    """
    Calculate predicted total points.

    Methodology:
    1. Estimate game pace (average of team paces, adjusted)
    2. Calculate expected efficiency for each team
    3. Projected points for each team

    Returns:
      Tuple of (total, home_score, away_score)
    """
    # Estimate game pace
    expected_pace = (home_pace + away_pace) / 2

    # Normalize pace to possessions per team
    # Pace is possessions per 48 minutes (both teams), so each teams get ~1/2
    possessions_per_team = expected_pace / 2

    # Calculate expected efficiency for each team
    # Home team offense vs Away team defense
    # The expected points = (ORtg + opponent DRtg) / 2 * possessions / 100
    # This accounts for both teams' tendencies

    # Home team expected points
    home_expected_efficiency = ((home_off_rating + away_def_rating) / 2)

    # Away team expected points
    away_expected_efficiency = ((away_off_rating + home_def_rating) / 2)

    # Calculate Scores
    # Points = efficiency * possessions / 100
    home_score = (home_expected_efficiency * possessions_per_team / 100) + self.HOME_COURT_BOOST
    away_score = (away_expected_efficiency * possessions_per_team / 100) + self.AWAY_PENALTY

    # Alternative simpler calculation as sanity check
    simple_total = (
      (home_off_rating + away_off_rating) / 2 *
      (expected_pace / 100)
    )

    # Blend the two approaches
    calculated_total = home_score + away_score
    final_total = (calculated_total * 0.8) + (simple_total * 0.2)

    # Adjust individual scores to match total
    score_ratio = home_score / (home_score + away_score) if (home_score + away_score) > 0 else 0.5
    final_home_score = final_total * score_ratio
    final_away_score = final_total * (1 - score_ratio)

    return final_total, final_home_score, final_away_score

  def calculate_recent_form(self, team_id: int, n_games: int = 5) -> float:
    """Calculate recent form adjustment. Returns adjustment in points based on last N games"""
    if self.games_df is None or len(self.games_df) == 0:
      return 0.0

    team_games = self.games_df[
      (self.games_df['HOME_TEAM_ID'] == team_id) | (self.games_df['AWAY_TEAM_ID'] == team_id)
    ].sort_values('GAME_DATE', ascending=False).head(n_games)

    if len(team_games) == 0:
      return 0.0

    margins = []
    for _, game in team_games.iterrows():
      if game['HOME_TEAM_ID'] == team_id:
        margin = game['HOME_PTS'] - game['AWAY_PTS']
      else:
        margin = game['AWAY_PTS'] - game['HOME_PTS']
      margins.append(margin)

    avg_margin = np.mean(margins)

    # Compare to expected margin based on net rating
    team_rating = self.get_team_ratings(team_id)
    expected_margin = team_rating['net_rating']

    # Return adjustment (positive = playing better than expected)
    adjustment = (avg_margin - expected_margin) * self.RECENT_GAME_DAMPENER # Dampen effect
    return np.clip(adjustment, -3.0, 3.0)

  def predict_game(self, home_team_id: int, away_team_id: int, home_team_name: str = "Home", away_team_name: str = "Away", neutral_site: bool = False, home_rest_days: int = 2, away_rest_days: int = 2) -> GamePrediction:
    """
    Generate complete game prediction

    Args:
      home_team_id: Home team ID
      away_team_id: Away team ID
      home_team_name: Display name for home team
      away_team_name: Display name for away team
      neutral_site: If True, no home court advantage
      home_rest_days: Days since home team's last game
      away_rest_days: Days since away team's last game

    Returns:
      GamePrediction object with all predictions
    """
    # Get ratings
    home_elo = self.get_elo(home_team_id)
    away_elo = self.get_elo(away_team_id)
    home_ratings = self.get_team_ratings(home_team_id)
    away_ratings = self.get_team_ratings(away_team_id)

    # Calculate home court adjustment
    home_court_adj = 0 if neutral_site else self.HOME_COURT_TOTAL_POINTS
    home_elo_adj = 0 if neutral_site else self.HOME_COURT_ELO

    # Calculate rest adjustment
    rest_adj = 0
    if home_rest_days <= 1:
      rest_adj -= 2 # Home team tired
    if away_rest_days <= 1:
      rest_adj += 2 # Away team tired
    if home_rest_days >= 3 and away_rest_days <= 1:
      rest_adj += 1 # Extra advantage for home
    if away_rest_days >= 3 and home_rest_days <= 1:
      rest_adj -= 1 # Extra advantage for away

    # Calculate recent form adjustments
    home_form_adj = self.calculate_recent_form(home_team_id)
    away_form_adj = self.calculate_recent_form(away_team_id)
    form_adj = home_form_adj - away_form_adj

    # Win probability
    home_win_probability = self.calculate_win_probability(home_elo, away_elo, home_elo_adj)

    # Spread prediction
    spread, spread_conf = self.calculate_spread(home_elo, away_elo, home_ratings['net_rating'], away_ratings['net_rating'])

    # Apply adjustments to spread
    adjusted_spread = spread - rest_adj - form_adj
    if neutral_site:
      adjusted_spread += self.HOME_COURT_TOTAL_POINTS # Remove home advantage

    # Total Prediction
    total, home_score, away_score = self.calculate_total(
      home_ratings['off_rating'], home_ratings['def_rating'],
      away_ratings['off_rating'], away_ratings['def_rating'],
      home_ratings['pace'], away_ratings['pace']
    )

    # Adjust scores for spread - spread affects the margin, not total
    score_adjustment = adjusted_spread / 2
    adjusted_home_score = home_score - score_adjustment
    adjusted_away_score = away_score + score_adjustment

    # Calculate total confidence
    # Higher confidence when teams have similar paces and many games played
    pace_diff = abs(home_ratings['pace'] - away_ratings['pace'])
    total_conf = max(0.4, min(0.9, 0.8 - pace_diff / 50))
    return GamePrediction(
      home_team_id=home_team_id,
      away_team_id=away_team_id,
      home_team_name=home_team_name,
      away_team_name=away_team_name,
      home_win_prob=home_win_probability,
      away_win_prob=1 - home_win_probability,
      predicted_spread=round(adjusted_spread, 1),
      spread_confidence=spread_conf,
      predicted_total=round(total, 1),
      total_confidence=total_conf,
      home_predicted_score=round(adjusted_home_score, 1),
      away_predicted_score=round(adjusted_away_score, 1),
      home_elo=home_elo,
      away_elo=away_elo,
      home_net_rating=home_ratings['net_rating'],
      away_net_rating=away_ratings['net_rating'],
      home_pace=home_ratings['pace'],
      away_pace=away_ratings['pace'],
      home_off_rating=home_ratings['off_rating'],
      home_def_rating=home_ratings['def_rating'],
      away_off_rating=away_ratings['off_rating'],
      away_def_rating=away_ratings['def_rating'],
      home_court_adjustment=home_court_adj,
      rest_adjustment=rest_adj,
      recent_form_adjustment=form_adj
    )