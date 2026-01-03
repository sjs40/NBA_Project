import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

class EloRating:
  """
  Elo rating system adapted for NBA basketball.

  Key concepts:
  - Every team stats at a base rating (1500)
  - After each game, ratings are updated based on result
  - The amount of change depends on:
    - K-factor (how much weight to give each game)
    - Expected outcome (based on rating difference)
    - Actual outcome
  - Home court advantage is built in
  """

  def __init__(
    self,
    k_factor: float = 20.0,
    home_advantage: float = 100.0,
    initial_rating: float = 1500.0,
    mean_reversion_factor: float = 0.75
  ):
    """
    Initialize Elo rating system.

    Args:
      k_factor: How much weight to give each game (higher = more volatile)
      home_advantage: Elo points added to home team's expected score
      initial_rating: Starting rating for new teams/seasons
      mean_reversion_factor: How much to regress to mean between seasons
    """
    self.k_factor = k_factor
    self.home_advantage = home_advantage
    self.initial_rating = initial_rating
    self.mean_reversion_factor = mean_reversion_factor
    self.ratings: Dict[int, float] = {}
    self.history: list = []

  def get_rating(self, team_id: int) -> float:
    """Get current rating for a team."""
    return self.ratings.get(team_id, self.initial_rating)

  def expected_score(self, rating_a: float, rating_b: float, home_advantage: float = 0) -> float:
    """
    Calculate expected score (win probability) for team A.

    Uses to logistic curve:
    E = 1 / (1 + 10^((Rb - Ra - HCA) / 400))

    Args:
      rating_a: Team A's Elo rating
      rating_b: Team B's Elo rating
      home_advantage: Additional Elo points for home team

    Returns:
      Probability that team A wins (0-1)
    """
    exponent = (rating_b - rating_a - home_advantage) / 400
    return 1 / (1 + 10 ** exponent)

  def update_ratings(self, home_team_id: int, away_team_id: int, home_score: int, away_score: int, game_date: Optional[datetime] = None, game_id: Optional[str] = None) -> Tuple[float, float]:
    """
    Update ratings after a game

    Args:
      home_team_id: ID of home team
      away_team_id: ID of away team
      home_score: Points scored by home team
      away_score: Points scored by away team
      game_date: Date of game (for history)
      game_id: Game ID (for history)

    Returns:
      Tuple of (new_home_rating, new_away_rating)
    """

    # Get current ratings
    home_rating = self.get_rating(home_team_id)
    away_rating = self.get_rating(away_team_id)

    # Calculate expected scores
    home_expected = self.expected_score(home_rating, away_rating, self.home_advantage)
    away_expected = 1 - home_expected

    # Determine actual scores (1 for win, 0 for loss, 0.5 for tie)
    if home_score > away_score:
      home_actual = 1.0
      away_actual = 0.0
    elif away_score > home_score:
      home_actual = 0.0
      away_actual = 1.0
    else:
      home_actual = 0.5
      away_actual = 0.5

    # Calculate margin of victory multiplier
    # This gives more weight to blowouts
    mov = abs(home_score - away_score)
    mov_multiplier = np.log(mov + 1) * (2.2 / ((home_rating - away_rating) * 0.001 + 2.2))
    mov_multiplier = max(1.0, min(mov_multiplier, 3.0)) # Cap between 1-3

    # Adjust K-factor with MOV
    adjusted_k = self.k_factor * mov_multiplier

    # Update ratings
    new_home_rating = home_rating + adjusted_k * (home_actual - home_expected)
    new_away_rating = away_rating + adjusted_k * (away_actual - away_expected)

    # Store updated ratings
    self.ratings[home_team_id] = new_home_rating
    self.ratings[away_team_id] = new_away_rating

    # Record history
    self.history.append({
      'game_id': game_id,
      'game_date': game_date,
      'home_team_id': home_team_id,
      'away_team_id': away_team_id,
      'home_rating_before': home_rating,
      'away_rating_before': away_rating,
      'home_rating_after': new_home_rating,
      'away_rating_after': new_away_rating,
      'home_expected': home_expected,
      'home_actual': home_actual,
      'home_score': home_score,
      'away_score': away_score
    })

    return new_home_rating, new_away_rating

  def predict_game(self, home_team_id: int, away_team_id: int) -> Dict:
    """
    Predict the outcome of a future game.

    Returns:
      Dict with prediction details
    """
    home_rating = self.get_rating(home_team_id)
    away_rating = self.get_rating(away_team_id)

    home_win_prob = self.expected_score(home_rating, away_rating, self.home_advantage)

    # Convert to spread (rough approx: 25 Elo points ~ 1 point spread)
    rating_diff = home_rating - away_rating + self.home_advantage
    spread = rating_diff / 25

    return {
      'home_rating': home_rating,
      'away_rating': away_rating,
      'home_win_prob': home_win_prob,
      'away_win_prob': 1 - home_win_prob,
      'predicted_spread': spread,
      'predicted_winner_id': home_team_id if home_win_prob > 0.5 else away_team_id
    }

  def new_season_regression(self):
    """
    Regress ratings towards the mean for a new season
    """
    for team_id in self.ratings:
      current = self.ratings[team_id]
      regressed = (current * self.mean_reversion_factor + self.initial_rating * (1 - self.mean_reversion_factor))
      self.ratings[team_id] = regressed

  def get_rankings(self) -> pd.DataFrame:
    """Get current rankings sorted by Elo rating."""
    rankings = pd.DataFrame([
      {'TEAM_ID': team_id, 'elo_rating': rating}
      for team_id, rating in self.ratings.items()
    ])
    rankings = rankings.sort_values('elo_rating', ascending=False)
    rankings['rank'] = range(1, len(rankings) + 1)
    return rankings

  def process_season(self, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process an entire season of games to build ratings.

    Args:
      games_df: DataFrame with colums:
        - game_id, game_date, home_team_id, away_team_id, home_pts, away_pts

    Returns:
      DataFrame with rating history
    """
    games_df = games_df.sort_values('GAME_DATE')

    for _, game in games_df.iterrows():
      self.update_ratings(
        home_team_id=game['HOME_TEAM_ID'],
        away_team_id=game['AWAY_TEAM_ID'],
        home_score=game['HOME_PTS'],
        away_score=game['AWAY_PTS'],
        game_date=game['GAME_DATE'],
        game_id=game['GAME_ID']
      )

    return pd.DataFrame(self.history)

