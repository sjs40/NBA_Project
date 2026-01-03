import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path

class NBAGamePredictor:
  """
  Ensemble model for NBA game prediction.

  Combines multiple models:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest (captures non-linear patters)
  3. Gradient Boosting (high accuracy)

  Uses probability calibration for better probability estimates.
  """

  def __init__(self, model_dir: str = "models"):
    self.model_dir = Path(model_dir)
    self.model_dir.mkdir(parents=True, exist_ok=True)

    self.scaler = StandardScaler()

    # Define base model
    self.models = {
      'logistic': LogisticRegression(
        max_iter=1000,
        random_state=42
      ),
      'random_forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_leaf=20,
        n_jobs=-1
      ),
      'gradient_boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
      )
    }

    # Ensemble weights (tunes based on validation performance)
    self.ensemble_weights = {
      'logistic': 0.25,
      'random_forest': 0.35,
      'gradient_boosting': 0.40
    }

    self.feature_columns: List[str] = []
    self.is_fitted = False

  def _get_feature_columns(self, X: pd.DataFrame) -> List[str]:
    """Get list of feature columns (excluding non-features)"""
    exclude = ['GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
    return [c for c in X.columns if c not in exclude]

  def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NBAGamePredictor':
    """
    Train all models on the training data.

    Args:
      X: Feature DataFrame
      y: Target Series (1 = home win)

    Returns:
      self
    """
    self.feature_columns = self._get_feature_columns(X)
    X_features = X[self.feature_columns].copy()

    # Handle missing values
    X_features = X_features.fillna(X_features.median())

    # Scale features
    X_scaled = self.scaler.fit_transform(X_features)

    # Train each model with probability calibration
    for name, model in self.models.items():
      print(f"Training {name} model...")

      # Calibrate probabilities using cross-validation
      calibrated = CalibratedClassifierCV(
        model,
        method='isotonic',
        cv=5
      )
      calibrated.fit(X_scaled, y)
      self.models[name] = calibrated

      # Evaluate with time-series cross-validation
      tscv = TimeSeriesSplit(n_splits=5)
      scores = cross_val_score(
        calibrated, X_scaled, y, cv=tscv, scoring='accuracy'
      )
      print(f"  CV Accuracy: {scores.mean():.3f} +/- {scores.std()*2:.3f}")

    self.is_fitted = True
    return self

  def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
    """
    Predict win probabilities for games

    Returns:
      Array of [away_win_prob, home_win_prob] for each game
    """
    if not self.is_fitted:
      raise ValueError("Model must be fit. Call fit() first.")

    X_features = X[self.feature_columns].copy()
    X_features = X_features.fillna(X_features.median())
    X_scaled = self.scaler.transform(X_features)

    # Get predictions from each model
    predictions = []
    for name, model in self.models.items():
      prob = model.predict_proba(X_scaled)
      predictions.append(prob * self.ensemble_weights[name])

    # Weighted average of probabilities
    ensemble_prob = np.sum(predictions, axis=0)

    return ensemble_prob

  def predict(self, X: pd.DataFrame) -> np.ndarray:
    """Predict game winners (1 = home win)"""
    proba = self.predict_proba(X)
    return (proba[:1] > 0.5).astype(int)

  def predict_games(self, X: pd.DataFrame, team_names: Optional[Dict[int, str]] = None) -> pd.DataFrame:
    """
    Make predictions with detailed output

    Returns:
      DataFrame with predictions for each game
    """
    proba = self.predict_proba(X)

    results = pd.DataFrame({
      'home_team_id': X.get('home_team_id', range(len(X))),
      'away_team_id': X.get('away_team_id', range(len(X))),
      'home_win_prob': proba[:, 1],
      'away_win_prob': proba[:, 0],
      'predicted_winner': np.where(proba[:, 1] > 0.5, 'HOME', 'AWAY'),
      'confidence': np.abs(proba[:, 1] - 0.5) * 2 # 0-1 scale
    })

    # Add spread prediction (rough approximation)
    # ~2.5 points per 10% probability difference
    prob_diff = proba[:, 1] - 0.5
    results['predicted_spread'] = prob_diff * 25 # Positive = home favored

    # Add team names if provided
    if team_names:
      results['home_team'] = results['home_team_id'].map(team_names)
      results['away_team'] = results['away_team_id'].map(team_names)

    return results

  def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance

    Returns:
      Dict with various metrics
    """

