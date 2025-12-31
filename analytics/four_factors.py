import pandas as pd
import numpy as np
from typing import Dict, Tuple

class FourFactors:
  """
  Calculate Dean Oliver's Four Factors

  The Four Factors explain ~90% of winning in basketball:
  - Shooting (40%)
  - Turnovers (25%)
  - Rebounding (20%)
  - Free Throws (15%)
  """

  # Weights for composite score
  WEIGHTS = {
    'efg': 0.40,
    'tov': 0.25,
    'orb': 0.20,
    'ft_rate': 0.15
  }

  @staticmethod
  def effective_fg_pct(fgm: float, fg3m: float, fga: float) -> float:
    """
    Calculate Effective FG Percentage.
    eFG% = (FGM + 0.5 * 3PM) / FGA

    This adjusts for the fact that 3P are worth more
    """
    if fga == 0:
      return 0.0
    return (fgm + 0.5 * fg3m) / fga

  @staticmethod
  def turnover_rate(tov: float, fga: float, fta: float) -> float:
    """
    Calculate Turnover Rate per 100 possessions.
    TOV% = TOV / (FGA + 0.44 * FTA + TOV)
    """
    possessions = fga + 0.44 * fta + tov
    if possessions == 0:
      return 0.0
    return tov / possessions

  @staticmethod
  def offensive_rebound_rate(oreb: float, opp_dreb: float) -> float:
    """
    Calculate Offensive Rebound Rate
    ORB% = OREB / (OREB + OPP_DREB)
    """
    total = oreb + opp_dreb
    if total == 0:
      return 0.0
    return oreb / total

  @staticmethod
  def free_throw_rate(ftm: float, fga: float) -> float:
    """
    Calculate Free Throw Rate
    FT Rate = FTM / FGA
    """
    if fga == 0:
      return 0.0
    return ftm / fga

  @classmethod
  def calculate_all(cls, team_stats: Dict, opp_stats: Dict) -> Dict[str, float]:
    """
    Calculate all four factors for offense and defense.

    Args:
      team_stats: Dict with keys fgm, fga, fg3m, ftm, fta, tov, oreb, dreb
      opp_stats: Dict with the same keys

    Returns:
      Dict with offensive and defensive four factors
    """
    return {
      # Offensive factors
      'off_efg': cls.effective_fg_pct(
          team_stats['fgm'], team_stats['fg3m'], team_stats['fga']
      ),
      'off_tov': cls.turnover_rate(
          team_stats['tov'], team_stats['fga'], team_stats['fta']
      ),
      'off_orb': cls.offensive_rebound_rate(
          team_stats['oreb'], opp_stats['dreb']
      ),
      'off_ft_rate': cls.free_throw_rate(
          team_stats['ftm'], team_stats['fga']
      ),

      # Defensive factors (opponent's offense = our defense)
      'def_efg': cls.effective_fg_pct(
          opp_stats['fgm'], opp_stats['fg3m'], opp_stats['fga']
      ),
      'def_tov': cls.turnover_rate(
          opp_stats['tov'], opp_stats['fga'], opp_stats['fta']
      ),
      'def_orb': cls.offensive_rebound_rate(
          opp_stats['oreb'], team_stats['dreb']
      ),
      'def_ft_rate': cls.free_throw_rate(
          opp_stats['ftm'], opp_stats['fga']
      ),
    }

  @classmethod
  def calculate_differentials(cls, factors: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate the differential (offense - defense) for each factor.
    Positive = good for offense, negative = good for defense

    NOTE: For TOV%, we flip it since lower offensive TOV% is better.
    """
    return {
      'efg_diff': factors['off_efg'] - factors['def_efg'],
      'tov_diff': factors['def_tov'] - factors['off_tov'],
      'orb_diff': factors['off_orb'] - factors['def_orb'],
      'ft_rate_diff': factors['off_ft_rate'] - factors['def_ft_rate']
    }