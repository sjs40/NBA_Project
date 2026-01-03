from nba_api.stats.endpoints import (
leaguegamefinder,
teamgamelogs,
leaguestandings,
teamestimatedmetrics,
boxscoretraditionalv2,
boxscoreadvancedv2
)

from nba_api.stats.static import teams, players
import pandas as pd
import time
from datetime import datetime
from typing import Optional, List, Dict

class NBADataCollector:
  """Collects NBA data from the official NBA API"""

  def __init__(self, season: str = "2024-25"):
    self.season = season
    self.teams_dict = {team['id']: team for team in teams.get_teams()}

  def _rate_limit(self, seconds: float = 0.6):
    """Respect API rate limits."""
    time.sleep(seconds)

  def get_all_teams(self) -> pd.DataFrame:
    """Get all NBA teams."""
    return pd.DataFrame(teams.get_teams())

  def get_team_game_logs(self, team_id: int, season: Optional[str] = None) -> pd.DataFrame:
    """Get game logs for a specific team."""
    season = season or self.season
    self._rate_limit()

    logs = teamgamelogs.TeamGameLogs(
      team_id_nullable=team_id,
      season_nullable=season
    )
    return logs.get_data_frames()[0]

  def get_all_games(self, season: Optional[str] = None) -> pd.DataFrame:
    """Get all games for a season"""
    season = season or self.season
    self._rate_limit()

    games = leaguegamefinder.LeagueGameFinder(
      season_nullable=season,
      league_id_nullable="00"
    )
    return games.get_data_frames()[0]

  def get_league_standings(self, season: Optional[str] = None) -> pd.DataFrame:
    """Get current league standings."""
    season = season or self.season
    self._rate_limit()

    standings = leaguestandings.LeagueStandings(season_nullable=season)
    return standings.get_data_frames()[0]

  def get_team_advanced_stats(self, season: Optional[str] = None) -> pd.DataFrame:
    """Get estimated advanced metrics for all teams."""
    season = season or self.season
    self._rate_limit()

    metrics = teamestimatedmetrics.TeamEstimatedMetrics(season=season)
    return metrics.get_data_frames()[0]

  def get_box_score(self, game_id: str) -> Dict[str, pd.DataFrame]:
    """Get detailed box score for a game."""
    self._rate_limit()

    traditional = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    advanced = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)

    return {
      'player_stats': traditional.get_data_frames()[0],
      'team_stats': traditional.get_data_frames()[1],
      'player_advanced': advanced.get_data_frames()[0],
      'team_advanced': advanced.get_data_frames()[1]
    }

  def get_season_games_with_details(self, season: Optional[str] = None) -> pd.DataFrame:
    """Get all games with consolidated home/away information. Returns one row per game (not two rows per team)."""
    season = season or self.season
    df = self.get_all_games(season)

    # Split into home and away
    home_games = df[df['MATCHUP'].str.contains(' vs. ')].copy()
    away_games = df[df['MATCHUP'].str.contains(' @ ')].copy()

    # Rename columns for merging
    home_games = home_games.add_prefix('HOME_')
    away_games = away_games.add_prefix('AWAY_')

    # Merge on game ID
    games = home_games.merge(
      away_games,
      left_on='HOME_GAME_ID',
      right_on='AWAY_GAME_ID'
    )

    # Clean up columns
    games['GAME_ID'] = games['HOME_GAME_ID']
    games['GAME_DATE'] = pd.to_datetime(games['HOME_GAME_DATE'])
    games['HOME_TEAM_ID'] = games['HOME_TEAM_ID']
    games['AWAY_TEAM_ID'] = games['AWAY_TEAM_ID']
    games['HOME_PTS'] = games['HOME_PTS']
    games['AWAY_PTS'] = games['AWAY_PTS']
    games['HOME_WINS'] = (games['HOME_PTS'] > games['AWAY_PTS']).astype(int)

    return games.sort_values(by='GAME_DATE')

  def get_all_team_game_stats(self, season: Optional[str] = None) -> pd.DataFrame:
    """Get detailed game stats for all teams, needed for possession calculation"""
    season = season or self.season
    all_stats = []
    team_list = teams.get_teams()

    for team in team_list:
      team_id = team['id']
      try:
        self._rate_limit(0.7)

        logs = teamgamelogs.TeamGameLogs(team_id_nullable=team_id, season_nullable=season)
        df = logs.get_data_frames()[0]

        if len(df) > 0:
          stats_df = pd.DataFrame({
            'GAME_ID': df['GAME_ID'],
            'TEAM_ID': df['TEAM_ID'],
            'team_name': df['TEAM_NAME'],
            'GAME_DATE': pd.to_datetime(df['GAME_DATE']),
            'pts': df['PTS'],
            'fgm': df['FGM'],
            'fga': df['FGA'],
            'fg3m': df['FG3M'],
            'fg3a': df['FG3A'],
            'ftm': df['FTM'],
            'fta': df['FTA'],
            'oreb': df['OREB'],
            'dreb': df['DREB'],
            'reb': df['REB'],
            'ast': df['AST'],
            'tov': df['TOV'],
            'stl': df['STL'],
            'blk': df['BLK']
          })
          all_stats.append(stats_df)
      except Exception as e:
        print(f"Error fetching team stats for {team_id}: {e}")
        continue

    if all_stats:
      return pd.concat(all_stats, ignore_index=True)
    return pd.DataFrame()



