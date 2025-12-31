from nba_api_collector import NBADataCollector

if __name__ == '__main__':
  ### nba_api_collector examples ###
  collector = NBADataCollector(season="2025-26")

  # Get all teams
  teams_df = collector.get_all_teams()
  print(f"Found {len(teams_df)} teams")
  print(teams_df.head())

  # Get all games this season
  games = collector.get_season_games_with_details()
  print(f"Found {len(games)} games")
  games.sort_values('GAME_DATE')
  print(games.head())

  # Get standings
  standings = collector.get_league_standings()
  print(standings[['TeamCity', 'TeamName', 'WINS', 'LOSSES', 'WinPCT']].head(10))