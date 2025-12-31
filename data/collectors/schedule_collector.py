from nba_api.stats.endpoints import leaguegamelog, scoreboardv2
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

class ScheduleCollector:
  """Collects upcoming NBA game schedules."""

  def __init__(self):
    self.base_url = "https://www.basketball-reference.com"

  def _rate_limit(self, seconds: float = 0.6):
    """Respect API rate limits."""
    time.sleep(seconds)

  def get_today_games(self) -> pd.DataFrame:
    """Get today's game from NBA API."""
    self._rate_limit()
    scoreboard = scoreboardv2.ScoreboardV2(game_date=datetime.now().strftime("%Y-%m-%d"))
    return scoreboard.get_data_frames()[0]

  def get_upcoming_games_bbref(self, days_ahead: int = 7) -> pd.DataFrame:
    """Scrape upcoming games from Basketball Reference."""
    games_list = []

    for i in range(days_ahead):
      date = datetime.now() + timedelta(days=i)
      month = date.strftime("%B").lower()
      day = date.day
      year = date.year

      url = f"{self.base_url}/leagues/NBA_{year}_games-{month}.html"

      try:
        self._rate_limit()
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
          soup = BeautifulSoup(response.text, 'html.parser')
          games = soup.find_all('tr', class_='game_row')
          games_list.extend(games)
      except Exception as e:
        print(f"Error fetching games for {date}: {e}")
        continue

    return pd.DataFrame(games_list)