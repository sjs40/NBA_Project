import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collectors.nba_api_collector import NBADataCollector
from data.storage.database import NBADatabase
from rankings.elo import EloRating
from rankings.composite import CompositeRankings

# Page configuration
st.set_page_config(
  page_title="NBA Analytics Dashboard",
  layout="wide",
  initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .prediction-win {
        color: #10B981;
        font-weight: bold;
    }
    .prediction-loss {
        color: #EF4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data(season: str = "2024-25"):
  """Load and cache NBA data."""
  collector = NBADataCollector(season=season)

  teams = collector.get_all_teams()
  games = collector.get_season_games_with_details(season)
  standings = collector.get_league_standings(season)

  return teams, games, standings

@st.cache_resource
def get_elo_system():
  """Initialize and cache Elo rating system. K-factor 32"""
  return EloRating(k_factor=20, home_advantage=100)

def create_rankings_chart(rankings_df: pd.DataFrame) -> go.Figure:
  """Create horizontal bar chart for power rankings."""
  fig = go.Figure()

  # Sort by rank (reverse for bottom-to-top display)
  df = rankings_df.sort_values('rank', ascending=True).head(15)

  fig.add_trace(go.Bar(
    x=df['team_name'],
    y=df['composite_score'],
    orientation='h',
    marker=dict(
      color=df['composite_score'],
      colorscale='Blues',
      showscale=True,
      colorbar=dict(title="Score")
    ),
    text=df['composite_score'].round(1),
    textposition='outside',
  ))

  fig.update_layout(
    title="Power Rankings (Top 15)",
    xaxis_title="Composite Score",
    yaxis_title="",
    height=500,
    yaxis=dict(autorange="reversed"),
    showlegend=False
  )

  return fig

def create_elo_history_chart(history_df: pd.DataFrame, team_ids: list) -> go.Figure:
  """Create line chart showing Elo rating history"""
  fig = go.Figure()

  for team_id in team_ids:
    team_history = history_df[(history_df['home_team_id'] == team_id) | (history_df['away_team_id'] == team_id)].copy()

    team_history['rating'] = team_history.apply(
      lambda x: x['home_rating_after'] if x['home_team_id'] == team_id else x['away_rating_after'], axis=1
    )

    fig.add_trace(go.Scatter(
      x=team_history['game_date'],
      y=team_history['rating'],
      mode='lines+markers',
      name=f"Team {team_id}"
    ))

  fig.update_layout(
    title="Elo Rating Over Time",
    xaxis_title="Date",
    yaxis_title="Elo Rating",
    height=400,
    hovermode='x unified'
  )

  return fig

def main():
  """Main dashboard application."""

  # Header
  st.markdown('<h1 class="main-header">🏀 NBA Analytics Dashboard</h1>', unsafe_allow_html=True)

  # Sidebar
  st.sidebar.title('Navigation')
  page = st.sidebar.radio(
    "Select Page",
    ["Power Rankings", "Team Analysis"]
  )

  # Season selector
  season = st.sidebar.selectbox(
    "Season",
    ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21"],
    index=0
  )

  # Load data
  with st.spinner("Loading data..."):
    teams, games, standings = load_data(season)

  # Initialize systems
  elo = get_elo_system()

  # Process games for Elo
  if len(games) > 0:
    elo_history = elo.process_season(games)

  # === Power Rankings Page ===
  if page == "Power Rankings":
    st.header("📊 Power Rankings")

    col1, col2 = st.columns([2, 1])

    with col1:
      # Calculate composite rankings
      elo_ratings = {r['team_id']: r['elo_rating']
                     for _, r in elo.get_rankings().iterrows()}

      # Simplified net ratings (calculate properly in production)
      net_ratings = {team_id: (rating - 1500) / 10
                     for team_id, rating in elo_ratings.items()}

      ranker = CompositeRankings()
      rankings = ranker.calculate_rankings(teams, games, elo_ratings, net_ratings)

      # Display chart
      fig = create_rankings_chart(rankings)
      st.plotly_chart(fig, use_container_width=True)

    with col2:
      st.subheader("Current Standings")
      if standings is not None and len(standings) > 0:
        display_cols = ['TeamCity', 'TeamName', 'WINS', 'LOSSES', 'WinPCT']
        standings_display = standings[display_cols].head(15)
        standings_display['WinPCT'] = standings_display['WinPCT'].apply(
          lambda x: f"{x:.3f}"
        )
        st.dataframe(standings_display, hide_index=True)

    # Full rankings table
    st.subheader("Full Power Rankings")
    display_rankings = rankings[[
      'rank', 'team_name', 'composite_score', 'elo', 'net_rating', 'win_pct', 'recent_form'
    ]].round(2)
    display_rankings.columns = [
      'Rank', 'Team', 'Score', 'Elo', 'Net Rtg', 'Win%', 'L10'
    ]
    st.dataframe(display_rankings, hide_index=True, use_container_width=True)

  elif page == "Team Analysis":
    st.header("🔍 Team Analysis")

    # Team selector
    team_names = teams['full_name'].tolist()
    selected_team = st.selectbox("Select Team", team_names)

    team_row = teams[teams['full_name'] == selected_team].iloc[0]
    team_id = team_row['id']

    col1, col2, col3, col4 = st.columns(4)

    # Get team stats
    team_games = games[(games['home_team_id'] == team_id) | (games['away_team_id'] == team_id)] if len(games) > 0 else pd.DataFrame()
    if len(team_games) > 0:
      wins = sum(
        1 for _, g in team_games.iterrows()
        if (g['home_team_id'] == team_id and g['home_pts'] > g['away_pts']) or (g['away_team_id'] == team_id and g['away_pts'] > g['home_pts'])
      )
      losses = len(team_games) - wins

      # Calculate PPG
      pts_scored = []
      pts_allowed = []
      for _, g in team_games.iterrows():
        if g['home_team_id'] == team_id:
          pts_scored.append(g['home_pts'])
          pts_allowed.append(g['away_pts'])
        else:
          pts_scored.append(g['away_pts'])
          pts_allowed.append(g['home_pts'])

      ppg = np.mean(pts_scored)
      opp_ppg = np.mean(pts_allowed)

      with col1:
        st.metric("Record", f"{wins}-{losses}")
      with col2:
        st.metric("PPG", f"{ppg:.1f}")
      with col3:
        st.metric("Opp PPG", f"{opp_ppg:.1f}")
      with col4:
        elo_rating = elo.get_rating(team_id)
        st.metric("Elo Rating", f"{elo_rating:.0f}")

    # Elo history chart
    if len(elo.history) > 0:
      st.subheader("Elo Rating History")
      history_df = pd.DataFrame(elo.history)
      fig = create_elo_history_chart(history_df, [team_id])
      st.plotly_chart(fig, use_container_width=True)

    # Recent games
    st.subheader("Recent Games")
    if len(team_games) > 0:
      recent = team_games.sort_values('game_date', ascending=False).head(10)
      st.dataframe(recent, hide_index=True)

  # Footer
  st.divider()
  st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
  st.caption(f"Data source: NBA API | Built with Streamlit")

if __name__ == "__main__":
  main()