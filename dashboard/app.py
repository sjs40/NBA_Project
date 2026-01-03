import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

from analytics.efficiency import PaceEfficiency
from predictions.game_predictor import GamePredictor, GamePrediction

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collectors.nba_api_collector import NBADataCollector
from data.storage.database import NBADatabase
from rankings.elo import EloRating
from rankings.composite import CompositeRankings
from nba_api.stats.endpoints import leaguegamefinder, teamgamelogs, leaguestandings, boxscoretraditionalv2
from nba_api.stats.static import teams

# Page configuration
st.set_page_config(
  page_title="NBA Analytics Dashboard",
  page_icon="🏀",
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
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_basic_data(season: str):
  collector = NBADataCollector(season=season)
  return collector.get_all_teams(), collector.get_season_games_with_details(season), collector.get_league_standings(season)

@st.cache_data(ttl=3600)
def load_detailed_stats(season: str):
  """Load detailed box score stats."""
  collector = NBADataCollector(season=season)
  return collector.get_all_team_game_stats(season)


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

# =============
# VIZUALIZATION
# =============

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
    team_history = history_df[(history_df['HOME_TEAM_ID'] == team_id) | (history_df['AWAY_TEAM_ID'] == team_id)].copy()

    team_history['rating'] = team_history.apply(
      lambda x: x['home_rating_after'] if x['HOME_TEAM_ID'] == team_id else x['away_rating_after'], axis=1
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

def create_net_rating_chart(ratings_df):
  fig = go.Figure()
  fig.add_trace(go.Bar(name='Offensive', y=ratings_df['team_name'], x=ratings_df['off_rating'],
                       orientation='h', marker_color='#10B981'))
  fig.add_trace(go.Bar(name='Defensive', y=ratings_df['team_name'], x=ratings_df['def_rating'],
                       orientation='h', marker_color='#EF4444'))

  fig.update_layout(
    title='Off vs Def Ratings',
    barmode='group',
    height=500,
    yaxis=dict(autorange="reversed")
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
    ["Power Rankings", "Net Ratings", "Game Predictor", "Team Analysis"]
  )

  # Season selector
  season = st.sidebar.selectbox(
    "Season",
    ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21"],
    index=0
  )

  # Load data
  with st.spinner("Loading data..."):
    teams_df, games, standings = load_basic_data(season)

  # Initialize systems
  elo = get_elo_system()

  # Process games for Elo
  if len(games) > 0:
    elo.process_season(games)

  elo_ratings = {r['TEAM_ID']: r['elo_rating'] for _, r in elo.get_rankings().iterrows()}

  # ===================
  # POWER RANKINGS PAGE
  # ===================

  if page == "Power Rankings":
    st.header("📊 Power Rankings")

    with st.spinner("Loading box scores for net ratings..."):
      try:
        game_stats = load_detailed_stats(season)
        if len(game_stats) > 0:
          pe = PaceEfficiency()
          net_ratings = pe.calculate_all_team_net_ratings(games, game_stats)
          st.success("✅ Net ratings from box scores.")
        else:
          net_ratings = {tid: 0.0 for tid in elo_ratings}
          st.warning("No box score data - using ELO ratings")
      except Exception as e:
        st.error(f"Error: {e}")
        net_ratings = {tid: 0.0 for tid in elo_ratings}

    ranker = CompositeRankings()
    rankings = ranker.calculate_rankings(teams_df, games, elo_ratings, net_ratings)

    if len(rankings) > 0:
      st.plotly_chart(create_rankings_chart(rankings), use_container_width=True)
      st.subheader("Full Rankings")

      disp = rankings[['rank', 'team_name', 'composite_score', 'elo',
                       'net_rating', 'win_pct', 'recent_form']].copy()
      disp.columns = ['Rank', 'Team', 'Score', 'Elo', 'NetRtg', 'Win%', 'L10']
      disp['Score'] = disp['Score'].round(1)
      disp['Elo'] = disp['Elo'].round(0)
      disp['NetRtg'] = disp['NetRtg'].round(1)
      disp['Win%'] = (disp['Win%']*100).round(1)
      disp['L10'] = (disp['L10']*100).round(1)
      st.dataframe(disp, hide_index=True, use_container_width=True)


  # ===========
  # NET RATINGS
  # ===========
  elif page == "Net Ratings":
    st.header("📈 Net Ratings Analysis")
    st.info("Net Rating = Offensive Rating - Defensive Rating (per 100 possessions)")

    with st.spinner("Calculating from box scores..."):
      try:
        game_stats = load_detailed_stats(season)
        if len(game_stats) > 0:
          pe = PaceEfficiency()
          ratings_df = pe.get_full_ratings_dataframe(games, game_stats, teams_df)
          st.plotly_chart(create_net_rating_chart(ratings_df), use_container_width=True)

          st.subheader("All Teams")
          disp = ratings_df[['team_name', 'games_played', 'off_rating', 'def_rating', 'net_rating', 'pace']].copy()
          disp.columns = ['Team', 'GP', 'ORtg', 'DRtg', 'NetRtg', 'Pace']
          for c in ['ORtg', 'DRtg', 'NetRtg', 'Pace']:
            disp[c] = disp[c].round(1)
          st.dataframe(disp, hide_index=True, use_container_width=True)
        else:
          st.warning("No box score data.")
      except Exception as e:
        st.error(f"Error: {e}")

  # ===================
  # GAME PREDICTOR PAGE
  # ===================
  elif page == "Game Predictor":
    st.header("🎯 Game Predictor")

    st.info("""
    **Select two teams to predict the game outcome.**
    
    Predictions include:
    - **Win Probability**: Based on Elo Ratings
    - **Spread**: Point differential prediction (negative = home favored)
    - **Total Points**: Combined score prediction based on pace & efficiency
    """)

    with st.spinner("Loading team statistics..."):
      try:
        game_stats = load_detailed_stats(season)
        if len(game_stats)>0:
          pe = PaceEfficiency()
          ratings_df = pe.get_full_ratings_dataframe(games, game_stats, teams_df)
          net_ratings = {row['TEAM_ID']: row['net_rating'] for _, row in ratings_df.iterrows()}
          has_detailed_stats = True
        else:
          ratings_df = pd.DataFrame()
          net_ratings = {tid: 0.0 for tid in elo_ratings}
          has_detailed_stats = False
      except Exception as e:
        st.warning(f"Could not load detailed box score stats: {e}")
        ratings_df = pd.DataFrame()
        net_ratings = {tid: 0.0 for tid in elo_ratings}
        has_detailed_stats = False

    # Team selection
    col1, col2 = st.columns(2)

    team_options = sorted(teams_df['full_name'].tolist())

    with col1:
      st.subheader("✈️ Away Team")
      away_team_name = st.selectbox("Select Away Team", team_options, key="away")
      away_row = teams_df[teams_df['full_name'] == away_team_name].iloc[0]
      away_id = away_row['id']

    with col2:
      st.subheader("🏠 Home Team")
      # Filter out away team from home options
      home_options = [t for t in team_options if t != away_team_name]
      home_team_name = st.selectbox("Select Home Team", home_options, key="home")
      home_row = teams_df[teams_df['full_name'] == home_team_name].iloc[0]
      home_id = home_row['id']

    # Additional options
    st.subheader("⚙️ Game Settings")
    settings_col1, settings_col2, settings_col3 = st.columns(3)

    with settings_col1:
      neutral_site = st.checkbox("Neutral Site", value=False)
    with settings_col2:
      away_rest = st.selectbox("Away Team Rest Days", [0, 1, 2, 3, 4, 5], index=2)
    with settings_col3:
      home_rest = st.selectbox("Home Team Rest Days", [0, 1, 2, 3, 4, 5], index=2)

    st.divider()

    # make prediction
    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):

      # Build team ratings DataFrame for GamePredictor
      if has_detailed_stats and len(ratings_df) > 0:
        team_ratings_for_predictor = ratings_df[['TEAM_ID', 'off_rating', 'def_rating', 'net_rating', 'pace']].copy()
      else:
        # Create default ratings if no detailed stats
        all_team_ids = list(set(teams_df['id'].tolist()))
        team_ratings_for_predictor = pd.DataFrame({
          'TEAM_ID': all_team_ids,
          'off_rating': [110.0] * len(all_team_ids),
          'def_rating': [110.0] * len(all_team_ids),
          'net_rating': [0.0] * len(all_team_ids),
          'pace': [100.0] * len(all_team_ids)
        })

      # Initialize GamePredictor with current data
      predictor = GamePredictor(
        elo_ratings=elo_ratings,
        team_ratings=team_ratings_for_predictor,
        games_df=games
      )

      # Generate prediction using GamePredictor class
      prediction = predictor.predict_game(
        home_team_id=home_id,
        away_team_id=away_id,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        neutral_site=neutral_site,
        home_rest_days=home_rest,
        away_rest_days=away_rest
      )

      # === DISPLAY RESULTS ===

      st.markdown("---")
      st.subheader(f"📊 Prediction: {prediction.away_team_name} @ {prediction.home_team_name}")

      # Main prediction cards
      pred_col1, pred_col2, pred_col3 = st.columns(3)

      with pred_col1:
        st.metric(
          label="🏆 Win Probability",
          value=f"{prediction.home_win_prob:.1%}",
          delta=f"{prediction.home_team_name.split()[-1]}"
        )

      with pred_col2:
        spread_display = f"{prediction.predicted_spread:+.1f}" if prediction.predicted_spread != 0 else "PICK"
        favored = prediction.home_team_name.split()[-1] if prediction.predicted_spread < 0 else prediction.away_team_name.split()[-1]
        st.metric(
          label="📏 Spread",
          value=spread_display,
          delta=f"{favored} favored"
        )

      with pred_col3:
        st.metric(
          label="📈 Total Points",
          value=f"{prediction.predicted_total:.1f}",
          delta=f"O/U {prediction.predicted_total:.0f}"
        )

      # Predicted Score
      st.markdown("---")
      st.subheader("🎯 Predicted Final Score")

      score_col1, score_col2, score_col3 = st.columns([2, 1, 2])

      with score_col1:
        st.markdown(f"### {prediction.away_team_name}")
        st.markdown(f"# {prediction.away_predicted_score:.0f}")

      with score_col2:
        st.markdown("### ")
        st.markdown("# @")

      with score_col3:
        st.markdown(f"### {prediction.home_team_name}")
        st.markdown(f"# {prediction.home_predicted_score:.0f}")

      # Win probability bar
      st.markdown("---")
      st.subheader("Win Probability")

      prob_col1, prob_col2 = st.columns(2)
      with prob_col1:
        st.write(f"**{prediction.away_team_name}**")
        st.progress(prediction.away_win_prob)
        st.write(f"{prediction.away_win_prob*100:.1f}%")
      with prob_col2:
        st.write(f"**{prediction.home_team_name}**")
        st.progress(prediction.home_win_prob)
        st.write(f"{prediction.home_win_prob*100:.1f}%")

      # Detailed breakdown using prediction object attributes
      st.markdown("---")
      with st.expander("📋 Detailed Breakdown", expanded=False):

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
          st.markdown(f"**{prediction.away_team_name} (Away)**")
          st.write(f"- Elo Rating: {prediction.away_elo:.0f}")
          st.write(f"- Offensive Rating: {prediction.away_off_rating:.1f}")
          st.write(f"- Defensive Rating: {prediction.away_def_rating:.1f}")
          st.write(f"- Net Rating: {prediction.away_net_rating:.1f}")
          st.write(f"- Pace: {prediction.away_pace:.1f}")
          st.write(f"- Rest Days: {away_rest}")

        with detail_col2:
          st.markdown(f"**{prediction.home_team_name} (Home)**")
          st.write(f"- Elo Rating: {prediction.home_elo:.0f}")
          st.write(f"- Offensive Rating: {prediction.home_off_rating:.1f}")
          st.write(f"- Defensive Rating: {prediction.home_def_rating:.1f}")
          st.write(f"- Net Rating: {prediction.home_net_rating:.1f}")
          st.write(f"- Pace: {prediction.home_pace:.1f}")
          st.write(f"- Rest Days: {home_rest}")

        st.markdown("---")
        st.markdown("**Adjustments Applied:**")
        st.write(f"- Home Court Advantage: {prediction.home_court_adjustment:+.1f} pts")
        st.write(f"- Rest Adjustment: {prediction.rest_adjustment:+.1f} pts")
        st.write(f"- Recent Form Adjustment: {prediction.recent_form_adjustment:+.1f} pts")
        st.write(f"- Spread Confidence: {prediction.spread_confidence:.1%}")
        st.write(f"- Total Confidence: {prediction.total_confidence:.1%}")

  # =============
  # TEAM ANALYSIS
  # =============
  elif page == "Team Analysis":
    st.header("🔍 Team Analysis")

    # Team selector
    team_names = teams_df['full_name'].tolist()
    selected_team = st.selectbox("Select Team", team_names)

    team_row = teams_df[teams_df['full_name'] == selected_team].iloc[0]
    team_id = team_row['id']

    col1, col2, col3, col4 = st.columns(4)

    # Get team stats
    team_games = games[(games['HOME_TEAM_ID'] == team_id) | (games['AWAY_TEAM_ID'] == team_id)] if len(games) > 0 else pd.DataFrame()
    if len(team_games) > 0:
      wins = sum(
        1 for _, g in team_games.iterrows()
        if (g['HOME_TEAM_ID'] == team_id and g['HOME_PTS'] > g['AWAY_PTS']) or (g['AWAY_TEAM_ID'] == team_id and g['AWAY_PTS'] > g['HOME_PTS'])
      )
      losses = len(team_games) - wins

      # Calculate PPG
      pts_scored = []
      pts_allowed = []
      for _, g in team_games.iterrows():
        if g['HOME_TEAM_ID'] == team_id:
          pts_scored.append(g['HOME_PTS'])
          pts_allowed.append(g['AWAY_PTS'])
        else:
          pts_scored.append(g['AWAY_PTS'])
          pts_allowed.append(g['HOME_PTS'])

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
      recent = team_games.sort_values('GAME_DATE', ascending=False).head(10)
      st.dataframe(recent, hide_index=True)

  # Footer
  st.divider()
  st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
  st.caption(f"Data source: NBA API | Built with Streamlit")

if __name__ == "__main__":
  main()