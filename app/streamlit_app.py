"""Streamlit web application for World Cup predictions."""

import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.helpers import setup_logger
from src.utils.constants import (
    PREDICTIONS_2022_CSV, PREDICTIONS_2026_CSV, WC_2022_TEAMS
)
from src.models.tournament_simulator import TournamentSimulator

# Configure streamlit
st.set_page_config(
    page_title="FIFA World Cup Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = setup_logger(__name__)


@st.cache_resource
def load_simulator():
    """Load model simulator."""
    simulator = TournamentSimulator(model_type='random_forest')
    if simulator.load_model() and simulator.load_matches_data():
        return simulator
    return None


@st.cache_data
def load_predictions_2022():
    """Load 2022 predictions."""
    try:
        return pd.read_csv(PREDICTIONS_2022_CSV)
    except:
        return None


@st.cache_data
def load_predictions_2026():
    """Load 2026 predictions."""
    try:
        return pd.read_csv(PREDICTIONS_2026_CSV)
    except:
        return None


def main():
    """Main app function."""
    
    # Sidebar
    with st.sidebar:
        st.title("⚽ FIFA World Cup Prediction")
        page = st.radio(
            "Select Page:",
            ["Home", "2022 Validation", "2026 Prediction", "Match Simulator", "About"]
        )
    
    # Home page
    if page == "Home":
        st.title("⚽ FIFA World Cup Prediction with Machine Learning")
        
        st.markdown("""
        ### Welcome!
        
        This application uses historical international football data (1930-2024) and machine learning
        to predict the winner of FIFA World Cups.
        
        #### Project Features:
        - **Historical Data Analysis**: 1930-2024 World Cup and continental tournament data
        - **Machine Learning Models**: Logistic Regression + Random Forest
        - **Monte Carlo Simulation**: 10,000+ tournament simulations for probability calculations
        - **Model Validation**: Backtesting on 2022 World Cup
        - **Interactive Predictions**: 2026 World Cup winner probabilities
        
        #### Technology Stack:
        - **Data Processing**: pandas, numpy
        - **ML Models**: scikit-learn
        - **Web App**: Streamlit
        - **Visualization**: Plotly
        
        #### How It Works:
        1. **Feature Engineering**: Extract historical team performance metrics
        2. **Model Training**: Train on historical matches
        3. **Validation**: Test predictions on 2022 World Cup
        4. **Prediction**: Simulate 2026 tournament 10,000 times
        5. **Probabilities**: Calculate win probability for each team
        """)
        
        st.info("📊 Check the sidebar to navigate to different sections!")
    
    # 2022 Validation
    elif page == "2022 Validation":
        st.title("2022 World Cup - Model Validation")
        
        df_2022 = load_predictions_2022()
        
        if df_2022 is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Predicted Teams")
                top_10 = df_2022.head(10)
                fig = px.bar(
                    top_10,
                    x='Team',
                    y='Win_Probability',
                    title="Predicted Win Probabilities (Top 10)",
                    labels={'Win_Probability': 'Probability (%)', 'Team': 'Team'},
                )
                fig.update_yaxes(tickformat=".2%")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Key Information")
                st.metric("Total Teams", len(df_2022))
                st.metric("Predicted Winner", df_2022.iloc[0]['Team'])
                st.metric("Winner Probability", f"{df_2022.iloc[0]['Win_Probability']:.2%}")
            
            st.subheader("Full Predictions")
            st.dataframe(
                df_2022.assign(**{'Win_Probability': df_2022['Win_Probability'].apply(lambda x: f"{x:.2%}")}),
                use_container_width=True
            )
            
            st.success("✓ Model validation completed!")
            st.info("2022 actual winner: Argentina")
        else:
            st.warning("2022 predictions not available. Please run the prediction script first.")
    
    # 2026 Prediction
    elif page == "2026 Prediction":
        st.title("2026 World Cup - Predictions")
        
        df_2026 = load_predictions_2026()
        
        if df_2026 is not None:
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["Top Predictions", "All Teams", "Visualizations"])
            
            with tab1:
                st.subheader("🏆 Top 10 Favorites")
                top_10 = df_2026.head(10)
                
                for idx, row in top_10.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(row['Win_Probability'], f"{row['Rank']}. {row['Team']}")
                    with col2:
                        st.metric("", f"{row['Win_Probability']:.2%}")
            
            with tab2:
                st.subheader("All Teams - Full Rankings")
                st.dataframe(
                    df_2026.assign(**{'Win_Probability': df_2026['Win_Probability'].apply(lambda x: f"{x:.2%}")}),
                    use_container_width=True,
                    hide_index=True
                )
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig_bar = px.bar(
                        df_2026.head(15),
                        x='Team',
                        y='Win_Probability',
                        title="Top 15 Teams - Win Probability",
                        labels={'Win_Probability': 'Probability'},
                    )
                    fig_bar.update_yaxes(tickformat=".2%")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Pie chart
                    top_5 = df_2026.head(5).copy()
                    others_prob = df_2026.iloc[5:]['Win_Probability'].sum()
                    pie_data = pd.concat([
                        top_5,
                        pd.DataFrame({'Team': ['Others'], 'Win_Probability': [others_prob], 'Rank': [0]})
                    ])
                    
                    fig_pie = px.pie(
                        pie_data,
                        names='Team',
                        values='Win_Probability',
                        title="Win Probability Distribution",
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Key statistics
            st.subheader("📈 Key Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Teams", len(df_2026))
            
            with col2:
                st.metric("Predicted Winner", df_2026.iloc[0]['Team'])
            
            with col3:
                top_5_prob = df_2026.head(5)['Win_Probability'].sum()
                st.metric("Top 5 Combined", f"{top_5_prob:.1%}")
            
            with col4:
                top_10_prob = df_2026.head(10)['Win_Probability'].sum()
                st.metric("Top 10 Combined", f"{top_10_prob:.1%}")
        else:
            st.warning("2026 predictions not available. Please run the prediction script first.")
    
    # Match Simulator
    elif page == "Match Simulator":
        st.title("⚽ Match Simulator")
        
        simulator = load_simulator()
        
        if simulator:
            st.subheader("Simulate a Match")
            
            col1, col2 = st.columns(2)
            
            with col1:
                home_team = st.text_input("Home Team", placeholder="e.g., Argentina")
            
            with col2:
                away_team = st.text_input("Away Team", placeholder="e.g., France")
            
            if st.button("Simulate Match"):
                if home_team and away_team:
                    # Get match probability
                    win_prob = simulator.calculate_match_probability(home_team, away_team)
                    
                    # Simulate match
                    goals_home, goals_away = simulator.simulate_match(home_team, away_team)
                    
                    st.subheader("Match Result")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(home_team, goals_home)
                    
                    with col2:
                        st.metric("vs", "")
                    
                    with col3:
                        st.metric(away_team, goals_away)
                    
                    st.metric(f"{home_team} Win Probability", f"{win_prob:.2%}")
                    
                    # Determine result
                    if goals_home > goals_away:
                        st.success(f"🎉 {home_team} Wins!")
                    elif goals_home < goals_away:
                        st.success(f"🎉 {away_team} Wins!")
                    else:
                        st.info("⚖️ Draw!")
                else:
                    st.error("Please enter both team names")
        else:
            st.error("Failed to load simulator. Please ensure models are trained.")
    
    # About
    elif page == "About":
        st.title("About This Project")
        
        st.markdown("""
        ## FIFA World Cup 2026 Prediction Project
        
        ### Author
        **Bruno Delgado Herrero**
        - Email: brunodelgado@msmk.university
        - GitHub: [@brrrr1](https://github.com/brrrr1)
        - University: Canterbury Christ Church University
        
        ### Project Description
        This is an individual study project (IS 40) that develops a robust, data-driven predictive
        model to estimate the probability of each national team winning the 2026 FIFA World Cup.
        
        ### Methodology
        
        #### Data Collection
        - World Cup results (1930-2022)
        - Continental tournament data (Euro, Copa América, AFCON, Asian Cup)
        - Team rankings and performance metrics
        - Player-level statistics (where available)
        
        #### Feature Engineering
        - Win/loss ratios and historical performance
        - Goal difference and scoring patterns
        - Recent form (last 5-10 matches)
        - FIFA rankings and Elo ratings
        - Head-to-head records
        - Home advantage adjustments
        
        #### Machine Learning Models
        
        **Logistic Regression:**
        - Baseline model for match outcome prediction
        - Interpretable coefficients
        - Linear decision boundaries
        
        **Random Forest:**
        - Ensemble method capturing non-linear patterns
        - Feature importance analysis
        - 200 estimators, max depth 20
        - Hyperparameter tuned via grid search
        
        #### Validation Strategy
        - Train on data up to 2022
        - Predict 2022 World Cup outcomes
        - Compare predictions with actual results
        - Calculate accuracy, precision, recall, log loss
        
        #### Tournament Simulation
        - Monte Carlo approach: 10,000+ simulations
        - Each simulation runs full tournament bracket
        - Calculates win probability for each team
        - Generates confidence intervals
        
        ### Key Features of 2026 Tournament
        - **48 teams** (expanded from 32)
        - **New group stage format** (12 groups of 4)
        - **Multiple host nations** (USA, Canada, Mexico)
        - **New teams participating** (some without historical data)
        
        ### Technologies Used
        - **Python 3.10+**
        - **Data Processing**: pandas, numpy
        - **Machine Learning**: scikit-learn, Random Forest
        - **Web Framework**: Streamlit
        - **Visualization**: Plotly, Seaborn, Matplotlib
        - **Version Control**: Git/GitHub
        
        ### Limitations & Future Work
        
        **Current Limitations:**
        - Limited data for recently promoted teams
        - Player injuries/transfers not considered in real-time
        - Political events and boycotts not modeled
        - COVID-19 effects on tournament scheduling
        
        **Future Improvements:**
        - Player-level rating system (player ratings aggregation)
        - Transfer market data integration
        - Injury prediction models
        - Real-time form updates
        - Alternative algorithms (XGBoost, Neural Networks)
        - Bayesian uncertainty quantification
        
        ### References
        - FiveThirtyEight Soccer Power Index (SPI)
        - Kaggle International Football Match Data
        - Academic papers on sports prediction (see dissertation)
        - FIFA Official Rankings and Statistics
        
        ### License
        This project is for educational purposes. Data sourced from public APIs and Kaggle.
        
        ---
        
        **Last Updated**: December 2025
        """)


if __name__ == "__main__":
    main()
