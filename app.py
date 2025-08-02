import streamlit as st
import pandas as pd
import joblib
import os
import io

from data_loader import load_data
from predictor import load_models, predict_best_11
from player_to_country import player_to_country,teams
from data_loader import encode_features  # assuming you put encode_features there

# ---------- Constants ----------
DATA_FOLDER = "data/"
MODEL_FOLDER = "models/"
STADIUMS = ["Dubai (DISC)", "Lahore", "Karachi", "Rawalpindi"]
TEAM_OPTIONS = sorted(set(player_to_country.values()))

# ---------- Page Setup ----------
st.set_page_config(page_title="🏏 Dream11 Predictor", layout="centered")
st.title("🏆 Dream11 Best 11 Predictor for Champions Trophy 2025")

# ---------- Sidebar Inputs ----------
st.sidebar.header("📋 Match Configuration")
stadium = st.sidebar.selectbox("🏟️ Select Stadium", STADIUMS)
team1 = st.sidebar.selectbox("🏏 Select Team 1", TEAM_OPTIONS)
team2 = st.sidebar.selectbox("🏏 Select Team 2", [t for t in TEAM_OPTIONS if t != team1])

# ---------- Predict Button ----------
if st.sidebar.button("⚡ Predict Best 11"):
    try:
        st.info("🔁 Loading data and models...")

        # Load data and encode
        df = load_data(DATA_FOLDER)
        df, encoder = encode_features(df)

        # Load or train models
        if not os.path.exists(MODEL_FOLDER) or len(os.listdir(MODEL_FOLDER)) == 0:
            st.warning("No models found. Please run model training script first.")
            st.stop()
        models = load_models(MODEL_FOLDER)

        # Prepare squad from team1 + team2
        match_squad = teams[team1] + teams[team2]

        # Predict
        best_11 = predict_best_11(stadium, team1, team2, match_squad, df, encoder, models)

        if best_11:
            st.success("✅ Optimized Best 11:")
            df_output = pd.DataFrame(best_11, columns=["Player Name", "Role", "Predicted Fantasy Points"])
            st.dataframe(df_output)

            # Write ONLY best 11 to Excel
            buffer = io.BytesIO()
            df_output.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)

            st.download_button(
                label="💾 Download Best 11 (as Excel)",
                data=buffer,
                file_name="best_11.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("⚠️ No valid predictions made.")
    except Exception as e:
        st.error(f"❌ Error: {e}")     