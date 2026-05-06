import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="F1 Winner Predictor", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #050505;
        color: #ffffff;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 2px solid #e10600;
    }

    [data-testid="stMetric"] {
        background-color: #111111;
        border: 1px solid #e10600;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #e10600;
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stDataFrame"] div {
        font-size: 16px !important;
    }

    .f1-section-divider {
        border-top: 3px solid #e10600;
        margin: 1.5rem 0;
    }
    
    table {
        width: 100%;
    }

    th {
        color: #ffffff !important;
        background-color: #e10600 !important;
    }

    td {
        color: #ffffff !important;
        background-color: #111111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# F1 Race Winner Predictor")
st.markdown("Post-qualifying Formula 1 race winner prediction dashboard.")
st.markdown('<div class="f1-section-divider"></div>', unsafe_allow_html=True)

try:
    df = pd.read_csv("predictions_2026.csv")
except FileNotFoundError:
    st.error("predictions_2026.csv not found. Run train_model.py first.")
    st.stop()

required_columns = [
    "race_id",
    "Abbreviation",
    "TeamName",
    "q_pos",
    "predicted_win_probability",
    "is_winner",
]

missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

race_list = sorted(df["race_id"].unique())

with st.sidebar:
    st.header("Controls")
    selected_race = st.selectbox("Select a 2026 race", race_list)
    with st.expander("Model information"):
        st.write(
            "The model is trained on 2020-2025 Formula 1 data and evaluated on 2026 races. "
            "Predictions are post-qualifying, meaning qualifying position is already known."
        )

race_df = df[df["race_id"] == selected_race].copy()
race_df = race_df.sort_values(by="predicted_win_probability", ascending=False)

predicted_winner = race_df.iloc[0]
actual_winner_df = race_df[race_df["is_winner"] == 1]
actual_winner = actual_winner_df.iloc[0] if not actual_winner_df.empty else None

st.subheader(selected_race)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Winner", predicted_winner["Abbreviation"])

with col2:
    if actual_winner is not None:
        st.metric("Actual Winner", actual_winner["Abbreviation"])
    else:
        st.metric("Actual Winner", "Not available")

with col3:
    confidence = predicted_winner["predicted_win_probability"] * 100
    st.metric("Model Confidence", f"{confidence:.2f}%")

if actual_winner is not None:
    if predicted_winner["Abbreviation"] == actual_winner["Abbreviation"]:
        st.success("Prediction result: Correct winner prediction")
    else:
        st.warning("Prediction result: Predicted winner did not match actual winner")

st.subheader("Top 5 Win Probabilities")

top_5 = race_df[
    ["Abbreviation", "TeamName", "q_pos", "predicted_win_probability"]
].head(5).copy()

top_5.columns = ["Driver", "Team", "Qualifying Position", "Win Probability (%)"]
top_5["Win Probability (%)"] = (top_5["Win Probability (%)"] * 100).round(2)

st.table(top_5)

st.subheader("Win Probability Chart")

chart_df = race_df[["Abbreviation", "predicted_win_probability"]].head(10).copy()
chart_df["win_probability_percent"] = chart_df["predicted_win_probability"] * 100

chart = (
    alt.Chart(chart_df)
    .mark_bar(color="#e10600")
    .encode(
        x=alt.X(
            "win_probability_percent:Q",
            title="Win Probability (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        y=alt.Y(
            "Abbreviation:N",
            title="Driver",
            sort="-x",
        ),
        tooltip=[
            alt.Tooltip("Abbreviation:N", title="Driver"),
            alt.Tooltip("win_probability_percent:Q", title="Win Probability (%)", format=".2f"),
        ],
    )
    .properties(height=420)
)

st.altair_chart(chart, use_container_width=True)

st.subheader("Full Race Prediction Table")

full_table = race_df[
    ["Abbreviation", "TeamName", "q_pos", "predicted_win_probability", "is_winner"]
].copy()

full_table.columns = [
    "Driver",
    "Team",
    "Qualifying Position",
    "Win Probability (%)",
    "Actual Winner",
]

full_table["Win Probability (%)"] = (full_table["Win Probability (%)"] * 100).round(2)
full_table["Actual Winner"] = full_table["Actual Winner"].map({1: "Yes", 0: "No"})

st.table(full_table)
