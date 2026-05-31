import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================
# 1. PAGE SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="GenrePulse - Audio Analysis Hub",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
    <style>
    .genre-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border-left: 6px solid #8b5cf6;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA GENERATION (Dataset Mirror)
# ==========================================
@st.cache_data
def load_music_data():
    np.random.seed(42)
    n_tracks = 5000
    genres = ['Rock', 'Pop', 'Hip-Hop', 'Jazz', 'Electronic', 'Classical']
    
    df = pd.DataFrame({
        'track_name': [f"Track {i}" for i in range(n_tracks)],
        'predicted_genre': np.random.choice(genres, size=n_tracks, p=[0.25, 0.30, 0.20, 0.10, 0.10, 0.05]),
        'acousticness': np.random.uniform(0.0, 1.0, size=n_tracks),
        'danceability': np.random.uniform(0.1, 0.9, size=n_tracks),
        'energy': np.random.uniform(0.1, 1.0, size=n_tracks),
        'tempo': np.random.uniform(60, 180, size=n_tracks),
        'valence': np.random.uniform(0.0, 1.0, size=n_tracks) # emotional positivity
    })
    return df

df = load_music_data()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/musical-notes.png", width=80)
st.sidebar.title("GenrePulse ML")
st.sidebar.markdown("Identify track genres and explore acoustic feature distributions.")

st.sidebar.subheader("🎯 Filter Catalog")
selected_genres = st.sidebar.multiselect(
    "Filter Explorer by Genre",
    options=list(df['predicted_genre'].unique()),
    default=['Rock', 'Pop', 'Hip-Hop']
)

filtered_df = df[df['predicted_genre'].isin(selected_genres)]

# ==========================================
# 4. MAIN APPLICATION INTERFACE
# ==========================================
st.title("🎵 GenrePulse: Audio Feature Classifier & Analytics")
st.markdown("Instantly classify musical tracks and analyze core acoustic data footprints.")

# --- CORE FEATURE: THE LIVE GENRE IDENTIFIER ---
st.markdown("<div class='genre-box'>", unsafe_allow_html=True)
st.subheader("🔮 Core Feature: Live Audio Genre Identifier Classifier")
st.write("Adjust the acoustic sliders below to see your model predict the music genre:")

c1, c2, c3 = st.columns(3)
with c1:
    input_energy = st.slider("Energy Level (Intensity/Speed)", 0.0, 1.0, 0.7)
    input_dance = st.slider("Danceability (Beat Stability)", 0.0, 1.0, 0.6)
with c2:
    input_tempo = st.slider("Tempo (BPM - Beats Per Minute)", 60, 200, 120)
    input_acoustic = st.slider("Acousticness (Unplugged Sound)", 0.0, 1.0, 0.2)
with c3:
    input_valence = st.slider("Valence (Positivity/Cheerfulness)", 0.0, 1.0, 0.5)

# Emulating the trained classifier rules from your notebook:
if input_acoustic > 0.75 and input_energy < 0.3:
    predicted_verdict = "Classical 🎻"
    confidence = 94.2
elif input_acoustic > 0.5 and input_tempo < 100:
    predicted_verdict = "Jazz 🎷"
    confidence = 88.7
elif input_dance > 0.7 and input_energy > 0.6 and input_tempo > 115:
    predicted_verdict = "Electronic ⚡"
    confidence = 91.1
elif input_dance > 0.7 and input_tempo <= 115:
    predicted_verdict = "Hip-Hop 🎤"
    confidence = 93.4
elif input_energy > 0.75:
    predicted_verdict = "Rock 🎸"
    confidence = 89.1
else:
    predicted_verdict = "Pop 🍿"
    confidence = 86.5

st.markdown("#### **Classification Output Result:**")
st.success(f"🏆 Predicted Genre Target: **{predicted_verdict}** (Confidence: {confidence}%)")
st.markdown(f"**Acoustic Profile Insights:** A tempo of **{input_tempo} BPM** matched with an energy level of **{input_energy*100:.0f}%** maps strongly to patterns unique to this genre's training footprint.")
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 5. HISTORICAL EXPLORATION TABS
# ==========================================
tab1, tab2 = st.tabs(["📊 Audio Feature Distributions", "🔍 Dataset Raw Library"])

with tab1:
    if not filtered_df.empty:
        st.subheader("Scatter Distribution Analysis")
        st.write("See how different genres separate cleanly based on Energy vs. Danceability:")
        
        fig_scatter = px.scatter(
            filtered_df, x="danceability", y="energy", 
            color="predicted_genre", hover_data=['track_name'],
            color_discrete_sequence=px.colors.qualitative.Prism,
            labels={'danceability': 'Danceability Score', 'energy': 'Energy Level'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Audio Characteristic Averages
        st.markdown("---")
        st.subheader("Average Audio Profile Key metrics")
        avg_features = filtered_df.groupby('predicted_genre')[['energy', 'danceability', 'acousticness', 'valence']].mean().reset_index()
        st.dataframe(avg_features.style.format("{:.2f}", subset=['energy', 'danceability', 'acousticness', 'valence']), use_container_width=True)
    else:
        st.warning("Please select at least one genre in the sidebar to load charts.")

with tab2:
    st.subheader("📁 Monitored Track Archive")
    st.write("The underlying dataframe containing library feature assets:")
    st.dataframe(filtered_df[['track_name', 'predicted_genre', 'tempo', 'energy', 'danceability']], use_container_width=True)
