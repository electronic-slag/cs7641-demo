import streamlit as st
import pandas as pd
from recommender import KKBoxRecommender
import sys, numpy, lightfm, joblib
import datetime

# Inject stronger custom CSS to force override Streamlit dark theme, fix dropdown and table black issues
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #f7f9fa !important;
        color: #222222 !important;
    }
    /* Force white background for dropdown */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #fff !important;
        color: #222 !important;
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        font-size: 16px !important;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover, .stSelectbox div[data-baseweb="select"] > div:focus {
        background-color: #eaf4ff !important;
        border: 1.5px solid #4f8cff !important;
    }
    .stSelectbox label {
        color: #2a4d7a !important;
        font-weight: 600 !important;
    }
    /* Force white background for DataFrame */
    .stDataFrame, .stDataFrame .css-1v0mbdj, .stDataFrame .css-1v0mbdj th, .stDataFrame .css-1v0mbdj td {
        background-color: #fff !important;
        color: #222 !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        border: 1px solid #e0e0e0 !important;
    }
    .stDataFrame .css-1v0mbdj tr:hover {
        background-color: #eaf4ff !important;
    }
    /* Compatible with new Streamlit DataFrame rendering */
    .stDataFrame table, .stDataFrame th, .stDataFrame td {
        background-color: #fff !important;
        color: #222 !important;
        border-bottom: 1px solid #e0e0e0 !important;
    }
    .stDataFrame tr:hover {
        background-color: #eaf4ff !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f8cff 0%, #38e8ff 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 0.5em 2em !important;
        margin-top: 1em !important;
        box-shadow: 0 2px 8px rgba(79,140,255,0.08);
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #38e8ff 0%, #4f8cff 100%) !important;
    }
    ::-webkit-scrollbar {
        width: 8px;
        background: #e0e0e0;
        border-radius: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #b3d1ff;
        border-radius: 8px;
    }
    .stSpinner>div>div {
        color: #4f8cff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="KKBox Music Recommendation System", layout="wide")

# Top banner
st.markdown("""
<div style='width:100%;background:linear-gradient(90deg,#4f8cff 0%,#38e8ff 100%);padding:1.2em 0 1.2em 1.5em;border-radius:12px;margin-bottom:1.5em;'>
    <span style='font-size:2.2em;font-weight:700;color:#fff;letter-spacing:1px;'>🎵 KKBox Enhanced Music Recommendation System</span>
</div>
""", unsafe_allow_html=True)

# Page sections
left, right = st.columns([1,2])

with left:
    st.markdown("#### Select Test User")
    st.write("Please select a user, and the system will recommend 10 songs for them.")
    st.write(f"python: {sys.version.split()[0]}")
    st.write(f"numpy: {numpy.__version__}")
    st.write(f"lightfm: {lightfm.__version__}")
    st.write(f"joblib: {joblib.__version__}")

    @st.cache_resource
    def load_recommender():
        return KKBoxRecommender(model_path='models', data_path='.')

    recommender = load_recommender()
    test_users = recommender.get_test_users()
    user_id = st.selectbox("Select User ID", test_users, key="user_select")
    recommend_btn = st.button("✨ Generate Recommendations", use_container_width=True)

with right:
    st.markdown("#### Recommendation Results")
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    if recommend_btn:
        with st.spinner("Generating recommendations..."):
            results = recommender.recommend(user_id, n=10)
        st.session_state['results'] = results
    results = st.session_state['results']
    if results is not None:
        if not results:
            st.warning("This user is not in the training set, cannot generate recommendations.")
        else:
            # User info card
            st.markdown(f"""
            <div style='background:#eaf4ff;padding:1em 1.5em;border-radius:10px;margin-bottom:1em;display:flex;align-items:center;'>
                <span style='font-size:1.1em;font-weight:600;color:#2a4d7a;'>User ID:</span>
                <span style='font-size:1.1em;color:#4f8cff;margin-left:0.5em;'>{user_id}</span>
                <span style='margin-left:2em;font-size:1em;color:#888;'>Recommendation time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
            """, unsafe_allow_html=True)
            df = pd.DataFrame(results)
            df = df[['name', 'artist', 'genre', 'cluster_id', 'valence', 'energy', 'danceability', 'ensemble_score']]
            df.columns = ['Song Name', 'Artist', 'Genre', 'Cluster', 'Valence', 'Energy', 'Danceability', 'Ensemble Score']
            df.insert(0, 'No.', range(1, len(df)+1))
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Please select a user on the left and click the 'Generate Recommendations' button.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#aaa;font-size:1em;'>© 2025 KKBox Recommendation System | Powered by Streamlit</div>", unsafe_allow_html=True) 