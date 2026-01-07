"""
Streamlit application for Agentic AI Movie & TV Recommendation System.
Uses Hugging Face Inference API instead of local model.
"""

import streamlit as st
import pandas as pd
import os
from orchestrator_api import RecommendationOrchestrator
from agents.preference_extraction_api import PreferenceExtractionAgent
from agents.movie_search import MovieSearchAgent
from data_loader import load_movies_and_tv_shows

# Page configuration
st.set_page_config(
    page_title="Agentic Movie & TV Recommender",
    page_icon="🎬",
    layout="wide"
)

# Title
st.title("🎬 Agentic AI Movie & TV Recommender")

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
    st.session_state.movies_loaded = False

# Load movies and TV shows data
@st.cache_data
def load_content():
    """Load movies and TV shows data."""
    try:
        df = load_movies_and_tv_shows()
        if len(df) == 0:
            st.error("❌ No content data found.")
            st.info("""
            Please ensure you have:
            1. **movies.csv** in the data/ directory (from MovieLens dataset)
            2. Optionally: **tv_shows.csv** in the data/ directory
            
            You can download MovieLens from: https://grouplens.org/datasets/movielens/
            """)
            return None
        return df
    except Exception as e:
        st.error(f"❌ Error loading content data: {str(e)}")
        return None

# Check for API key (supports both .env file and Streamlit secrets)
try:
    # Try Streamlit secrets first (for deployed apps)
    api_key = st.secrets.get("HF_API_KEY") or st.secrets.get("HUGGINGFACE_API_KEY")
except:
    api_key = None

# Fallback to environment variables (for local development)
if not api_key:
    api_key = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")

if not api_key:
    st.error("⚠️ **Hugging Face API key not found.**")
    st.info("""
    Please set your Hugging Face API key:
    
    **For Local Development:**
    1. Get your API key from: https://huggingface.co/settings/tokens
    2. Create a `.env` file in the project root
    3. Add: `HF_API_KEY=your_api_key_here`
    
    **For Streamlit Cloud:**
    1. Go to your app settings → Secrets
    2. Add: `HF_API_KEY = "your_api_key_here"`
    
    **Free tier:** 1000 requests/month | **Pro tier:** Unlimited requests
    """)
    st.stop()

# Load movies and TV shows
content_df = load_content()

if content_df is not None and not st.session_state.movies_loaded:
    try:
        # Initialize agents
        preference_agent = PreferenceExtractionAgent(api_key=api_key)
        search_agent = MovieSearchAgent(content_df)
        
        # Initialize orchestrator
        st.session_state.orchestrator = RecommendationOrchestrator(
            preference_agent=preference_agent,
            search_agent=search_agent,
            api_key=api_key
        )
        st.session_state.movies_loaded = True
        st.session_state.content_df = content_df  # Store for UI
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        st.exception(e)
        st.stop()

# Main interface
if st.session_state.orchestrator is not None:
    # Input section with better styling
    st.markdown("---")
    st.markdown("### 📺 Your Favorite Movies & TV Series")
    st.markdown("""
    Select **3-5 titles** you love, and we'll analyze your preferences to recommend movies and TV series 
    that match your taste. You can type to search for titles.
    """)
    
    available_titles = st.session_state.content_df['title'].tolist()
    favorite_titles = st.multiselect(
        "Choose your favorites:",
        options=available_titles,
        max_selections=5,
        help="Select 3-5 movies or TV series you love. The AI will analyze why you like them.",
        label_visibility="collapsed"
    )
    
    # Show selected count
    if favorite_titles:
        st.caption(f"✓ Selected {len(favorite_titles)} title(s)")
    
    # Generate recommendations button - centered and styled
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_btn = st.button("🎯 Get AI-Powered Recommendation", type="primary", use_container_width=True)
    
    if generate_btn:
        if len(favorite_titles) < 3:
            st.warning("⚠️ Please select at least 3 favorite titles for better recommendations.")
        elif len(favorite_titles) > 5:
            st.warning(f"⚠️ Please select no more than 5 titles (selected: {len(favorite_titles)}).")
        else:
            with st.spinner("🤔 AI agents are analyzing your preferences and reasoning about recommendations..."):
                try:
                    recommendation = st.session_state.orchestrator.recommend(
                        favorite_titles=favorite_titles,
                        exclude_titles=favorite_titles
                    )
                    
                    # Display results with improved styling
                    st.markdown("---")
                    st.markdown("## ✨ Your Personalized Recommendations")
                    st.markdown("")
                    
                    # Primary recommendation - highlighted
                    primary = recommendation['primary_recommendation']
                    with st.container():
                        st.markdown("### 🎯 Primary Recommendation")
                        # Use columns for better layout
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            # Show type icon (movie or TV)
                            type_icon = "📺" if primary.get('type') == 'tv_show' else "🎬"
                            st.markdown(f"#### {type_icon} {primary['title']}")
                            if 'genres' in primary:
                                st.markdown(f"<span style='color: #666; font-size: 14px;'>📂 {primary['genres']}</span>", unsafe_allow_html=True)
                        st.markdown("")
                        # Why this matches - in a styled box
                        st.info(f"💡 **Why this matches:** {primary['reason']}")
                        st.markdown("")
                    
                    # Alternatives - cleaner layout
                    if 'alternatives' in recommendation and recommendation['alternatives']:
                        st.markdown("### 🔀 Alternative Recommendations")
                        st.markdown("")
                        for idx, alt in enumerate(recommendation['alternatives'], 1):
                            with st.container():
                                # Show type icon (movie or TV)
                                type_icon = "📺" if alt.get('type') == 'tv_show' else "🎬"
                                st.markdown(f"**{idx}. {type_icon} {alt['title']}**")
                                if 'genres' in alt:
                                    st.caption(f"📂 {alt['genres']}")
                                st.markdown(f"💭 {alt.get('reason', 'Strong alternative match.')}")
                                if idx < len(recommendation['alternatives']):
                                    st.markdown("")
                        st.markdown("")
                    
                    # Overall explanation - in a callout
                    if 'explanation' in recommendation and recommendation['explanation']:
                        st.markdown("### 📝 Why These Recommendations?")
                        st.markdown(f"<div style='background-color: #1e1e1e; color: #ffffff; padding: 15px; border-radius: 5px; margin: 10px 0; border: 1px solid #333;'>{recommendation['explanation']}</div>", unsafe_allow_html=True)
                        st.markdown("")
                    
                    # Preferences summary (collapsible)
                    if 'preferences_summary' in recommendation and recommendation['preferences_summary']:
                        with st.expander("🔍 What the AI learned about your preferences", expanded=False):
                            st.write(recommendation['preferences_summary'])
                    
                except Exception as e:
                    st.error(f"❌ Error generating recommendation: {str(e)}")
                    st.exception(e)
    

else:
    st.info("👆 Please ensure the movie data file (movies.csv) is available in the data/ directory and Hugging Face API key is configured.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px 0; color: #888; font-size: 12px;'>
    🤖 <strong>Agentic AI System</strong> | Powered by Hugging Face Inference API | Session-based | Platform-independent
</div>
""", unsafe_allow_html=True)

