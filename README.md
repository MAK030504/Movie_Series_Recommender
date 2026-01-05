# 🎬 Agentic AI Movie & TV Recommendation System

An intelligent recommendation system that uses **multi-agent AI reasoning** to understand *why* you like certain movies and TV series, then provides thoughtful, personalized recommendations.

Unlike traditional recommenders that focus on *what* you watched, this system analyzes themes, tone, pacing, and narrative style to recommend content that matches your deeper preferences.

## 🌟 Features

- **Agentic AI Architecture**: Three specialized AI agents work together to understand your taste
- **LLM-Powered Reasoning**: Uses Hugging Face Inference API (Mistral-7B-Instruct) to extract preferences and reason about recommendations
- **Focused Output**: Returns ONE primary recommendation + 2-3 alternatives with explanations
- **Session-Based**: No login or account required
- **Platform-Independent**: Works with any streaming service
- **Spoiler-Free**: All explanations are designed to avoid spoilers
- **Cloud-Based**: No local model required - uses Hugging Face Inference API

## 🏗️ Architecture

The system uses three core agents:

1. **Preference Extraction Agent**: Analyzes your favorite titles to extract structured preferences (themes, tone, pacing, narrative style)
2. **Movie/TV Search Agent**: Finds candidate titles from the MovieLens dataset based on extracted preferences using content-based filtering
3. **Recommendation Orchestrator**: Coordinates agents, reasons over candidates, and selects the best recommendation with explanations

```
User Input (3-5 favorites)
   ↓
Preference Extraction Agent (LLM reasoning via Hugging Face API)
   ↓
Movie/TV Search Agent (candidate finding)
   ↓
Recommendation Orchestrator (final selection + explanation via Hugging Face API)
   ↓
ONE Primary Recommendation + 2-3 Alternatives
```

## 📋 Requirements

- Python 3.8+
- Hugging Face API key (free tier available)
- MovieLens dataset (movies.csv)
- TV shows dataset (tv_shows.csv) - included with 150 popular shows

## 🚀 Installation

1. **Clone or download this repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up Hugging Face API key:**
   - Get your free API key from https://huggingface.co/settings/tokens
   - Create a `.env` file in the project root:
   ```bash
   HF_API_KEY=your_api_key_here
   ```

4. **Download the MovieLens dataset:**
   - Visit https://grouplens.org/datasets/movielens/
   - Download the MovieLens dataset (1M, 10M, 25M, or 32M - all work)
   - Extract the `movies.csv` file to the `data/` directory
   - Ensure the CSV has columns: `movieId`, `title`, `genres`

5. **TV Shows dataset:**
   - The project includes `data/tv_shows.csv` with 150 popular TV series
   - No additional setup required

## 💻 Usage

1. **Run the Streamlit application:**
```bash
streamlit run app_api.py
```

2. **Open your browser** to the URL shown (usually http://localhost:8501)

3. **Select 3-5 favorite movies/TV series** from the dropdown

4. **Click "Get AI-Powered Recommendation"** to see your personalized recommendation

## 📁 Project Structure

```
Movie_Series_Recommender/
│
├── agents/
│   ├── __init__.py
│   ├── preference_extraction_api.py    # Preference Extraction Agent
│   └── movie_search.py                  # Movie/TV Search Agent
│
├── data/
│   ├── movies.csv                       # MovieLens dataset (download separately)
│   └── tv_shows.csv                      # TV shows dataset (included)
│
├── orchestrator_api.py                   # Recommendation Orchestrator
├── llm_helper_api_v2.py                 # Hugging Face API helper
├── data_loader.py                        # Data loading utilities
├── app_api.py                           # Streamlit UI
├── requirements.txt                     # Python dependencies
├── .env                                 # API keys (create manually)
├── README.md                            # This file
└── instruction.md                       # Project context and guidelines
```

## 🔧 Configuration

### Model Selection

By default, the system uses `mistralai/Mistral-7B-Instruct-v0.2` via Hugging Face Inference API. You can change this in:

- `agents/preference_extraction_api.py` - `model_name` parameter in `PreferenceExtractionAgent`
- `orchestrator_api.py` - `model_name` parameter in `RecommendationOrchestrator`

### Candidate Pool Size

Adjust the number of candidates searched in `orchestrator_api.py`:
```python
candidates = self.search_agent.search_candidates(preferences, n_candidates=50)
```

## 🎯 How It Works

1. **User provides 3-5 favorite titles** (movies or TV series)

2. **Preference Extraction Agent** uses LLM reasoning (via Hugging Face API) to analyze:
   - Themes (e.g., existentialism, survival, family)
   - Tone (dark, hopeful, humorous, intense)
   - Pacing (slow-burn, fast-paced)
   - Narrative style (character-driven vs plot-driven)
   - Genre blend

3. **Movie/TV Search Agent** finds 20-50 candidate titles from the dataset using content-based filtering (TF-IDF + cosine similarity)

4. **Recommendation Orchestrator** uses LLM reasoning (via Hugging Face API) to:
   - Evaluate candidates against extracted preferences
   - Select ONE primary recommendation
   - Select 2-3 alternatives
   - Generate spoiler-free explanations

## 🔐 Privacy & Data

- **Session-based**: No data persists beyond your session
- **No user accounts**: No login or authentication required
- **No tracking**: Personal data is not stored or tracked
- **API-based**: Uses Hugging Face Inference API - requests are sent to Hugging Face servers
- **No local model**: No need to download large model files

## 🛠️ Technology Stack

- **Python 3.8+**
- **Hugging Face Inference API** (Mistral-7B-Instruct-v0.2)
- **Streamlit** (UI)
- **pandas, numpy** (data processing)
- **scikit-learn** (content-based filtering for candidate search)
- **huggingface_hub** (Hugging Face API client)
- **python-dotenv** (environment variable management)

## 📝 Notes

- This is a **personal AI/ML project**, not production-scale
- Focus is on **reasoning and taste understanding**, not scale
- **API-based**: Requires internet connection and Hugging Face API key
- **Free tier available**: Hugging Face offers free API access with rate limits
- **No local model download**: Much faster setup compared to local LLM versions
- MovieLens dataset provides the candidate pool for movies
- TV shows dataset (150 shows) is included in the project

## 🚧 Future Enhancements

Potential extensions (not currently implemented):
- Support for other LLM providers (OpenAI, Anthropic, etc.)
- Long-term memory for preference patterns
- TMDB API integration for richer metadata
- Multi-language support
- Enhanced TV series dataset
- User preference history (optional)

## 📄 License

This project is provided as-is for educational and personal use.

## 🙏 Acknowledgments

- MovieLens dataset: https://grouplens.org/datasets/movielens/
- Mistral AI for the Mistral-7B-Instruct model
- Hugging Face for the Inference API and infrastructure
- Streamlit for the UI framework
