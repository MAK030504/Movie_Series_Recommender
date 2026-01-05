"""
Movie/TV Search Agent

Finds a broad set of candidate movies and TV series aligned with extracted preferences.
Focuses on coverage, not precision - generates 20-50 candidates.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieSearchAgent:
    """
    Agent that searches for candidate movies/TV series based on extracted preferences.
    Uses content-based filtering to find broad matches.
    """
    
    def __init__(self, content_df: pd.DataFrame):
        """
        Initialize the Movie Search Agent.
        
        Parameters:
        -----------
        content_df : pd.DataFrame
            DataFrame with columns: 'id', 'title', 'genres', 'type' (optional)
            Can contain both movies and TV shows
        """
        self.content_df = content_df.copy()
        self.vectorizer = None
        self.feature_matrix = None
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare TF-IDF feature matrix from content genres."""
        self.content_df['content'] = self.content_df['genres'].fillna('')
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.feature_matrix = self.vectorizer.fit_transform(self.content_df['content'])
    
    def search_candidates(self, preferences: Dict[str, Any], n_candidates: int = 50) -> List[Dict[str, Any]]:
        """
        Find candidate movies/TV series based on extracted preferences.
        
        Parameters:
        -----------
        preferences : Dict[str, Any]
            Structured preferences from Preference Extraction Agent
        n_candidates : int
            Number of candidates to return (default: 50)
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of candidate titles with their information
            Each dict contains: 'movieId', 'title', 'genres', 'match_score'
        """
        # Build search query from preferences
        search_query = self._build_search_query(preferences)
        
        # Vectorize the search query
        query_vector = self.vectorizer.transform([search_query])
        
        # Compute similarity scores
        similarities = cosine_similarity(query_vector, self.feature_matrix)[0]
        
        # Create results DataFrame
        results_df = self.content_df.copy()
        results_df['match_score'] = similarities
        
        # Filter by format preference if specified
        if 'format_preference' in preferences:
            format_pref = preferences['format_preference'].lower()
            if format_pref == 'movie' and 'type' in results_df.columns:
                results_df = results_df[results_df['type'] == 'movie']
            elif format_pref == 'tv' and 'type' in results_df.columns:
                results_df = results_df[results_df['type'] == 'tv_show']
            # 'both' means no filtering
        
        # Sort by similarity and get top N
        results_df = results_df.sort_values('match_score', ascending=False)
        results_df = results_df.head(n_candidates)
        
        # Convert to list of dictionaries
        candidates = []
        for _, row in results_df.iterrows():
            candidate = {
                'id': row['id'],
                'title': row['title'],
                'genres': row['genres'],
                'match_score': float(row['match_score'])
            }
            # Add type if available
            if 'type' in row:
                candidate['type'] = row['type']
            candidates.append(candidate)
        
        return candidates
    
    def _build_search_query(self, preferences: Dict[str, Any]) -> str:
        """
        Build a search query string from preferences.
        
        Parameters:
        -----------
        preferences : Dict[str, Any]
            Structured preferences
            
        Returns:
        --------
        str
            Query string for search
        """
        query_parts = []
        
        # Add themes
        if 'themes' in preferences and preferences['themes']:
            query_parts.extend(preferences['themes'])
        
        # Add genre blend
        if 'genre_blend' in preferences and preferences['genre_blend']:
            query_parts.append(preferences['genre_blend'])
        
        # Add genres from the original titles (if available)
        # This helps maintain genre coverage
        
        return " ".join(query_parts)

