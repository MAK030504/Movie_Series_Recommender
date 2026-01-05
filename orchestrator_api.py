"""
Recommendation Orchestrator - Hugging Face Inference API Version

Coordinates agents, reasons over candidates, and produces the final recommendation.
Uses Hugging Face Inference API instead of local model.
"""

from typing import List, Dict, Any, Optional
import json
from agents.preference_extraction_api import PreferenceExtractionAgent
from agents.movie_search import MovieSearchAgent
from llm_helper_api_v2 import HuggingFaceInferenceLLM


class RecommendationOrchestrator:
    """
    Orchestrator that coordinates the recommendation process.
    Selects ONE primary recommendation and 2-3 alternatives with explanations.
    Uses Hugging Face Inference API for LLM reasoning.
    """
    
    def __init__(self, preference_agent: PreferenceExtractionAgent, 
                 search_agent: MovieSearchAgent,
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 api_key: str = None):
        """
        Initialize the Recommendation Orchestrator.
        
        Parameters:
        -----------
        preference_agent : PreferenceExtractionAgent
            Agent for extracting user preferences
        search_agent : MovieSearchAgent
            Agent for finding candidate titles
        model_name : str
            Hugging Face model identifier (default: mistralai/Mistral-7B-Instruct-v0.3)
        api_key : str, optional
            Hugging Face API key
        """
        self.preference_agent = preference_agent
        self.search_agent = search_agent
        self.llm = HuggingFaceInferenceLLM(model_name=model_name, api_key=api_key)
        
        # Session state
        self.session_state = {
            'favorite_titles': [],
            'extracted_preferences': None,
            'current_recommendation': None
        }
    
    def recommend(self, favorite_titles: List[str], exclude_titles: List[str] = None) -> Dict[str, Any]:
        """
        Generate recommendations based on user's favorite titles.
        
        Parameters:
        -----------
        favorite_titles : List[str]
            List of 3-5 favorite movie/TV titles
        exclude_titles : List[str], optional
            Titles to exclude from recommendations (default: favorite_titles)
            
        Returns:
        --------
        Dict[str, Any]
            Recommendation result with:
            - primary_recommendation: Dict with 'title', 'genres', 'reason'
            - alternatives: List[Dict] with 2-3 alternatives
            - explanation: str (spoil-free explanation)
            - preferences_summary: str
        """
        exclude_titles = exclude_titles or favorite_titles
        
        # Update session state
        self.session_state['favorite_titles'] = favorite_titles
        
        # Step 1: Extract preferences
        preferences = self.preference_agent.extract_preferences(favorite_titles)
        self.session_state['extracted_preferences'] = preferences
        
        # Step 2: Find candidates
        candidates = self.search_agent.search_candidates(preferences, n_candidates=50)
        
        # Filter out excluded titles
        candidates = [c for c in candidates if c['title'] not in exclude_titles]
        
        # Step 3: Use LLM to reason over candidates and select best matches
        recommendation = self._reason_and_select(candidates, preferences, favorite_titles)
        
        # Update session state
        self.session_state['current_recommendation'] = recommendation
        
        return recommendation
    
    def _reason_and_select(self, candidates: List[Dict[str, Any]], 
                          preferences: Dict[str, Any],
                          favorite_titles: List[str]) -> Dict[str, Any]:
        """
        Use LLM reasoning to select the best recommendation from candidates.
        
        Parameters:
        -----------
        candidates : List[Dict[str, Any]]
            List of candidate titles
        preferences : Dict[str, Any]
            Extracted preferences
        favorite_titles : List[str]
            User's favorite titles (for context)
            
        Returns:
        --------
        Dict[str, Any]
            Final recommendation structure
        """
        # Prepare candidate list for LLM (top 20 for reasoning efficiency)
        top_candidates = candidates[:20]
        
        candidates_str = "\n".join([f"- {c['title']} ({c['genres']})" for c in top_candidates])
        favorites_str = "\n".join([f"- {title}" for title in favorite_titles])
        
        system_prompt = "You are an expert film critic who provides thoughtful, spoil-free recommendations."
        
        user_prompt = f"""Based on a user's favorite titles and their extracted preferences, select the BEST recommendation from the candidate list.

User's favorite titles:
{favorites_str}

Extracted preferences:
- Themes: {', '.join(preferences.get('themes', []))}
- Tone: {preferences.get('tone', 'N/A')}
- Pacing: {preferences.get('pacing', 'N/A')}
- Narrative style: {preferences.get('narrative_style', 'N/A')}
- Genre blend: {preferences.get('genre_blend', 'N/A')}
- Reasoning: {preferences.get('reasoning', 'N/A')}

Candidate titles (top matches):
{candidates_str}

Select:
1. ONE primary recommendation (the best match)
2. 2-3 alternative recommendations (different but still strong matches)

Return a JSON object:
{{
    "primary_recommendation": {{
        "title": "Movie/Series Title",
        "reason": "Brief explanation of why this is the best match (spoil-free)"
    }},
    "alternatives": [
        {{"title": "Title 1", "reason": "Brief reason"}},
        {{"title": "Title 2", "reason": "Brief reason"}}
    ],
    "explanation": "Overall explanation connecting the recommendation to the user's preferences (2-3 sentences, spoil-free)"
}}

Be thoughtful - select based on the deeper preferences, not just genre similarity. Ensure the explanation is compelling and spoil-free. Return ONLY the JSON object, no additional text."""

        try:
            response = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.8,
                max_tokens=2000,
                format_json=True
            )
            
            # Parse JSON response
            result = json.loads(response)
            
            # Enrich with genre information and type from candidates
            result = self._enrich_with_genres(result, candidates)
            
            # Add preferences summary
            result['preferences_summary'] = preferences.get('reasoning', '')
            
            return result
            
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            try:
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                result = json.loads(cleaned)
                result = self._enrich_with_genres(result, candidates)
                result['preferences_summary'] = preferences.get('reasoning', '')
                return result
            except:
                # Fallback to simple selection if LLM fails
                return self._fallback_selection(candidates, preferences)
        except Exception as e:
            # Fallback to simple selection if LLM fails
            return self._fallback_selection(candidates, preferences)
    
    def _enrich_with_genres(self, recommendation: Dict[str, Any], 
                           candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrich recommendation with genre information and type from candidates."""
        candidate_dict = {c['title']: c for c in candidates}
        
        # Enrich primary recommendation
        if recommendation['primary_recommendation']['title'] in candidate_dict:
            rec = recommendation['primary_recommendation']
            candidate = candidate_dict[rec['title']]
            rec['genres'] = candidate['genres']
            if 'type' in candidate:
                rec['type'] = candidate['type']
        
        # Enrich alternatives
        for alt in recommendation['alternatives']:
            if alt['title'] in candidate_dict:
                candidate = candidate_dict[alt['title']]
                alt['genres'] = candidate['genres']
                if 'type' in candidate:
                    alt['type'] = candidate['type']
        
        return recommendation
    
    def _fallback_selection(self, candidates: List[Dict[str, Any]], 
                           preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback selection if LLM reasoning fails."""
        primary = candidates[0]
        alternatives = candidates[1:4] if len(candidates) >= 4 else candidates[1:]
        
        return {
            "primary_recommendation": {
                "title": primary['title'],
                "genres": primary['genres'],
                "reason": "Strong match based on genre similarity and preferences."
            },
            "alternatives": [
                {"title": alt['title'], "genres": alt['genres'], "reason": "Alternative strong match."}
                for alt in alternatives
            ],
            "explanation": f"Based on your preferences for {preferences.get('genre_blend', 'these genres')}, these recommendations align with your taste.",
            "preferences_summary": preferences.get('reasoning', '')
        }
    
    def get_session_state(self) -> Dict[str, Any]:
        """Get current session state."""
        return self.session_state.copy()
    
    def clear_session(self):
        """Clear session state."""
        self.session_state = {
            'favorite_titles': [],
            'extracted_preferences': None,
            'current_recommendation': None
        }

