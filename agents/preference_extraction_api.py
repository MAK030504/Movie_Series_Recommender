"""
Preference Extraction Agent - Hugging Face Inference API Version

Understands WHY users liked their favourite movies/series by analyzing
themes, tone, pacing, narrative style, genre blend, and format preferences.
Uses Hugging Face Inference API instead of local model.
"""

from typing import List, Dict, Any
import json
import re
from llm_helper_api_v2 import HuggingFaceInferenceLLM


class PreferenceExtractionAgent:
    """
    Agent that extracts structured preferences from user's favorite titles
    using LLM reasoning via Hugging Face Inference API.
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 api_key: str = None):
        """
        Initialize the Preference Extraction Agent.
        
        Parameters:
        -----------
        model_name : str
            Hugging Face model identifier (default: mistralai/Mistral-7B-Instruct-v0.3)
        api_key : str, optional
            Hugging Face API key. If not provided, reads from HF_API_KEY env variable.
        """
        self.llm = HuggingFaceInferenceLLM(model_name=model_name, api_key=api_key)
    
    def extract_preferences(self, favorite_titles: List[str]) -> Dict[str, Any]:
        """
        Extract structured preferences from user's favorite movies/series.
        
        Parameters:
        -----------
        favorite_titles : List[str]
            List of 3-5 movie/TV titles the user likes
            
        Returns:
        --------
        Dict[str, Any]
            Structured preferences containing:
            - themes: List[str]
            - tone: str
            - pacing: str
            - narrative_style: str
            - genre_blend: str
            - format_preference: str (optional)
            - reasoning: str (brief explanation)
        """
        titles_str = "\n".join(f"- {title}" for title in favorite_titles)
        
        system_prompt = "You are an expert film and TV critic who understands narrative structure, themes, and audience preferences."
        
        user_prompt = f"""Analyze the following movies/series that a user loves and extract their underlying preferences.

User's favorite titles:
{titles_str}

Based on these titles, identify the user's taste preferences. Think about WHY they might like these titles - what themes, tones, pacing, and narrative styles connect them.

CRITICAL: Return a COMPLETE, valid JSON object with ALL fields filled. The JSON must be properly closed.

Return a JSON object with the following structure:
{{
    "themes": ["theme1", "theme2", "theme3"],
    "tone": "dark/hopeful/humorous/intense/serious/etc",
    "pacing": "slow-burn/fast-paced/moderate",
    "narrative_style": "character-driven/plot-driven/balanced",
    "genre_blend": "description of genre combinations (e.g., sci-fi + drama, thriller + mystery)",
    "format_preference": "movie/TV/both",
    "reasoning": "Brief explanation of what connects these titles in the user's taste"
}}

IMPORTANT REQUIREMENTS:
- All fields must have values (no empty strings or null)
- The JSON must end with a closing brace }}
- Return ONLY the JSON object, no markdown code blocks, no additional text
- Ensure the JSON is complete and valid before ending your response

Focus on understanding the deeper preferences, not just surface-level genre labels. Be specific and thoughtful."""

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Increase max_tokens to ensure complete response
                response = self.llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    max_tokens=2000,  # Increased from 1500
                    format_json=True
                )
                
                # Clean response
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                # Try to parse JSON
                try:
                    result = json.loads(cleaned)
                    # Validate required fields
                    required_fields = ['themes', 'tone', 'pacing', 'narrative_style', 'genre_blend']
                    for field in required_fields:
                        if field not in result:
                            result[field] = self._get_default_value(field)
                    return result
                except json.JSONDecodeError as e:
                    # Try to fix incomplete JSON
                    result = self._fix_incomplete_json(cleaned, str(e))
                    if result:
                        return result
                    
                    # If this is the last attempt, raise error
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Error parsing JSON response after {max_retries} attempts: {str(e)}\n"
                            f"Response (first 500 chars): {response[:500]}"
                        )
                    # Otherwise, retry with a more explicit prompt
                    user_prompt += "\n\nIMPORTANT: Return a COMPLETE, valid JSON object. Ensure all fields have values and the JSON is properly closed with a closing brace."
                    continue
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Error extracting preferences: {str(e)}")
                continue
        
        # Should not reach here, but just in case
        raise RuntimeError("Failed to extract preferences after multiple attempts")
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields."""
        defaults = {
            'themes': ['General Entertainment'],
            'tone': 'balanced',
            'pacing': 'moderate',
            'narrative_style': 'balanced',
            'genre_blend': 'mixed genres',
            'format_preference': 'both',
            'reasoning': 'Preferences extracted from user selections'
        }
        return defaults.get(field, '')
    
    def _fix_incomplete_json(self, text: str, error_msg: str) -> Dict[str, Any]:
        """Try to fix incomplete JSON by adding missing closing braces and default values."""
        try:
            # Check if JSON is incomplete (doesn't end with })
            if not text.rstrip().endswith('}'):
                # Try to complete the JSON by finding the last complete field
                # Remove any trailing comma
                cleaned_text = text.rstrip().rstrip(',')
                
                # Determine which fields are missing
                has_themes = '"themes":' in cleaned_text
                has_tone = '"tone":' in cleaned_text
                has_pacing = '"pacing":' in cleaned_text
                has_narrative_style = '"narrative_style":' in cleaned_text
                has_genre_blend = '"genre_blend":' in cleaned_text
                
                # Build completion string for missing fields
                missing_fields = []
                
                if not has_genre_blend:
                    missing_fields.append('"genre_blend": "mixed genres"')
                if not has_narrative_style:
                    missing_fields.insert(0, '"narrative_style": "balanced"')
                if not has_pacing:
                    missing_fields.insert(0, '"pacing": "moderate"')
                if not has_tone:
                    missing_fields.insert(0, '"tone": "balanced"')
                if not has_themes:
                    missing_fields.insert(0, '"themes": ["General Entertainment"]')
                
                # Always add format_preference and reasoning if missing
                if '"format_preference":' not in cleaned_text:
                    missing_fields.append('"format_preference": "both"')
                if '"reasoning":' not in cleaned_text:
                    missing_fields.append('"reasoning": "Preferences extracted from user selections"')
                
                # Complete the JSON
                if missing_fields:
                    # Ensure there's a comma before adding missing fields
                    if not cleaned_text.rstrip().endswith(',') and not cleaned_text.rstrip().endswith('{'):
                        cleaned_text += ','
                    fixed = cleaned_text + ' ' + ', '.join(missing_fields) + '}'
                    
                    try:
                        result = json.loads(fixed)
                        return result
                    except Exception as parse_error:
                        # If fixing didn't work, fall through to extraction method
                        pass
            
            # Try to extract what we can and fill in defaults
            result = {
                'themes': ['General Entertainment'],
                'tone': 'balanced',
                'pacing': 'moderate',
                'narrative_style': 'balanced',
                'genre_blend': 'mixed genres',
                'format_preference': 'both',
                'reasoning': 'Preferences extracted from user selections'
            }
            
            # Try to extract any complete fields
            if '"themes":' in text:
                themes_match = re.search(r'"themes":\s*\[(.*?)\]', text, re.DOTALL)
                if themes_match:
                    try:
                        themes_str = '[' + themes_match.group(1) + ']'
                        themes = json.loads(themes_str)
                        result['themes'] = themes
                    except:
                        pass
            
            if '"tone":' in text:
                tone_match = re.search(r'"tone":\s*"([^"]*)"', text)
                if tone_match:
                    result['tone'] = tone_match.group(1)
            
            if '"pacing":' in text:
                pacing_match = re.search(r'"pacing":\s*"([^"]*)"', text)
                if pacing_match:
                    result['pacing'] = pacing_match.group(1)
            
            if '"narrative_style":' in text:
                style_match = re.search(r'"narrative_style":\s*"([^"]*)"', text)
                if style_match:
                    result['narrative_style'] = style_match.group(1)
            
            return result
            
        except Exception:
            return None

