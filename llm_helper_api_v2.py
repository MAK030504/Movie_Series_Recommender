"""
LLM Helper Module - Hugging Face Inference API Version (using huggingface_hub)

Uses Hugging Face InferenceClient which automatically handles the correct endpoint.
"""

import json
import re
import os
from typing import Dict, Any, Optional
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()


class HuggingFaceInferenceLLM:
    """
    Wrapper for Hugging Face Inference API using InferenceClient.
    Automatically uses the correct endpoint format.
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 api_key: str = None):
        """
        Initialize the Hugging Face Inference API client.
        
        Parameters:
        -----------
        model_name : str
            Hugging Face model identifier (default: mistralai/Mistral-7B-Instruct-v0.3)
        api_key : str, optional
            Hugging Face API key. If not provided, reads from HF_API_KEY env variable.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Hugging Face API key is required. "
                "Set HF_API_KEY environment variable or pass api_key parameter. "
                "Get your API key from: https://huggingface.co/settings/tokens"
            )
        
        # Use InferenceClient which handles endpoint automatically
        self.client = InferenceClient(model=model_name, token=self.api_key)
    
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, 
                 max_tokens: int = 2000, format_json: bool = False) -> str:
        """
        Generate text using the Hugging Face Inference API.
        
        Parameters:
        -----------
        system_prompt : str
            System instruction/context
        user_prompt : str
            User query/prompt
        temperature : float
            Sampling temperature (default: 0.7)
        max_tokens : int
            Maximum tokens to generate (default: 2000)
        format_json : bool
            Whether to extract JSON from response (default: False)
            
        Returns:
        --------
        str
            Generated text
        """
        # Format messages for chat completion (conversational task)
        # Mistral-7B-Instruct requires conversational/chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Use chat completion for conversational models
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract generated text from chat completion response
            if isinstance(response, dict):
                # Response format: {"choices": [{"message": {"content": "..."}}]}
                if "choices" in response and len(response["choices"]) > 0:
                    generated_text = response["choices"][0].get("message", {}).get("content", "")
                elif "content" in response:
                    generated_text = response["content"]
                else:
                    generated_text = str(response)
            else:
                generated_text = str(response)
            
            # If JSON format requested, try to extract JSON
            if format_json:
                generated_text = self._extract_json(generated_text)
            
            return generated_text.strip()
            
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON object from text response.
        
        Parameters:
        -----------
        text : str
            Text that may contain JSON
            
        Returns:
        --------
        str
            Extracted JSON string
        """
        # Try to find JSON in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return text

