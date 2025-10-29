"""Base Agent class with Gemini API wrapper"""

import google.generativeai as genai
from typing import Optional, Dict, Any
from config import GEMINI_API_KEY, GEMINI_MODEL, AGENT_CONFIG


class BaseAgent:
    """Base class for all agents with Gemini API integration"""
    
    def __init__(self, model_name: str = GEMINI_MODEL, config: Dict[str, Any] = None):
        """Initialize the agent with Gemini API"""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.model_name = model_name
        self.config = config or AGENT_CONFIG.copy()
        
        # Try to create model with fallback options
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": self.config.get("temperature", 0.7),
                    "max_output_tokens": self.config.get("max_output_tokens", 2048),
                    "top_p": self.config.get("top_p", 0.8),
                    "top_k": self.config.get("top_k", 40),
                }
            )
        except Exception as e:
            # Try fallback models (order matters - try newer ones first)
            fallback_models = [
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-flash-latest",
                "gemini-pro-latest",
                "gemini-2.5-flash-lite"
            ]
            fallback_models = [m for m in fallback_models if m != model_name]
            
            for fallback in fallback_models:
                try:
                    print(f"⚠️  Model '{model_name}' not available. Trying fallback: {fallback}")
                    self.model_name = fallback
                    self.model = genai.GenerativeModel(
                        model_name=fallback,
                        generation_config={
                            "temperature": self.config.get("temperature", 0.7),
                            "max_output_tokens": self.config.get("max_output_tokens", 2048),
                            "top_p": self.config.get("top_p", 0.8),
                            "top_k": self.config.get("top_k", 40),
                        }
                    )
                    print(f"✅ Using model: {fallback}")
                    break
                except:
                    continue
            else:
                # If all fallbacks fail, raise original error with helpful message
                available_models = self._list_available_models()
                raise ValueError(
                    f"Model '{model_name}' not found. Available models: {available_models}\n"
                    f"Original error: {str(e)}"
                )
    
    @staticmethod
    def _list_available_models() -> list:
        """List available models (for error messages)"""
        try:
            models = genai.list_models()
            available = []
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    # Return just the model identifier (after models/)
                    model_id = m.name.split('/')[-1] if '/' in m.name else m.name
                    available.append(model_id)
            return available
        except:
            return ["Unable to list models - check API key"]
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from Gemini model"""
        try:
            response = self.model.generate_content(prompt, **kwargs)
            if not response.text:
                raise Exception("Empty response from model")
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            # Check if it's a model not found error
            if "not found" in error_msg.lower() or "404" in error_msg:
                # Try to reinitialize with a fallback model
                print(f"\n⚠️  Model '{self.model_name}' error. Attempting to switch to fallback model...")
                fallback_models = [
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                    "gemini-flash-latest",
                    "gemini-pro-latest",
                    "gemini-2.5-flash-lite"
                ]
                for fallback in fallback_models:
                    if fallback != self.model_name:
                        try:
                            self.model_name = fallback
                            self.model = genai.GenerativeModel(
                                model_name=fallback,
                                generation_config={
                                    "temperature": self.config.get("temperature", 0.7),
                                    "max_output_tokens": self.config.get("max_output_tokens", 2048),
                                    "top_p": self.config.get("top_p", 0.8),
                                    "top_k": self.config.get("top_k", 40),
                                }
                            )
                            print(f"✅ Switched to: {fallback}")
                            # Retry generation
                            response = self.model.generate_content(prompt, **kwargs)
                            if not response.text:
                                raise Exception("Empty response from model")
                            return response.text.strip()
                        except Exception as fallback_error:
                            # Continue to next fallback
                            continue
                
                # If fallbacks fail, provide helpful error
                available = self._list_available_models()
                raise Exception(
                    f"Model '{self.model_name}' not available.\n"
                    f"Available models: {available}\n"
                    f"Please update GEMINI_MODEL in config.py\n"
                    f"Original error: {error_msg}"
                )
            raise Exception(f"Error generating response: {error_msg}")
    
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate structured JSON response"""
        # Add JSON format instruction to prompt
        json_prompt = f"{prompt}\n\nRespond only with valid JSON, no additional text."
        response = self.generate(json_prompt)
        
        # Try to extract JSON from response
        import json
        try:
            # Look for JSON in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # If parsing fails, return error dict
        return {"error": "Failed to parse JSON response", "raw_response": response}
    
    def update_config(self, **kwargs):
        """Update agent configuration"""
        self.config.update(kwargs)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.config.get("temperature", 0.7),
                "max_output_tokens": self.config.get("max_output_tokens", 2048),
                "top_p": self.config.get("top_p", 0.8),
                "top_k": self.config.get("top_k", 40),
            }
        )

