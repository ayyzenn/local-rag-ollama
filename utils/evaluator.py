"""Answer evaluation utilities"""

import json
from typing import Dict, Any
from agents.base_agent import BaseAgent
from config import EVALUATION_CONFIG


class AnswerEvaluator:
    """Evaluates answer quality and sufficiency"""
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.min_confidence = EVALUATION_CONFIG["min_confidence_score"]
        self.min_completeness = EVALUATION_CONFIG["min_completeness_score"]
    
    def evaluate_answer_sufficiency(
        self, 
        query: str, 
        answer: str, 
        context: str
    ) -> Dict[str, Any]:
        """Evaluate if an answer is sufficient using LLM"""
        from utils.prompt_templates import ROUTER_EVALUATION_PROMPT
        
        prompt = ROUTER_EVALUATION_PROMPT.format(
            query=query,
            answer=answer,
            context=context[:500]  # Limit context length
        )
        
        try:
            response = self.agent.generate(prompt)
            # Try to parse JSON from response
            evaluation = self._parse_evaluation_response(response)
            return evaluation
        except Exception as e:
            # Fallback: simple heuristics
            return self._fallback_evaluation(query, answer)
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON evaluation from LLM response"""
        # Try to extract JSON from response
        try:
            # Look for JSON in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                evaluation = json.loads(json_str)
                return evaluation
        except:
            pass
        
        # If JSON parsing fails, try to infer from text
        response_lower = response.lower()
        sufficient = "sufficient" in response_lower and "not sufficient" not in response_lower
        
        return {
            "sufficient": sufficient,
            "completeness_score": 0.7 if sufficient else 0.4,
            "relevance_score": 0.7,
            "confidence_score": 0.7,
            "reasoning": response
        }
    
    def _fallback_evaluation(self, query: str, answer: str) -> Dict[str, Any]:
        """Fallback evaluation using simple heuristics"""
        # Simple heuristics
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Check if answer is too short
        word_count = len(answer.split())
        is_too_short = word_count < 10
        
        # Check if answer contains relevant keywords
        overlap = len(query_words.intersection(answer_words))
        keyword_coverage = overlap / len(query_words) if query_words else 0
        
        # Check for uncertainty indicators
        uncertainty_indicators = ["i don't know", "i'm not sure", "cannot", "unable", "no information"]
        has_uncertainty = any(indicator in answer.lower() for indicator in uncertainty_indicators)
        
        sufficient = not is_too_short and keyword_coverage > 0.3 and not has_uncertainty
        
        return {
            "sufficient": sufficient,
            "completeness_score": 0.8 if sufficient else 0.4,
            "relevance_score": keyword_coverage,
            "confidence_score": 0.9 if not has_uncertainty else 0.3,
            "reasoning": f"Heuristic evaluation: word_count={word_count}, keyword_coverage={keyword_coverage:.2f}, uncertainty={has_uncertainty}"
        }
    
    def is_sufficient(self, evaluation: Dict[str, Any]) -> bool:
        """Check if evaluation indicates sufficient answer"""
        if not evaluation.get("sufficient", False):
            return False
        
        completeness = evaluation.get("completeness_score", 0)
        confidence = evaluation.get("confidence_score", 0)
        
        return (completeness >= self.min_completeness and 
                confidence >= self.min_confidence)

