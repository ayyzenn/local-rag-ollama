"""Router Agent that decides between Basic and Advanced agents"""

from agents.base_agent import BaseAgent
from agents.basic_generator import BasicGeneratorAgent
from agents.advanced_generator import AdvancedGeneratorAgent
from utils.evaluator import AnswerEvaluator
from config import ROUTER_CONFIG
from typing import Dict, Any, Optional


class RouterAgent(BaseAgent):
    """Router agent that routes queries to appropriate generator agent"""
    
    def __init__(
        self, 
        basic_agent: BasicGeneratorAgent,
        advanced_agent: AdvancedGeneratorAgent
    ):
        super().__init__(config=ROUTER_CONFIG)
        self.basic_agent = basic_agent
        self.advanced_agent = advanced_agent
        self.evaluator = AnswerEvaluator(self)
    
    def route_and_generate(
        self, 
        query: str,
        mode: str = "silent",  # silent, verbose, debug
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Route query through agents and generate answer
        
        Strategy: Always try basic first, use advanced if basic is insufficient
        """
        debug_mode = (debug or mode == "debug")
        verbose_mode = (mode == "verbose" or mode == "debug")
        
        if debug_mode:
            print("[Router] Analyzing query...")
            print(f"[Router] Routing to Basic Agent first (strategy: try-basic-then-advanced)")
        
        # Step 1: Try Basic Generator
        if verbose_mode:
            print("[Router] → Using Basic Generator Agent")
        
        basic_result = self.basic_agent.generate_answer(query, debug=debug_mode)
        
        # Step 2: Evaluate basic answer
        if debug_mode:
            print("[Router] Evaluating basic answer...")
        
        evaluation = self.evaluator.evaluate_answer_sufficiency(
            query=query,
            answer=basic_result["answer"],
            context=basic_result["context"]
        )
        
        is_sufficient = self.evaluator.is_sufficient(evaluation)
        
        if debug_mode:
            print(f"[Router] Basic answer evaluation:")
            print(f"  - Sufficient: {is_sufficient}")
            print(f"  - Completeness: {evaluation.get('completeness_score', 0):.2f}")
            print(f"  - Confidence: {evaluation.get('confidence_score', 0):.2f}")
        
        # Step 3: If sufficient, return basic answer
        if is_sufficient:
            if verbose_mode:
                print("[Router] ✓ Basic answer is sufficient")
            
            return {
                "answer": basic_result["answer"],
                "context": basic_result["context"],
                "retrieved_chunks": basic_result["retrieved_chunks"],
                "metadata": {
                    **basic_result["metadata"],
                    "routing": {
                        "strategy": "basic_only",
                        "evaluation": evaluation
                    }
                }
            }
        
        # Step 4: Basic insufficient, use Advanced Generator
        if verbose_mode:
            print("[Router] ✗ Basic answer insufficient")
            print("[Router] → Using Advanced Generator Agent")
        
        if debug_mode:
            print("[Router] Activating Advanced Agent with all techniques...")
        
        advanced_result = self.advanced_agent.generate_answer(
            query=query,
            techniques=["decomposition", "hyde", "multi_query"],
            debug=debug_mode
        )
        
        # Step 5: Evaluate advanced answer
        if debug_mode:
            print("[Router] Evaluating advanced answer...")
        
        adv_evaluation = self.evaluator.evaluate_answer_sufficiency(
            query=query,
            answer=advanced_result["answer"],
            context=advanced_result["context"]
        )
        
        adv_is_sufficient = self.evaluator.is_sufficient(adv_evaluation)
        
        if debug_mode:
            print(f"[Router] Advanced answer evaluation:")
            print(f"  - Sufficient: {adv_is_sufficient}")
            print(f"  - Completeness: {adv_evaluation.get('completeness_score', 0):.2f}")
            print(f"  - Confidence: {adv_evaluation.get('confidence_score', 0):.2f}")
        
        if verbose_mode:
            if adv_is_sufficient:
                print("[Router] ✓ Advanced answer is sufficient")
            else:
                print("[Router] ⚠ Advanced answer may still have limitations")
        
        return {
            "answer": advanced_result["answer"],
            "context": advanced_result["context"],
            "retrieved_chunks": advanced_result["retrieved_chunks"],
            "metadata": {
                **advanced_result["metadata"],
                "routing": {
                    "strategy": "basic_then_advanced",
                    "basic_evaluation": evaluation,
                    "advanced_evaluation": adv_evaluation,
                    "advanced_used": True
                }
            }
        }

