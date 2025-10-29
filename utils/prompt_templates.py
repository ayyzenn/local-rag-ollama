"""Prompt templates for all agents in the RAG system"""

# Router Agent Prompts
ROUTER_EVALUATION_PROMPT = """You are a routing agent that evaluates answer quality. 
Analyze the following answer and determine if it sufficiently addresses the user's question.

User Question: {query}

Generated Answer: {answer}

Context Used: {context}

Rate the answer on the following criteria:
1. Completeness: Does it fully answer the question?
2. Relevance: Is it relevant to the question asked?
3. Confidence: Does it seem confident and well-grounded?

Respond in JSON format:
{{
    "sufficient": true/false,
    "completeness_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "confidence_score": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""

# Basic Generator Prompt
BASIC_GENERATOR_PROMPT = """Answer the following question using only the information provided in the context. 
Be concise, factual, and directly address the question.

Context:
{context}

Question: {query}

Answer:"""

# Advanced Generator - Query Decomposition Prompt
DECOMPOSITION_PROMPT = """Break down the following complex question into simpler sub-questions that can be answered independently.

Question: {query}

Analyze the question and identify the main components. Generate 3-5 focused sub-questions that cover different aspects.

Respond in JSON format:
{{
    "sub_queries": [
        "sub-question 1",
        "sub-question 2",
        "sub-question 3"
    ],
    "reasoning": "why this decomposition helps"
}}
"""

DECOMPOSITION_SYNTHESIS_PROMPT = """Synthesize the following sub-answers into a coherent, comprehensive answer to the original question.

Original Question: {query}

Sub-Answers:
{sub_answers}

Create a well-structured answer that combines all relevant information from the sub-answers."""

# Advanced Generator - HyDE Prompt
HYDE_PROMPT = """Based on the following question, generate a hypothetical ideal answer that would fully address it.
This hypothetical answer represents what a perfect response would look like, even if you don't know the actual answer.

Question: {query}

Generate a detailed hypothetical answer that:
1. Directly addresses all aspects of the question
2. Includes relevant details and examples
3. Is structured and well-organized

Hypothetical Answer:"""

HYDE_GENERATION_PROMPT = """Using the actual documents retrieved based on a hypothetical answer, generate a grounded response to the question.

Question: {query}

Retrieved Context (from real documents):
{context}

Generate a factual answer using only the information from the retrieved context."""

# Multi-Query Prompt
MULTI_QUERY_PROMPT = """Generate multiple alternative phrasings of the following question to capture different perspectives and improve retrieval.

Original Question: {query}

Generate 4 different ways to ask this question that:
1. Use different terminology
2. Emphasize different aspects
3. Vary the question structure

Respond in JSON format:
{{
    "variations": [
        "variation 1",
        "variation 2",
        "variation 3",
        "variation 4"
    ]
}}
"""

# Answer Sufficiency Check Prompt
ANSWER_SUFFICIENCY_PROMPT = """Evaluate whether the following answer sufficiently addresses the question.

Question: {query}

Answer: {answer}

Based on the context available, is this answer complete and satisfactory?

Respond in JSON format:
{{
    "sufficient": true/false,
    "missing_information": ["what's missing if insufficient"],
    "reasoning": "explanation"
}}
"""

# Advanced Generator - Multi-Technique Combination Prompt
ADVANCED_GENERATION_PROMPT = """Generate a comprehensive answer using all the retrieved context from multiple advanced retrieval techniques.

Question: {query}

Retrieved Context:
{context}

Synthesize all the information above into a complete, well-structured answer that fully addresses the question."""

