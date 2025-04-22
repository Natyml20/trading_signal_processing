"""
Chain of Density module for trading signal enhancement.

This module implements the Chain of Density approach for refining trading signals
by iteratively finding patterns in successful/unsuccessful signals and providing
reasons for signal overrides.
"""

import logging
import time
from typing import List, Dict, Tuple, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from openai_utils.openai_call import openai_request

logger = logging.getLogger(__name__)

class SignalOverrideResponse(BaseModel):
    """Model for the Chain of Density response format"""
    previous_reasons: List[str] = Field(default_factory=list)
    missing_reasons: List[str] = Field(default_factory=list)
    definitive_reasons: List[str] = Field(default_factory=list)
    verdict: str  # BUY, SELL, or HOLD

class Trade(BaseModel):
    """Model for trade history"""
    timestamp: str
    action: str  # BUY, SELL, HOLD
    signal: int  # 1, -1, 0
    price: float
    quantity: float = 0.0
    cost: float = 0.0

def _build_prompt(current_signal: Dict[str, Any], 
                 historical_signals: List[Dict[str, Any]], 
                 trade_history: List[Trade],
                 iteration: int = 0, 
                 previous_result: Dict[str, Any] = None) -> List[Dict[str, str]]:
    """
    Build the Chain of Density prompt for trading signal enhancement.
    
    Args:
        current_signal: The current signal data.
        historical_signals: Similar historical signals from vector search.
        trade_history: Recent trade history.
        iteration: Current iteration number (0-based).
        previous_result: Results from the previous iteration.
        
    Returns:
        List of messages for the LLM prompt.
    """
    # Format similar signals as context
    similar_signals_text = ""
    for i, signal in enumerate(historical_signals):
        similar_signals_text += f"Similar Signal #{i+1} (Similarity: {signal.get('score', 0):.2f}):\n"
        similar_signals_text += f"Date: {signal.get('datetime', 'Unknown')}\n"
        similar_signals_text += f"Original Signal: {signal.get('Signal', 'Unknown')} "
        similar_signals_text += f"({'BUY' if signal.get('Signal') == 1 else 'SELL' if signal.get('Signal') == -1 else 'HOLD'})\n"
        similar_signals_text += f"Strategy Success: {signal.get('strategy_success', 'Unknown')}\n"
        similar_signals_text += f"Summary: {signal.get('summary', 'No summary available')}\n"
        similar_signals_text += f"Key Factors: {signal.get('key_factors', 'No factors available')}\n"
        similar_signals_text += f"Sentiment: {signal.get('sentiment', 'Unknown')} (Intensity: {signal.get('intensity', 'Unknown')})\n\n"
    
    # Format trade history
    trade_history_text = "\n".join([
        f"- {trade.timestamp}: {trade.action} ({trade.signal}) at {trade.price:.2f}, Quantity: {trade.quantity:.4f}, Cost: {trade.cost:.2f}"
        for trade in trade_history[-5:]  # Last 5 trades
    ])
    
    if not trade_history_text:
        trade_history_text = "No recent trades."
    
    # Format current signal details
    signal_details = (
        f"Timestamp: {current_signal.get('datetime')}\n"
        f"Close Price: {current_signal.get('close'):.2f}\n"
        f"Original Signal: {current_signal.get('Signal')} "
        f"({'BUY' if current_signal.get('Signal') == 1 else 'SELL' if current_signal.get('Signal') == -1 else 'HOLD'})\n"
        f"MVRV Momentum: {current_signal.get('mvrv_btc_momentum')}\n"
        f"Volume: {current_signal.get('spot_volume_daily_sum')}\n"
        f"Sentiment: {current_signal.get('sentiment')}\n"
        f"Dominant Emotions: {current_signal.get('dominant_emotions', 'Not available')}\n"
        f"Summary: {current_signal.get('summary')}\n"
    )
    
    # Initial analysis guidelines
    INITIAL_ANALYSIS = """
    Step 1: Initial Analysis
    Analyze the current signal and similar historical signals:
    1. Identify patterns in successful and unsuccessful signals
    2. Consider market sentiment and conditions
    3. Look for correlations between technical indicators and outcomes
    """
    
    PATTERN_IDENTIFICATION = """
    Step 2: Pattern Development
    Organize findings into key patterns:
    1. Cluster similar outcomes based on market conditions
    2. Note frequency of each pattern
    3. Identify contradictions between the original signal and market context
    """
    
    DECISION_MAKING = """
    Step 3: Decision Evaluation
    For each potential action (BUY, SELL, HOLD):
    1. Count supporting evidence
    2. Assess confidence level
    3. Consider potential risks and rewards
    """
    
    # Style constraints
    VERBOSITY = """The Reasons must:
    - Be clearly stated and specific
    - Reference historical patterns
    - Connect to concrete data points
    - Avoid ambiguous language
    """
    
    # Guidelines for subsequent iterations
    FUSION = """Guidelines for Subsequent Iterations:
    1. Maintain all previous reasons that remain valid.
    2. Add 3-5 new insights identified in this iteration.
    3. Combine similar categories/reasons where appropriate.
    4. Refine reasoning to be more precise and data-driven.
    5. Ensure verdict is well-supported by definitive reasons.
    """
    
    RESULT_FORMAT = """{
    "previous_reasons": ["Reason 1", "Reason 2",...],
    "missing_reasons": ["Missing reason 1", "reason 2",...],
    "definitive_reasons": ["Definitive reason 1", "Definitive Reason 2",...],
    "verdict": "BUY" or "SELL" or "HOLD"
    }"""
    
    # Base prompt structure based on iteration
    if iteration == 0:
        instruction_prompt = (
            "You are an AI trading assistant that enhances trading signals. "
            "Analyze historical similar signals and current market conditions to determine if the original signal "
            "should be overridden. Your goal is to improve trading performance by identifying patterns in "
            "successful and unsuccessful signals."
        )
        
        user_prompt = (
            f"Please review the current trading signal and similar historical signals. Using Chain of Density, "
            f"determine whether to override the original signal ({current_signal.get('Signal')}) with BUY, SELL, or HOLD.\n\n"
            f"Follow these steps for the initial analysis:\n"
            f"{INITIAL_ANALYSIS}\n{PATTERN_IDENTIFICATION}\n{DECISION_MAKING}\n"
            f"Style Guidelines:\n{VERBOSITY}\n\n"
            f"Current Signal:\n{signal_details}\n\n"
            f"Historical Similar Signals:\n{similar_signals_text}\n\n"
            f"Recent Trade History:\n{trade_history_text}\n\n"
            f"Response Format:\n{RESULT_FORMAT}"
        )
    else:
        # Format previous results
        prev_reasons = ", ".join([f'"{r}"' for r in previous_result.get('previous_reasons', [])])
        prev_definitive = ", ".join([f'"{r}"' for r in previous_result.get('definitive_reasons', [])])
        prev_verdict = previous_result.get('verdict', 'HOLD')
        
        instruction_prompt = (
            "You are an AI trading assistant refining signal enhancement through Chain of Density. "
            "Continue to analyze the trading context and build upon previous iterations to improve the decision."
        )
        
        user_prompt = (
            f"This is iteration {iteration+1} of the Chain of Density process. "
            f"Review your previous analysis and refine your reasoning about whether to override "
            f"the original signal ({current_signal.get('Signal')}).\n\n"
            f"Previous iteration verdict: {prev_verdict}\n"
            f"Previous definitive reasons: [{prev_definitive}]\n\n"
            f"Follow these guidelines for refinement:\n"
            f"{FUSION}\n"
            f"Style Guidelines:\n{VERBOSITY}\n\n"
            f"Current Signal:\n{signal_details}\n\n"
            f"Historical Similar Signals:\n{similar_signals_text}\n\n"
            f"Recent Trade History:\n{trade_history_text}\n\n"
            f"Response Format:\n{RESULT_FORMAT}"
        )
    
    payload = [
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return payload

async def run_chain_of_density(
    current_signal: Dict[str, Any],
    historical_signals: List[Dict[str, Any]],
    trade_history: List[Trade],
    iterations: int = 2,
    max_validation_retries: int = 3
) -> Tuple[bool, SignalOverrideResponse]:
    """
    Run the Chain of Density process for trading signal enhancement.
    
    Args:
        current_signal: The current signal data.
        historical_signals: Similar historical signals from vector search.
        trade_history: Recent trade history.
        iterations: Number of iterations to run (default: 2).
        max_validation_retries: Maximum number of validation retries.
        
    Returns:
        Tuple of (success flag, final response)
    """
    logger.info(f"Starting Chain of Density with {iterations} iterations for signal on {current_signal.get('datetime')}")
    
    # Define response schema
    response_format = {"jsonSchema": {
        "type": "object",
        "properties": {
            "previous_reasons": {"type": "array", "items": {"type": "string"}},
            "missing_reasons": {"type": "array", "items": {"type": "string"}},
            "definitive_reasons": {"type": "array", "items": {"type": "string"}},
            "verdict": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]}
        },
        "required": ["previous_reasons", "missing_reasons", "definitive_reasons", "verdict"]
    }}
    
    previous_result = None
    final_response = None
    
    for i in range(iterations):
        logger.info(f"Starting iteration {i+1}/{iterations}")
        
        # Build prompt based on iteration
        prompt = _build_prompt(
            current_signal=current_signal,
            historical_signals=historical_signals,
            trade_history=trade_history,
            iteration=i,
            previous_result=previous_result
        )
        
        # Call OpenAI API
        success, response_obj = await openai_request(prompt, response_format)
        
        if not success:
            logger.error(f"OpenAI API call failed at iteration {i+1}: {response_obj}")
            
            # If first iteration fails, return failure
            if i == 0:
                return False, SignalOverrideResponse(
                    previous_reasons=[],
                    missing_reasons=[],
                    definitive_reasons=["Failed to generate analysis due to API error."],
                    verdict="HOLD"  # Default to HOLD on failure
                )
            
            # If later iteration fails, return last successful iteration
            if final_response:
                logger.warning(f"Using results from iteration {i} as final due to API failure")
                return True, final_response
            
            # If no successful iterations, return failure
            return False, SignalOverrideResponse(
                previous_reasons=[],
                missing_reasons=[],
                definitive_reasons=["Failed to generate analysis due to API error."],
                verdict="HOLD"  # Default to HOLD on failure
            )
        
        # Parse response
        try:
            validated_response = SignalOverrideResponse(
                previous_reasons=response_obj.get("previous_reasons", []),
                missing_reasons=response_obj.get("missing_reasons", []),
                definitive_reasons=response_obj.get("definitive_reasons", []),
                verdict=response_obj.get("verdict", "HOLD")
            )
            
            # Store current iteration result for next iteration
            previous_result = response_obj
            final_response = validated_response
            
            logger.info(f"Iteration {i+1} successful: Verdict={validated_response.verdict}, Reasons={len(validated_response.definitive_reasons)}")
            
        except ValidationError as e:
            logger.error(f"Validation error at iteration {i+1}: {e}")
            
            # If first iteration fails, return failure
            if i == 0:
                return False, SignalOverrideResponse(
                    previous_reasons=[],
                    missing_reasons=[],
                    definitive_reasons=["Failed to validate analysis due to response format error."],
                    verdict="HOLD"  # Default to HOLD on validation failure
                )
            
            # If later iteration fails, return last successful iteration
            if final_response:
                logger.warning(f"Using results from iteration {i} as final due to validation failure")
                return True, final_response
                
            # If no successful iterations, return failure
            return False, SignalOverrideResponse(
                previous_reasons=[],
                missing_reasons=[],
                definitive_reasons=["Failed to validate analysis due to response format error."],
                verdict="HOLD"  # Default to HOLD on validation failure
            )
    
    return True, final_response 