import os
import pandas as pd
import logging
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from datetime import datetime

# Import Chain of Density module
from cod_trading import SignalOverrideResponse, Trade, run_chain_of_density

# Import vector database function
from vector_db import search_vector_database, check_collection_status, fallback_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("trading_enhancer")

# Initialize FastAPI app
app = FastAPI(
    title="Trading Signal Enhancer API",
    description="API for enhancing trading signals using generative AI and vector similarity search",
    version="1.0.0",
)

# Pydantic models
class SignalInput(BaseModel):
    """Model for a single trading signal input"""
    datetime: str
    close: float
    mvrv_btc_momentum: float
    spot_volume_daily_sum: float
    Signal: int  # Original signal: 1 (BUY), -1 (SELL), 0 (HOLD)
    summary: str
    next_news_prediction: str
    sentiment: str
    index: Optional[int] = None
    key_factors: Optional[str] = None
    dominant_emotions: Optional[str] = None
    dominant_sentiment: Optional[str] = None
    intensity: Optional[float] = None
    psychology_explanation: Optional[str] = None
    original_index: Optional[int] = None

# Convert string verdict to signal integer
def verdict_to_signal(verdict: str) -> int:
    verdict = verdict.upper()
    if verdict == "BUY":
        return 1
    elif verdict == "SELL":
        return -1
    else:  # HOLD
        return 0

# Define similar signals search from dataframe
async def search_similar_signals_from_dataframe(df: pd.DataFrame, current_idx: int, current_signal: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar signals within the dataframe based on text similarity and other features.
    
    Args:
        df: The dataframe containing all signals
        current_idx: The index of the current signal
        current_signal: The current signal data
        limit: Maximum number of similar signals to return
        
    Returns:
        List of similar signals with their payloads
    """
    try:
        # Call vector database function instead of implementing similarity search here
        text_query = f"{current_signal['summary']} {current_signal['next_news_prediction']}"
        similar_signals = await search_vector_database(text_query, limit=limit)
        
        # If we got results, return them
        if similar_signals and len(similar_signals) > 0:
            logger.info(f"Successfully found {len(similar_signals)} signals using vector search")
            return similar_signals
        
        # If vector search returned no results, fall back to TF-IDF based search
        logger.info("Vector search returned no results, falling back to TF-IDF search")
        fallback_results = await fallback_search(df, current_signal, limit)
        
        if fallback_results and len(fallback_results) > 0:
            logger.info(f"Successfully found {len(fallback_results)} signals using fallback search")
            return fallback_results
            
        # Return empty list if both methods failed
        logger.warning("Both vector and fallback search methods failed to find any similar signals")
        return []
        
    except Exception as e:
        logger.error(f"Error searching for similar signals: {e}")
        
        # Try fallback search if vector search failed
        try:
            logger.info("Trying fallback search due to vector search error")
            fallback_results = await fallback_search(df, current_signal, limit)
            return fallback_results
        except Exception as fallback_e:
            logger.error(f"Fallback search also failed: {fallback_e}")
            # Return empty list on error
            return []

# Generate trade history from dataframe
def generate_trade_history(df: pd.DataFrame) -> List[Trade]:
    """
    Generate trade history from signals in the dataframe.
    
    Args:
        df: Dataframe containing trading signals
        
    Returns:
        List of Trade objects
    """
    trade_history = []
    
    try:
        # Sort by datetime to ensure chronological order
        df_sorted = df.sort_values('datetime')
        
        # Generate trades from signals
        for idx, row in df_sorted.iterrows():
            signal_value = row.get('Signal', 0)
            action = "BUY" if signal_value == 1 else "SELL" if signal_value == -1 else "HOLD"
            
            # Skip HOLD signals as they don't create trades
            if action == "HOLD":
                continue
                
            # Create Trade object from signal data
            trade = Trade(
                timestamp=row.get('datetime', ''),
                action=action,
                signal=signal_value,
                price=row.get('close', 0.0),
                quantity=0.5,  # Default quantity
                cost=row.get('close', 0.0) * 0.5  # Simple cost calculation
            )
            
            trade_history.append(trade)
    
    except Exception as e:
        logger.error(f"Error generating trade history: {e}")
        # Provide a minimal default trade history on error
        trade_history = [
            Trade(timestamp="2025-02-24 14:00:00", action="BUY", signal=1, price=90000.0, quantity=0.5, cost=45000.0),
            Trade(timestamp="2025-02-24 15:30:00", action="SELL", signal=-1, price=91000.0, quantity=0.5, cost=45500.0)
        ]
    
    # Return last 10 trades (or all if less than 10)
    return trade_history[-10:]

# Main endpoint to process a CSV file of signals
@app.post("/enhance-signals")
async def enhance_signals():
    """
    Process the gold_test.csv dataset of trading signals and enhance them using generative AI.
    
    The CSV contains all the columns defined in the SignalInput model.
    """
    try:
        # Load the gold_test.csv file directly
        file_path = "./data/gold_test.csv"
        df = pd.read_csv(file_path)
        logger.info(f"Loaded gold_test.csv with {len(df)} signals")
        
        # Initialize output DataFrame
        output_df = df.copy()
        output_df['enhanced_signal'] = None
        output_df['enhancement_reasons'] = None
        
        # Generate trade history from the dataframe
        trade_history = generate_trade_history(df)
        logger.info(f"Generated trade history with {len(trade_history)} trades")
        
        # Process each signal
        for idx, row in df.iterrows():
            try:
                # Create signal dictionary from row
                signal = row.to_dict()
                
                # Search for similar signals using the vector database function
                similar_signals = await search_similar_signals_from_dataframe(
                    df=df, 
                    current_idx=idx, 
                    current_signal=signal, 
                    limit=10
                )
                
                # Generate enhancement using Chain of Density
                success, enhancement = await run_chain_of_density(
                    current_signal=signal,
                    historical_signals=similar_signals,
                    trade_history=trade_history,
                    iterations=3  # Number of Chain of Density iterations
                )
                
                # Update output DataFrame
                output_df.at[idx, 'enhanced_signal'] = verdict_to_signal(enhancement.verdict)
                output_df.at[idx, 'enhancement_reasons'] = "; ".join(enhancement.definitive_reasons)
                
                logger.info(f"Processed signal {idx+1}/{len(df)}: Original={signal['Signal']}, Enhanced={enhancement.verdict}")
                
            except Exception as e:
                logger.error(f"Error processing signal {idx}: {e}")
                output_df.at[idx, 'enhanced_signal'] = row['Signal']  # Default to original signal
                output_df.at[idx, 'enhancement_reasons'] = f"Error: {str(e)}"
        
        # Save enhanced signals to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"gold_test_enhanced_{timestamp}.csv"
        output_df.to_csv(output_path, index=False)
        
        # Calculate performance metrics
        metrics = {
            "total_signals": len(df),
            "original_signals": {
                "buy_count": len(df[df['Signal'] == 1]),
                "sell_count": len(df[df['Signal'] == -1]),
                "hold_count": len(df[df['Signal'] == 0]),
            },
            "enhanced_signals": {
                "buy_count": len(output_df[output_df['enhanced_signal'] == 1]),
                "sell_count": len(output_df[output_df['enhanced_signal'] == -1]),
                "hold_count": len(output_df[output_df['enhanced_signal'] == 0]),
            },
            "changes": {
                "total_changes": len(output_df[output_df['Signal'] != output_df['enhanced_signal']]),
                "buy_to_sell": len(output_df[(output_df['Signal'] == 1) & (output_df['enhanced_signal'] == -1)]),
                "buy_to_hold": len(output_df[(output_df['Signal'] == 1) & (output_df['enhanced_signal'] == 0)]),
                "sell_to_buy": len(output_df[(output_df['Signal'] == -1) & (output_df['enhanced_signal'] == 1)]),
                "sell_to_hold": len(output_df[(output_df['Signal'] == -1) & (output_df['enhanced_signal'] == 0)]),
                "hold_to_buy": len(output_df[(output_df['Signal'] == 0) & (output_df['enhanced_signal'] == 1)]),
                "hold_to_sell": len(output_df[(output_df['Signal'] == 0) & (output_df['enhanced_signal'] == -1)]),
            }
        }
        
        return JSONResponse(content={
            "message": "Successfully enhanced trading signals from gold_test.csv",
            "output_file": output_path,
            "metrics": metrics
        })
    
    except Exception as e:
        logger.error(f"Error processing gold_test.csv: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing signals: {str(e)}")

# Endpoint to download the enhanced signals CSV
@app.get("/download/{filename}")
async def download_enhanced_signals(filename: str):
    """
    Download the enhanced signals CSV file.
    """
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(path=filename, filename=filename, media_type="text/csv")

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Performs initialization tasks when the application starts.
    """
    logger.info("Trading Signal Enhancer API starting up...")
    
    # Check if Qdrant vector database is accessible
    status, error = await check_collection_status()
    if not status and error:
        logger.warning(f"Vector database check failed: {error}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("trading_enhancer_api:app", host="0.0.0.0", port=8501, reload=True) 