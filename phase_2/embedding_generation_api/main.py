import os
import logging
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# -------------------------------------------------------------------
# 1) Logger setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
lgguru = logging.getLogger("lgguru")

# --------------------------------------------------------------------
# Qdrant Client Initialization
# --------------------------------------------------------------------
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

# Initialize Qdrant client with default encoder
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    prefer_grpc=True
)
lgguru.info(f"Qdrant client configured for host: {QDRANT_HOST}:{QDRANT_PORT}")

# Default collection name
DEFAULT_COLLECTION = "bitcoin_train_set"

# -------------------------------------------------------------------
# 3) FastAPI initialization
# -------------------------------------------------------------------
app = FastAPI(
    title="Trading Signal Vector API",
    description="API to search and manage trading signal data with Qdrant.",
    version="1.2.0",
)

# -------------------------------------------------------------------
# 4) Request/Response Models
# -------------------------------------------------------------------
class SummarySearchRequest(BaseModel):
    summary: str = Field(
        ..., 
        description="The summary text to find similar matches for",
        example="Bitcoin's price has plummeted below $89,000, driven by fears of inflation, unfulfilled promises from the Trump administration, and a significant hack of the Bybit exchange, leading to a broader downturn in the cryptocurrency market."
    )
    next_news_prediction: str = Field(
        None,
        description="Optional prediction of future news (not used for matching)",
        example="Analysts predict further volatility as markets react to recent developments."
    )
    sentiment: str = Field(
        None,
        description="Optional sentiment value (not used for matching)",
        example="Extreme Fear"
    )

class SummarySearchResponse(BaseModel):
    message: str = Field(..., example="Found similar entry")
    similarity_score: float = Field(..., example=0.89)
    match: Dict[str, Any]

# -------------------------------------------------------------------
# 6) Qdrant Upload Endpoint
# -------------------------------------------------------------------
@app.post("/upload-train-data", status_code=200)
async def upload_train_data(file: UploadFile = File(...)):
    collection_name = DEFAULT_COLLECTION
    lgguru.info(f"Received file upload request for collection '{collection_name}'. Filename: {file.filename}")

    try:
        # 1) Read CSV
        try:
            df = pd.read_csv(file.file)
            lgguru.info(f"Read {len(df)} rows from '{file.filename}'.")
        except Exception as e:
            lgguru.error(f"Failed to read CSV '{file.filename}': {e}")
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")
        finally:
            await file.close()

        # 2) Prepare data
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)
        
        # Check for required columns
        text_columns = ['summary', 'next_news_prediction']
        required_columns = text_columns + ['datetime', 'close', 'Signal', 'sentiment']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            msg = f"Missing required columns: {missing_columns}"
            lgguru.error(msg)
            raise HTTPException(status_code=400, detail=msg)
        
        # Convert DataFrame to lists for Qdrant
        documents = df['summary'].fillna('').tolist()
        
        # Create metadata array
        metadata = []
        for _, row in df.iterrows():
            # Convert row to dictionary and handle special types
            row_dict = {col: str(val) if isinstance(val, (pd.Timestamp, np.integer, np.floating)) 
                        else val for col, val in row.items()}
            metadata.append(row_dict)
        
        # Create IDs array
        ids = df.index.tolist()
        
        # Ensure collection exists
        try:
            collections = qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                lgguru.info(f"Creating collection '{collection_name}'")
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=384,  
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            lgguru.error(f"Failed to check/create collection: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating collection: {e}")
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            
            qdrant_client.add(
                collection_name=collection_name,
                documents=documents[i:batch_end],
                metadata=metadata[i:batch_end],
                ids=ids[i:batch_end]
            )
            lgguru.info(f"Uploade384d batch {i//batch_size + 1}")
        
        return JSONResponse(content={"message": f"Uploaded {len(documents)} records to '{collection_name}'."})
    
    except HTTPException:
        raise
    except Exception as e:
        lgguru.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# -------------------------------------------------------------------
# Additional utility functions
# -------------------------------------------------------------------
async def find_closest_summary_match(row: Dict[str, Any], collection_name: str = DEFAULT_COLLECTION) -> Dict[str, Any]:
    """
    Find the closest match to the provided row based on the summary field.
    
    Args:
        row: A dictionary containing the same structure as the dataset, must include 'summary' field
        collection_name: The name of the Qdrant collection to search in
        
    Returns:
        The closest matching row from the database
    """
    if 'summary' not in row:
        lgguru.error("Input row must contain a 'summary' field")
        raise ValueError("Input row must contain a 'summary' field")
        
    try:
        # Search directly using the summary text
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_text=row['summary'],
            limit=1
        )
        
        if not search_result:
            lgguru.warning(f"No matches found for summary in collection '{collection_name}'")
            return None
            
        # Get the closest match
        closest_match = search_result[0]
        similarity_score = closest_match.score
        
        # Convert payload to dictionary
        match_data = closest_match.metadata
        
        # Add similarity score to the result
        match_data['similarity_score'] = similarity_score
        
        lgguru.info(f"Found closest match with similarity score: {similarity_score:.4f}")
        return match_data
        
    except Exception as e:
        lgguru.error(f"Error finding closest match: {e}")
        raise

# Add REST endpoint to use the function
@app.post("/find-similar-summary", response_model=SummarySearchResponse)
async def find_similar_summary(request: SummarySearchRequest):
    """
    Find the closest match to a provided row based on summary text.
    
    This endpoint searches the vector database for entries with similar summaries
    to the one provided, using semantic similarity rather than keyword matching.
    
    ## Example Use Case
    - Search for similar market conditions to understand historical patterns
    - Find similar news events and their impact on trading signals
    - Analyze how sentiment correlates with similar market narratives
    
    ## Notes
    - Only the 'summary' field is required and used for matching
    - Other fields are ignored in the similarity calculation
    - Returns a 404 if no matches are found
    """
    try:
        # Convert Pydantic model to dict
        row_dict = request.dict(exclude_unset=True)
        
        if 'summary' not in row_dict:
            lgguru.error("Input must contain a 'summary' field")
            raise ValueError("Input must contain a 'summary' field")
            
        # Search directly using the summary text
        search_result = qdrant_client.search(
            collection_name=DEFAULT_COLLECTION,
            query_text=row_dict['summary'],
            limit=1
        )
        
        if not search_result:
            return JSONResponse(content={"message": "No similar entries found"}, status_code=404)
            
        # Get the closest match
        closest_match = search_result[0]
        similarity_score = closest_match.score
        
        # Convert payload to dictionary
        match_data = closest_match.metadata
        
        # Add similarity score to the result
        match_data['similarity_score'] = similarity_score
        
        lgguru.info(f"Found closest match with similarity score: {similarity_score:.4f}")
        
        return {
            "message": "Found similar entry",
            "similarity_score": similarity_score,
            "match": match_data
        }
            
    except Exception as e:
        lgguru.error(f"Error processing similar summary request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# 7) Run the app
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8500, reload=True)
