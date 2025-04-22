import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("vector_db")

# Configure Qdrant client - adjust host/port as needed
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", 6334))
COLLECTION_NAME = "bitcoin_train_set"

# Initialize Qdrant client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Model will be initialized during startup
model = None

async def check_collection_status() -> Tuple[bool, Optional[str]]:
    """
    Check if the Qdrant collection exists and is accessible.
    
    Returns:
        Tuple of (success status, error message if any)
    """
    try:
        # List collections to check connectivity
        collections = qdrant_client.get_collections()
        
        # Check if our collection exists
        collection_names = [col.name for col in collections.collections]
        if COLLECTION_NAME not in collection_names:
            logger.warning(f"Collection '{COLLECTION_NAME}' not found in Qdrant. Vector similarity search may not work.")
            return False, f"Collection '{COLLECTION_NAME}' not found"
        else:
            # Get collection info
            collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            
            # Log collection info without assuming specific attributes
            logger.info(f"Collection '{COLLECTION_NAME}' found in Qdrant")
            logger.info(f"Collection info: {collection_info}")
            
            return True, None
            
    except Exception as e:
        error_msg = f"Error connecting to Qdrant: {e}"
        logger.error(error_msg)
        logger.warning("Vector similarity search will not be available!")
        return False, error_msg

async def search_vector_database(query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar signals in the vector database based on the provided text.
    
    Args:
        query_text: The text to search for (e.g., summary or news prediction)
        limit: The maximum number of results to return
        
    Returns:
        List of similar signals with their payloads
    """
    try:
        # Check if model is initialized
        if model is None:
            logger.error("Sentence transformer model is not initialized")
            return []
        
        # Generate embedding vector for the query text
        summary_embedding = model.encode(query_text, convert_to_numpy=False)

        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector={
                "summary": summary_embedding  
            },
            limit=limit,  
            with_payload=True  
        )
        
        # Process results
        similar_signals = []
        for result in search_result:
            payload = result.payload
            payload["score"] = result.score  # Add similarity score
            similar_signals.append(payload)
            
        logger.info(f"Found {len(similar_signals)} similar signals in vector database")
        return similar_signals
     
    except Exception as e:
        logger.error(f"Error searching vector database: {e}")
        logger.error(f"Error details: {str(e)}")
        
        # Return empty list on error - we'll implement a fallback method for search
        # that doesn't rely on vector DB
        return []

# Fallback method for when vector search is not available
async def fallback_search(df: pd.DataFrame, current_signal: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Fallback search using pandas dataframe when vector search fails"""
    try:
        # Extract summary and news prediction
        current_text = f"{current_signal.get('summary', '')} {current_signal.get('next_news_prediction', '')}"
        
        # Create a text column combining summary and news prediction
        df['combined_text'] = df['summary'] + ' ' + df['next_news_prediction']
        
        # Vectorize the text using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['combined_text'].fillna(''))
        
        # Vectorize the current text
        current_vector = vectorizer.transform([current_text])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(current_vector, tfidf_matrix)[0]
        
        # Add similarity scores to dataframe
        df['similarity'] = similarities
        
        # Sort by similarity and get top matches
        similar_signals = df.sort_values('similarity', ascending=False).head(limit)
        
        # Format results similar to vector search output
        result = []
        for _, row in similar_signals.iterrows():
            signal_dict = row.to_dict()
            payload = {
                "datetime": signal_dict.get('datetime', ''),
                "close": signal_dict.get('close', 0.0),
                "Signal": signal_dict.get('Signal', 0),
                "summary": signal_dict.get('summary', ''),
                "next_news_prediction": signal_dict.get('next_news_prediction', ''),
                "sentiment": signal_dict.get('sentiment', ''),
                "score": signal_dict.get('similarity', 0.0),
                "key_factors": signal_dict.get('key_factors', 'Not available'),
                "intensity": signal_dict.get('intensity', 0.0)
            }
            result.append(payload)
            
        return result
        
    except Exception as e:
        logger.error(f"Fallback search failed: {e}")
        return [] 