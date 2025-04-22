from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import logging
import boto3
from botocore.exceptions import ClientError
import json
from decimal import Decimal
from dotenv import load_dotenv
import pandas as pd
import io
from urllib.parse import unquote

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("Starting application...")
logger.info(f"Environment variables: {dict(os.environ)}")

# Initialize AWS clients
try:   
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('TradingPortfolio')
    logger.info("Successfully connected to DynamoDB")
except Exception as e:
    logger.error(f"Failed to initialize DynamoDB: {str(e)}")
    # Initialize with None to allow the app to start even if DynamoDB is not available
    table = None

app = FastAPI(title="Trading Signal Processing System")
logger.info("FastAPI app initialized")

# Add middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Received request: {request.method} {request.url.path}")
    logger.info(f"Headers: {dict(request.headers)}")
    response = await call_next(request)
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
logger.info("CORS middleware added")

# Pydantic models for request/response validation
class Signal(BaseModel):
    datetime: str
    close: float
    signal: int

    def get_datetime(self) -> datetime:
        """Convert the string datetime to a datetime object"""
        try:
            return datetime.strptime(self.datetime, "%m/%d/%Y %H:%M")
        except ValueError:
            try:
                return datetime.fromisoformat(self.datetime)
            except ValueError:
                raise ValueError("Invalid datetime format. Use 'MM/DD/YYYY HH:MM' or ISO format")

class Performance(BaseModel):
    cumulative_return: float
    current_position: str
    portfolio_value: float
    number_of_trades: int
    win_rate: float

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Global variables to store portfolio state
PERCENTAGE_OF_CASH_TO_BUY = 0.3 # 30% of cash holdings
portfolio = {
    "current_position": "HOLD",
    "portfolio_value": 1000000.0,
    "bitcoin_holdings": 0.0,
    "cash_holdings": 1000000.0,
    "trades": [],
    "total_return": 0.0,
    "number_of_trades": 0,
    "winning_trades": 0
}

def parse_timestamp(timestamp_str: str) -> str:
    """Parse timestamp from format '2/24/2025 0:00' to ISO format"""
    try:
        # Try to parse the input format
        dt = datetime.strptime(timestamp_str, "%m/%d/%Y %H:%M")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        # If it fails, try ISO format
        try:
            datetime.fromisoformat(timestamp_str)
            return timestamp_str
        except ValueError:
            raise ValueError("Invalid timestamp format. Use 'MM/DD/YYYY HH:MM' or ISO format")

def save_to_dynamodb(data: dict):
    """Save portfolio data to DynamoDB with separate trade and state information"""
    try:
        # Extract the last trade if it exists
        last_trade = data["trades"][-1] if data["trades"] else None
        
        # Convert float values to Decimal
        item = {
            'timestamp': parse_timestamp(str(last_trade["timestamp"])),
            'current_position': data["current_position"],
            'portfolio_value': Decimal(str(data["portfolio_value"])),
            'bitcoin_holdings': Decimal(str(data["bitcoin_holdings"])),
            'cash_holdings': Decimal(str(data["cash_holdings"])),
            'total_return': Decimal(str(data["total_return"])),
            'number_of_trades': int(data["number_of_trades"]),
            'winning_trades': int(data["winning_trades"])
        }
        
        # Add trade data if it exists
        if last_trade:
            if last_trade["price"] is not None:
                item['data'] = json.dumps({
                    'trade_type': last_trade["type"],
                    'trade_price': Decimal(str(last_trade["price"])),
                    'trade_amount': Decimal(str(last_trade["amount"])),
                    'trade_timestamp': parse_timestamp(str(last_trade["timestamp"]))
                }, default=str)
            else:
                item['data'] = json.dumps({
                    'trade_type': last_trade["type"],
                    'trade_price': 0,
                    'trade_amount': 0,
                    'trade_timestamp': parse_timestamp(str(last_trade["timestamp"])),
                    'reason': last_trade["reason"]
                }, default=str)
        
        table.put_item(Item=item)
        logger.info("Successfully saved portfolio data to DynamoDB")
    except ClientError as e:
        logger.error(f"Error saving to DynamoDB: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving portfolio data")

def load_from_dynamodb():
    """Load latest portfolio data from DynamoDB"""
    try:
        response = table.scan(Limit=1) # Get the latest item
        if response['Items']:
            item = response['Items'][0]
            # Reconstruct the portfolio state
            portfolio_data = {
                "current_position": item['current_position'],
                "portfolio_value": float(item['portfolio_value']),
                "bitcoin_holdings": float(item['bitcoin_holdings']),
                "cash_holdings": float(item['cash_holdings']),
                "total_return": float(item['total_return']),
                "number_of_trades": int(item['number_of_trades']),
                "winning_trades": int(item['winning_trades'])
            }
            
            # Add the last trade if it exists
            if 'data' in item:
                trade_data = json.loads(item['data'])
                portfolio_data['trades'] = [{
                    "type": trade_data['trade_type'],
                    "price": float(trade_data['trade_price']),
                    "amount": float(trade_data['trade_amount']),
                    "timestamp": trade_data['trade_timestamp']
                }]
            else:
                portfolio_data['trades'] = []
                
            return portfolio_data
        return None
    except ClientError as e:
        logger.error(f"Error loading from DynamoDB: {str(e)}")
        return None

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    logger.info(f"Health check requested from {request.client.host}")
    try:
        # Check DynamoDB connection
        table.meta.client.describe_table(TableName='TradingPortfolio')
        logger.info("Successfully connected to DynamoDB")
        return {"status": "healthy", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.post("/signal")
async def process_signal(file: UploadFile = File(None), signal: Signal | None = None):
    """Process trading signals from CSV file or single signal"""
    try:
        if file is not None:
            # Process CSV file
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            df = df.sort_values(by='datetime', ascending=True)
            
            # Validate required columns
            required_columns = ['datetime', 'close', 'Signal']
            if not all(col in df.columns for col in required_columns):
                raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_columns}")
            
            # Process each row in the CSV
            for _, row in df.iterrows():
                signal_data = { 
                    "datetime": str(row['datetime']),
                    "close": float(row['close']),
                    "signal": int(row['Signal'])  # Note: 'Signal' is capitalized in the CSV
                }
                process_trading_signal(signal_data)
                save_to_dynamodb(portfolio)
            return {
                "message": f"Processed {len(df)} signals successfully",
                "current_position": portfolio["current_position"],
                "portfolio_value": portfolio["portfolio_value"],
                "cumulative_return": portfolio["total_return"],
                "number_of_trades": portfolio["number_of_trades"],
                "winning_trades": portfolio["winning_trades"],
                "cash_holdings": portfolio["cash_holdings"],
                "bitcoin_holdings": portfolio["bitcoin_holdings"]
            }
        
        elif signal is not None:
            try:
                # Process single signal
                signal_datetime = signal.get_datetime()
                logger.info(f"Processing signal: {signal.signal} at {signal_datetime}")
                
                if signal.signal not in [-1, 0, 1]:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid signal value. Must be -1, 0, or 1"
                    )

                signal_dict = signal.model_dump()
                signal_dict['datetime'] = signal_datetime
                
                try:
                    process_trading_signal(signal_dict)
                    save_to_dynamodb(portfolio)
                except Exception as e:
                    logger.error(f"Error processing signal: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing signal: {str(e)}"
                    )
                
                return {
                    "message": "Signal processed successfully",
                    "current_position": portfolio["current_position"],
                    "portfolio_value": portfolio["portfolio_value"]
                }
            except ValueError as e:
                logger.error(f"Validation error: {str(e)}")
                raise HTTPException(
                    status_code=422,
                    detail=str(e)
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error: {str(e)}"
                )
            
        raise HTTPException(status_code=400, detail="Either a CSV file or a single signal must be provided")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing signal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance")
async def get_performance():
    """Get current portfolio performance"""
    try:
        # Try to load latest data from DynamoDB
        latest_data = load_from_dynamodb()
        if latest_data:
            portfolio.update(latest_data)
        
        # Calculate win rate (percentage of winning trades)
        win_rate = (portfolio["winning_trades"] / portfolio["number_of_trades"]) * 100 if portfolio["number_of_trades"] > 0 else 0

        return Performance(
            cumulative_return=portfolio["total_return"],
            current_position=portfolio["current_position"],
            portfolio_value=portfolio["portfolio_value"],
            number_of_trades=portfolio["number_of_trades"],
            win_rate=win_rate
        )
    except Exception as e:
        logger.error(f"Error getting performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def process_trading_signal(signal: dict):
    """Process trading signal and update portfolio"""
    global portfolio
    
    try:
        # Convert signal values to Decimal
        close_price = Decimal(str(signal["close"]))
        
        # Handle BUY signal
        if signal["signal"] == 1:
            if portfolio["cash_holdings"] > 0:                  
                # Calculate Bitcoin amount to buy with certain percentage of all available cash
                cash_to_use = Decimal(str(PERCENTAGE_OF_CASH_TO_BUY)) * Decimal(str(portfolio["cash_holdings"]))
                bitcoin_amount = cash_to_use / close_price
                
                portfolio["bitcoin_holdings"] = float(bitcoin_amount)
                portfolio["cash_holdings"] = float(Decimal(str(portfolio["cash_holdings"])) - cash_to_use)
                portfolio["current_position"] = "BUY"
                portfolio["number_of_trades"] += 1
                
                portfolio["trades"].append({
                    "type": "BUY",
                    "price": float(close_price),
                    "amount": float(bitcoin_amount),
                    "timestamp": signal["datetime"],
                    "return": 0.0  # No return for buy trades
                })
                logger.info(f"Executed BUY order for {bitcoin_amount} BTC at {close_price}")
            else:
                logger.info("No cash holdings to buy")
                portfolio["trades"].append({
                    "type": "BUY",
                    "price": None,
                    "amount": None,
                    "timestamp": signal["datetime"],
                    "reason": "No cash holdings to buy",
                    "return": 0.0
                })
        
        # Handle SELL signal
        elif signal["signal"] == -1:
            if portfolio["bitcoin_holdings"] > 0:
                # Sell all Bitcoin holdings
                cash_amount = Decimal(str(portfolio["bitcoin_holdings"])) * close_price
                portfolio["cash_holdings"] = float(Decimal(str(portfolio["cash_holdings"])) + cash_amount)
                portfolio["bitcoin_holdings"] = 0.0
                portfolio["current_position"] = "SELL"
                portfolio["number_of_trades"] += 1
                
                # Calculate trade return
                last_trade = portfolio["trades"][-1] if portfolio["trades"] else None
                trade_return = 0.0
                if last_trade and last_trade["type"] == "BUY" and last_trade["price"] is not None:
                    buy_price = Decimal(str(last_trade["price"]))
                    trade_return = float(((close_price - buy_price) / buy_price) * Decimal('100'))
                    if trade_return > 0:
                        portfolio["winning_trades"] += 1
                
                portfolio["trades"].append({
                    "type": "SELL",
                    "price": float(close_price),
                    "amount": float(cash_amount),
                    "timestamp": signal["datetime"],
                    "return": trade_return
                })
                logger.info(f"Executed SELL order for {cash_amount} USD at {close_price} with return: {trade_return}%")
            else:
                portfolio["trades"].append({
                    "type": "SELL",
                    "price": None,
                    "amount": None,
                    "timestamp": signal["datetime"],
                    "reason": "No Bitcoin holdings to sell",
                    "return": 0.0
                })
                logger.info("No Bitcoin holdings to sell")
        
        # Update portfolio value and calculate returns
        initial_value = Decimal(str(portfolio["portfolio_value"]))
        portfolio["portfolio_value"] = float(Decimal(str(portfolio["cash_holdings"])) + (Decimal(str(portfolio["bitcoin_holdings"])) * close_price))
        portfolio["total_return"] = float(((Decimal(str(portfolio["portfolio_value"])) - initial_value) / initial_value) * Decimal('100'))
        
    except Exception as e:
        logger.error(f"Error in process_trading_signal: {str(e)}")
        raise

def get_signal_by_timestamp(timestamp: str):
    """Get signal data from DynamoDB by timestamp"""
    try:
        logger.info(f"Querying DynamoDB for timestamp: {timestamp}")
        response = table.query(
            KeyConditionExpression='#ts = :ts',
            ExpressionAttributeNames={
                '#ts': 'timestamp'
            },
            ExpressionAttributeValues={
                ':ts': timestamp
            }
        )
        
        if response['Items']:
            logger.info("Signal found in DynamoDB")
            item = response['Items'][0]
            return {
                "current_position": item['current_position'],
                "portfolio_value": float(item['portfolio_value']),
                "bitcoin_holdings": float(item['bitcoin_holdings']),
                "cash_holdings": float(item['cash_holdings']),
                "total_return": float(item['total_return']),
                "number_of_trades": int(item['number_of_trades']),
                "winning_trades": int(item['winning_trades']),
                "trade": json.loads(item['data']) if 'data' in item else None
            }
        logger.info("No signal found in DynamoDB")
        return None
    except Exception as e:
        logger.error(f"Error querying DynamoDB: {str(e)}")
        return None

@app.get("/signal/{timestamp}")
async def get_signal(timestamp: str, request: Request):
    """Get signal data for a specific timestamp
    Args:
        timestamp: Date in ISO format (e.g., "2025-02-24T18:00:00")
    Example:
        GET /signal/2025-02-24T18:00:00
        
        Note: If you have a date in format "2/24/2025 18:00", you should use the following format instead:
        "2025-02-24T18:00:00".
    """
    try:
        logger.info(f"Signal request from {request.client.host} for timestamp: {timestamp}")
        # Decode URL-encoded timestamp
        decoded_timestamp =  unquote(timestamp)
        
        # Get signal data
        signal_data = get_signal_by_timestamp(decoded_timestamp)
        
        if signal_data:
            return signal_data
        else:
            raise HTTPException(status_code=404, detail="Signal not found for the given timestamp")
    except Exception as e:
        logger.error(f"Error getting signal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 
    