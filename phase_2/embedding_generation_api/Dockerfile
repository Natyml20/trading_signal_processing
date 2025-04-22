# Start from a lightweight Python image
FROM python:3.12-slim

# Create a working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI application code into the container
COPY . .

# Expose the port on which your FastAPI app will run
EXPOSE 8000

# Run the server on container startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

