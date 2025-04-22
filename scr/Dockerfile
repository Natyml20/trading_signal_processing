# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port 80
EXPOSE 80

# Set environment variables
ENV PORT=80
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Create a script to run the application
RUN echo '#!/bin/sh\nuvicorn main:app --host 0.0.0.0 --port 80 --log-level info' > /app/run.sh && \
    chmod +x /app/run.sh

# Command to run the application
CMD ["/app/run.sh"]