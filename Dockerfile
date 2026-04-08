FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency manifest first (layer cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]