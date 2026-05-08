# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]