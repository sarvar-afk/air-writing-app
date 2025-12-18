# Base image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose ports (Azure default ingress)
EXPOSE 8000

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
