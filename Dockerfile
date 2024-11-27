FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-surprise pydantic

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "recommendation_system:app", "--host", "0.0.0.0", "--port", "8000"]
