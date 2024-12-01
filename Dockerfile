# Use a compatible Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for scikit-surprise and other Python libraries
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libstdc++6 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files into the container
COPY . /app

# Install Python dependencies with a pinned NumPy version
RUN pip install --no-cache-dir scikit-learn "numpy<2" fastapi uvicorn pandas scikit-surprise pydantic

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "recommendation_system:app", "--host", "0.0.0.0", "--port", "8000"]
