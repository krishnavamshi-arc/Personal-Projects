# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by some packages
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Create uploads folder
RUN mkdir -p uploads

# Expose port
EXPOSE 8080

# Environment variable for Flask
ENV PORT=8080
ENV FLASK_ENV=production

# Run the app
CMD ["python", "app.py"]
