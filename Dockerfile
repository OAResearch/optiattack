FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY core ./core
COPY main.py .
COPY gradio_ui.py .
RUN mkdir ".cache"

EXPOSE 13000

# Command to run the application
ENTRYPOINT ["python",  "-u", "main.py"]