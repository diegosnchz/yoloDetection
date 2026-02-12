# Use an official Python runtime as a parent image, pinning to 3.10 for best compatibility
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libusb-1.0-0 \
    libgtk-3-0 \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Set environment variables for GUI (optional, requires host config)
ENV DISPLAY=host.docker.internal:0.0

# Expose port for Streamlit
EXPOSE 8501

# Run app_dashboard.py when the container launches
CMD ["streamlit", "run", "app_dashboard.py", "--server.address=0.0.0.0"]
