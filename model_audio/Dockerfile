# Stage 1: Build stage
FROM python:3.11.9-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA repository keyring and CUDA toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Stage 2: Final stage
FROM python:3.11.9-slim

# Copy only the necessary files from the builder stage
COPY --from=builder /usr/local /usr/local

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . .

# Set the entrypoint
# ENTRYPOINT ["python3"]

# Commands to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
EXPOSE 8001