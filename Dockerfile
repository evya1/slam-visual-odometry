FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV dependencies (placeholder for future use)
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace

# Default command
CMD ["/bin/bash"]
