# SLAM Visual Odometry - Makefile

.PHONY: build test clean docker-build docker-shell docker-run help

# Default target
all: build

# Build the project
build:
	@echo "Building project..."
	@mkdir -p build
	@cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$$(nproc)

# Run tests
test:
	@echo "Running tests..."
	@cd build && ctest --output-on-failure

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	@docker build -t slam-vo:latest .

# Open interactive shell in Docker container
docker-shell:
	@echo "Opening Docker shell..."
	@docker run -it --rm \
		-v $$(pwd):/workspace \
		-v $$(pwd)/data:/workspace/data \
		slam-vo:latest /bin/bash

# Run Docker container with data mount
docker-run:
	@echo "Running Docker container..."
	@docker run -it --rm \
		-v $$(pwd):/workspace \
		-v $$(pwd)/data:/workspace/data \
		slam-vo:latest

# Display help
help:
	@echo "SLAM Visual Odometry - Available targets:"
	@echo "  make build         - Build the project"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-shell  - Open interactive Docker shell"
	@echo "  make docker-run    - Run Docker container with data mount"
	@echo "  make help          - Display this help message"
