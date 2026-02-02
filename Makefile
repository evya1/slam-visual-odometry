# ---- Config ----
IMAGE      ?= slam-vo
WORKDIR    ?= /workspace
BUILD_DIR  ?= build
GENERATOR  ?= Ninja
BUILD_TYPE ?= RelWithDebInfo

# ---- Helpers ----
.PHONY: help build configure test clean docker-build docker-shell docker-run

help:
	@echo "Targets:"
	@echo "  build         Configure + build locally"
	@echo "  configure     Configure locally (no build)"
	@echo "  test          Run tests (CTest) from $(BUILD_DIR)"
	@echo "  clean         Remove build directory"
	@echo "  docker-build  Build Docker image $(IMAGE)"
	@echo "  docker-shell  Open an interactive shell in the container"
	@echo "  docker-run    Run container with ./data and ./results mounted"

# ---- Local build ----
configure:
	cmake -S . -B $(BUILD_DIR) -G "$(GENERATOR)" -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR) -j

test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf $(BUILD_DIR)

# ---- Docker ----
docker-build:
	docker build -t $(IMAGE) .

docker-shell:
	docker run -it --rm \
	  -v "$(PWD)":$(WORKDIR) -w $(WORKDIR) \
	  $(IMAGE) bash

docker-run:
	@mkdir -p data results
	docker run -it --rm \
	  -v "$(PWD)":$(WORKDIR) -w $(WORKDIR) \
	  -v "$(PWD)/data":$(WORKDIR)/data \
	  -v "$(PWD)/results":$(WORKDIR)/results \
	  $(IMAGE) bash

docker-run-gui:
	@mkdir -p data results
	docker run -it --rm \
	  -p 6080:6080 \
	  -v "$(PWD)":$(WORKDIR) -w $(WORKDIR) \
	  -v "$(PWD)/data":$(WORKDIR)/data \
	  -v "$(PWD)/results":$(WORKDIR)/results \
	  $(IMAGE) bash

