FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Toolchain + OpenCV/Eigen + OpenGL/X11 deps (needed for Pangolin GUI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config ninja-build \
    ca-certificates \
    libopencv-dev libopencv-contrib-dev \
    libeigen3-dev \
    libgl1-mesa-dev libglu1-mesa-dev libglew-dev \
    libx11-dev libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev libxxf86vm-dev \
 && rm -rf /var/lib/apt/lists/*

# Pangolin (optionally pin via PANGOLIN_COMMIT later)
ARG PANGOLIN_COMMIT=master
RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git /opt/Pangolin \
 && cd /opt/Pangolin \
 && git checkout "${PANGOLIN_COMMIT}" \
 && git submodule update --init --recursive \
 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build \
 && cmake --install build \
 && ldconfig

# Expected mount points: dataset in /workspace/data, outputs in /workspace/results
RUN mkdir -p /workspace/data /workspace/results

# Better Docker caching: configure before copying full source
COPY CMakeLists.txt /workspace/
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo

COPY . /workspace
RUN cmake --build build

CMD ["bash"]
