FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Toolchain + OpenCV/Eigen + OpenGL/X11 deps (for Pangolin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config ninja-build \
    ca-certificates \
    libopencv-dev libopencv-contrib-dev \
    libeigen3-dev \
    libgl1-mesa-dev libglu1-mesa-dev libglew-dev \
    libx11-dev libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev libxxf86vm-dev \
    libgl1-mesa-dri mesa-utils \
    xvfb x11vnc openbox novnc websockify \
 && rm -rf /var/lib/apt/lists/*

# Pangolin
ARG PANGOLIN_COMMIT=master
RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git /opt/Pangolin \
 && cd /opt/Pangolin \
 && git checkout "${PANGOLIN_COMMIT}" \
 && git submodule update --init --recursive \
 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build \
 && cmake --install build \
 && ldconfig

RUN mkdir -p /workspace/data /workspace/results

# Configure early for caching (placeholder-safe)
COPY CMakeLists.txt /workspace/
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo

COPY . /workspace
RUN cmake --build build || true

ENV DISPLAY=:1
ENV LIBGL_ALWAYS_SOFTWARE=1
EXPOSE 6080

RUN cat >/usr/local/bin/start-novnc <<'EOF'\n\
#!/usr/bin/env bash\n\
set -e\n\
: "${VNC_RESOLUTION:=1280x720x24}"\n\
Xvfb :1 -screen 0 "${VNC_RESOLUTION}" -ac +extension GLX +render -noreset &\n\
export DISPLAY=:1\n\
openbox-session &\n\
x11vnc -display :1 -forever -shared -nopw -rfbport 5901 -bg\n\
websockify --web=/usr/share/novnc/ 6080 localhost:5901 &\n\
echo "noVNC: http://localhost:6080/vnc.html"\n\
exec \"$@\"\n\
EOF\n\
 && chmod +x /usr/local/bin/start-novnc

CMD ["bash"]
