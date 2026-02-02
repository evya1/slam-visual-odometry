# SLAM Visual Odometry

Monocular visual odometry in C++ (OpenCV, Eigen, Pangolin) for a SLAM course - feature matching + epipolar geometry + live trajectory viewer.

## Quickstart

### Prerequisites

- CMake 3.16+
- C++17 compatible compiler (GCC 7+, Clang 5+)
- Ninja (recommended)
- Docker (optional)

### Make Targets

```bash
make build
make test
make clean
make docker-build
make docker-shell
make docker-run
make docker-run-gui
```

### Manual Build (optional)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

### Docker Notes

The `docker-run` and `docker-run-gui` targets mount `./data` and `./results` to `/workspace/data` and `/workspace/results` in the container.

### GUI Workflow

`make docker-run-gui` starts Xvfb + openbox + x11vnc + noVNC. Open http://localhost:6080/vnc.html in your browser; Pangolin/OpenCV windows appear in that VNC session.

## Directory Structure

```
.
├── include/         # Header files
├── src/             # Source files
├── tests/           # Unit tests
├── scripts/         # Utility scripts
├── configs/         # Configuration files
├── data/            # Dataset directory (not tracked)
└── results/         # Output results (not tracked)
```

## Dataset Notes

Place `Dataset_VO.tar` in `data/` (gitignored). When running in Docker, `data/` is mounted to `/workspace/data`. Extract the archive into a subfolder like `data/sequence/`.

```bash
mkdir -p data/sequence
tar -xf data/Dataset_VO.tar -C data/sequence
```

## Development

This is a minimal scaffold. Source code will be added incrementally.

## License

See LICENSE file for details.
