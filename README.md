# SLAM Visual Odometry

Monocular visual odometry in C++ (OpenCV, Eigen, Pangolin) for a SLAM course - feature matching + epipolar geometry + live trajectory viewer.

## Quickstart

### Prerequisites

- CMake 3.16+
- C++17 compatible compiler
- Ninja (recommended, used by the Makefile)
- Docker (optional)

### Building

```bash
make build
```

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

- `make build`: Configure + build locally.
- `make test`: Run tests via CTest from `build/`.
- `make clean`: Remove the `build/` directory.
- `make docker-build`: Build the Docker image.
- `make docker-shell`: Open an interactive container shell.
- `make docker-run`: Run a container with `./data` and `./results` mounted to `/workspace`.
- `make docker-run-gui`: Run with Xvfb + openbox + x11vnc + noVNC for GUI output.

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

## GUI Workflow (Docker)

`make docker-run-gui` starts Xvfb + openbox + x11vnc + noVNC in the container. Open
`http://localhost:6080/vnc.html` in a browser; Pangolin/OpenCV windows will appear there.

## Dataset Notes

Place `Dataset_VO.tar` in `data/` (gitignored). The directory is mounted to
`/workspace/data` when using Docker.

```bash
mkdir -p data/sequence
tar -xf data/Dataset_VO.tar -C data/sequence
```

## Development

This is a minimal scaffold. Source code will be added incrementally.

## License

See LICENSE file for details.
