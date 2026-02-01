# SLAM Visual Odometry

Monocular visual odometry in C++ (OpenCV, Eigen, Pangolin) for a SLAM course - feature matching + epipolar geometry + live trajectory viewer.

## Quickstart

### Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+)
- Docker (optional)

### Building

```bash
# Build using Makefile
make build

# Or manually with CMake
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Running Tests

```bash
make test
```

### Docker

```bash
# Build Docker image
make docker-build

# Open interactive shell
make docker-shell

# Run container with data mount
make docker-run
```

### Clean

```bash
make clean
```

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

Place your SLAM/VO datasets in the `data/` directory. This directory is ignored by git and mounted when using Docker.

Supported dataset formats:
- KITTI Odometry Dataset
- TUM RGB-D Dataset
- EuRoC MAV Dataset

### Example Dataset Structure

```
data/
├── kitti/
│   ├── sequences/
│   │   ├── 00/
│   │   ├── 01/
│   │   └── ...
│   └── poses/
└── tum/
    └── rgbd_dataset_freiburg1_xyz/
```

## Development

This is a minimal scaffold. Source code will be added incrementally.

## License

See LICENSE file for details.
