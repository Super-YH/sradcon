#!/bin/bash
# Build script for Sradcon

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build"
INSTALL=false
CLEAN=false
TESTS=ON
EXAMPLES=OFF
SANITIZERS=OFF
LTO=ON
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -i|--install)
            INSTALL=true
            shift
            ;;
        --no-tests)
            TESTS=OFF
            shift
            ;;
        --examples)
            EXAMPLES=ON
            shift
            ;;
        --sanitizers)
            SANITIZERS=ON
            shift
            ;;
        --no-lto)
            LTO=OFF
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Sradcon Build Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --debug         Build in Debug mode (default: Release)"
            echo "  -r, --release       Build in Release mode"
            echo "  -c, --clean         Clean build directory before building"
            echo "  -i, --install       Install after building"
            echo "  --no-tests          Disable building tests"
            echo "  --examples          Enable building examples"
            echo "  --sanitizers        Enable address and UB sanitizers"
            echo "  --no-lto            Disable Link Time Optimization"
            echo "  -j, --jobs N        Use N parallel jobs (default: auto-detect)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Build in Release mode"
            echo "  $0 --debug          # Build in Debug mode"
            echo "  $0 --clean -i       # Clean, build, and install"
            echo "  $0 --sanitizers     # Build with sanitizers for testing"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${GREEN}=== Sradcon Build Configuration ===${NC}"
echo "Build type: $BUILD_TYPE"
echo "Build directory: $BUILD_DIR"
echo "Tests: $TESTS"
echo "Examples: $EXAMPLES"
echo "Sanitizers: $SANITIZERS"
echo "LTO: $LTO"
echo "Parallel jobs: $JOBS"
echo -e "${GREEN}===================================${NC}"
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo -e "${GREEN}Configuring CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_TESTING="$TESTS" \
    -DBUILD_EXAMPLES="$EXAMPLES" \
    -DENABLE_SANITIZERS="$SANITIZERS" \
    -DENABLE_LTO="$LTO"

# Build
echo -e "${GREEN}Building...${NC}"
cmake --build . --config "$BUILD_TYPE" -j"$JOBS"

# Run tests if enabled
if [ "$TESTS" = "ON" ]; then
    echo -e "${GREEN}Running tests...${NC}"
    ctest --output-on-failure -C "$BUILD_TYPE"
fi

# Install if requested
if [ "$INSTALL" = true ]; then
    echo -e "${GREEN}Installing...${NC}"
    sudo cmake --install .
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "Executable location: $BUILD_DIR/sradcon"
echo ""
echo "To run:"
echo "  $BUILD_DIR/sradcon --help"
echo ""
if [ "$INSTALL" = true ]; then
    echo "Installed to system. Run with:"
    echo "  sradcon --help"
fi
