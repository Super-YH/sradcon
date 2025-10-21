# Contributing to Sradcon

Thank you for your interest in contributing to Sradcon! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected behavior**
- **Actual behavior**
- **System information**: OS, compiler version, Sradcon version
- **Log files** if applicable (use `--debug --log debug.log`)
- **Audio file characteristics** if relevant: sample rate, bit depth, channels

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **Clear use case** for the enhancement
- **Expected benefits**
- **Potential implementation approach** (if you have ideas)
- **Impact on existing functionality**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding style** (see below)
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass**
6. **Write a clear commit message**

## Development Setup

### Prerequisites

Install development dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install cmake build-essential libsndfile1-dev libfftw3-dev \
    libeigen3-dev catch2 clang-format clang-tidy cppcheck

# macOS
brew install cmake libsndfile fftw eigen catch2 clang-format
```

### Building

```bash
git clone https://github.com/yourusername/sradcon.git
cd sradcon
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build . -j$(nproc)
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

## Coding Guidelines

### C++ Style

- **C++ Standard**: C++17
- **Naming Conventions**:
  - Classes: `PascalCase` (e.g., `RDONoiseShaper`)
  - Functions: `camelCase` (e.g., `processChannel`)
  - Variables: `snake_case` (e.g., `sample_rate`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_BUFFER_SIZE`)
  - Private members: suffix with `_` (e.g., `data_`)

### Code Formatting

Use clang-format for automatic formatting:

```bash
find src include -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
```

### Documentation

- Use Doxygen-style comments for public APIs
- Document complex algorithms with inline comments
- Update docs/ when adding features

Example:

```cpp
/**
 * @brief Perform resampling with psychoacoustic optimization
 * @param signal Input audio signal (channels x samples)
 * @param is_ms_processing Enable Mid/Side processing for stereo
 * @param disable_hfe Disable high-frequency enhancement
 * @return Resampled audio signal
 * @throws ProcessingException if processing fails
 */
MatrixF resample(const MatrixF& signal, bool is_ms_processing, bool disable_hfe);
```

### Error Handling

- Use custom exception classes from `sradcon/exception.hpp`
- Validate inputs and throw appropriate exceptions
- Use RAII for resource management
- Log errors with appropriate log levels

### Performance Considerations

- Profile code before optimizing
- Use `PERF_TIMER` macro for performance-critical sections
- Prefer Eigen operations over manual loops
- Consider SIMD-friendly algorithms
- Avoid unnecessary memory allocations

### Testing

- Write unit tests for new functions
- Test edge cases and error conditions
- Use descriptive test names
- Group related tests in sections

Example:

```cpp
TEST_CASE("DSP Utilities - Mel scale conversions", "[dsp][mel]") {
    SECTION("Hz to Mel conversion") {
        REQUIRE(DspUtils::hz_to_mel(0.0f) == Approx(0.0f));
        REQUIRE(DspUtils::hz_to_mel(1000.0f) > 0.0f);
    }
}
```

## Project Structure

```
sradcon/
├── include/sradcon/    # Public headers
├── src/                # Implementation files
├── tests/              # Unit tests
├── docs/               # Documentation
├── examples/           # Example programs
├── cmake/              # CMake modules
└── .github/workflows/  # CI/CD configuration
```

## Commit Messages

Follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system changes
- `ci`: CI/CD changes

Example:

```
feat(resampler): add Kaiser window support

Implement Kaiser window function for improved filtering flexibility.
The beta parameter can be configured via the config file.

Closes #123
```

## Release Process

1. Update version in `CMakeLists.txt` and `include/sradcon/version.hpp`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag -a v1.0.0 -m "Release 1.0.0"`
4. Push tag: `git push origin v1.0.0`
5. Create GitHub release with changelog

## Questions?

If you have questions about contributing, please:

1. Check existing documentation
2. Search closed issues
3. Open a new issue with the `question` label

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
