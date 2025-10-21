# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-21

### Added

#### Core Features
- Professional-grade audio resampling engine with psychoacoustic optimization
- Intelligent mixed-phase filtering (adaptive linear/minimum phase selection)
- 128-band mel-scale frequency analysis
- RDO (Rate-Distortion Optimized) noise shaping
- High-frequency enhancement for upsampling
- Mid/Side stereo processing

#### Commercial-Grade Infrastructure
- Comprehensive error handling system with custom exception classes
- Thread-safe logging system with multiple log levels
- Performance monitoring and profiling capabilities
- Configuration file support (key=value format)
- Multi-threaded processing with FFTW3
- Input validation and parameter checking

#### Build System & Development
- Modern CMake build system (3.16+)
- Support for GCC, Clang, and MSVC compilers
- Optional LTO/IPO optimization
- Address and undefined behavior sanitizers support
- Unit testing framework with Catch2
- Comprehensive CI/CD pipeline (GitHub Actions)
- Cross-platform support (Linux, macOS, Windows)

#### Documentation
- Complete user guide (English/Japanese)
- API documentation
- Inline code documentation
- Usage examples
- Performance benchmarks

#### Quality Assurance
- Unit tests for core DSP functions
- Continuous integration on multiple platforms
- Static analysis with cppcheck and clang-tidy
- Code formatting checks with clang-format

### Technical Details

#### Dependencies
- libsndfile: Audio file I/O
- FFTW3: High-performance FFT
- Eigen3: Linear algebra operations
- cxxopts: Command-line parsing (bundled)
- Catch2: Unit testing (optional)

#### Algorithms
- Levinson-Durbin algorithm for LPC coefficient calculation
- Griffin-Lim algorithm for phase reconstruction
- ATH (Absolute Threshold of Hearing) modeling
- Temporal masking and spreading function
- Spectral flatness measure for tonality detection

### Performance
- Multi-core processing support
- Optimized memory management with FFTW memory alignment
- SIMD-friendly operations with Eigen
- Zero-copy audio buffer management where possible

### Notes
- This project utilizes LLM technology for development assistance
- Designed for professional audio production workflows
- Extensive testing with various audio formats and sample rates

## [Unreleased]

### Planned Features
- JSON/YAML configuration file support
- Additional window functions (Blackman, Kaiser, etc.)
- Batch processing mode
- GUI frontend
- VST3/AU plugin version
- Additional audio format support (FLAC, MP3, etc.)
- Preset management system
- Real-time processing mode

---

[1.0.0]: https://github.com/yourusername/sradcon/releases/tag/v1.0.0
