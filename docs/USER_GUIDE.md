# Sradcon User Guide

## Overview

Sradcon is a professional-grade audio resampler and requantizer with advanced psychoacoustic processing capabilities. It provides high-quality audio resampling comparable to commercial solutions like Saracon, but as free and open-source software.

## Features

- **Intelligent Resampling**: Adaptive mixed-phase filtering based on psychoacoustic analysis
- **Advanced Dithering**: RDO (Rate-Distortion Optimized) noise shaping
- **High-Frequency Enhancement**: Intelligent synthesis of high-frequency content during upsampling
- **Multi-threaded Processing**: Efficient use of modern multi-core processors
- **Flexible Configuration**: Command-line arguments and configuration files
- **Comprehensive Logging**: Detailed operation logs and performance monitoring

## Installation

### Prerequisites

- CMake 3.16 or later
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- libsndfile
- FFTW3
- Eigen3

### Building from Source

```bash
git clone https://github.com/yourusername/sradcon.git
cd sradcon
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
sudo cmake --install .
```

## Basic Usage

### Simple Resampling

Convert a 96kHz file to 44.1kHz:

```bash
sradcon -i input.wav -o output.wav --target_sr 44100
```

### Upsampling with HFE

Upsample to 192kHz with high-frequency enhancement:

```bash
sradcon -i input_44k.wav -o output_192k.wav --target_sr 192000
```

### Custom Bit Depth

Resample and requantize to 24-bit:

```bash
sradcon -i input.wav -o output.wav --target_sr 48000 --target_bit_depth 24
```

## Configuration File

Create a configuration file `sradcon.conf`:

```ini
# Resampling settings
target_sr=44100
chunk_size=8192
filter_taps=32767
num_bands=128
disable_hfe=false

# Dithering settings
target_bit_depth=16
lpc_order=16
enable_dither=true
float32_output=false

# Processing settings
enable_ms_processing=true
enable_debug=false
debug_path=debug_plots
num_threads=0
```

Use the configuration file:

```bash
sradcon -i input.wav -o output.wav -c sradcon.conf
```

## Advanced Options

### Performance Tuning

```bash
# Use 8 threads for processing
sradcon -i input.wav -o output.wav --threads 8

# Larger chunk size for better quality (slower)
sradcon -i input.wav -o output.wav --chunk_size 16384

# More filter taps for sharper filtering
sradcon -i input.wav -o output.wav --filter_taps 65535
```

### Debug Mode

Enable debug mode to save intermediate processing data:

```bash
sradcon -i input.wav -o output.wav --debug --log processing.log
```

### Performance Monitoring

View detailed performance statistics:

```bash
sradcon -i input.wav -o output.wav --perf
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input WAV file path | Required |
| `-o, --output` | Output WAV file path | Required |
| `-c, --config` | Configuration file path | None |
| `--target_sr` | Target sample rate (Hz) | 44100 |
| `--chunk_size` | Processing chunk size | 8192 |
| `--filter_taps` | FIR filter length | 32767 |
| `--disable_hfe` | Disable high-frequency enhancement | false |
| `--target_bit_depth` | Target bit depth (8/16/24/32/64) | 16 |
| `--float32_output` | Force 32-bit float output | false |
| `--no_dither` | Disable dithering | false |
| `--lpc_order` | LPC filter order for noise shaping | 16 |
| `--threads` | Number of threads (0=auto) | 0 |
| `--debug` | Enable debug mode | false |
| `--log` | Log file path | None |
| `--perf` | Show performance report | false |
| `-v, --version` | Show version information | - |
| `-h, --help` | Show help message | - |

## Quality Settings

### Maximum Quality

For the highest quality resampling:

```bash
sradcon -i input.wav -o output.wav \
  --target_sr 44100 \
  --chunk_size 16384 \
  --filter_taps 65535 \
  --lpc_order 32 \
  --threads 0
```

### Balanced Quality/Speed

For a good balance between quality and speed:

```bash
sradcon -i input.wav -o output.wav \
  --target_sr 44100 \
  --chunk_size 8192 \
  --filter_taps 16383 \
  --lpc_order 16
```

### Fast Processing

For quick processing with acceptable quality:

```bash
sradcon -i input.wav -o output.wav \
  --target_sr 44100 \
  --chunk_size 4096 \
  --filter_taps 8191 \
  --lpc_order 8
```

## Tips and Best Practices

1. **Sample Rate Conversion**: Use common ratios when possible (e.g., 96kHz → 48kHz) for better results
2. **High-Frequency Enhancement**: Enable for upsampling from significantly lower rates
3. **Dithering**: Always use dithering when reducing bit depth
4. **Threading**: Let the auto-detection choose thread count unless you have specific requirements
5. **Debugging**: Use debug mode only when analyzing problems - it creates large data files

## Troubleshooting

### Out of Memory Errors

Reduce `chunk_size` or `filter_taps`:

```bash
sradcon -i input.wav -o output.wav --chunk_size 4096 --filter_taps 8191
```

### Slow Processing

Reduce quality settings or enable more threads:

```bash
sradcon -i input.wav -o output.wav --chunk_size 4096 --threads 0
```

### Audio Artifacts

Increase quality settings:

```bash
sradcon -i input.wav -o output.wav --chunk_size 16384 --filter_taps 65535
```

## Support

- GitHub Issues: https://github.com/yourusername/sradcon/issues
- Documentation: https://github.com/yourusername/sradcon/docs

## License

Sradcon is released under the MIT License. See LICENSE file for details.
