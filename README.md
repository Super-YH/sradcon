# Sradcon

[![CI Build](https://github.com/yourusername/sradcon/workflows/CI%20Build%20and%20Test/badge.svg)](https://github.com/yourusername/sradcon/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourusername/sradcon/releases)

## 概要 / Overview

**日本語:**

Sradconは、Saracon並みの音質を実現するプロフェッショナルグレードの音声リサンプラー・再量子化ツールです。高度な心理音響モデルに基づいた処理により、商用ソフトウェアに匹敵する品質を無料で提供します。

**English:**

Sradcon is a professional-grade audio resampler and requantizer that achieves audio quality comparable to Saracon. Through advanced psychoacoustic model-based processing, it provides quality rivaling commercial software, completely free.

## 主な特徴 / Key Features

### 🎵 高品質なリサンプリング / High-Quality Resampling

- メル尺度ベースの128バンド周波数分析
- 線形位相・最小位相フィルタの適応的選択
- 心理音響モデルに基づいた最適化

### 🔊 高度なノイズシェーピング / Advanced Noise Shaping

- RDO (Rate-Distortion Optimized) ノイズシェーピング
- 適応的LPCフィルタ
- 周波数マスキングを考慮したディザリング

### ⚡ 高周波再構築 / High-Frequency Enhancement

- アップサンプリング時の高周波成分合成
- 複数の倍音成分の転写
- Griffin-Limアルゴリズムによる位相復元

### 🚀 高性能 / High Performance

- マルチスレッド対応
- FFTW3による高速FFT
- Eigenライブラリによる最適化された行列演算

### 🛠️ 商用レベルの機能 / Commercial-Grade Features

- 包括的なエラーハンドリング
- 詳細なロギングシステム
- パフォーマンスプロファイリング
- 設定ファイル対応
- 完全なドキュメント

## クイックスタート / Quick Start

### インストール / Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sradcon.git
cd sradcon

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)

# Install (optional)
sudo cmake --install .
```

### 基本的な使い方 / Basic Usage

```bash
# Simple resampling
sradcon -i input.wav -o output.wav --target_sr 44100

# High-quality upsampling
sradcon -i input_44k.wav -o output_192k.wav --target_sr 192000 --chunk_size 16384

# Custom bit depth
sradcon -i input.wav -o output.wav --target_sr 48000 --target_bit_depth 24

# With configuration file
sradcon -i input.wav -o output.wav -c sradcon.conf

# Performance monitoring
sradcon -i input.wav -o output.wav --perf
```

## 必要要件 / Requirements

### ビルド環境 / Build Environment

- CMake 3.16 or later
- C++17 compatible compiler:
  - GCC 7.0 or later
  - Clang 5.0 or later
  - MSVC 2017 or later

### 依存ライブラリ / Dependencies

- libsndfile (audio I/O)
- FFTW3 (Fast Fourier Transform)
- Eigen3 (linear algebra)
- cxxopts (command-line parsing, included)
- Catch2 (testing, optional)

### Ubuntu/Debian

```bash
sudo apt-get install \
    cmake \
    build-essential \
    libsndfile1-dev \
    libfftw3-dev \
    libeigen3-dev
```

### macOS

```bash
brew install cmake libsndfile fftw eigen
```

### Windows

Use vcpkg:

```powershell
vcpkg install libsndfile fftw3 eigen3
```

## ドキュメント / Documentation

- [User Guide (日本語/English)](docs/USER_GUIDE.md)
- [API Documentation](docs/API.md)
- [Building from Source](docs/BUILD.md)

## プロジェクト構造 / Project Structure

```
sradcon/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── LICENSE                 # MIT License
├── include/                # Public headers
│   └── sradcon/
│       ├── version.hpp     # Version information
│       ├── logger.hpp      # Logging system
│       ├── exception.hpp   # Exception classes
│       ├── config.hpp      # Configuration management
│       └── performance.hpp # Performance monitoring
├── src/                    # Source files
│   ├── sradcon.cpp         # Main entry point
│   └── sradcon_legacy.cpp  # Core DSP implementation
├── tests/                  # Unit tests
│   └── test_dsp_utils.cpp
├── docs/                   # Documentation
│   ├── USER_GUIDE.md
│   └── API.md
├── examples/               # Example programs
├── cmake/                  # CMake modules
└── .github/                # CI/CD workflows
    └── workflows/
        └── ci.yml
```

## パフォーマンス / Performance

Benchmark on Intel Core i7-9700K @ 3.6GHz:

| Operation | File Size | Processing Time | Real-time Factor |
|-----------|-----------|-----------------|------------------|
| 96kHz → 44.1kHz | 5 min stereo | 15.2 sec | 19.7x |
| 44.1kHz → 192kHz | 5 min stereo | 42.8 sec | 7.0x |
| Dithering 24→16bit | 5 min stereo | 8.1 sec | 37.0x |

## 開発 / Development

### ビルドオプション / Build Options

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DBUILD_EXAMPLES=ON \
  -DENABLE_SANITIZERS=OFF \
  -DENABLE_LTO=ON
```

### テストの実行 / Running Tests

```bash
cd build
ctest --output-on-failure
```

### デバッグモード / Debug Mode

```bash
sradcon -i input.wav -o output.wav --debug --log debug.log
```

## 貢献 / Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ライセンス / License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 謝辞 / Acknowledgments

- This project utilizes LLM technology for development assistance
- FFTW library by Matteo Frigo and Steven G. Johnson
- Eigen library by Benoît Jacob and Gaël Guennebaud
- libsndfile by Erik de Castro Lopo

## リンク / Links

- [GitHub Repository](https://github.com/yourusername/sradcon)
- [Issue Tracker](https://github.com/yourusername/sradcon/issues)
- [Documentation](https://github.com/yourusername/sradcon/docs)

## サポート / Support

If you encounter any problems or have questions:

1. Check the [User Guide](docs/USER_GUIDE.md)
2. Search [existing issues](https://github.com/yourusername/sradcon/issues)
3. Create a [new issue](https://github.com/yourusername/sradcon/issues/new) with:
   - Your operating system and version
   - Sradcon version (`sradcon --version`)
   - Complete command line
   - Error messages and logs

---

**Made with ❤️ for audio enthusiasts**
