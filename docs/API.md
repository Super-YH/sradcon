# Sradcon API Documentation

## Core Classes

### Logger

Thread-safe logging system with multiple log levels.

```cpp
#include "sradcon/logger.hpp"

// Set log level
Sradcon::Logger::Instance().SetLevel(Sradcon::LogLevel::DEBUG);

// Enable console output
Sradcon::Logger::Instance().EnableConsole(true);

// Set log file
Sradcon::Logger::Instance().SetLogFile("sradcon.log");

// Log messages
LOG_DEBUG("Debug message");
LOG_INFO("Info message");
LOG_WARNING("Warning message");
LOG_ERROR("Error message");
LOG_CRITICAL("Critical error");
```

### Config

Configuration management with validation.

```cpp
#include "sradcon/config.hpp"

// Create default configuration
Sradcon::Config config;

// Load from file
config = Sradcon::Config::LoadFromFile("config.conf");

// Set values programmatically
config.resampler.target_sr = 48000;
config.resampler.chunk_size = 8192;
config.dither.target_bit_depth = 24;

// Validate configuration
config.Validate(); // Throws exception if invalid
```

### PerformanceMonitor

Performance profiling and monitoring.

```cpp
#include "sradcon/performance.hpp"

// Automatic timing with RAII
{
    PERF_TIMER("my_function");
    // Code to measure
}

// Manual timing
Sradcon::PerformanceMonitor::Instance().RecordTiming("operation", 1234);

// Print performance report
Sradcon::PerformanceMonitor::Instance().PrintReport();

// Get specific timing
long long total_us = Sradcon::PerformanceMonitor::Instance()
    .GetTotalMicroseconds("operation");
```

### Exception Classes

Custom exception hierarchy for error handling.

```cpp
#include "sradcon/exception.hpp"

try {
    // Operations
} catch (const Sradcon::FileIOException& e) {
    // Handle file I/O errors
} catch (const Sradcon::AudioFormatException& e) {
    // Handle audio format errors
} catch (const Sradcon::InvalidParameterException& e) {
    // Handle invalid parameters
} catch (const Sradcon::ProcessingException& e) {
    // Handle processing errors
} catch (const Sradcon::MemoryException& e) {
    // Handle memory errors
} catch (const Sradcon::SradconException& e) {
    // Handle any Sradcon exception
}
```

## DSP Utilities

### FFT Operations

```cpp
#include "sradcon/dsp_utils.hpp"

// Enable FFTW threading
DspUtils::FFTWManager::Instance().EnableThreads(4);

// Real FFT
DspUtils::VectorF signal(1024);
// Fill signal...
DspUtils::VectorCF spectrum = DspUtils::rfft(signal, 1024);

// Inverse Real FFT
DspUtils::VectorF reconstructed = DspUtils::irfft(spectrum, 1024);
```

### STFT/ISTFT

```cpp
// Short-Time Fourier Transform
DspUtils::VectorF signal(44100);
int n_fft = 2048;
int hop_length = 512;
DspUtils::VectorF window = DspUtils::getWindow("hann", n_fft);

Eigen::MatrixXf magnitude;
Eigen::MatrixXcf phase;
DspUtils::stft(signal, n_fft, hop_length, window, magnitude, phase);

// Inverse STFT
DspUtils::VectorF reconstructed = DspUtils::istft(magnitude, phase, hop_length, window);
```

### Filter Design

```cpp
// FIR lowpass filter
float cutoff_hz = 8000.0f;
float sample_rate = 44100.0f;
int num_taps = 101;
DspUtils::VectorF filter_taps = DspUtils::firwin(num_taps, cutoff_hz, sample_rate);

// Apply filter
DspUtils::VectorF filtered = DspUtils::lfilter(filter_taps,
    DspUtils::VectorF::Ones(1), signal);
```

### Psychoacoustic Analysis

```cpp
AdvancedPsychoacousticModel model(sample_rate, fft_size, num_bands);
auto result = model.analyze_chunk(audio_chunk);

// Access analysis results
const auto& power_db = result.power_db;
const auto& masking_threshold = result.masking_threshold_db;
const auto& tonality = result.tonality;
const auto& power_spectrum = result.power_spectrum;
```

## Processing Classes

### IntelligentRDOResampler

```cpp
IntelligentRDOResampler resampler(
    original_sr,    // Original sample rate
    target_sr,      // Target sample rate
    chunk_size,     // Processing chunk size
    filter_taps,    // Filter length
    num_bands       // Number of frequency bands
);

// Resample audio
DspUtils::MatrixF output = resampler.resample(
    input_audio,           // Input audio matrix
    is_ms_processing,      // Use Mid/Side processing
    disable_hfe            // Disable high-frequency enhancement
);
```

### RDONoiseShaper

```cpp
RDONoiseShaper shaper(
    target_bit_depth,  // Target bit depth
    sample_rate,       // Sample rate
    chunk_size,        // Processing chunk size
    lpc_order          // LPC filter order
);

// Apply noise shaping
DspUtils::MatrixF shaped = shaper.process(audio_data);
```

### HighFrequencySynthesizer

```cpp
HighFrequencySynthesizer synthesizer(
    original_sr,    // Original sample rate
    target_sr,      // Target sample rate
    n_fft,          // FFT size
    hop_length,     // Hop length
    iterations      // Griffin-Lim iterations
);

// Synthesize high frequencies
DspUtils::VectorF enhanced = synthesizer.synthesize(
    audio_channel,
    channel_name
);
```

## Utility Functions

### Mel Scale Conversion

```cpp
float hz = 1000.0f;
float mel = DspUtils::hz_to_mel(hz);
float hz_back = DspUtils::mel_to_hz(mel);
```

### Linspace

```cpp
DspUtils::VectorF values = DspUtils::linspace(0.0f, 1.0f, 100);
```

### Convolution

```cpp
DspUtils::VectorF result = DspUtils::convolve(signal, kernel, "same");
// Modes: "same", "full"
```

### Toeplitz Solver

```cpp
DspUtils::VectorF r(order + 1);  // Autocorrelation
DspUtils::VectorF a(order);       // LPC coefficients
DspUtils::solve_toeplitz(r, a);
```

## Thread Safety

The following components are thread-safe:

- `Logger`
- `PerformanceMonitor`
- `FFTWManager` (when properly initialized)

The processing classes (`IntelligentRDOResampler`, `RDONoiseShaper`, `HighFrequencySynthesizer`)
are NOT thread-safe and should not be shared between threads. Create separate instances for
each thread.

## Error Handling Best Practices

```cpp
try {
    // Initialize
    Sradcon::Config config = Sradcon::Config::LoadFromFile("config.conf");
    config.Validate();

    // Process
    IntelligentRDOResampler resampler(...);
    auto output = resampler.resample(input, true, false);

} catch (const Sradcon::InvalidParameterException& e) {
    LOG_ERROR(std::string("Invalid parameter: ") + e.what());
    return ERROR_INVALID_PARAM;
} catch (const Sradcon::FileIOException& e) {
    LOG_ERROR(std::string("File I/O error: ") + e.what());
    return ERROR_FILE_IO;
} catch (const Sradcon::ProcessingException& e) {
    LOG_ERROR(std::string("Processing error: ") + e.what());
    return ERROR_PROCESSING;
} catch (const std::exception& e) {
    LOG_CRITICAL(std::string("Unexpected error: ") + e.what());
    return ERROR_UNKNOWN;
}
```

## Version Information

```cpp
#include "sradcon/version.hpp"

// Get version string
const char* version = Sradcon::Version::GetVersionString();

// Get build information
const char* build = Sradcon::Version::GetBuildInfo();

// Access version components
int major = Sradcon::Version::Major;
int minor = Sradcon::Version::Minor;
int patch = Sradcon::Version::Patch;
```
