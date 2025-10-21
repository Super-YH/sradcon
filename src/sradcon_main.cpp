#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <thread>

// Sradcon headers
#include "sradcon/version.hpp"
#include "sradcon/logger.hpp"
#include "sradcon/exception.hpp"
#include "sradcon/config.hpp"
#include "sradcon/performance.hpp"

// Third-party library includes
#include <sndfile.h>
#include <Eigen/Dense>
#include "cxxopts.hpp"
#include <fftw3.h>

// Global settings
bool DEBUG_MODE = false;
std::filesystem::path DEBUG_PATH;

// ===================================================================================
// Include all DSP code from legacy implementation
// This preserves the original high-quality resampling algorithms
// ===================================================================================

// Copy all the DspUtils namespace and processing classes from sradcon_legacy.cpp
// We'll compile sradcon_legacy.cpp separately and link it

// Forward declarations
namespace DspUtils {
    using VectorF = Eigen::VectorXf;
    using VectorCF = Eigen::VectorXcf;
    using MatrixF = Eigen::MatrixXf;
}

class AudioFile;
class AdvancedPsychoacousticModel;
class RDONoiseShaper;
class HighFrequencySynthesizer;
class IntelligentRDOResampler;

// These are implemented in sradcon_legacy.cpp
extern "C" {
    // We'll create a C-compatible interface
}

// ===================================================================================
// Main execution with commercial-grade infrastructure
// ===================================================================================
int main(int argc, char* argv[]) {
    try {
        // Initialize logging
        Sradcon::Logger::Instance().SetLevel(Sradcon::LogLevel::INFO);
        Sradcon::Logger::Instance().EnableConsole(true);

        LOG_INFO("Sradcon v" + std::string(Sradcon::Version::GetVersionString()));
        LOG_INFO("Build: " + std::string(Sradcon::Version::GetBuildInfo()));

        // Parse command line options
        cxxopts::Options options("Sradcon",
            "Professional audio resampler with advanced psychoacoustic processing");

        options.add_options()
            ("i,input", "Input WAV file path", cxxopts::value<std::string>())
            ("o,output", "Output WAV file path", cxxopts::value<std::string>())
            ("c,config", "Configuration file path", cxxopts::value<std::string>())
            ("target_sr", "Target sample rate in Hz", cxxopts::value<int>()->default_value("44100"))
            ("chunk_size", "Resampling chunk size", cxxopts::value<int>()->default_value("8192"))
            ("filter_taps", "Number of FIR filter taps", cxxopts::value<int>()->default_value("32767"))
            ("disable_hfe", "Disable High-Frequency Excitation", cxxopts::value<bool>()->default_value("false"))
            ("target_bit_depth", "Target bit depth (8/16/24/32/64)", cxxopts::value<int>()->default_value("16"))
            ("float32_output", "Force 32-bit float output", cxxopts::value<bool>()->default_value("false"))
            ("no_dither", "Disable dither and noise shaping", cxxopts::value<bool>()->default_value("false"))
            ("lpc_order", "LPC noise shaping filter order", cxxopts::value<int>()->default_value("16"))
            ("threads", "Number of processing threads (0=auto)", cxxopts::value<int>()->default_value("0"))
            ("debug", "Enable detailed debug mode", cxxopts::value<bool>()->default_value("false"))
            ("log", "Log file path", cxxopts::value<std::string>())
            ("perf", "Print performance report", cxxopts::value<bool>()->default_value("false"))
            ("v,version", "Print version information")
            ("h,help", "Print usage");

        auto result = options.parse(argc, argv);

        // Handle special flags
        if (result.count("version")) {
            std::cout << "Sradcon version " << Sradcon::Version::GetVersionString() << std::endl;
            std::cout << "Build: " << Sradcon::Version::GetBuildInfo() << std::endl;
            std::cout << "\nHigh-quality audio resampler with psychoacoustic optimization" << std::endl;
            std::cout << "License: MIT" << std::endl;
            return 0;
        }

        if (result.count("help") || !result.count("input") || !result.count("output")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        // Set up log file if specified
        if (result.count("log")) {
            Sradcon::Logger::Instance().SetLogFile(result["log"].as<std::string>());
            LOG_INFO("Logging to file: " + result["log"].as<std::string>());
        }

        // Enable debug mode if requested
        DEBUG_MODE = result["debug"].as<bool>();
        if (DEBUG_MODE) {
            Sradcon::Logger::Instance().SetLevel(Sradcon::LogLevel::DEBUG);
            DEBUG_PATH = "debug_plots_cpp";
            std::filesystem::create_directories(DEBUG_PATH);
            LOG_INFO("Debug mode enabled. Data will be saved to '" + DEBUG_PATH.string() + "'");
        }

        // Load configuration
        Sradcon::Config config;
        if (result.count("config")) {
            LOG_INFO("Loading configuration from: " + result["config"].as<std::string>());
            try {
                config = Sradcon::Config::LoadFromFile(result["config"].as<std::string>());
            } catch (const Sradcon::SradconException& e) {
                LOG_WARNING("Failed to load config file: " + std::string(e.what()));
                LOG_INFO("Using command-line arguments and defaults");
            }
        }

        // Override config with command-line arguments
        config.resampler.target_sr = result["target_sr"].as<int>();
        config.resampler.chunk_size = result["chunk_size"].as<int>();
        config.resampler.filter_taps = result["filter_taps"].as<int>();
        config.resampler.disable_hfe = result["disable_hfe"].as<bool>();
        config.dither.target_bit_depth = result["target_bit_depth"].as<int>();
        config.dither.lpc_order = result["lpc_order"].as<int>();
        config.dither.enable = !result["no_dither"].as<bool>();
        config.dither.float32_output = result["float32_output"].as<bool>();
        config.processing.num_threads = result["threads"].as<int>();

        // Validate configuration
        LOG_DEBUG("Validating configuration...");
        config.Validate();

        // Print configuration summary
        LOG_INFO("Configuration:");
        LOG_INFO("  Target sample rate: " + std::to_string(config.resampler.target_sr) + " Hz");
        LOG_INFO("  Chunk size: " + std::to_string(config.resampler.chunk_size));
        LOG_INFO("  Filter taps: " + std::to_string(config.resampler.filter_taps));
        LOG_INFO("  Target bit depth: " + std::to_string(config.dither.target_bit_depth));
        LOG_INFO("  Dithering: " + std::string(config.dither.enable ? "enabled" : "disabled"));

        // Note: The actual processing would use the classes from sradcon_legacy.cpp
        // This requires proper linking and compilation setup

        LOG_ERROR("IMPORTANT: This main file needs to be linked with the processing classes");
        LOG_ERROR("Please use the original sradcon_legacy.cpp for now, or complete the integration");
        LOG_ERROR("The commercial-grade infrastructure is ready, but class integration is pending");

        // Show performance report if requested
        if (result["perf"].as<bool>()) {
            Sradcon::PerformanceMonitor::Instance().PrintReport();
        }

        return 1; // Return error until integration is complete

    } catch (const Sradcon::SradconException& e) {
        LOG_CRITICAL("Sradcon error: " + std::string(e.what()));
        return 1;
    } catch (const std::exception& e) {
        LOG_CRITICAL("Unexpected error: " + std::string(e.what()));
        return 1;
    } catch (...) {
        LOG_CRITICAL("Unknown error occurred");
        return 1;
    }
}
