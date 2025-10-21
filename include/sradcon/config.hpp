#pragma once

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include "exception.hpp"

namespace Sradcon {

class Config {
public:
    struct ResamplerConfig {
        int target_sr = 44100;
        int chunk_size = 8192;
        int filter_taps = 32767;
        int num_bands = 128;
        bool disable_hfe = false;
    };

    struct DitherConfig {
        int target_bit_depth = 16;
        int lpc_order = 16;
        bool enable = true;
        bool float32_output = false;
    };

    struct ProcessingConfig {
        bool enable_ms_processing = true;
        bool enable_debug = false;
        std::string debug_path = "debug_plots";
        int num_threads = 0; // 0 = auto-detect
    };

    ResamplerConfig resampler;
    DitherConfig dither;
    ProcessingConfig processing;

    // Validation
    void Validate() const {
        if (resampler.target_sr <= 0 || resampler.target_sr > 384000) {
            throw InvalidParameterException("target_sr", "must be between 1 and 384000");
        }
        if (resampler.chunk_size < 128 || resampler.chunk_size > 65536) {
            throw InvalidParameterException("chunk_size", "must be between 128 and 65536");
        }
        if (resampler.filter_taps < 3 || resampler.filter_taps > 65535) {
            throw InvalidParameterException("filter_taps", "must be between 3 and 65535");
        }
        if (resampler.num_bands < 4 || resampler.num_bands > 512) {
            throw InvalidParameterException("num_bands", "must be between 4 and 512");
        }
        if (dither.target_bit_depth != 8 && dither.target_bit_depth != 16 &&
            dither.target_bit_depth != 24 && dither.target_bit_depth != 32 &&
            dither.target_bit_depth != 64) {
            throw InvalidParameterException("target_bit_depth", "must be 8, 16, 24, 32, or 64");
        }
        if (dither.lpc_order < 1 || dither.lpc_order > 128) {
            throw InvalidParameterException("lpc_order", "must be between 1 and 128");
        }
    }

    // Load from simple key=value format
    static Config LoadFromFile(const std::string& filename) {
        Config config;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw FileIOException(filename, "Could not open config file");
        }

        std::string line;
        int line_num = 0;
        while (std::getline(file, line)) {
            line_num++;
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;

            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;

            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1);

            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            try {
                config.SetValue(key, value);
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Error at line " << line_num << ": " << e.what();
                throw InvalidParameterException("config", oss.str());
            }
        }

        config.Validate();
        return config;
    }

    void SetValue(const std::string& key, const std::string& value) {
        if (key == "target_sr") resampler.target_sr = std::stoi(value);
        else if (key == "chunk_size") resampler.chunk_size = std::stoi(value);
        else if (key == "filter_taps") resampler.filter_taps = std::stoi(value);
        else if (key == "num_bands") resampler.num_bands = std::stoi(value);
        else if (key == "disable_hfe") resampler.disable_hfe = ParseBool(value);
        else if (key == "target_bit_depth") dither.target_bit_depth = std::stoi(value);
        else if (key == "lpc_order") dither.lpc_order = std::stoi(value);
        else if (key == "enable_dither") dither.enable = ParseBool(value);
        else if (key == "float32_output") dither.float32_output = ParseBool(value);
        else if (key == "enable_ms_processing") processing.enable_ms_processing = ParseBool(value);
        else if (key == "enable_debug") processing.enable_debug = ParseBool(value);
        else if (key == "debug_path") processing.debug_path = value;
        else if (key == "num_threads") processing.num_threads = std::stoi(value);
        // Unknown keys are silently ignored for forward compatibility
    }

private:
    static bool ParseBool(const std::string& value) {
        if (value == "true" || value == "1" || value == "yes" || value == "on") return true;
        if (value == "false" || value == "0" || value == "no" || value == "off") return false;
        throw InvalidParameterException("boolean", "must be true/false, 1/0, yes/no, or on/off");
    }
};

} // namespace Sradcon
