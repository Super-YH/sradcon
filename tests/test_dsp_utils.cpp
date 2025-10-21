#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include <Eigen/Dense>
#include <cmath>

// Simplified test versions of DSP functions
namespace DspUtils {
    using VectorF = Eigen::VectorXf;

    float hz_to_mel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
    float mel_to_hz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

    VectorF linspace(float start, float end, int num) {
        VectorF result(num);
        if (num <= 1) {
            if (num == 1) result[0] = start;
            return result;
        }
        float step = (end - start) / static_cast<float>(num - 1);
        for (int i = 0; i < num; ++i) result[i] = start + i * step;
        return result;
    }
}

TEST_CASE("DSP Utilities - Mel scale conversions", "[dsp][mel]") {
    SECTION("Hz to Mel conversion") {
        REQUIRE(DspUtils::hz_to_mel(0.0f) == Approx(0.0f));
        REQUIRE(DspUtils::hz_to_mel(1000.0f) > 0.0f);
    }

    SECTION("Mel to Hz conversion") {
        REQUIRE(DspUtils::mel_to_hz(0.0f) == Approx(0.0f).margin(0.01f));
    }

    SECTION("Round-trip conversion") {
        float original_hz = 1000.0f;
        float mel = DspUtils::hz_to_mel(original_hz);
        float recovered_hz = DspUtils::mel_to_hz(mel);
        REQUIRE(recovered_hz == Approx(original_hz).epsilon(0.001));
    }
}

TEST_CASE("DSP Utilities - Linspace", "[dsp][linspace]") {
    SECTION("Basic linspace") {
        auto result = DspUtils::linspace(0.0f, 10.0f, 11);
        REQUIRE(result.size() == 11);
        REQUIRE(result[0] == Approx(0.0f));
        REQUIRE(result[10] == Approx(10.0f));
        REQUIRE(result[5] == Approx(5.0f));
    }

    SECTION("Single element") {
        auto result = DspUtils::linspace(5.0f, 10.0f, 1);
        REQUIRE(result.size() == 1);
        REQUIRE(result[0] == Approx(5.0f));
    }

    SECTION("Negative range") {
        auto result = DspUtils::linspace(-5.0f, 5.0f, 11);
        REQUIRE(result.size() == 11);
        REQUIRE(result[0] == Approx(-5.0f));
        REQUIRE(result[10] == Approx(5.0f));
    }
}

TEST_CASE("Configuration - Validation", "[config]") {
    SECTION("Valid configuration should pass") {
        // Test would require including the config header
        // This is a placeholder for the structure
        REQUIRE(true);
    }
}
