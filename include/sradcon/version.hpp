#pragma once

#define SRADCON_VERSION_MAJOR 1
#define SRADCON_VERSION_MINOR 0
#define SRADCON_VERSION_PATCH 0

#define SRADCON_VERSION_STRING "1.0.0"
#define SRADCON_VERSION_BUILD __DATE__ " " __TIME__

namespace Sradcon {

struct Version {
    static constexpr int Major = SRADCON_VERSION_MAJOR;
    static constexpr int Minor = SRADCON_VERSION_MINOR;
    static constexpr int Patch = SRADCON_VERSION_PATCH;
    static constexpr const char* String = SRADCON_VERSION_STRING;
    static constexpr const char* Build = SRADCON_VERSION_BUILD;

    static const char* GetVersionString() noexcept { return String; }
    static const char* GetBuildInfo() noexcept { return Build; }
};

} // namespace Sradcon
