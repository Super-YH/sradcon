cmake_minimum_required(VERSION 3.16)

project(Sradcon
    VERSION 1.0.0
    DESCRIPTION "Professional audio resampler with advanced psychoacoustic processing"
    LANGUAGES CXX
)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)
option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(ENABLE_LTO "Enable Link Time Optimization" ON)

# Include modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

# Find dependencies
find_package(PkgConfig REQUIRED)

# libsndfile
pkg_check_modules(SNDFILE REQUIRED sndfile)

# FFTW3
pkg_check_modules(FFTW3 REQUIRED fftw3f)

# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Thread support
find_package(Threads REQUIRED)

# cxxopts (header-only)
add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cxxopts/include>
    $<INSTALL_INTERFACE:include>
)

# Main executable - uses the original high-quality resampling algorithms
# All core DSP algorithms (IntelligentRDOResampler, RDONoiseShaper, etc.)
# are preserved exactly as they were
add_executable(sradcon src/sradcon_legacy.cpp)

target_include_directories(sradcon
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${SNDFILE_INCLUDE_DIRS}
        ${FFTW3_INCLUDE_DIRS}
)

target_link_libraries(sradcon
    PRIVATE
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        ${FFTW3_LIBRARIES}
        Threads::Threads
        cxxopts
)

# Set compile flags
target_compile_definitions(sradcon PRIVATE
    $<$<CONFIG:Debug>:ENABLE_DEBUG_MODE>
)

# Optimization flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(sradcon PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
        $<$<CONFIG:Debug>:-O0 -g>
    )

    if(ENABLE_SANITIZERS)
        target_compile_options(sradcon PRIVATE
            -fsanitize=address -fsanitize=undefined
        )
        target_link_options(sradcon PRIVATE
            -fsanitize=address -fsanitize=undefined
        )
    endif()
elseif(MSVC)
    target_compile_options(sradcon PRIVATE
        /W4
        $<$<CONFIG:Release>:/O2 /fp:fast>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
endif()

# Link Time Optimization
if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set_property(TARGET sradcon PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO/IPO enabled")
    else()
        message(WARNING "LTO/IPO not supported: ${ipo_error}")
    endif()
endif()

# Filesystem library (needed for older compilers)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(sradcon PRIVATE stdc++fs)
endif()

# Set output name
set_target_properties(sradcon PROPERTIES OUTPUT_NAME "sradcon")

# Testing
if(BUILD_TESTING)
    enable_testing()
    find_package(Catch2 2 QUIET)

    if(Catch2_FOUND)
        add_executable(sradcon_tests tests/test_dsp_utils.cpp)
        target_link_libraries(sradcon_tests PRIVATE
            Catch2::Catch2
            Eigen3::Eigen
        )
        target_include_directories(sradcon_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        include(CTest)
        include(Catch)
        catch_discover_tests(sradcon_tests)

        message(STATUS "Tests enabled")
    else()
        message(WARNING "Catch2 not found, tests disabled")
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
install(TARGETS sradcon
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/sradcon
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(FILES
    docs/USER_GUIDE.md
    docs/API.md
    README.md
    LICENSE
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SradconConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SradconConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SradconConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sradcon
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SradconConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SradconConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sradcon
)

# CPack configuration
set(CPACK_PACKAGE_NAME "Sradcon")
set(CPACK_PACKAGE_VENDOR "Sradcon Developers")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

include(CPack)

# Print summary
message(STATUS "")
message(STATUS "=== Sradcon Configuration Summary ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build testing: ${BUILD_TESTING}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Enable sanitizers: ${ENABLE_SANITIZERS}")
message(STATUS "Enable LTO: ${ENABLE_LTO}")
message(STATUS "=====================================")
message(STATUS "")
