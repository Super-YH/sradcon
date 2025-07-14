cmake_minimum_required(VERSION 3.16)
project(IntelligentResampler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 依存ライブラリの探索 ---

find_package(PkgConfig REQUIRED)

# libsndfile (pkg-configを利用)
pkg_check_modules(SNDFILE REQUIRED sndfile)

# FFTW3 (pkg-configを利用)
# libfftw3-dev パッケージなどによって提供される fftw3f.pc ファイルを探す
pkg_check_modules(FFTW3 REQUIRED fftw3f)

# Eigen (ヘッダオンリー)
# third_party/eigen ディレクトリにEigenのヘッダを配置することを想定
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)

# cxxopts (ヘッダオンリー)
# third_party/cxxopts/include ディレクトリにcxxopts.hppを配置することを想定
add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cxxopts/include)


# --- 実行ファイルの生成 ---
add_executable(resampler intelligent_resampler.cpp)


# --- リンクとインクルードパスの設定 ---
# pkg_check_modules が見つけたライブラリとインクルードパスをリンクする
target_link_libraries(resampler PRIVATE
    # sndfile
    ${SNDFILE_LIBRARIES}

    # fftw
    ${FFTW3_LIBRARIES}

    # ヘッダオンリーライブラリ
    eigen
    cxxopts
)

# pkg_check_modules が見つけたインクルードパスを追加
target_include_directories(resampler PRIVATE
    ${SNDFILE_INCLUDE_DIRS}
    ${FFTW3_INCLUDE_DIRS}
)


# --- デバッグと最適化オプション ---
target_compile_definitions(resampler PRIVATE
    $<$<CONFIG:Debug>:ENABLE_DEBUG_MODE>
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX)
    # march=native は実行するマシンに最適化するため、配布時には注意
    target_compile_options(resampler PRIVATE -O3 -march=native -ffast-math)
elseif(MSVC)
    target_compile_options(resampler PRIVATE /O2 /fp:fast)
endif()

# C++17のfilesystemを使うために必要 (特に古いGCC/Clangの場合)
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(resampler PRIVATE stdc++fs)
endif()