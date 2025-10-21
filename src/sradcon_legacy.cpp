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

// --- サードパーティライブラリのインクルード ---
#include <sndfile.h>
#include <Eigen/Dense>
#include "cxxopts.hpp"
#include <fftw3.h> // pocketfftからFFTWに変更


// --- グローバル設定 ---
bool DEBUG_MODE = false;
std::filesystem::path DEBUG_PATH;

// ===================================================================================
// 1. DSPユーティリティ (DspUtils)
// ===================================================================================
namespace DspUtils {

using VectorF = Eigen::VectorXf;
using VectorCF = Eigen::VectorXcf;
using MatrixF = Eigen::MatrixXf;
using ArrayF = Eigen::ArrayXf;
using ArrayCF = Eigen::ArrayXcf;

// --- 数学ヘルパー ---
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

std::vector<int> searchsorted(const VectorF& sorted_array, const VectorF& values_to_find) {
    std::vector<int> indices;
    indices.reserve(values_to_find.size());
    for (int i = 0; i < values_to_find.size(); ++i) {
        auto it = std::lower_bound(sorted_array.data(), sorted_array.data() + sorted_array.size(), values_to_find[i]);
        indices.push_back(std::distance(sorted_array.data(), it));
    }
    return indices;
}

void solve_toeplitz(const VectorF& r, VectorF& a) {
    int order = a.size();
    if (r.size() < order + 1) { a.setZero(); return; }
    if (std::abs(r[0]) < 1e-9f) { a.setZero(); return; }

    a.setZero();
    VectorF k(order);
    VectorF temp_a(order);
    float alpha = r[0];

    for (int i = 0; i < order; ++i) {
        float num = r[i + 1];
        for (int j = 0; j < i; ++j) num += a[j] * r[i - j];
        k[i] = -num / alpha;

        temp_a.head(i) = a.head(i);
        a[i] = k[i];
        for (int j = 0; j < i; ++j) a[j] += k[i] * temp_a[i - 1 - j];

        alpha *= (1.0f - k[i] * k[i]);
        if (alpha <= 1e-12f) { a.setZero(); return; }
    }
}


// --- FFT (FFTWラッパー) ---
VectorCF rfft(const VectorF& signal, size_t n_fft) {
    size_t n_spec = n_fft / 2 + 1;
    VectorCF output(n_spec);

    // FFTWが推奨するアラインメントされたメモリを確保
    float* in_buf = static_cast<float*>(fftwf_malloc(sizeof(float) * n_fft));
    fftwf_complex* out_buf = reinterpret_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex) * n_spec));

    // 入力信号をFFTWのバッファにコピー（必要に応じてゼロパディング）
    if (signal.size() < n_fft) {
        VectorF padded_signal = VectorF::Zero(n_fft);
        padded_signal.head(signal.size()) = signal;
        Eigen::Map<VectorF>(in_buf, n_fft) = padded_signal;
    } else {
        Eigen::Map<VectorF>(in_buf, n_fft) = signal.head(n_fft);
    }

    // FFTプランを作成
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(n_fft, in_buf, out_buf, FFTW_ESTIMATE);

    // FFTを実行
    fftwf_execute(plan);

    // 結果をEigenのVectorにコピー
    // fftwf_complexはstd::complex<float>とメモリレイアウト互換
    std::memcpy(output.data(), out_buf, sizeof(fftwf_complex) * n_spec);

    // リソースを解放
    fftwf_destroy_plan(plan);
    fftwf_free(in_buf);
    fftwf_free(out_buf);

    return output;
}

VectorF irfft(const VectorCF& spectrum, size_t out_len) {
    size_t n_spec = spectrum.size();
    VectorF output(out_len);

    // FFTWが推奨するアラインメントされたメモリを確保
    fftwf_complex* in_buf = reinterpret_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex) * n_spec));
    float* out_buf = static_cast<float*>(fftwf_malloc(sizeof(float) * out_len));

    // スペクトルデータをFFTWのバッファにコピー
    std::memcpy(in_buf, spectrum.data(), sizeof(fftwf_complex) * n_spec);

    // IFFTプランを作成
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(out_len, in_buf, out_buf, FFTW_ESTIMATE);

    // IFFTを実行
    fftwf_execute(plan);

    // 結果をEigenのVectorにコピー
    Eigen::Map<VectorF>(output.data(), out_len) = Eigen::Map<VectorF>(out_buf, out_len);

    // リソースを解放
    fftwf_destroy_plan(plan);
    fftwf_free(in_buf);
    fftwf_free(out_buf);

    // FFTWの逆変換は正規化されないため、手動で正規化
    output /= static_cast<float>(out_len);

    return output;
}


// --- 窓関数 ---
VectorF getWindow(const std::string& name, int size, bool sym = true) {
    VectorF window(size);
    if (size == 0) return window;
    int den = sym ? (size - 1) : size;
    if (den == 0) {
        if(size > 0) window.setOnes();
        return window;
    }
    if (name == "hann") {
        for (int i = 0; i < size; ++i) {
            window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / den));
        }
    } else {
        throw std::invalid_argument("Unsupported window type: " + name);
    }
    return window;
}

// --- 信号処理 ---
VectorF convolve(const VectorF& signal, const VectorF& kernel, const std::string& mode = "same") {
    int sig_len = signal.size();
    int ker_len = kernel.size();
    int conv_len = sig_len + ker_len - 1;

    size_t fft_size = 1;
    while(fft_size < conv_len) fft_size <<= 1;

    VectorF sig_padded = VectorF::Zero(fft_size);
    sig_padded.head(sig_len) = signal;
    VectorF ker_padded = VectorF::Zero(fft_size);
    ker_padded.head(ker_len) = kernel;

    VectorCF sig_fft = rfft(sig_padded, fft_size);
    VectorCF ker_fft = rfft(ker_padded, fft_size);

    VectorCF conv_fft = sig_fft.array() * ker_fft.array();
    VectorF result_full = irfft(conv_fft, fft_size);

    if (mode == "full") {
        return result_full.head(conv_len);
    }
    if (mode == "same") {
        int start = (ker_len - 1) / 2;
        return result_full.segment(start, sig_len);
    }
    throw std::invalid_argument("Unsupported convolve mode: " + mode);
}

VectorF firwin(int numtaps, float cutoff, float fs, const std::string& window_name = "hann") {
    float nyquist = fs / 2.0f;
    cutoff /= nyquist;

    VectorF taps(numtaps);
    int alpha = (numtaps - 1) / 2;
    for (int i = 0; i < numtaps; ++i) {
        if (i == alpha) {
            taps[i] = cutoff;
        } else {
            taps[i] = cutoff * std::sin(M_PI * cutoff * (i - alpha)) / (M_PI * cutoff * (i - alpha));
        }
    }
    VectorF win = getWindow(window_name, numtaps);
    return taps.array() * win.array();
}

VectorF lfilter(const VectorF& b, const VectorF& a, const VectorF& x) {
    if (a.size() != 1 || a[0] != 1.0f) {
        throw std::runtime_error("lfilter currently only supports FIR filters (a=[1.0])");
    }
    VectorF result(x.size());
    result.setZero();

    for (int n = 0; n < x.size(); ++n) {
        for (int k = 0; k < b.size(); ++k) {
            if (n - k >= 0) {
                result[n] += b[k] * x[n - k];
            }
        }
    }
    return result;
}

void stft(const VectorF& signal, int n_fft, int hop_length, const VectorF& window, Eigen::MatrixXf& mag, Eigen::MatrixXcf& phase) {
    int num_frames = (signal.size() > n_fft) ? (1 + (signal.size() - n_fft) / hop_length) : 1;
    int n_spec = n_fft / 2 + 1;
    mag.resize(n_spec, num_frames);
    phase.resize(n_spec, num_frames);
    mag.setZero();
    phase.setZero();

    for (int i = 0; i < num_frames; ++i) {
        int start = i * hop_length;
        int chunk_size = std::min((int)signal.size() - start, n_fft);
        VectorF chunk_raw = VectorF::Zero(n_fft);
        chunk_raw.head(chunk_size) = signal.segment(start, chunk_size);

        VectorF chunk = chunk_raw.array() * window.array();
        VectorCF spectrum = rfft(chunk, n_fft);
        mag.col(i) = spectrum.array().abs();
        ArrayF mag_col = mag.col(i).array();
        phase.col(i) = spectrum.array() / mag_col.max(1e-9f);
    }
}

VectorF istft(const Eigen::MatrixXf& mag, const Eigen::MatrixXcf& phase, int hop_length, const VectorF& window) {
    int n_fft = (mag.rows() - 1) * 2;
    int num_frames = mag.cols();
    long long out_len = n_fft + (num_frames - 1) * hop_length;
    VectorF output = VectorF::Zero(out_len);
    VectorF win_sq_sum = VectorF::Zero(out_len);

    VectorF win_sq = window.array().square();

    for (int i = 0; i < num_frames; ++i) {
        int start = i * hop_length;
        VectorCF spectrum = mag.col(i).array() * phase.col(i).array();
        VectorF chunk = irfft(spectrum, n_fft);
        output.segment(start, n_fft).array() += chunk.array() * window.array();
        win_sq_sum.segment(start, n_fft) += win_sq;
    }

    for(long long i=0; i < out_len; ++i) {
        if (win_sq_sum[i] > 1e-9f) {
            output[i] /= win_sq_sum[i];
        }
    }

    return output;
}

} // namespace DspUtils

// ===================================================================================
// 2. 音声ファイル入出力 (AudioFile)
// (変更なし)
// ===================================================================================
class AudioFile {
public:
    int sr = 0;
    long long frames = 0;
    int channels = 0;
    DspUtils::MatrixF data;

    bool load(const std::string& path) {
        SF_INFO sfinfo;
        sfinfo.format = 0;
        SNDFILE* infile = sf_open(path.c_str(), SFM_READ, &sfinfo);
        if (!infile) {
            std::cerr << "Error: Could not open input file: " << sf_strerror(NULL) << std::endl;
            return false;
        }

        sr = sfinfo.samplerate;
        frames = sfinfo.frames;
        channels = sfinfo.channels;

        std::vector<float> buffer(frames * channels);
        sf_read_float(infile, buffer.data(), frames * channels);
        sf_close(infile);

        data.resize(frames, channels);
        if (channels == 1) {
            data.col(0) = Eigen::Map<DspUtils::VectorF>(buffer.data(), frames);
        } else {
            for (long long i = 0; i < frames; ++i) {
                for (int j = 0; j < channels; ++j) {
                    data(i, j) = buffer[i * channels + j];
                }
            }
        }
        return true;
    }

    bool save(const std::string& path, int target_sr, const std::string& subtype) {
        SF_INFO sfinfo;
        sfinfo.samplerate = target_sr;
        sfinfo.channels = data.cols();

        if (subtype == "PCM_16") sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        else if (subtype == "PCM_24") sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_24;
        else if (subtype == "PCM_32") sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_32;
        else if (subtype == "PCM_U8") sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_U8;
        else if (subtype == "FLOAT") sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
        else if (subtype == "DOUBLE") sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_DOUBLE;
        else {
            std::cerr << "Error: Unsupported output subtype " << subtype << std::endl;
            return false;
        }

        SNDFILE* outfile = sf_open(path.c_str(), SFM_WRITE, &sfinfo);
        if (!outfile) {
            std::cerr << "Error: Could not open output file: " << sf_strerror(NULL) << std::endl;
            return false;
        }

        // Eigenのデータをインターリーブされたstd::vectorにコピー
        std::vector<float> buffer(data.rows() * data.cols());
        for (Eigen::Index i = 0; i < data.rows(); ++i) {
            for (Eigen::Index j = 0; j < data.cols(); ++j) {
                buffer[i * data.cols() + j] = data(i, j);
            }
        }

        sf_write_float(outfile, buffer.data(), buffer.size());
        sf_close(outfile);
        return true;
    }
};


// ===================================================================================
// 3. 高度な心理音響モデル (AdvancedPsychoacousticModel)
// (変更なし)
// ===================================================================================
class AdvancedPsychoacousticModel {
public:
    struct AnalysisResult {
        DspUtils::VectorF power_db;
        DspUtils::VectorF masking_threshold_db;
        DspUtils::VectorF tonality;
        DspUtils::VectorF power_spectrum;
    };

    AdvancedPsychoacousticModel(int sr, int fft_size, int num_bands = 16, float alpha = 0.8f);
    AnalysisResult analyze_chunk(const DspUtils::VectorF& signal_chunk);

    const DspUtils::VectorF& get_freqs() const { return freqs; }
    const std::vector<int>& get_band_indices() const { return band_indices; }
    const DspUtils::VectorF& get_ath_db() const { return ath_db; }

private:
    void _precompute_band_indices();
    DspUtils::VectorF _spreading_function(const DspUtils::VectorF& power_db);
    DspUtils::VectorF _calculate_tonality(const DspUtils::VectorF& power_spectrum);

    int sr, fft_size, num_bands;
    float alpha;
    DspUtils::VectorF freqs, ath_db;
    std::vector<int> band_indices;
};

AdvancedPsychoacousticModel::AdvancedPsychoacousticModel(int p_sr, int p_fft_size, int p_num_bands, float p_alpha)
    : sr(p_sr), fft_size(p_fft_size), num_bands(p_num_bands), alpha(p_alpha) {
    freqs = DspUtils::linspace(0, sr / 2.0f, fft_size / 2 + 1);
    _precompute_band_indices();

    ath_db.resize(freqs.size());
    for (int i = 0; i < freqs.size(); ++i) {
        float f_khz = freqs[i] / 1000.0f;
        f_khz = std::max(f_khz, 1e-12f);
        ath_db[i] = (3.64f * std::pow(f_khz, -0.8f) -
                     6.5f * std::exp(-0.6f * std::pow(f_khz - 3.3f, 2.0f)) +
                     1e-3f * std::pow(f_khz, 4.0f));
        if (freqs[i] > sr / 2.2f) {
            ath_db[i] = std::numeric_limits<float>::infinity();
        }
    }
}

void AdvancedPsychoacousticModel::_precompute_band_indices() {
    float max_mel = DspUtils::hz_to_mel(sr / 2.0f);
    DspUtils::VectorF mel_points = DspUtils::linspace(0, max_mel, num_bands + 1);
    DspUtils::VectorF hz_points(mel_points.size());
    for(int i = 0; i < mel_points.size(); ++i) hz_points[i] = DspUtils::mel_to_hz(mel_points[i]);

    band_indices = DspUtils::searchsorted(freqs, hz_points);
    if (!band_indices.empty()) {
        band_indices.back() = freqs.size();
    }
}

DspUtils::VectorF AdvancedPsychoacousticModel::_spreading_function(const DspUtils::VectorF& power_db) {
    DspUtils::VectorF kernel(8);
    kernel << 0.05f, 0.1f, 0.2f, 1.0f, 0.4f, 0.2f, 0.1f, 0.05f;
    return DspUtils::convolve(power_db, kernel, "same");
}

DspUtils::VectorF AdvancedPsychoacousticModel::_calculate_tonality(const DspUtils::VectorF& power_spectrum) {
    DspUtils::VectorF tonality_per_band = DspUtils::VectorF::Zero(num_bands);
    for (int i = 0; i < num_bands; ++i) {
        int start = band_indices[i];
        int end = band_indices[i + 1];
        if (start >= end) continue;

        auto band_spectrum = power_spectrum.segment(start, end - start);
        float log_sum = (band_spectrum.array() + 1e-12f).log().sum();
        float gmean = std::exp(log_sum / band_spectrum.size());
        float amean = band_spectrum.mean();
        float sfm = (amean > 1e-12f) ? (gmean / (amean + 1e-12f)) : 1.0f;
        tonality_per_band[i] = 1.0f - sfm;
    }
    return tonality_per_band;
}

AdvancedPsychoacousticModel::AnalysisResult AdvancedPsychoacousticModel::analyze_chunk(const DspUtils::VectorF& signal_chunk_const) {
    DspUtils::VectorF signal_chunk = signal_chunk_const;
    if (signal_chunk.size() != fft_size) {
        DspUtils::VectorF padded_chunk = DspUtils::VectorF::Zero(fft_size);
        padded_chunk.head(signal_chunk.size()) = signal_chunk;
        signal_chunk = padded_chunk;
    }

    DspUtils::VectorF window = DspUtils::getWindow("hann", fft_size);
    DspUtils::VectorF windowed_signal = signal_chunk.array() * window.array();

    DspUtils::VectorCF spectrum = DspUtils::rfft(windowed_signal, fft_size);
    DspUtils::VectorF power_spectrum = spectrum.array().abs2();
    DspUtils::VectorF power_db = 10.0f * power_spectrum.array().max(1e-12f).log10() + 96.0f;

    DspUtils::VectorF spreading_db = _spreading_function(power_db);
    DspUtils::VectorF masking_threshold_db = ath_db.cwiseMax((spreading_db.array() - 10.0f).matrix());
    DspUtils::VectorF tonality = _calculate_tonality(power_spectrum);

    return {power_db, masking_threshold_db, tonality, power_spectrum};
}


// ===================================================================================
// 4. RDOノイズシェーパー (RDONoiseShaper)
// (変更なし)
// ===================================================================================
class RDONoiseShaper {
public:
    RDONoiseShaper(int target_bit_depth, int sr, int chunk_size, int lpc_order);
    DspUtils::MatrixF process(const DspUtils::MatrixF& signal);

private:
    DspUtils::VectorF process_channel(const DspUtils::VectorF& channel_data);
    DspUtils::VectorF _calculate_lpc_coeffs(const DspUtils::VectorF& power_spectrum, int order);
    DspUtils::VectorF _design_adaptive_filter(const DspUtils::VectorF& chunk_data, const DspUtils::VectorF& power_spectrum, int chunk_index);

    int target_bit_depth, sr, chunk_size, lpc_order;
    float quantization_step;
    std::unique_ptr<AdvancedPsychoacousticModel> psy_model;
    DspUtils::VectorF error_hist, simple_b;
};

RDONoiseShaper::RDONoiseShaper(int p_target_bit_depth, int p_sr, int p_chunk_size, int p_lpc_order)
    : target_bit_depth(p_target_bit_depth), sr(p_sr), chunk_size(p_chunk_size), lpc_order(p_lpc_order) {

    if (target_bit_depth != 8 && target_bit_depth != 16 && target_bit_depth != 24 && target_bit_depth != 32) {
        throw std::invalid_argument("Dithering is only supported for 8, 16, 24, 32-bit PCM.");
    }

    quantization_step = 1.0f / static_cast<float>(1LL << (target_bit_depth - 1));
    psy_model = std::make_unique<AdvancedPsychoacousticModel>(sr, chunk_size, 32);
    error_hist = DspUtils::VectorF::Zero(lpc_order + 1);

    simple_b = DspUtils::VectorF::Zero(lpc_order + 1);
    if (lpc_order >= 2) { simple_b[0] = 1.0f; simple_b[1] = -1.5f; simple_b[2] = 0.6f; }
    else if (lpc_order == 1) { simple_b[0] = 1.0f; simple_b[1] = -1.0f; }
    else { simple_b[0] = 1.0f; }
}

DspUtils::VectorF RDONoiseShaper::_calculate_lpc_coeffs(const DspUtils::VectorF& power_spectrum, int order) {
    DspUtils::VectorF autocorr = DspUtils::irfft(power_spectrum, (power_spectrum.size() - 1) * 2);
    if (autocorr.size() < order + 1) return simple_b;

    DspUtils::VectorF r = autocorr.head(order + 1);
    DspUtils::VectorF a(order);

    try {
        DspUtils::solve_toeplitz(r, a);
        DspUtils::VectorF lpc_filter(order + 1);
        lpc_filter[0] = 1.0f;
        lpc_filter.tail(order) = a;
        if (lpc_filter.hasNaN()) return simple_b;
        return lpc_filter;
    } catch (...) {
        return simple_b;
    }
}

DspUtils::VectorF RDONoiseShaper::_design_adaptive_filter(const DspUtils::VectorF& chunk_data, const DspUtils::VectorF& power_spectrum, int chunk_index) {
    auto analysis = psy_model->analyze_chunk(chunk_data);
    const auto& freqs = psy_model->get_freqs();

    float margin_sum = 0.0f;
    int audible_count = 0;
    for (int i = 0; i < freqs.size(); ++i) {
        if (freqs[i] > 20.0f && freqs[i] < 20000.0f) {
            margin_sum += analysis.power_db[i] - analysis.masking_threshold_db[i];
            audible_count++;
        }
    }
    float avg_margin = (audible_count > 0) ? margin_sum / audible_count : 0.0f;
    float shaping_gain = std::min(4.0f, std::max(0.7f, (avg_margin - 5.0f) / 20.0f));
    DspUtils::VectorF lpc_b = _calculate_lpc_coeffs(power_spectrum, lpc_order);
    DspUtils::VectorF adaptive_b = simple_b * (1.0f - shaping_gain) + lpc_b * shaping_gain;

    if (DEBUG_MODE) {
        std::cout << "  [DEBUG RDO] Chunk " << std::setw(4) << std::setfill('0') << chunk_index << ": Avg Margin=" << std::fixed << std::setprecision(2) << avg_margin << "dB, Shaping Gain=" << std::setprecision(3) << shaping_gain << std::endl;
    }
    return adaptive_b;
}

DspUtils::VectorF RDONoiseShaper::process_channel(const DspUtils::VectorF& channel_data) {
    long long num_samples = channel_data.size();
    DspUtils::VectorF output = DspUtils::VectorF::Zero(num_samples);

    error_hist.setZero();

    for (long long i = 0; i < num_samples; i += chunk_size) {
        int chunk_index = i / chunk_size;
        long long end = std::min(i + chunk_size, num_samples);
        auto chunk = channel_data.segment(i, end - i);

        DspUtils::VectorF padded_chunk = chunk;
        if (chunk.size() < chunk_size) {
            padded_chunk.resize(chunk_size);
            padded_chunk.tail(chunk_size - chunk.size()).setZero();
        }
        auto analysis = psy_model->analyze_chunk(padded_chunk);
        DspUtils::VectorF b = _design_adaptive_filter(padded_chunk, analysis.power_spectrum, chunk_index);

        for (int n = 0; n < chunk.size(); ++n) {
            float error_shaped = b.tail(lpc_order).dot(error_hist.head(lpc_order));
            float sample_to_quantize = chunk[n] + error_shaped;
            float quantized_sample = std::round(sample_to_quantize / quantization_step) * quantization_step;
            output[i + n] = quantized_sample;
            float current_error = quantized_sample - sample_to_quantize;

            std::memmove(error_hist.data() + 1, error_hist.data(), lpc_order * sizeof(float));
            error_hist[0] = current_error;
        }
    }
    return output.cwiseMax(-1.0f).cwiseMin(1.0f);
}

DspUtils::MatrixF RDONoiseShaper::process(const DspUtils::MatrixF& signal) {
    DspUtils::MatrixF output(signal.rows(), signal.cols());
    for(int i=0; i < signal.cols(); ++i){
        output.col(i) = process_channel(signal.col(i));
    }
    return output;
}


// ===================================================================================
// 5. 高音補完モジュール (HighFrequencySynthesizer)
// (変更なし)
// ===================================================================================
class HighFrequencySynthesizer {
public:
    HighFrequencySynthesizer(int original_sr, int target_sr, int n_fft, int hop_length, int iterations);
    DspUtils::VectorF synthesize(const DspUtils::VectorF& signal, const std::string& ch_name);

private:
    DspUtils::MatrixF _predict_high_freq_envelope(const DspUtils::MatrixF& S_mag);
    DspUtils::VectorF _griffin_lim(const DspUtils::MatrixF& S_mag);
    void _debug_plot_synthesis(const DspUtils::MatrixF& S_mag, const DspUtils::MatrixF& S_mag_hfe, const std::string& ch_name);

    int original_sr, target_sr, n_fft, hop_length, iterations;
    DspUtils::VectorF freqs;
    int cutoff_bin, source_start_bin;
    float noise_floor_factor;
    std::vector<std::pair<float, float>> transpositions;
};

HighFrequencySynthesizer::HighFrequencySynthesizer(int osr, int tsr, int fft, int hop, int iter)
    : original_sr(osr), target_sr(tsr), n_fft(fft), hop_length(hop), iterations(iter), noise_floor_factor(0.05f) {
    freqs = DspUtils::linspace(0, target_sr / 2.0f, n_fft / 2 + 1);
    cutoff_bin = DspUtils::searchsorted(freqs, DspUtils::VectorF::Constant(1, original_sr / 2.0f * 0.98f))[0];
    source_start_bin = cutoff_bin / 2;
    transpositions = {{2.0f, 0.6f}, {3.0f, 0.3f}, {4.0f, 0.15f}};
}

DspUtils::MatrixF HighFrequencySynthesizer::_predict_high_freq_envelope(const DspUtils::MatrixF& S_mag) {
    long num_freq_bins = S_mag.rows();
    long num_frames = S_mag.cols();
    DspUtils::MatrixF S_hfe_high_only = DspUtils::MatrixF::Zero(num_freq_bins - cutoff_bin, num_frames);
    auto source_band_mag = S_mag.block(source_start_bin, 0, cutoff_bin - source_start_bin, num_frames);

    DspUtils::VectorF noise_level(num_frames);
    for (int t = 0; t < num_frames; ++t) {
        DspUtils::VectorF col = source_band_mag.col(t);
        std::sort(col.data(), col.data() + col.size());
        noise_level[t] = col[static_cast<int>(col.size() * 0.05)] * noise_floor_factor;
    }
    S_hfe_high_only = (DspUtils::MatrixF::Random(S_hfe_high_only.rows(), S_hfe_high_only.cols()).array() * 0.5f + 1.0f).matrix();
    for (int t = 0; t < num_frames; ++t) S_hfe_high_only.col(t) *= noise_level[t];

    auto source_freqs = freqs.segment(source_start_bin, cutoff_bin - source_start_bin);
    auto target_high_freqs = freqs.segment(cutoff_bin, num_freq_bins - cutoff_bin);

    for (const auto& pair : transpositions) {
        float factor = pair.first;
        float gain = pair.second;
        DspUtils::VectorF transposed_freqs = source_freqs * factor;

        for (int t = 0; t < num_frames; ++t) {
            auto source_mag_subset = source_band_mag.col(t);
            DspUtils::VectorF interpolated_mag(target_high_freqs.size());
            interpolated_mag.setZero();
            for(int i=0; i<target_high_freqs.size(); ++i) {
                auto it = std::upper_bound(transposed_freqs.data(), transposed_freqs.data() + transposed_freqs.size(), target_high_freqs[i]);
                if (it != transposed_freqs.data() && it != transposed_freqs.data() + transposed_freqs.size()){
                    int idx = std::distance(transposed_freqs.data(), it);
                    float x1 = transposed_freqs[idx-1], x2 = transposed_freqs[idx];
                    float y1 = source_mag_subset[idx-1], y2 = source_mag_subset[idx];
                    if (x2 > x1) {
                       interpolated_mag[i] = y1 + (y2 - y1) * (target_high_freqs[i] - x1) / (x2 - x1);
                    }
                }
            }

            DspUtils::ArrayF decay = DspUtils::linspace(1.0, 0.2, target_high_freqs.size()).array().square();
            S_hfe_high_only.col(t).array() += interpolated_mag.array() * gain * decay;
        }
    }

    DspUtils::MatrixF S_mag_synthesized = S_mag;
    S_mag_synthesized.bottomRows(num_freq_bins - cutoff_bin) = S_hfe_high_only;
    return S_mag_synthesized;
}

DspUtils::VectorF HighFrequencySynthesizer::_griffin_lim(const DspUtils::MatrixF& S_mag) {
    Eigen::MatrixXcf angles = Eigen::MatrixXcf::Random(S_mag.rows(), S_mag.cols());
    angles = angles.array().unaryExpr([](std::complex<float> c){ return std::exp(std::complex<float>(0, c.real() * M_PI * 2.0f)); });

    DspUtils::VectorF win = DspUtils::getWindow("hann", n_fft);
    DspUtils::VectorF y = DspUtils::istft(S_mag, angles, hop_length, win);

    for (int i = 0; i < iterations; ++i) {
        DspUtils::MatrixF current_mag;
        Eigen::MatrixXcf current_phase;
        DspUtils::stft(y, n_fft, hop_length, win, current_mag, current_phase);
        y = DspUtils::istft(S_mag, current_phase, hop_length, win);
    }
    return y;
}

void HighFrequencySynthesizer::_debug_plot_synthesis(const DspUtils::MatrixF& S_mag, const DspUtils::MatrixF& S_mag_hfe, const std::string& ch_name){
    if (!DEBUG_MODE) return;
    std::cout << "  [DEBUG HFE] Writing synthesis data for " << ch_name << "..." << std::endl;
    auto path_orig = DEBUG_PATH / ("hfe_mag_orig_" + ch_name + ".csv");
    auto path_synth = DEBUG_PATH / ("hfe_mag_synth_" + ch_name + ".csv");

    std::ofstream ofs_orig(path_orig);
    ofs_orig << (20.0f * (S_mag.array() + 1e-9f).log10()).matrix();
    ofs_orig.close();

    std::ofstream ofs_synth(path_synth);
    ofs_synth << (20.0f * (S_mag_hfe.array() + 1e-9f).log10()).matrix();
    ofs_synth.close();

    std::cout << "  [DEBUG HFE] Saved synthesis data to " << path_orig.string() << " and " << path_synth.string() << std::endl;
}

DspUtils::VectorF HighFrequencySynthesizer::synthesize(const DspUtils::VectorF& signal, const std::string& ch_name) {
    std::cout << "  - Synthesizing high frequencies for " << ch_name << "..." << std::endl;
    DspUtils::VectorF win = DspUtils::getWindow("hann", n_fft);
    DspUtils::MatrixF S_mag;
    Eigen::MatrixXcf phase;
    DspUtils::stft(signal, n_fft, hop_length, win, S_mag, phase);

    DspUtils::MatrixF S_mag_hfe = _predict_high_freq_envelope(S_mag);

    _debug_plot_synthesis(S_mag, S_mag_hfe, ch_name);

    DspUtils::VectorF y_hfe = _griffin_lim(S_mag_hfe);

    float cutoff_hz = original_sr / 2.0f * 0.95f;
    DspUtils::VectorF lpf_taps = DspUtils::firwin(101, cutoff_hz, target_sr);
    DspUtils::VectorF hpf_taps = DspUtils::firwin(101, cutoff_hz, target_sr);
    for(int i=0; i < hpf_taps.size(); ++i) hpf_taps[i] = -hpf_taps[i];
    hpf_taps[(101-1)/2] += 1.0f;

    DspUtils::VectorF low_passed = DspUtils::lfilter(lpf_taps, DspUtils::VectorF::Ones(1), signal);
    DspUtils::VectorF high_passed = DspUtils::lfilter(hpf_taps, DspUtils::VectorF::Ones(1), y_hfe);

    long long max_len = std::max(low_passed.size(), high_passed.size());
    low_passed.conservativeResize(max_len);
    high_passed.conservativeResize(max_len);

    DspUtils::VectorF combined_signal = low_passed + high_passed;

    return combined_signal.head(signal.size());
}


// ===================================================================================
// 6. インテリジェントRDOリサンプラー (IntelligentRDOResampler)
// (変更なし)
// ===================================================================================
class IntelligentRDOResampler {
public:
    IntelligentRDOResampler(int original_sr, int target_sr, int chunk_size, int filter_taps, int num_bands);
    DspUtils::MatrixF resample(const DspUtils::MatrixF& signal, bool is_ms_processing, bool disable_hfe);

private:
    DspUtils::VectorF _resample_channel(const DspUtils::VectorF& channel_data, const std::string& ch_name);
    DspUtils::VectorF _to_minimum_phase(const DspUtils::VectorF& fir_taps);
    DspUtils::VectorCF _design_intelligent_mixed_phase_filter(const DspUtils::VectorF& chunk, int chunk_index, const std::string& ch_name);
    void _debug_plot_filter_selection(const DspUtils::VectorF& power_db, const DspUtils::VectorF& mask_db, const DspUtils::VectorF& tonality, const DspUtils::VectorF& linear_mask, int chunk_idx, const std::string& ch_name);

    int original_sr, target_sr, chunk_size, filter_taps, num_bands, hop_size;
    float resample_ratio;
    bool is_upsampling;
    DspUtils::VectorF window;
    std::unique_ptr<AdvancedPsychoacousticModel> psy_model;
    std::unique_ptr<HighFrequencySynthesizer> hfe_synthesizer;

    DspUtils::VectorCF H_linear, H_min;
};

IntelligentRDOResampler::IntelligentRDOResampler(int osr, int tsr, int csize, int ftaps, int nbands)
    : original_sr(osr), target_sr(tsr), chunk_size(csize), filter_taps(ftaps | 1), num_bands(nbands) {

    hop_size = chunk_size / 2;
    resample_ratio = static_cast<float>(target_sr) / original_sr;
    window = DspUtils::getWindow("hann", chunk_size);
    psy_model = std::make_unique<AdvancedPsychoacousticModel>(original_sr, chunk_size, num_bands);
    is_upsampling = resample_ratio > 1.0f;

    if (is_upsampling) {
        int hfe_fft_size = std::min(4096, chunk_size * 2);
        hfe_synthesizer = std::make_unique<HighFrequencySynthesizer>(original_sr, target_sr, hfe_fft_size, hfe_fft_size / 4, 10);
    }

    float cutoff_hz = std::min(original_sr, target_sr) / 2.0f * 0.9f;
    DspUtils::VectorF h_linear_taps = DspUtils::firwin(filter_taps, cutoff_hz, original_sr);
    DspUtils::VectorF h_min_taps = _to_minimum_phase(h_linear_taps);

    H_linear = DspUtils::rfft(h_linear_taps, chunk_size);
    H_min = DspUtils::rfft(h_min_taps, chunk_size);

    std::cout << "Filter pre-calculation complete." << std::endl;
}

DspUtils::VectorF IntelligentRDOResampler::_to_minimum_phase(const DspUtils::VectorF& fir_taps) {
    int n = fir_taps.size();
    size_t fft_size = 1;
    while(fft_size < n) fft_size <<= 1;

    DspUtils::VectorF taps_padded = DspUtils::VectorF::Zero(fft_size);
    taps_padded.head(n) = fir_taps;

    DspUtils::VectorCF H = DspUtils::rfft(taps_padded, fft_size);
    DspUtils::VectorF log_mag = (H.array().abs() + 1e-12f).log();

    DspUtils::VectorF cepstrum_full = DspUtils::irfft(log_mag, fft_size);

    DspUtils::VectorF w = DspUtils::VectorF::Zero(fft_size);
    w[0] = 1;
    for(int i=1; i < (fft_size+1)/2; ++i) w[i] = 2;
    if (fft_size % 2 == 0) w[fft_size/2] = 1;

    DspUtils::VectorF cepstrum = cepstrum_full.array() * w.array();

    DspUtils::VectorCF H_min_spec = DspUtils::rfft(cepstrum, fft_size).array().exp();
    return DspUtils::irfft(H_min_spec, fft_size).head(n);
}

void IntelligentRDOResampler::_debug_plot_filter_selection(const DspUtils::VectorF& power_db, const DspUtils::VectorF& mask_db, const DspUtils::VectorF& tonality, const DspUtils::VectorF& linear_mask, int chunk_idx, const std::string& ch_name) {
    if(!DEBUG_MODE) return;
    std::cout << "  [DEBUG Filter] Writing filter analysis for Chunk " << chunk_idx << ", " << ch_name << "..." << std::endl;
    auto path = DEBUG_PATH / ("filter_analysis_ch" + ch_name + "_chunk" + std::to_string(chunk_idx) + ".csv");
    std::ofstream ofs(path);

    ofs << "Freq,PowerDB,MaskDB,ATH,Tonality,IsLinear\n";
    const auto& freqs = psy_model->get_freqs();
    const auto& ath_db = psy_model->get_ath_db();
    const auto& band_indices = psy_model->get_band_indices();

    int band_idx = 0;
    for(int i=0; i<freqs.size(); ++i) {
        if(band_idx + 1 < band_indices.size() && i >= band_indices[band_idx+1]) {
            band_idx++;
        }
        ofs << freqs[i] << "," << power_db[i] << "," << mask_db[i] << "," << ath_db[i] << "," << (band_idx < tonality.size() ? tonality[band_idx] : 0) << "," << (band_idx < linear_mask.size() ? linear_mask[band_idx] : 0) << "\n";
    }
    ofs.close();
    std::cout << "  [DEBUG Filter] Saved filter plot data to " << path.string() << std::endl;
}

DspUtils::VectorCF IntelligentRDOResampler::_design_intelligent_mixed_phase_filter(const DspUtils::VectorF& chunk, int chunk_index, const std::string& ch_name) {
    auto analysis = psy_model->analyze_chunk(chunk);
    DspUtils::VectorCF mixed_filter_spectrum = DspUtils::VectorCF::Zero(H_linear.size());
    DspUtils::VectorF linear_phase_mask = DspUtils::VectorF::Zero(num_bands);

    if(DEBUG_MODE && chunk_index < 5)
        std::cout << "\n--- Analyzing Chunk " << std::setw(4) << std::setfill('0') << chunk_index << " for Channel " << ch_name << " ---" << std::endl;

    for (int i = 0; i < num_bands; ++i) {
        int start = psy_model->get_band_indices()[i];
        int end = psy_model->get_band_indices()[i+1];
        if (start >= end) continue;

        float avg_power = analysis.power_db.segment(start, end-start).mean();
        float avg_margin = (analysis.power_db.segment(start, end-start) - analysis.masking_threshold_db.segment(start, end-start)).mean();
        float tonality = analysis.tonality[i];

        bool use_linear_phase = false;
        if (avg_power < psy_model->get_ath_db().segment(start, end-start).mean()) {
             use_linear_phase = false;
        } else if (tonality > 0.6) {
            use_linear_phase = true;
        } else if (avg_margin > 9.0) {
            use_linear_phase = true;
        } else {
            use_linear_phase = false;
        }

        float band_start_freq = psy_model->get_freqs()[start];
        if (ch_name == "Mid" && band_start_freq < 9000) use_linear_phase = false;
        if (ch_name == "Side" && band_start_freq < 7000) use_linear_phase = false;

        if (use_linear_phase) {
            mixed_filter_spectrum.segment(start, end-start) = H_linear.segment(start, end-start);
            linear_phase_mask[i] = 1.0;
        } else {
            mixed_filter_spectrum.segment(start, end-start) = H_min.segment(start, end-start);
        }
    }

    if (DEBUG_MODE && chunk_index < 5)
      _debug_plot_filter_selection(analysis.power_db, analysis.masking_threshold_db, analysis.tonality, linear_phase_mask, chunk_index, ch_name);

    return mixed_filter_spectrum;
}

DspUtils::VectorF IntelligentRDOResampler::_resample_channel(const DspUtils::VectorF& channel_data, const std::string& ch_name) {
    std::cout << "  - Resampling Channel " << ch_name << "..." << std::endl;
    long long num_chunks_calc = (channel_data.size() > chunk_size) ? (1 + (channel_data.size() - chunk_size) / hop_size) : 1;
    long long num_chunks = std::max(0LL, num_chunks_calc);

    long long output_len = static_cast<long long>(std::ceil(channel_data.size() * resample_ratio));
    DspUtils::VectorF output_signal = DspUtils::VectorF::Zero(output_len);
    int resampled_hop_size = static_cast<int>(hop_size * resample_ratio);
    long long output_ptr = 0;

    for (long long i = 0; i < num_chunks; ++i) {
        long long start = i * hop_size;
        DspUtils::VectorF chunk_raw = DspUtils::VectorF::Zero(chunk_size);
        long long signal_part_len = std::min(static_cast<long long>(chunk_size), channel_data.size() - start);
        if(signal_part_len > 0) chunk_raw.head(signal_part_len) = channel_data.segment(start, signal_part_len);

        DspUtils::VectorCF filter_spectrum = _design_intelligent_mixed_phase_filter(chunk_raw, i, ch_name);
        DspUtils::VectorF chunk_windowed = chunk_raw.array() * window.array();
        DspUtils::VectorCF chunk_spectrum = DspUtils::rfft(chunk_windowed, chunk_size);
        DspUtils::VectorCF filtered_spectrum = chunk_spectrum.array() * filter_spectrum.array();

        int target_fft_size = static_cast<int>(chunk_size * resample_ratio);
        DspUtils::VectorCF resampled_spectrum = DspUtils::VectorCF::Zero(target_fft_size / 2 + 1);
        int n_min = std::min(static_cast<int>(filtered_spectrum.size()), static_cast<int>(resampled_spectrum.size()));
        resampled_spectrum.head(n_min) = filtered_spectrum.head(n_min);

        DspUtils::VectorF processed_chunk = DspUtils::irfft(resampled_spectrum, target_fft_size);

        long long output_end = output_ptr + processed_chunk.size();
        if (output_end > output_signal.size()) {
            processed_chunk.conservativeResize(std::max(0LL, output_signal.size() - output_ptr));
        }

        if (processed_chunk.size() > 0) {
            output_signal.segment(output_ptr, processed_chunk.size()).array() += processed_chunk.array();
        }
        output_ptr += resampled_hop_size;
    }
    return output_signal * resample_ratio;
}


DspUtils::MatrixF IntelligentRDOResampler::resample(const DspUtils::MatrixF& signal, bool is_ms_processing, bool disable_hfe) {
    long long out_rows = static_cast<long long>(signal.rows() * resample_ratio);
    if(out_rows <= 0) return DspUtils::MatrixF(0, signal.cols());
    DspUtils::MatrixF output(out_rows, signal.cols());

    if (signal.cols() == 2) {
        std::string ch1_name = is_ms_processing ? "Mid" : "1";
        std::string ch2_name = is_ms_processing ? "Side" : "2";
        output.col(0) = _resample_channel(signal.col(0), ch1_name);
        output.col(1) = _resample_channel(signal.col(1), ch2_name);
    } else {
        output.col(0) = _resample_channel(signal.col(0), "Mono");
    }

    if (is_upsampling && !disable_hfe) {
        std::cout << "\nApplying High-Frequency Excitation post-resampling..." << std::endl;
        if (output.cols() == 2) {
             std::string ch1_name = is_ms_processing ? "Mid" : "1";
             std::string ch2_name = is_ms_processing ? "Side" : "2";
            output.col(0) = hfe_synthesizer->synthesize(output.col(0), ch1_name);
            output.col(1) = hfe_synthesizer->synthesize(output.col(1), ch2_name);
        } else {
            output.col(0) = hfe_synthesizer->synthesize(output.col(0), "Mono");
        }
    }
    return output;
}

// ===================================================================================
// 7. メイン実行部 (main)
// (変更なし)
// ===================================================================================
int main(int argc, char* argv[]) {
    cxxopts::Options options("Sradcon", "Sradcon audio resampler");

    options.add_options()
        ("i,input", "Input WAV file path", cxxopts::value<std::string>())
        ("o,output", "Output WAV file path", cxxopts::value<std::string>())
        ("target_sr", "Target sample rate in Hz", cxxopts::value<int>()->default_value("44100"))
        ("chunk_size", "Resampling chunk size", cxxopts::value<int>()->default_value("8192"))
        ("filter_taps", "Number of FIR filter taps", cxxopts::value<int>()->default_value("32767"))
        ("disable_hfe", "Disable High-Frequency Excitation on upsampling", cxxopts::value<bool>()->default_value("false"))
        ("target_bit_depth", "Target bit depth (8, 16, 24, 32 for PCM, 64 for float)", cxxopts::value<int>()->default_value("16"))
        ("float32_output", "Force 32-bit float output", cxxopts::value<bool>()->default_value("false"))
        ("no_dither", "Disable dither and noise shaping", cxxopts::value<bool>()->default_value("false"))
        ("lpc_order", "Order of the LPC noise shaping filter", cxxopts::value<int>()->default_value("16"))
        ("debug", "Enable detailed debug mode", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }

    if (result.count("help") || !result.count("input") || !result.count("output")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    DEBUG_MODE = result["debug"].as<bool>();
    if (DEBUG_MODE) {
        DEBUG_PATH = "debug_plots_cpp";
        std::filesystem::create_directories(DEBUG_PATH);
        std::cout << "--- DEBUG MODE ENABLED --- (Data will be saved to '" << DEBUG_PATH.string() << "')" << std::endl;
    }

    try {
        std::cout << "Reading input file: " << result["input"].as<std::string>() << std::endl;
        AudioFile audio_in;
        if (!audio_in.load(result["input"].as<std::string>())) return 1;
        std::cout << "  - Original SR: " << audio_in.sr << " Hz, Channels: " << audio_in.channels << ", Duration: " << (float)audio_in.frames / audio_in.sr << "s" << std::endl;

        DspUtils::MatrixF signal_to_process = audio_in.data;
        bool is_stereo = audio_in.channels == 2;
        bool is_ms_processed = false;
        if (is_stereo) {
            std::cout << "\nApplying Mid/Side encoding for stereo signal." << std::endl;
            DspUtils::VectorF mid = (audio_in.data.col(0) + audio_in.data.col(1)) / 2.0f;
            DspUtils::VectorF side = (audio_in.data.col(0) - audio_in.data.col(1)) / 2.0f;
            signal_to_process.col(0) = mid;
            signal_to_process.col(1) = side;
            is_ms_processed = true;
        }

        int target_sr = result["target_sr"].as<int>();
        DspUtils::MatrixF resampled_signal;
        if (audio_in.sr == target_sr) {
            std::cout << "\nTarget sample rate is the same as original. Skipping resampling." << std::endl;
            resampled_signal = signal_to_process;
        } else {
            std::cout << "\nResampling from " << audio_in.sr << " Hz to " << target_sr << " Hz..." << std::endl;
            IntelligentRDOResampler resampler(audio_in.sr, target_sr,
                result["chunk_size"].as<int>(),
                result["filter_taps"].as<int>(),
                128);
            resampled_signal = resampler.resample(signal_to_process, is_ms_processed, result["disable_hfe"].as<bool>());
            std::cout << "Resampling complete." << std::endl;
        }

        DspUtils::MatrixF output_signal = resampled_signal;
        std::string subtype = "FLOAT";
        int target_bit_depth = result["target_bit_depth"].as<int>();

        if (result["float32_output"].as<bool>()) {
            std::cout << "\nOutput format set to 32-bit float. Dithering is skipped." << std::endl;
            subtype = "FLOAT";
        } else if (target_bit_depth == 64) {
            std::cout << "\nOutput format set to 64-bit float (DOUBLE). Dithering is skipped." << std::endl;
            subtype = "DOUBLE";
        } else if (target_bit_depth == 8 || target_bit_depth == 16 || target_bit_depth == 24 || target_bit_depth == 32) {
            if (!result["no_dither"].as<bool>()) {
                int lpc_order = result["lpc_order"].as<int>();
                std::cout << "\nApplying " << target_bit_depth << "-bit dither and adaptive noise shaping (LPC Order: " << lpc_order << ")..." << std::endl;
                int shaper_chunk_size = std::max(1024, result["chunk_size"].as<int>() / 4);
                RDONoiseShaper shaper(target_bit_depth, target_sr, shaper_chunk_size, lpc_order);
                output_signal = shaper.process(resampled_signal);
                std::cout << "Dithering and shaping complete." << std::endl;
            } else {
                 std::cout << "\nDithering disabled by user." << std::endl;
            }
            subtype = (target_bit_depth == 8) ? "PCM_U8" : "PCM_" + std::to_string(target_bit_depth);
        } else {
            std::cerr << "Error: Unsupported target bit depth '" << target_bit_depth << "'." << std::endl;
            return 1;
        }

        DspUtils::MatrixF final_output = output_signal;
        if (is_ms_processed) {
            std::cout << "\nConverting signal back from Mid/Side to Left/Right." << std::endl;
            DspUtils::VectorF mid_ch = output_signal.col(0);
            DspUtils::VectorF side_ch = output_signal.col(1);
            final_output.col(0) = mid_ch + side_ch;
            final_output.col(1) = mid_ch - side_ch;
        }

        final_output = final_output.cwiseMax(-1.0f).cwiseMin(1.0f);

        AudioFile audio_out;
        audio_out.data = final_output;
        audio_out.sr = target_sr;
        audio_out.channels = final_output.cols();
        std::cout << "\nWriting output file: " << result["output"].as<std::string>() << " (SR: " << target_sr << ", Format: " << subtype << ")" << std::endl;
        if(!audio_out.save(result["output"].as<std::string>(), target_sr, subtype)) return 1;

        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An unhandled error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
