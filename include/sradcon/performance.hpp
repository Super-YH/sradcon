#pragma once

#include <chrono>
#include <string>
#include <map>
#include <mutex>
#include <iostream>
#include <iomanip>

namespace Sradcon {

class PerformanceMonitor {
public:
    static PerformanceMonitor& Instance() {
        static PerformanceMonitor instance;
        return instance;
    }

    class Timer {
    public:
        Timer(const std::string& name)
            : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
            PerformanceMonitor::Instance().RecordTiming(name_, duration);
        }

    private:
        std::string name_;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };

    void RecordTiming(const std::string& name, long long microseconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& stats = timings_[name];
        stats.count++;
        stats.total_us += microseconds;
        stats.min_us = std::min(stats.min_us, microseconds);
        stats.max_us = std::max(stats.max_us, microseconds);
    }

    void PrintReport(std::ostream& os = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);

        os << "\n========== Performance Report ==========\n";
        os << std::left << std::setw(30) << "Operation"
           << std::right << std::setw(10) << "Count"
           << std::setw(15) << "Total (ms)"
           << std::setw(15) << "Avg (ms)"
           << std::setw(15) << "Min (ms)"
           << std::setw(15) << "Max (ms)" << "\n";
        os << std::string(100, '-') << "\n";

        for (const auto& [name, stats] : timings_) {
            double total_ms = stats.total_us / 1000.0;
            double avg_ms = stats.total_us / 1000.0 / stats.count;
            double min_ms = stats.min_us / 1000.0;
            double max_ms = stats.max_us / 1000.0;

            os << std::left << std::setw(30) << name
               << std::right << std::setw(10) << stats.count
               << std::setw(15) << std::fixed << std::setprecision(2) << total_ms
               << std::setw(15) << std::fixed << std::setprecision(2) << avg_ms
               << std::setw(15) << std::fixed << std::setprecision(2) << min_ms
               << std::setw(15) << std::fixed << std::setprecision(2) << max_ms << "\n";
        }
        os << "========================================\n";
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        timings_.clear();
    }

    long long GetTotalMicroseconds(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = timings_.find(name);
        return (it != timings_.end()) ? it->second.total_us : 0;
    }

private:
    PerformanceMonitor() = default;
    PerformanceMonitor(const PerformanceMonitor&) = delete;
    PerformanceMonitor& operator=(const PerformanceMonitor&) = delete;

    struct TimingStats {
        long long count = 0;
        long long total_us = 0;
        long long min_us = LLONG_MAX;
        long long max_us = 0;
    };

    mutable std::mutex mutex_;
    std::map<std::string, TimingStats> timings_;
};

// Convenience macro for automatic timing
#define PERF_TIMER(name) \
    Sradcon::PerformanceMonitor::Timer _perf_timer_##__LINE__(name)

} // namespace Sradcon
