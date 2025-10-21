#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <memory>

namespace Sradcon {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

class Logger {
public:
    static Logger& Instance() {
        static Logger instance;
        return instance;
    }

    void SetLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = level;
    }

    void SetLogFile(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_file_.open(filename, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Warning: Could not open log file: " << filename << std::endl;
        }
    }

    void EnableConsole(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        console_enabled_ = enable;
    }

    void Log(LogLevel level, const std::string& message, const char* file = nullptr, int line = 0) {
        if (level < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        std::ostringstream oss;
        oss << GetTimestamp() << " [" << LevelToString(level) << "] ";

        if (file && line > 0) {
            oss << "[" << file << ":" << line << "] ";
        }

        oss << message << std::endl;

        std::string log_line = oss.str();

        if (console_enabled_) {
            if (level >= LogLevel::ERROR) {
                std::cerr << log_line;
            } else {
                std::cout << log_line;
            }
        }

        if (log_file_.is_open()) {
            log_file_ << log_line;
            log_file_.flush();
        }
    }

    ~Logger() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

private:
    Logger() : level_(LogLevel::INFO), console_enabled_(true) {}
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string GetTimestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    const char* LevelToString(LogLevel level) const {
        switch (level) {
            case LogLevel::DEBUG:    return "DEBUG";
            case LogLevel::INFO:     return "INFO";
            case LogLevel::WARNING:  return "WARN";
            case LogLevel::ERROR:    return "ERROR";
            case LogLevel::CRITICAL: return "CRITICAL";
            default:                 return "UNKNOWN";
        }
    }

    LogLevel level_;
    bool console_enabled_;
    std::ofstream log_file_;
    std::mutex mutex_;
};

// Convenience macros
#define LOG_DEBUG(msg) \
    Sradcon::Logger::Instance().Log(Sradcon::LogLevel::DEBUG, msg, __FILE__, __LINE__)

#define LOG_INFO(msg) \
    Sradcon::Logger::Instance().Log(Sradcon::LogLevel::INFO, msg)

#define LOG_WARNING(msg) \
    Sradcon::Logger::Instance().Log(Sradcon::LogLevel::WARNING, msg, __FILE__, __LINE__)

#define LOG_ERROR(msg) \
    Sradcon::Logger::Instance().Log(Sradcon::LogLevel::ERROR, msg, __FILE__, __LINE__)

#define LOG_CRITICAL(msg) \
    Sradcon::Logger::Instance().Log(Sradcon::LogLevel::CRITICAL, msg, __FILE__, __LINE__)

} // namespace Sradcon
