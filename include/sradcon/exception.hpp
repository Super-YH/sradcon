#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace Sradcon {

class SradconException : public std::exception {
public:
    explicit SradconException(const std::string& message)
        : message_(message) {}

    virtual ~SradconException() noexcept = default;

    virtual const char* what() const noexcept override {
        return message_.c_str();
    }

protected:
    std::string message_;
};

class FileIOException : public SradconException {
public:
    explicit FileIOException(const std::string& filename, const std::string& reason)
        : SradconException(BuildMessage(filename, reason)) {}

private:
    static std::string BuildMessage(const std::string& filename, const std::string& reason) {
        std::ostringstream oss;
        oss << "File I/O error for '" << filename << "': " << reason;
        return oss.str();
    }
};

class AudioFormatException : public SradconException {
public:
    explicit AudioFormatException(const std::string& message)
        : SradconException("Audio format error: " + message) {}
};

class InvalidParameterException : public SradconException {
public:
    explicit InvalidParameterException(const std::string& param_name, const std::string& reason)
        : SradconException(BuildMessage(param_name, reason)) {}

private:
    static std::string BuildMessage(const std::string& param_name, const std::string& reason) {
        std::ostringstream oss;
        oss << "Invalid parameter '" << param_name << "': " << reason;
        return oss.str();
    }
};

class ProcessingException : public SradconException {
public:
    explicit ProcessingException(const std::string& stage, const std::string& reason)
        : SradconException(BuildMessage(stage, reason)) {}

private:
    static std::string BuildMessage(const std::string& stage, const std::string& reason) {
        std::ostringstream oss;
        oss << "Processing error in '" << stage << "': " << reason;
        return oss.str();
    }
};

class MemoryException : public SradconException {
public:
    explicit MemoryException(const std::string& reason)
        : SradconException("Memory error: " + reason) {}
};

} // namespace Sradcon
