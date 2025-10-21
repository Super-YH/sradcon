#pragma once

#include <Eigen/Dense>
#include <memory>

namespace Sradcon {

// Forward declarations for DSP utility types
namespace DspUtils {
    using VectorF = Eigen::VectorXf;
    using VectorCF = Eigen::VectorXcf;
    using MatrixF = Eigen::MatrixXf;
    using ArrayF = Eigen::ArrayXf;
    using ArrayCF = Eigen::ArrayXcf;
}

// Forward declarations
class AdvancedPsychoacousticModel;
class RDONoiseShaper;
class HighFrequencySynthesizer;
class IntelligentRDOResampler;

} // namespace Sradcon
