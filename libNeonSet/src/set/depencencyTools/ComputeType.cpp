#include "Neon/set/dependency/Pattern.h"

namespace Neon {
    auto PatternUtils::toString(Pattern val) -> std::string {
        switch (val) {
            case Pattern::MAP: {
                return "MAP";
            }
            case Pattern::STENCIL: {
                return "STENCIL";
            }
            case Pattern::REDUCE: {
                return "REDUCE";
            }
        }
        NEON_THROW_UNSUPPORTED_OPTION();
    }

    auto PatternUtils::fromInt(int val) -> Pattern {
        switch (val) {
            case 0: {
                return Pattern::MAP;
            }
            case 1: {
                return Pattern::STENCIL;
            }
            case 2: {
                return Pattern::REDUCE;
            }
            default:
                NEON_THROW_UNSUPPORTED_OPTION();
        }
    }
} // namespace Neon
