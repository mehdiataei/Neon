#pragma once
#include "Neon/set/Backend.h"

namespace Neon {
    /**
     * Enumeration for the supported type of computation by the skeleton
     * */
    enum struct Pattern {
        MAP /**< Map operation */ = 0,
        STENCIL /**< Stencil operation */ = 1,
        REDUCE /**< Reduction operation */ = 2
    };

    struct PatternUtils {
        /**
         * Returns a string for the selected allocator
         *
         * @param allocator
         * @return
         */
        static auto toString(Pattern val) -> std::string;

        static auto fromInt(int val) -> Pattern;

    };
} // namespace Neon
