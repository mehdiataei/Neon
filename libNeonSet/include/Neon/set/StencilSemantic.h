#pragma once
#include <string>
#include <vector>

#include "Neon/core/core.h"

namespace Neon::set {

enum struct StencilSemantic
{
    standard = 0 /*<  Transfer for halo update on grid structure    */,
    streaming = 1 /*< Transfer for halo update on lattice structure */
};


struct StencilSemanticUtils
{
    static constexpr int nOptions = 2;

    static auto toString(StencilSemantic opt) -> std::string;
    static auto fromString(const std::string& opt) -> StencilSemantic;
    static auto getOptions() -> std::array<StencilSemantic, nOptions>;
    
    struct Cli
    {
        explicit Cli(std::string);
        explicit Cli(StencilSemantic model);
        Cli();

        auto getOption() -> StencilSemantic;
        auto set(const std::string& opt) -> void;
        auto getStringOptions() -> std::string;

       private:
        bool mSet = false;
        StencilSemantic mOption;
    };
};


}  // namespace Neon::set
