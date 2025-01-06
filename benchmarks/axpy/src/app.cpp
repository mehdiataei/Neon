#include "axpy.h"

#include "Neon/Neon.h"
#include "Neon/core/tools/clipp.h"
#include "Neon/domain/dGrid.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    Neon::init();
    // Default values
    int dim = 10;
    int ngpus = 1;
    std::string dtype = "int";
    int cardinality = 1;
    int iterations = 10;
    int repetitions = 1;
    bool show_help = false;

    // Create clipp parser
    auto cli = (
        clipp::option("--help").set(show_help) % "Show help",
        clipp::option("--dim") & clipp::value("dimension", dim) % "Dimension size (default: 10)",
        clipp::option("--ngpus") & clipp::value("number of GPUs", ngpus) % "Number of GPUs (default: 1)",
        clipp::option("--dtype") & clipp::value("data type", dtype) % "Data type (default: int)",
        clipp::option("--cardinality") & clipp::value("cardinality", cardinality) % "Cardinality (default: 1)",
        clipp::option("--iterations") & clipp::value("iterations", iterations) % "Number of iterations (default: 10)",
        clipp::option("--repetitions") & clipp::value("repetitions", repetitions) % "Number of repetitions (default: 1)"
    );

    // Parse command-line arguments
    if (!clipp::parse(argc, argv, cli) || show_help) {
        std::cout << "Usage:\n" << clipp::make_man_page(cli, argv[0]);
        return 0;
    }

    // Print parsed values
    std::cout << "Parsed arguments:\n";
    std::cout << "  dim: " << dim << "\n";
    std::cout << "  ngpus: " << ngpus << "\n";
    std::cout << "  dtype: " << dtype << "\n";
    std::cout << "  cardinality: " << cardinality << "\n";
    std::cout << "  iterations: " << iterations << "\n";
    std::cout << "  repetitions: " << repetitions << "\n";

    if (dtype == "int") {
        if (cardinality == 1) {
            axpy_repetition<int, 1>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 3) {
            axpy_repetition<int, 3>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 5) {
            axpy_repetition<int, 5>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 19) {
            axpy_repetition<int, 5>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 27) {
            axpy_repetition<int, 5>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
    }

    if (dtype == "float") {
        if (cardinality == 1) {
            axpy_repetition<float, 1>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 3) {
            axpy_repetition<float, 3>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 5) {
            axpy_repetition<float, 5>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 19) {
            axpy_repetition<float, 5>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 27) {
            axpy_repetition<float, 5>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
    }

    if (dtype == "double") {
        if (cardinality == 1) {
            axpy_repetition<double, 1>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 3) {
            axpy_repetition<double, 3>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 5) {
            axpy_repetition<double, 5>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 19) {
            axpy_repetition<double, 5>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
        if (cardinality == 27) {
            axpy_repetition<double, 5>(Neon::index_3d(dim), ngpus, iterations, repetitions, argc, argv);
        }
    }

    return 0;
}
