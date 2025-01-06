#pragma once
#include "Neon/Neon.h"

template<typename dType, int cardinality>
auto axpy_repetition(Neon::index_3d dim,
                     int ngpus,
                     int iterations,
                     int repetitions, int argc, char **argv) ;

extern template auto axpy_repetition<int, 1>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<int, 3>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<int, 5>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<int, 19>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<int, 27>(Neon::index_3d, int, int, int, int, char **);

extern template auto axpy_repetition<float, 1>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<float, 3>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<float, 5>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<float, 19>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<float, 27>(Neon::index_3d, int, int, int, int, char **);

extern template auto axpy_repetition<double, 1>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<double, 3>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<double, 5>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<double, 19>(Neon::index_3d, int, int, int, int, char **);
extern template auto axpy_repetition<double, 27>(Neon::index_3d, int, int, int, int, char **);
