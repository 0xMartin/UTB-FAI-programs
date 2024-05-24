#ifndef __OPT_FUNCTION_H__
#define __OPT_FUNCTION_H__

#include <cuda_runtime.h>
#include <cmath>

/// @brief Rastrigin testovaci funkce
/// @param x X pozice
/// @param y Y pozice
static __host__ __device__ float opt(float x, float y)
{
    const double A = 10.0;
    const double pi = 3.14159265358979323846;

    double term1 = x * x - A * cos(2 * pi * x);
    double term2 = y * y - A * cos(2 * pi * y);

    return 2 * A + term1 + term2;
}

#endif