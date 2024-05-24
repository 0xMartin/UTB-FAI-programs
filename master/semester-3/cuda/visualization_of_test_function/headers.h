#ifndef __HEADERS_H__
#define __HEADERS_H__

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK_RETURN(value)                                  \
	{                                                             \
		cudaError_t err = value;                                  \
		if (err != cudaSuccess)                                   \
		{                                                         \
			fprintf(stderr, "Error %s at line %d in file %s\n",   \
					cudaGetErrorString(err), __LINE__, __FILE__); \
			exit(1);                                              \
		}                                                         \
	}

#define BLOCK_SIZE (16u)

static __host__ __device__ void rainbowPalette(float value, unsigned char *r, unsigned char *g, unsigned char *b)
{
	// souctova kombinace barev
	*r = *g = *b = 0;

	// color 1 (blue)
	*b += max((1.0 - abs(value * 4.0)), 0.0) * 255;

	// color 2 (cyan)
	*g += max((1.0 - abs(value * 4.0 - 1.0)), 0.0) * 255;
	*b += max((1.0 - abs(value * 4.0 - 1.0)), 0.0) * 255;

	// color 3 (green)
	*g += max((1.0 - abs(value * 4.0 - 2.0)), 0.0) * 255;

	// color 4 (yellow)
	*r += max((1.0 - abs(value * 4.0 - 3.0)), 0.0) * 255;
	*g += max((1.0 - abs(value * 4.0 - 3.0)), 0.0) * 255;

	// color 5 (read)
	*r += max((1.0 - abs(value * 4.0 - 4.0)), 0.0) * 255;
}

#endif