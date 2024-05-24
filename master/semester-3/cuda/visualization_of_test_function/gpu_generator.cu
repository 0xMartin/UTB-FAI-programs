#ifndef __GPU_GENERATOR_CU__
#define __GPU_GENERATOR_CU__

#include "headers.h"
#include "opt_function.h"

/// @brief Generator matice hodnot z testovaci funkce
namespace GPUGenerator
{
    /// @brief Kernel pro generovani matici hodnot testovaci funkce v rozsahu (X ∈ [minX,maxX], Y ∈ [minY,maxY])
    ///        Pomer (maxX - minX)/(maxY-minY) se musi rovnat (img_width)/(img_height) jinak dojde k nerovnomernemu meritku
    ///
    /// @param matrix Pointer na matici do ktere se budou vysledne hodnoty zapisovat (GPU)
    /// @param width Sirka matice
    /// @param height Vyska matice
    /// @param minX Minimalni hodnota na ose X
    /// @param maxX Maximalni hodnota na ose X
    /// @param minY Minimalni hodnota na ose Y
    /// @param maxY Maximalni hodnota na ose Y
    static __global__ void generateKernel(float *matrix_d,
                                          int width,
                                          int height,
                                          float minX,
                                          float maxX,
                                          float minY,
                                          float maxY)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < height && j < width)
        {
            // vypocet souradni x, y
            float x = minX + j * (maxX - minX) / (width - 1);
            float y = minY + i * (maxY - minY) / (height - 1);
            // pro souradnice x,y vypocita hodnotu testovaci funkce
            matrix_d[i * width + j] = opt(x, y);
        }
    }

    /// @brief Na GPU vygeneruje matici hodnot testovaci funkce v rozsahu (X ∈ [minX,maxX], Y ∈ [minY,maxY]).
    ///        Pomer (maxX - minX)/(maxY-minY) se musi rovnat (img_width)/(img_height) jinak dojde k nerovnomernemu meritku
    ///
    /// @param matrix Pointer na matici do ktere se budou vysledne hodnoty zapisovat (GPU)
    /// @param width Sirka matice
    /// @param height Vyska matice
    /// @param minX Minimalni hodnota na ose X
    /// @param maxX Maximalni hodnota na ose X
    /// @param minY Minimalni hodnota na ose Y
    /// @param maxY Maximalni hodnota na ose Y
    static void generateGPU(float *matrix_d,
                            int width,
                            int height,
                            float minX,
                            float maxX,
                            float minY,
                            float maxY)
    {
        if (abs(((float)width/height) - ((maxX - minX)/(maxY - minY))) > 0.01) {
            std::cout << "Wrong ratio, the image becomes distorted!!" << std::endl;
        }

        dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

        generateKernel<<<grid_size, block_size>>>(matrix_d, width, height, minX, maxX, minY, maxY);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
}

#endif