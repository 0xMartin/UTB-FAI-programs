#ifndef __GPU_IMAGE_CU__
#define __GPU_IMAGE_CU__

#include "headers.h"
#include "utils/pngio.h"

/// @brief Nastroje pro generovani obrazu z matice pomoci GPU
namespace GPUImage
{
    /// @brief Kernel pro vygenerovani obrazku z matice. Obrazek bude zobrazovat hodnoty z testovaci funkce pro 2D. [Modra = minimum, Cervena = maximum]
    /// @param matrix Pointer na matici vyslednych hodnot optimalizacni funkce (GPU)
    /// @param image Pointer na image buffer (GPU)
    /// @param width Sirka obrazku
    /// @param height Vyska obrazku
    /// @param minVal Minimalni hodnota v matici (pro normalizaci pixelu ve vyslednem obrazku)
    /// @param maxVal Maximalni hodnota v matici (pro normalizaci pixelu ve vyslednem obrazku)
    static __global__ void imageKernel(float *matrix_d,
                                       unsigned char *image_d,
                                       int width,
                                       int height,
                                       float minVal,
                                       float maxVal)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < height && j < width)
        {
            // normalizuje aktualni hodnotu matice[x, y]: 0.0 az 1.0 => 0.0 minimum, 1.0 maximum
            float normalized = (matrix_d[i * width + j] - minVal) / (maxVal - minVal);

            // vypocita index pixelu v image bufferu 3b RGB
            int pixel_i = (i * width + j) * 3;

            // vypocita barvy pixelu [minimu = modra, maximu = cervena]
            unsigned char r, g, b;
            rainbowPalette(normalized, &r, &g, &b);
            image_d[pixel_i] = r;
            image_d[pixel_i + 1] = g;
            image_d[pixel_i + 2] = b;
        }
    }

    /// @brief Metoda pro vygenerovani obrazku pomoci GPU [Modra = minimum, Cervena = maximum]
    /// @param matrix Pointer na matici vyslednych hodnot optimalizacni funkce (GPU)
    /// @param width Sirka obrazku
    /// @param height Vyska obrazku
    /// @param minVal Minimalni hodnota v matici (pro normalizaci pixelu ve vyslednem obrazku)
    /// @param maxVal Maximalni hodnota v matici (pro normalizaci pixelu ve vyslednem obrazku)
    /// @param fileName Nazev souboru, do ktereho bude ulozen vysledny PNG obrazek
    static void generateImage(float *matrix_d,
                              int width,
                              int height,
                              float minVal,
                              float maxVal,
                              const char *fileName)
    {
        // alokace pameti pro CPU
        unsigned char *h_data = new unsigned char[width * height * 3];

        // alokace pameti pro GPU
        size_t size = width * height * 3 * sizeof(unsigned char);
        unsigned char *d_data = NULL;
        CUDA_CHECK_RETURN(cudaMalloc((void **)&d_data, size));

        dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

        // spusti kernel na vygenerovani obrazku z matice
        imageKernel<<<grid_size, block_size>>>(matrix_d, d_data, width, height, minVal, maxVal);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // kopirovani hodnot z GPU na CPU
        CUDA_CHECK_RETURN(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

        // ulozi obrazek na disk
        png::image<png::rgb_pixel> img(width, height);
        pvg::rgbToPng(img, h_data);
        img.write(fileName);

        // dealokace pameti na GPU a CPU
        CUDA_CHECK_RETURN(cudaFree(d_data));
        delete[] h_data;
    }
}

#endif