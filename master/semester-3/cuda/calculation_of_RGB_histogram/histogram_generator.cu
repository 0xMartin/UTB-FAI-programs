#ifndef __HISTOGRAM_GENERATOR_H__
#define __HISTOGRAM_GENERATOR_H__

#include <iostream>
#include <cuda_runtime.h>
#include "image_rgb.cu"

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

// velikosti bloku pro jednotlive kernely
#define BLOCK_SIZE_HISTOGRAM 32
#define BLOCK_SIZE_COMPUTE_MAX 32
#define BLOCK_SIZE_IMAGE 64

// pruhlednost histogramu
#define HISTOGRAM_COLOR_INTENSITY 1.0

// delka pole histogramu
#define HISTOGRAM_LEN 256

namespace HistogramGenerator
{

    /// @brief Struktura histogramu pro RGB obrazek
    struct RGBHistogram
    {
        unsigned int *red;   /** Histogram cervene barvy */
        unsigned int *green; /** Histogram zelene barvy */
        unsigned int *blue;  /** Histogram modre barvy */
        unsigned int *max;   /** Maximalni hodnota histogramu. Jedna spolecna pro vsechny kanaly */
    };

    /// @brief Kernel pro vygenerovani histogramu
    /// @param in_channel Vstupni barevny kanal obrazku (1D)
    /// @param histogram Vystupni histogram. Musi mit velikost minimalne 256 (1D)
    /// @param width Sirka obrazku
    /// @param height Vyska obrazku
    __global__ static void generate_histogram_kernel(unsigned char *in_channel,
                                                     unsigned int *histogram,
                                                     unsigned int width,
                                                     unsigned int height)
    {
        // minimalni velikost bloku musi byt takova aby platilo: blockDim.x * blockDim.y >= 256
        // pozice pixelu v celem obrazku 2D
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        // 1D pozice v bloku 2D (i = x + y * width)
        int i = threadIdx.x + threadIdx.y * blockDim.x;

        // inicializace lokalniho histogramu pro blok (pro indexaci vyuzije 1D souradnici vlakna v bloku)
        __shared__ unsigned int histogram_private[256];
        if (i < HISTOGRAM_LEN)
        {
            histogram_private[i] = 0;
        }

        // nacte cast obrazku pro tento blok
        __shared__ unsigned char in_channel_shared[BLOCK_SIZE_HISTOGRAM * BLOCK_SIZE_HISTOGRAM];
        if (x < width && y < height)
        {
            in_channel_shared[i] = in_channel[x + y * width];
        }

        __syncthreads();

        // provede vypocet histogramu (lokalne)
        if (x < width && y < height)
        {
            atomicAdd(&(histogram_private[in_channel_shared[i]]), 1);
        }

        __syncthreads();

        // vsechny hodnoty lokalniho histogramu pricte ke globalnimu histogramu (pro indexaci vyuzije 1D souradnici vlakna v bloku)
        if (i < HISTOGRAM_LEN)
        {
            atomicAdd(&(histogram[i]), histogram_private[i]);
        }
    }

    /// @brief Kernel, ktery nalezne maximalni hodnotu v histogramu
    /// @param histogram Vstupni histogram jeden vybrany barevny kanal obrazku (1D)
    /// @param max Vystupni prommena ve ktere bude po provedeni kernelu zapsana maximalni hodnota (musi byt nastaveno na 0!!)
    __global__ static void compute_max_kernel(unsigned int *histogram,
                                              unsigned int *max)
    {
        // nacte cast histogramu do sdilene pamety (pro vsechny kanaly)
        __shared__ unsigned int histogram_shared[BLOCK_SIZE_COMPUTE_MAX];
        if (threadIdx.x < BLOCK_SIZE_COMPUTE_MAX)
        {
            histogram_shared[threadIdx.x] = histogram[threadIdx.x + blockIdx.x * blockDim.x];
        }

        // sdilena max hodnota
        __shared__ unsigned int max_private;
        if (threadIdx.x == 0)
        {
            max_private = 0;
        }

        __syncthreads();

        // max pro lokalni sdilenou hodnotu
        if (threadIdx.x < BLOCK_SIZE_COMPUTE_MAX)
        {
            atomicMax(&max_private, histogram_shared[threadIdx.x]);
        }

        __syncthreads();

        // max pro globalni hodnotu
        if (threadIdx.x == 0)
        {
            atomicMax(max, max_private);
        }
    }

    /// @brief Kernel, ktery vygeneruje obrazek histogramu. Osa x je rozdelna na bloky, kazde vlakno bloku vykresli vsechny pixely y = <0, height>
    /// @param out_channel Vstupni barevny kanal obrazku (1D)
    /// @param histogram Histogram ktery bude do obrazku vykreslovan
    /// @param maxHist Maximali hodnota histogramu (pro normalizaci) jde o pointer na cislo v poli v prommene max v histogramu
    /// @param width Sirka generovaneho obrazku
    /// @param height Vyska generovaneho obrazku
    __global__ static void generate_image_kernel(unsigned char *out_channel,
                                                 unsigned int *histogram,
                                                 unsigned int *maxHist,
                                                 unsigned int width,
                                                 unsigned int height)
    {
        // x pozice v obrazku
        int x = threadIdx.x + blockIdx.x * blockDim.x;

        if (x < width)
        {
            // prepocita pozici x souradnice ve vyslednem obrazku na i souradnici v histogramu <0, 255>
            float step = (float)width / HISTOGRAM_LEN;
            int i = (x - 1) / step;
            if (i < 0)
                i = 0;

            // vypocet vysky histogramu pro indexe "i". v tomto pripade vypocet y_min jelikoz cim nizssi je souradnci tim vyssi bude v obrazku vykreslena
            int y_min = (int)((1.0 - ((float)histogram[i]) / *maxHist) * height);

            // vykresleni radu pixelu
            for (int y = height - 1; y > y_min && y >= 0; --y)
            {
                out_channel[x + y * width] = 255 * HISTOGRAM_COLOR_INTENSITY;
            }
        }
    }

    /// @brief Inicializuje histogram. Histogramu bude alokovan jen na pameti GPU
    /// @param hitogram Ukazatel na strukturu histogramu
    /// @return True v pripade uspesne inicializace
    static bool initHistogram(RGBHistogram *histogram)
    {
        if (histogram == nullptr)
        {
            return false;
        }

        histogram->red = nullptr;
        histogram->green = nullptr;
        histogram->blue = nullptr;
        histogram->max = nullptr;

        size_t ch_siz = HISTOGRAM_LEN * sizeof(unsigned int);
        CUDA_CHECK_RETURN(cudaMalloc(&histogram->red, ch_siz));
        CUDA_CHECK_RETURN(cudaMalloc(&histogram->green, ch_siz));
        CUDA_CHECK_RETURN(cudaMalloc(&histogram->blue, ch_siz));
        CUDA_CHECK_RETURN(cudaMalloc(&histogram->max, sizeof(unsigned int)));
        int *max = new int[1];
        max[0] = 0;
        CUDA_CHECK_RETURN(cudaMemcpy(histogram->max,
                                     max,
                                     sizeof(unsigned int),
                                     cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        delete[] max;

        return true;
    }

    /// @brief Odstrani histogram z pameti GPU
    /// @param hitogram Ukazatel na strukturu histogramu
    static void deleteHistogram(RGBHistogram *histogram)
    {
        if (histogram != nullptr)
        {
            if (histogram->red)
                CUDA_CHECK_RETURN(cudaFree(histogram->red));
            if (histogram->green)
                CUDA_CHECK_RETURN(cudaFree(histogram->green));
            if (histogram->blue)
                CUDA_CHECK_RETURN(cudaFree(histogram->blue));
            if (histogram->max)
                CUDA_CHECK_RETURN(cudaFree(histogram->max));
            histogram->red = nullptr;
            histogram->green = nullptr;
            histogram->blue = nullptr;
            histogram->max = nullptr;
        }
    }

    /// @brief Vygeneruje RGB histogram ze vstupniho obrazku s vyuzitim stream
    /// @param img Ukazatel na RGB obrazek ze ktereho bude histogram generovan [GPU]
    /// @param histogram Ukazatel na histogram do ktereho budou vygenerovany vysledky [GPU]
    /// @return True v pripade uspesneho spusteni generovani histogramu
    static bool generateRGBHistogramAsync(ImageRGB::Image *img, RGBHistogram *histogram)
    {
        if (img == nullptr || histogram == nullptr)
        {
            return false;
        }
        // generovani histogramu pro jednotlive barvene slozky
        dim3 blockSize(BLOCK_SIZE_HISTOGRAM, BLOCK_SIZE_HISTOGRAM);
        dim3 gridSize((img->width + BLOCK_SIZE_HISTOGRAM - 1) / BLOCK_SIZE_HISTOGRAM,
                      (img->height + BLOCK_SIZE_HISTOGRAM - 1) / BLOCK_SIZE_HISTOGRAM);

        generate_histogram_kernel<<<gridSize, blockSize, 0, ImageRGB::stream_red>>>(
            img->red_channel,
            histogram->red,
            img->width,
            img->height);

        generate_histogram_kernel<<<gridSize, blockSize, 0, ImageRGB::stream_green>>>(
            img->green_channel,
            histogram->green,
            img->width,
            img->height);

        generate_histogram_kernel<<<gridSize, blockSize, 0, ImageRGB::stream_blue>>>(
            img->blue_channel,
            histogram->blue,
            img->width,
            img->height);

        // vypocet maxima pro kazdy histogram
        int gridSize2 = (HISTOGRAM_LEN + BLOCK_SIZE_COMPUTE_MAX - 1) / BLOCK_SIZE_COMPUTE_MAX;
        compute_max_kernel<<<gridSize2, BLOCK_SIZE_COMPUTE_MAX, 0, ImageRGB::stream_red>>>(
            histogram->red,
            histogram->max);

        compute_max_kernel<<<gridSize2, BLOCK_SIZE_COMPUTE_MAX, 0, ImageRGB::stream_green>>>(
            histogram->green,
            histogram->max);

        compute_max_kernel<<<gridSize2, BLOCK_SIZE_COMPUTE_MAX, 0, ImageRGB::stream_blue>>>(
            histogram->blue,
            histogram->max);

        return true;
    }

    /// @brief Vygeneruje obrazek histogramu s vyuzitim stream
    /// @param img Vystupni obrazek do ktereho bude histogram vykreslen [GPU]
    /// @param histogram Vekreslovany hitogram [GPU]
    /// @return True v pripade uspesneho spusteni generovani obrazku
    static bool generateImageAsync(ImageRGB::Image *img, RGBHistogram *histogram)
    {
        if (img == nullptr || histogram == nullptr)
        {
            return false;
        }

        int gridSize = (img->width + BLOCK_SIZE_IMAGE - 1) / BLOCK_SIZE_IMAGE;

        // vygeneruje obrazek
        generate_image_kernel<<<gridSize, BLOCK_SIZE_IMAGE, 0, ImageRGB::stream_red>>>(
            img->red_channel,
            histogram->red,
            histogram->max,
            img->width,
            img->height);

        generate_image_kernel<<<gridSize, BLOCK_SIZE_IMAGE, 0, ImageRGB::stream_green>>>(
            img->green_channel,
            histogram->green,
            histogram->max,
            img->width,
            img->height);

        generate_image_kernel<<<gridSize, BLOCK_SIZE_IMAGE, 0, ImageRGB::stream_blue>>>(
            img->blue_channel,
            histogram->blue,
            histogram->max,
            img->width,
            img->height);
        
        return true;
    }

    /// @brief Vygeneruje RGB histogram ze vstupniho obrazku
    /// @param img Ukazatel na RGB obrazek ze ktereho bude histogram generovan [GPU]
    /// @param histogram Ukazatel na histogram do ktereho budou vygenerovany vysledky [GPU]
    /// @return True v pripade uspesneho spusteni generovani histogramu
    static bool generateRGBHistogram(ImageRGB::Image *img, RGBHistogram *histogram)
    {
        if (img == nullptr || histogram == nullptr)
        {
            return false;
        }
        // generovani histogramu pro jednotlive barvene slozky
        dim3 blockSize(BLOCK_SIZE_HISTOGRAM, BLOCK_SIZE_HISTOGRAM);
        dim3 gridSize((img->width + BLOCK_SIZE_HISTOGRAM - 1) / BLOCK_SIZE_HISTOGRAM,
                      (img->height + BLOCK_SIZE_HISTOGRAM - 1) / BLOCK_SIZE_HISTOGRAM);
        generate_histogram_kernel<<<gridSize, blockSize>>>(
            img->red_channel,
            histogram->red,
            img->width,
            img->height);
        generate_histogram_kernel<<<gridSize, blockSize>>>(
            img->green_channel,
            histogram->green,
            img->width,
            img->height);
        generate_histogram_kernel<<<gridSize, blockSize>>>(
            img->blue_channel,
            histogram->blue,
            img->width,
            img->height);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // vypocet maxima pro kazdy histogram
        int gridSize2 = (HISTOGRAM_LEN + BLOCK_SIZE_COMPUTE_MAX - 1) / BLOCK_SIZE_COMPUTE_MAX;
        compute_max_kernel<<<gridSize2, BLOCK_SIZE_COMPUTE_MAX>>>(
            histogram->red,
            histogram->max);

        compute_max_kernel<<<gridSize2, BLOCK_SIZE_COMPUTE_MAX>>>(
            histogram->green,
            histogram->max);

        compute_max_kernel<<<gridSize2, BLOCK_SIZE_COMPUTE_MAX>>>(
            histogram->blue,
            histogram->max);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        return true;
    }

    /// @brief Vygeneruje obrazek histogramu
    /// @param img Vystupni obrazek do ktereho bude histogram vykreslen [GPU]
    /// @param histogram Vekreslovany hitogram [GPU]
    /// @return True v pripade uspesneho spusteni generovani obrazku
    static bool generateImage(ImageRGB::Image *img, RGBHistogram *histogram)
    {
        if (img == nullptr || histogram == nullptr)
        {
            return false;
        }

        int gridSize = (img->width + BLOCK_SIZE_IMAGE - 1) / BLOCK_SIZE_IMAGE;

        // vygeneruje obrazek
        generate_image_kernel<<<gridSize, BLOCK_SIZE_IMAGE>>>(
            img->red_channel,
            histogram->red,
            histogram->max,
            img->width,
            img->height);

        generate_image_kernel<<<gridSize, BLOCK_SIZE_IMAGE>>>(
            img->green_channel,
            histogram->green,
            histogram->max,
            img->width,
            img->height);

        generate_image_kernel<<<gridSize, BLOCK_SIZE_IMAGE>>>(
            img->blue_channel,
            histogram->blue,
            histogram->max,
            img->width,
            img->height);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        return true;
    }

    /// @brief Vygeneruje RGB histogram ze vstupniho obrazku
    /// @param img Ukazatel na RGB obrazek ze ktereho bude histogram generovan [GPU]
    /// @param histogram Ukazatel na histogram do ktereho budou vygenerovany vysledky [GPU]
    /// @param type Zpusob generovani
    /// @return True v pripade uspesneho spusteni generovani histogramu
    static bool generateRGBHistogramSyncOrAsync(ImageRGB::Image *img, RGBHistogram *histogram, ImageRGB::ProcessMode type)
    {
        switch (type)
        {
        case ImageRGB::DEFAULT:
            return generateRGBHistogram(img, histogram);
        case ImageRGB::STREAM:
            return generateRGBHistogramAsync(img, histogram);
        default:
            return false;
        }
    }

    /// @brief Vygeneruje obrazek histogramu
    /// @param img Vystupni obrazek do ktereho bude histogram vykreslen [GPU]
    /// @param histogram Vekreslovany hitogram [GPU]
    /// @param type Zpusob generovani
    /// @return True v pripade uspesneho spusteni generovani obrazku
    static bool generateImageSyncOrAsync(ImageRGB::Image *img, RGBHistogram *histogram, ImageRGB::ProcessMode type)
    {
        switch (type)
        {
        case ImageRGB::DEFAULT:
            return generateImage(img, histogram);
        case ImageRGB::STREAM:
            return generateImageAsync(img, histogram);
        default:
            return false;
        }
    }

}

#endif