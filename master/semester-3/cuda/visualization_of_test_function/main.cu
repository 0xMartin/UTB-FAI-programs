#include <iostream>
#include <string.h>
#include <chrono>

#include "headers.h"
#include "opt_function.h"

#include "cpu_generator.h"
#include "gpu_generator.cu"

#include "cpu_image.h"
#include "gpu_image.cu"

// konfigurace
static bool mode_gpu;
static float minX;
static float maxX;
static float minY;
static float maxY;
static float pixelsPerRealUnit;
static char *fileName;

static unsigned int __img_width;
static unsigned int __img_height;

static bool __parseArgumnets(int argc, char *argv[])
{
    if (argc < 8)
    {
        // Výpis informací o správném použití aplikace
        std::cout << "Incorrect format. Correct usage of app is: ./app_name {MODE} {MIN_X} {MAX_X} {MIN_Y} {MAX_Y} {PIXELS_PER_REAL_UNIT} {FILE_NAME}" << std::endl
                  << std::endl;
        std::cout << "Argument 'MODE' (computing mode):" << std::endl;
        std::cout << "\tGPU or CPU" << std::endl
                  << std::endl;
        std::cout << "Argument 'MIN_X', 'MAX_X' (Range of X axis):" << std::endl
                  << std::endl;
        std::cout << "Argument 'MIN_Y', 'MAX_Y' (Range of Y axis):" << std::endl
                  << std::endl;
        std::cout << "Argument 'PIXELS_PER_REAL_UNIT' (Number of pixels per one unit of real range):" << std::endl
                  << std::endl;
        std::cout << "Argument 'FILE_NAME' (Name of output image file):" << std::endl
                  << std::endl;
        std::cout << "Example: ./app_name GPU -5.12 5.12 -5.12 5.12 300 output.png" << std::endl;
        return false;
    }
    else
    {
        mode_gpu = (strcmp(argv[1], "GPU") == 0);
        minX = std::atof(argv[2]);
        maxX = max(minX + 1, std::atof(argv[3]));
        minY = std::atof(argv[4]);
        maxY = max(minY + 1, std::atof(argv[5]));
        pixelsPerRealUnit = std::atoi(argv[6]);
        fileName = argv[7];

        __img_width = (unsigned int)((maxX - minX) * pixelsPerRealUnit);
        __img_height = (unsigned int)((maxY - minY) * pixelsPerRealUnit);

        std::cout << "Mode: " << (mode_gpu ? "GPU" : "CPU") << std::endl;
        std::cout << "Range X: " << minX << "," << maxX << std::endl;
        std::cout << "Range Y: " << minY << "," << maxY << std::endl;
        std::cout << "Pixels per real unit: " << pixelsPerRealUnit << std::endl;
        std::cout << "Real image size: " << __img_width << ", " << __img_height << std::endl;
        std::cout << "Ouput file: " << fileName << std::endl;

        return true;
    }
}

static void __cpu_generate()
{
    // alokace matice pro CPU
    float *matrix_h = new float[__img_width * __img_height];

    // generovani hodnot matice na CPU
    CPUGenerator::generate(matrix_h, __img_width, __img_height, minX, maxX, minY, maxY);

    // nalezeni minima a maxima v matici
    float minVal = matrix_h[0];
    float maxVal = matrix_h[0];
    for (int i = 0; i < __img_width * __img_height; ++i)
    {
        minVal = fmin(minVal, matrix_h[i]);
        maxVal = fmax(maxVal, matrix_h[i]);
    }

    // generovani obrazku na CPU
    CPUImage::generateImage(matrix_h, __img_width, __img_height, minVal, maxVal, fileName);

    delete[] matrix_h;
}

static void __gpu_generate()
{
    // alokace matice pro CPU
    float *matrix_h = new float[__img_width * __img_height];

    // alokace matice pro GPU
    size_t size = __img_width * __img_height * sizeof(float);
    float *matrix_d;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&matrix_d, size));

    // kopirovani matice z CPU na GPU
    CUDA_CHECK_RETURN(cudaMemcpy(matrix_d, matrix_h, size, cudaMemcpyHostToDevice));

    // generovani hodnot matice na GPU
    GPUGenerator::generateGPU(matrix_d, __img_width, __img_height, minX, maxX, minY, maxY);

    // kopirovani matice z GPU na CPU
    CUDA_CHECK_RETURN(cudaMemcpy(matrix_h, matrix_d, size, cudaMemcpyDeviceToHost));

    // nalezeni minima a maxima v matici
    float minVal = matrix_h[0];
    float maxVal = matrix_h[0];
    for (int i = 0; i < __img_width * __img_height; ++i)
    {
        minVal = fmin(minVal, matrix_h[i]);
        maxVal = fmax(maxVal, matrix_h[i]);
    }

    // generovani obrazku na gpu
    GPUImage::generateImage(matrix_d, __img_width, __img_height, minVal, maxVal, fileName);

    // dealokace pameti
    delete[] matrix_h;
    CUDA_CHECK_RETURN(cudaFree(matrix_d));
}

int main(int argc, char *argv[])
{
    // zpracovani argumentu
    if (!__parseArgumnets(argc, argv))
    {
        return -1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // spusti prislusnou cast programu pro vybranou varianut
    std::cout << "Start image generating ..." << std::endl;
    if (mode_gpu)
    {
        __gpu_generate();
    }
    else
    {
        __cpu_generate();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;

    return 0;
}