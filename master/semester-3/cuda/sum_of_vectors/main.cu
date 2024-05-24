#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <limits>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>

/*********************************************************************************************************************/
// MACROS
/*********************************************************************************************************************/

// prumerovani pro presnejsi vysledky (pocet opakovani jednoho testu)
#define AVG_REPEAT_COUNT 50

#define CSV_DELIMITER ","

// NUMBER LIMITS
#define LIM_MIN(VAR) std::numeric_limits<typeof(VAR)>::min()
#define LIM_MAX(VAR) std::numeric_limits<typeof(VAR)>::max()

// CSV WRITE HELPER
#define TRY_WRITE_TO_CSV(v_size, cpu_time, gpu_time, gpu_mm_time)       \
    {                                                                   \
        if (__csv_stream != NULL)                                       \
        {                                                               \
            *__csv_stream << v_size << CSV_DELIMITER << cpu_time        \
                          << CSV_DELIMITER << gpu_time                  \
                          << CSV_DELIMITER << gpu_mm_time << std::endl; \
        }                                                               \
    }

// TIMING
#define TIME_INIT                                                      \
    std::chrono::time_point<std::chrono::high_resolution_clock> start; \
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
#define TIME_START start = std::chrono::high_resolution_clock::now();
#define TIME_END end = std::chrono::high_resolution_clock::now();
#define TIME_ELAPSED (std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count()

// APP STATUS
#define APP_OK 0
#define APP_ERROR_ARGS 1000
#define APP_ERROR_INIT 1001
#define APP_ERROR_RUNTIME 1002
#define APP_ERROR_CSV_OPENING 1002

// ERROR CHECKING
#define CUDA_CHECK_ERROR(VALUE)                                                                           \
    {                                                                                                     \
        cudaError_t error = VALUE;                                                                        \
        if (error != cudaSuccess)                                                                         \
        {                                                                                                 \
            printf("cuda error %s at line %d in file %s\n", cudaGetErrorName(error), __LINE__, __FILE__); \
            exit(-1);                                                                                     \
        }                                                                                                 \
    }

#define APP_CHECK_ERROR(VALUE) \
    {                          \
        int status = VALUE;    \
        if (status != APP_OK)  \
        {                      \
            return status;     \
        }                      \
    }

/*********************************************************************************************************************/
// VARIALBES
/*********************************************************************************************************************/

// konfigurace (mozne nastavovat)
static u_int32_t __repeat_count;     /** Pocet opakovani operace */
static unsigned long __start_length; /** Pocatecni delka vektoru */
static u_int32_t __increment;        /** Velikost navyseni delky vektoru v dalsim opakovani operace */
static char *__csv_file_name = NULL; /** Jmeno vystupniho csv souboru */

// pracovni pamet CPU
static int __blockSize;                    /** Maximalni pocet vlaken na blok*/
static std::ofstream *__csv_stream = NULL; /** Ukazatel na stream pro zapis do vystupniho csv souboru */
static unsigned long __final_size;         /** Finalni velikost vektory. Jeho velikost pri poslednim opakovani souctu */

static unsigned int *__cpu_vector_a = NULL;
static unsigned int *__cpu_vector_b = NULL;
static unsigned int *__cpu_vector_out = NULL;

// pracovni pamet GPU
static unsigned int *__gpu_vector_a = NULL;
static unsigned int *__gpu_vector_b = NULL;
static unsigned int *__gpu_vector_out = NULL;

static unsigned int *__gpu_mm_vector_a = NULL;
static unsigned int *__gpu_mm_vector_b = NULL;
static unsigned int *__gpu_mm_vector_out = NULL;

/*********************************************************************************************************************/
// DECLARATIONS
/*********************************************************************************************************************/

/**
 * Zpracuje vstupni argumenty
 *
 * @param argc      Pocet vstupni argumentu
 * @param argv      Vstupni argumenty
 *
 * @return  Status kod
 */
static int __parse_app_argumets(int argc, char *argv[]);

/**
 * Inicializuje celou aplikaci
 *
 * @return  Status kod
 */
static int __init();

/**
 * Otevre csv soubor pro zapis vyslednych casu.
 *
 * @return  Status kod
 */
static int __open_csv_file();

/**
 * Spusti provadeni operaci, pro kazdou zmeri cas jak dlouho trvalo jeji vykonani. Vysledek
 * vypise na stdout a pokud je nastaven zapis do vystupniho csv soubory, vypise vysledky i do nej
 *
 * @return  Status kod
 */
static int __run();

/**
 * Dealokace pameti cele aplikace + uzavreni streamu.
 *
 * @return  Status kod
 */
static int __clear();

/*********************************************************************************************************************/
// VECTOR FUNCTIONS
/*********************************************************************************************************************/

/**
 * Soucet prvku ve vektoru (vektorovy soucet). Vektory A, B, OUT musi byt
 * minimalne tak velike jak je specifikovane v parametru pro velikost vektoru
 *
 * @param a         Ukazatel na vektor A
 * @param b         Ukazatel na vektor B
 * @param out       Ukazatel na vystupni vektor
 * @param size      Velikost vektoru
 */
__global__ void vector_add_gpu(unsigned int *const a,
                               unsigned int *const b,
                               unsigned int *const out,
                               unsigned long size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        out[tid] = a[tid] + b[tid];
    }
}

/**
 * Soucet prvku ve vektoru (vektorovy soucet). Vektory A, B, OUT musi byt
 * minimalne tak velike jak je specifikovane v parametru pro velikost vektoru
 *
 * @param a         Ukazatel na vektor A
 * @param b         Ukazatel na vektor B
 * @param out       Ukazatel na vystupni vektor
 * @param size      Velikost vektoru
 *
 * @return Navrati true v pripade uspeneho provedeni operace souctu vektorus
 */
bool vector_add_cpu(unsigned int *const a,
                    unsigned int *const b,
                    unsigned int *const out,
                    unsigned long size)
{
    if (a == NULL || b == NULL || out == NULL)
        return false;

    for (size_t i = 0; i < size; ++i)
    {
        out[i] = a[i] + b[i];
    }

    return true;
}

/*********************************************************************************************************************/
// MAIN PROGRAM
/*********************************************************************************************************************/

int main(int argc, char *argv[])
{
    ///////////////////////////////////////////////////////////////////////////
    APP_CHECK_ERROR(__parse_app_argumets(argc, argv));
    APP_CHECK_ERROR(__init());
    APP_CHECK_ERROR(__open_csv_file());
    APP_CHECK_ERROR(__run());
    APP_CHECK_ERROR(__clear());
    ///////////////////////////////////////////////////////////////////////////
    return APP_OK;
}

static int __parse_app_argumets(int argc, char *argv[])
{
    // zpracovani argumentu
    if (argc < 4)
    {
        // vypis infa o aplikaci v pripade chybneho pouziti
        std::cout << "Incorrenct format. Correct usage of app is: ./app_name {start-length} {increment} {count} {file}" << std::endl
                  << std::endl;

        std::cout << "Argument 'start-length' (Start length of vector):" << std::endl;
        std::cout << "\tFrom 1 to " << LIM_MAX(__start_length) << ", inclusive" << std::endl
                  << std::endl;

        std::cout << "Argument 'increment' (Increment of vector length):" << std::endl;
        std::cout << "\tFrom 1 to " << LIM_MAX(__increment) << ", inclusive" << std::endl
                  << std::endl;

        std::cout << "Argument 'count' (Count of iterations):" << std::endl;
        std::cout << "\tFrom " << LIM_MIN(__repeat_count) << " to " << LIM_MAX(__repeat_count) << ", inclusive" << std::endl
                  << std::endl;

        std::cout << "Argument 'file' (Ouput CSV file name):" << std::endl;
        std::cout << "\tNOT REQUIRED ..." << std::endl;

        std::cout << "Example: ./app_name 100000 10000 100 out.csv" << std::endl;
        return APP_ERROR_ARGS;
    }
    else
    {
        // nacteni argumentu aplikace
        char *string, *stopstring;

        __start_length = strtoul(argv[1], &stopstring, 10);
        __increment = strtoul(argv[2], &stopstring, 10);
        __repeat_count = strtoul(argv[3], &stopstring, 10);

        if (argc > 4)
        {
            __csv_file_name = new char[strlen(argv[4]) + 1];
            if (__csv_file_name == NULL)
                return APP_ERROR_ARGS;
            strcpy(__csv_file_name, argv[4]);
        }

        return APP_OK;
    }
}

static int __init()
{
    // zjisti info o GPU a ziska maximalni pocet vlaken na blok
    int device;
    CUDA_CHECK_ERROR(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, device));
    __blockSize = prop.maxThreadsPerBlock;
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Compute compability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Grid size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
    std::cout << "Block size: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;

    // inicializace vektoru
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    if (__repeat_count < 1 || __start_length < 1)
        return APP_ERROR_INIT;

    __final_size = __start_length + (__repeat_count - 1) * __increment;
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    unsigned long i;

    // CPU vektory varianta
    __cpu_vector_a = new unsigned int[__final_size];
    __cpu_vector_b = new unsigned int[__final_size];
    __cpu_vector_out = new unsigned int[__final_size];
    for (i = 0; i < __final_size; ++i)
    {
        __cpu_vector_a[i] = static_cast<unsigned int>(std::rand());
        __cpu_vector_b[i] = static_cast<unsigned int>(std::rand());
    }

    // GPU vektory varianta
    CUDA_CHECK_ERROR(cudaMalloc((void **)&__gpu_vector_a, sizeof(unsigned int) * __final_size));
    CUDA_CHECK_ERROR(cudaMalloc((void **)&__gpu_vector_b, sizeof(unsigned int) * __final_size));
    CUDA_CHECK_ERROR(cudaMalloc((void **)&__gpu_vector_out, sizeof(unsigned int) * __final_size));
    CUDA_CHECK_ERROR(cudaMemcpy(__gpu_vector_a, __cpu_vector_a, sizeof(unsigned int) * __final_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(__gpu_vector_b, __cpu_vector_b, sizeof(unsigned int) * __final_size, cudaMemcpyHostToDevice));

    // GPU memory managed varianta
    CUDA_CHECK_ERROR(cudaMallocManaged(&__gpu_mm_vector_a, __final_size * sizeof(unsigned int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&__gpu_mm_vector_b, __final_size * sizeof(unsigned int)));
    CUDA_CHECK_ERROR(cudaMallocManaged(&__gpu_mm_vector_out, __final_size * sizeof(unsigned int)));
    for (i = 0; i < __final_size; ++i)
    {
        __gpu_mm_vector_a[i] = static_cast<unsigned int>(std::rand());
        __gpu_mm_vector_b[i] = static_cast<unsigned int>(std::rand());
    }

    std::cout << "Init done" << std::endl;
    return APP_OK;
}

static int __open_csv_file()
{
    // otevre stream jen pokud je specifikovane jmeno souboru do ktereho se ma zapisovat
    if (__csv_file_name != NULL)
    {
        __csv_stream = new std::ofstream(__csv_file_name);

        if (__csv_stream == NULL)
            return APP_ERROR_CSV_OPENING;
        if (!__csv_stream->is_open())
            return APP_ERROR_CSV_OPENING;

        *__csv_stream << "VECTOR_SIZE" << CSV_DELIMITER << "CPU"
                      << CSV_DELIMITER << "GPU" << CSV_DELIMITER << "GPU_MM" << std::endl;

        std::cout << "Output CSV file [" << __csv_file_name << "] openned" << std::endl;
    }
    return APP_OK;
}

static int __run()
{
    std::cout << "Block size set on: " << __blockSize << std::endl;
    std::cout << "Running now ..." << std::endl;

    if (__cpu_vector_a == NULL || __cpu_vector_b == NULL || __cpu_vector_out == NULL)
        return APP_ERROR_RUNTIME;

    int gridSize;
    TIME_INIT;

    for (unsigned long size = __start_length, i = 0; i < __repeat_count; ++i, size += __increment)
    {
        long cpu_time = 0, gpu_time = 0, gpu_mm_time = 0;

        // vypoci velikost gridu
        gridSize = (size + __blockSize - 1) / __blockSize;

        for (int j = 0; j < AVG_REPEAT_COUNT; ++j)
        {
            //////////////////////////////////////////////////////////////////////////////////////////////
            // CPU
            //////////////////////////////////////////////////////////////////////////////////////////////
            TIME_START;
            vector_add_cpu(__cpu_vector_a, __cpu_vector_b, __cpu_vector_out, size);
            TIME_END;
            cpu_time += TIME_ELAPSED;

            //////////////////////////////////////////////////////////////////////////////////////////////
            // GPU
            //////////////////////////////////////////////////////////////////////////////////////////////
            TIME_START;
            vector_add_gpu<<<gridSize, __blockSize>>>(__gpu_vector_a, __gpu_vector_b, __gpu_vector_out, size);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());
            CUDA_CHECK_ERROR(cudaMemcpy(__cpu_vector_out, __gpu_vector_out, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost));
            TIME_END;
            gpu_time += TIME_ELAPSED;

            //////////////////////////////////////////////////////////////////////////////////////////////
            // GPU Memory Managed
            //////////////////////////////////////////////////////////////////////////////////////////////
            TIME_START;
            vector_add_gpu<<<gridSize, __blockSize>>>(__gpu_mm_vector_a, __gpu_mm_vector_b, __gpu_mm_vector_out, size);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());
            TIME_END;
            gpu_mm_time += TIME_ELAPSED;
        }
        cpu_time /= AVG_REPEAT_COUNT;
        gpu_time /= AVG_REPEAT_COUNT;
        gpu_mm_time /= AVG_REPEAT_COUNT;

        //////////////////////////////////////////////////////////////////////////////////////////////
        TRY_WRITE_TO_CSV(size, cpu_time, gpu_time, gpu_mm_time);
        std::cout << "Vector Size: " << size << ", Grid size: " << gridSize << " = CPU: " << cpu_time << "us, GPU: " << gpu_time << "us, GPU MM: " << gpu_mm_time << "us" << std::endl;
    }

    std::cout << "End ..." << std::endl;
    return APP_OK;
}

static int __clear()
{
    if (__csv_file_name != NULL)
    {
        delete __csv_file_name;
        __csv_file_name = NULL;
    }

    if (__csv_stream != NULL)
    {
        __csv_stream->flush();
        __csv_stream->close();
        std::cout << "Output CSV file closing" << std::endl;
        delete __csv_stream;
        __csv_stream = NULL;
    }

    // CPU
    if (__cpu_vector_a)
        delete __cpu_vector_a;
    if (__cpu_vector_b)
        delete __cpu_vector_b;
    if (__cpu_vector_out)
        delete __cpu_vector_out;
    __cpu_vector_a = __cpu_vector_b = __cpu_vector_out = NULL;

    // GPU
    if (__gpu_vector_a)
        CUDA_CHECK_ERROR(cudaFree(__gpu_vector_a));
    if (__gpu_vector_b)
        CUDA_CHECK_ERROR(cudaFree(__gpu_vector_b));
    if (__gpu_vector_out)
        CUDA_CHECK_ERROR(cudaFree(__gpu_vector_out));
    __gpu_vector_a = __gpu_vector_b = __gpu_vector_out = NULL;

    // GPU MM
    if (__gpu_mm_vector_a)
        CUDA_CHECK_ERROR(cudaFree(__gpu_mm_vector_a));
    if (__gpu_mm_vector_b)
        CUDA_CHECK_ERROR(cudaFree(__gpu_mm_vector_b));
    if (__gpu_mm_vector_out)
        CUDA_CHECK_ERROR(cudaFree(__gpu_mm_vector_out));
    __gpu_mm_vector_a = __gpu_mm_vector_b = __gpu_mm_vector_out = NULL;

    std::cout << "Clear done" << std::endl;
    return APP_OK;
}
