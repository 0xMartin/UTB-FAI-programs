#include <iostream>
#include <chrono>
#include <string.h>

#include "argument_parser.h"

#include "image_rgb.cu"
#include "histogram_generator.cu"

// makro po jednoduche mereni aktualniho casu
#define CURRENT_TIME(start_time) (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count())

// error codes
#define ERR_ARGS_PARSE -1
#define ERR_INIT -2
#define ERR_HITOGRAM_DEVICE_INIT -3
#define ERR_IN_IMG_DEVICE_INIT -4
#define ERR_OUT_IMG_DEVICE_INIT -5
#define ERR_IN_IMG_LOAD -6
#define ERR_IN_IMG_MOVE_TO_DEVICE -7
#define ERR_HISTOGRAM_GENERATE -8
#define ERR_OUT_IMG_HOST_INIT -9
#define ERR_OUT_IMG_GENERATE -10
#define ERR_OUT_IMG_MOVE_TO_HOST -11
#define ERR_OUT_IMG_SAVE -12
#define ERR_IN_IMG_HOST_INIT -13
#define ERR_IN_IMG_GET_SIZE -14

// konfigurace (vstupni obrazke, nazev vystupniho obrazku, sirka+vyska vystupniho obrazku z histogramem, mod generovani)
static char __input_img_path[256];
static char __ouput_img_path[256];
static unsigned int __width;
static unsigned int __height;
static ImageRGB::ProcessMode __p_mode;

static bool __init(std::vector<ArgumentParser::Argument> &arguments)
{
    __input_img_path[0] = __ouput_img_path[0] = 0x0;
    __width = __height = 0;
    __p_mode = ImageRGB::DEFAULT;

    ArgumentParser::Argument buffer;

    // help
    if (ArgumentParser::findByName(arguments, buffer, "--help"))
    {
        std::cout << "Usage: appname [options...]" << std::endl;
        std::cout << "-i, --input <file>\t\tPath to the input image [required]" << std::endl;
        std::cout << "-o, --output <file>\t\tPath to the output image [required]" << std::endl;
        std::cout << "-w, --width <size>\t\tWidth of output image with histogram [required]" << std::endl;
        std::cout << "-h, --height <size>\t\tHeight of output image with histogram [required]" << std::endl;
        std::cout << "-t, --type <type>\t\tType of process method. Default or using multiple streams. (default value: DEFAULT)" << std::endl;

        std::cout << std::endl
                  << "Memory copy method types: DEFAULT, STREAM" << std::endl;
        return false;
    }

    // input
    if (ArgumentParser::findByName(arguments, buffer, "--input") ||
        ArgumentParser::findByName(arguments, buffer, "-i"))
    {
        strncpy(__input_img_path, buffer.value, 256);
    }

    // output
    if (ArgumentParser::findByName(arguments, buffer, "--output") ||
        ArgumentParser::findByName(arguments, buffer, "-o"))
    {
        strncpy(__ouput_img_path, buffer.value, 256);
    }

    // width
    if (ArgumentParser::findByName(arguments, buffer, "--width") ||
        ArgumentParser::findByName(arguments, buffer, "-w"))
    {
        __width = atoi(buffer.value);
    }

    // height
    if (ArgumentParser::findByName(arguments, buffer, "--height") ||
        ArgumentParser::findByName(arguments, buffer, "-h"))
    {
        __height = atoi(buffer.value);
    }

    // process method
    if (ArgumentParser::findByName(arguments, buffer, "--type") ||
        ArgumentParser::findByName(arguments, buffer, "-t"))
    {
        if (strcasecmp(buffer.value, "DEFAULT") == 0)
        {
            __p_mode = ImageRGB::DEFAULT;
        }
        else if (strcasecmp(buffer.value, "STREAM") == 0)
        {
            __p_mode = ImageRGB::STREAM;
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    // zpracovani argumentu
    std::vector<ArgumentParser::Argument> argumets;
    if (!ArgumentParser::parseArguments(argumets, argc, argv))
    {
        std::cerr << "Incorrect value of some argument, check help (--help)" << std::endl;
        return ERR_ARGS_PARSE;
    }

    // inicialzicace
    if (!__init(argumets))
        return 0;

    // init error check
    if (__input_img_path[0] == 0x0 || __ouput_img_path[0] == 0x0 || __width == 0 || __height == 0)
    {
        std::cerr << "Incorrect value of some argument, check help (--help)" << std::endl;
        return ERR_INIT;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    //////////////////////////////////////////////////////////////////////////////

    std::cout << "Mode: " << (__p_mode == ImageRGB::STREAM ? "STREAM" : "DEFAULT") << std::endl;

    // inicializace streamu
    if (__p_mode == ImageRGB::STREAM)
    {
        ImageRGB::initStreams();
        std::cout << CURRENT_TIME(start_time) << "ms : Streams for RGB channels created" << std::endl;
    }

    // ALOKACE NA HOST A DEVICE *************************************************************************************************

    // inicializace histogramu na devicce
    HistogramGenerator::RGBHistogram histogram;
    if (!HistogramGenerator::initHistogram(&histogram))
        return ERR_HITOGRAM_DEVICE_INIT;
    std::cout << CURRENT_TIME(start_time) << "ms : Histogram init done" << std::endl;

    // zjisteni velikosti vstupniho obrazku
    unsigned int in_width, in_height;
    if (!ImageRGB::getImageSize(__input_img_path, &in_width, &in_height))
        return ERR_IN_IMG_GET_SIZE;

    // alokace vstupniho obrazku na hostu
    ImageRGB::Image h_input_img;
    ImageRGB::init(&h_input_img, ImageRGB::HOST);
    if (!ImageRGB::allocateIfIsNull(&h_input_img, in_width, in_height))
        return ERR_IN_IMG_HOST_INIT;
    std::cout << CURRENT_TIME(start_time) << "ms : Input image allocated on host" << std::endl;

    // alokace vstupniho obrazku na device
    ImageRGB::Image d_input_img;
    ImageRGB::init(&d_input_img, ImageRGB::DEVICE);
    if (!ImageRGB::allocateIfIsNull(&d_input_img, in_width, in_height, __p_mode))
        return ERR_IN_IMG_DEVICE_INIT;
    std::cout << CURRENT_TIME(start_time) << "ms : Input image allocated on device" << std::endl;

    // alokace vystupniho obrazku na hostu
    ImageRGB::Image h_output_img;
    ImageRGB::init(&h_output_img, ImageRGB::HOST);
    if (!ImageRGB::allocateIfIsNull(&h_output_img, __width, __height))
        return ERR_OUT_IMG_HOST_INIT;
    std::cout << CURRENT_TIME(start_time) << "ms : Output image allocated on host" << std::endl;

    // alokace vystupniho obrazku na device
    ImageRGB::Image d_output_img;
    ImageRGB::init(&d_output_img, ImageRGB::DEVICE);
    if (!ImageRGB::allocateIfIsNull(&d_output_img, __width, __height, __p_mode))
        return ERR_OUT_IMG_DEVICE_INIT;
    std::cout << CURRENT_TIME(start_time) << "ms : Output image allocated on device" << std::endl;

    // NACITANI VSTUONIHO OBRAZKU A PRESUN NA DEVICE ****************************************************************************

    // nacist vstupni obrazek na hosta
    if (!ImageRGB::loadImageFromFile(&h_input_img, __input_img_path, __p_mode))
        return ERR_IN_IMG_LOAD;
    std::cout << CURRENT_TIME(start_time) << "ms : Input image load done" << std::endl;

    auto start_time_2 = std::chrono::high_resolution_clock::now();

    // presun vstupniho obrazku na device
    if (!ImageRGB::copy(&h_input_img, &d_input_img, __p_mode))
        return ERR_IN_IMG_MOVE_TO_DEVICE;
    std::cout << CURRENT_TIME(start_time) << "ms : Image moved to GPU" << std::endl;

    // GENEROVANI ***************************************************************************************************************

    // vygenerovani histogramu
    if (!HistogramGenerator::generateRGBHistogramSyncOrAsync(&d_input_img, &histogram, __p_mode))
        return ERR_HISTOGRAM_GENERATE;
    std::cout << CURRENT_TIME(start_time) << "ms : Histogram generating done" << std::endl;

    // synchronizace streamu (cekani na dokonceni vypoctu max hodnoty ze vsech streamu)
    if (__p_mode == ImageRGB::STREAM)
        ImageRGB::synchronizeRGBStreams();
    // vygenerovani obrazku histogramu
    if (!HistogramGenerator::generateImageSyncOrAsync(&d_output_img, &histogram, __p_mode))
        return ERR_OUT_IMG_GENERATE;
    std::cout << CURRENT_TIME(start_time) << "ms : Output image generating done" << std::endl;

    // PRESUN VYSTUPNIHO OBRAZKU NA HOSTA A ULOZENI ******************************************************************************

    // presun vygenerovaneho obrazku na hosta
    if (!ImageRGB::copy(&d_output_img, &h_output_img, __p_mode))
        return ERR_OUT_IMG_MOVE_TO_HOST;
    std::cout << CURRENT_TIME(start_time) << "ms : Output image moved to host" << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();

    // ulozeni obrazku
    if (!ImageRGB::saveToFile(&h_output_img, __ouput_img_path))
        return ERR_OUT_IMG_SAVE;
    std::cout << CURRENT_TIME(start_time) << "ms : Output image with histogram saved" << std::endl;

    // UKONCENI PROGRAMU *********************************************************************************************************

    // dealokace
    ImageRGB::deleteImage(&h_input_img, __p_mode);
    ImageRGB::deleteImage(&d_input_img, __p_mode);
    ImageRGB::deleteImage(&h_output_img, __p_mode);
    ImageRGB::deleteImage(&d_output_img, __p_mode);
    HistogramGenerator::deleteHistogram(&histogram);

    if (__p_mode == ImageRGB::STREAM)
    {
        std::cout << CURRENT_TIME(start_time) << "ms : Streams closed" << std::endl;
        ImageRGB::closeStreams();
    }

    //////////////////////////////////////////////////////////////////////////////
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;

    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_2);
    std::cout << "Time (Data copy + kernel): " << duration2.count() << " us" << std::endl;

    return 0;
}