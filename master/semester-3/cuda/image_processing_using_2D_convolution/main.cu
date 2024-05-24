#include <iostream>
#include <chrono>
#include <string.h>

#include "mask_utils.h"
#include "argument_parser.h"

#include "convolution.cu"
#include "image_rgb.cu"

// error codes
#define ERR_ARGS_PARSE -1000
#define ERR_INIT -2000
#define ERR_IMG_LOAD -3000
#define ERR_MOVE_IMG_TO_GPU -4000
#define ERR_MASK_CREATE -5000
#define ERR_CONV -6000
#define ERR_MOVE_IMG_TO_CPU -7000
#define ERR_IMG_SAVE -8000

// konfigurace
static const float *__selected_mask = nullptr;
static char __input_img_path[256];
static char __ouput_img_path[256];

static bool __init(std::vector<ArgumentParser::Argument> &arguments)
{
    __input_img_path[0] = __ouput_img_path[0] = 0x0;

    ArgumentParser::Argument buffer;

    // help
    if (ArgumentParser::findByName(arguments, buffer, "--help") ||
        ArgumentParser::findByName(arguments, buffer, "-h"))
    {
        std::cout << "Usage: appname [options...]" << std::endl;
        std::cout << "-i, --input <file>\t\tPath to the input image [required]" << std::endl;
        std::cout << "-o, --output <file>\t\tPath to the output image [required]" << std::endl;
        std::cout << "-f, --filter <name>\t\tName of filter [required]" << std::endl
                  << std::endl;

        std::cout << "Filter names: blur, laplacian, sharpen, sobel_x, sobel_y, prewitt_x, prewitt_y" << std::endl
                  << std::endl;
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

    // filter
    if (ArgumentParser::findByName(arguments, buffer, "--filter") ||
        ArgumentParser::findByName(arguments, buffer, "-f"))
    {
        if (strcmp(buffer.value, "blur") == 0)
        {
            __selected_mask = MaskUtils::blur_mask_5;
        }
        else if (strcmp(buffer.value, "laplacian") == 0)
        {
            __selected_mask = MaskUtils::laplacian_mask_5;
        }
        else if (strcmp(buffer.value, "sharpen") == 0)
        {
            __selected_mask = MaskUtils::sharpen_mask_5;
        }
        else if (strcmp(buffer.value, "sobel_x") == 0)
        {
            __selected_mask = MaskUtils::sobel_mask_x_5;
        }
        else if (strcmp(buffer.value, "sobel_y") == 0)
        {
            __selected_mask = MaskUtils::sobel_mask_y_5;
        }
        else if (strcmp(buffer.value, "prewitt_x") == 0)
        {
            __selected_mask = MaskUtils::prewitt_mask_x_5;
        }
        else if (strcmp(buffer.value, "prewitt_y") == 0)
        {
            __selected_mask = MaskUtils::prewitt_mask_y_5;
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
        std::cerr << "Incorrect value of some argument, check help (--help or -h)" << std::endl;
        return ERR_ARGS_PARSE;
    }

    // inicialzicace
    if (!__init(argumets))
    {
        return 0;
    }

    // init error check
    if (__selected_mask == nullptr || __input_img_path[0] == 0x0 || __ouput_img_path[0] == 0x0)
    {
        std::cerr << "Incorrect value of some argument, check help (--help or -h)" << std::endl;
        return ERR_INIT;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    //////////////////////////////////////////////////////////////////////////////

    // inicializace obrazku
    ImageRGB::Image h_img;
    ImageRGB::Image d_img;
    ImageRGB::init(&h_img, ImageRGB::HOST);
    ImageRGB::init(&d_img, ImageRGB::DEVICE);

    // nacte obrazek na hosta
    if (!ImageRGB::loadImageFromFile(&h_img, __input_img_path))
    {
        std::cerr << "Failed to load image" << std::endl;
        return ERR_IMG_LOAD;
    }
    std::cout << "Image loaded" << std::endl;

    // presune obrazek i na device
    if (!ImageRGB::copy(&h_img, &d_img))
    {
        std::cerr << "Failed to move image to GPU" << std::endl;
        return ERR_MOVE_IMG_TO_GPU;
    }
    std::cout << "Image moved to gpu" << std::endl;

    // vytvori masku + normalizace masky
    MaskUtils::Mask_t mask;
    if (!MaskUtils::createFromTemplate(&mask, __selected_mask, 5))
    {
        std::cerr << "Failed to create convolution mask" << std::endl;
        return ERR_MASK_CREATE;
    }
    std::cout << "Convolution mask created" << std::endl;

    // provede konvolu s vybranou maskou "filtrem"
    std::cout << "Convolution start" << std::endl;
    if (!Conv2D::applyConvMask(&d_img, &mask))
    {
        std::cerr << "Convolution process failed" << std::endl;
        return ERR_CONV;
    }
    std::cout << "Convolution done" << std::endl;

    // presune obrazek z device na hosta
    if (!ImageRGB::copy(&d_img, &h_img))
    {
        std::cerr << "Failed to move image to CPU" << std::endl;
        return ERR_MOVE_IMG_TO_CPU;
    }
    std::cout << "Image moved back on CPU" << std::endl;

    // zapise vysledny obrazek na disk
    if (!ImageRGB::saveToFile(&h_img, __ouput_img_path))
    {
        std::cerr << "Failed save final image" << std::endl;
        return ERR_IMG_SAVE;
    }
    std::cout << "Image saved" << std::endl;

    // odstraneni alokovanych obrazku
    ImageRGB::deleteImage(&h_img);
    ImageRGB::deleteImage(&d_img);

    // odstaneni masky
    MaskUtils::deleteMask(&mask);

    //////////////////////////////////////////////////////////////////////////////
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;

    return 0;
}