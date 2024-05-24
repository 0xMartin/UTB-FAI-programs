#ifndef __IMAGE_RGB_H__
#define __IMAGE_RGB_H__

#include <iostream>
#include <filesystem>
#include <png.h>
#include "utils/pngio.h"

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

/// @brief Ulehcuje praci s obrazky a jejich presouvani s hosta na device
namespace ImageRGB
{

    /// @brief Umisteni alokovane pameti
    enum MemLocation
    {
        HOST,  /** Pamet je alokovana na hostu */
        DEVICE /** Pamet je alokovana na device (GPU) */
    };

    /// @brief Struktura uchovavajici vsechny barevne kanaly RGB obrazku
    struct Image
    {
        unsigned char *red_channel;
        unsigned char *green_channel;
        unsigned char *blue_channel;
        unsigned int width;
        unsigned int height;
        MemLocation location;
    };

    /// @brief Inicializuje obrazek
    /// @param img3c Pointer na strukturu obrazku
    /// @param loc Umisteni obrazku (alokovane pro HOST nebo DEVICE)
    static void init(ImageRGB::Image *img3c, ImageRGB::MemLocation loc)
    {
        if (img3c != nullptr)
        {
            img3c->red_channel = nullptr;
            img3c->green_channel = nullptr;
            img3c->blue_channel = nullptr;
            img3c->width = img3c->height = 0;
            img3c->location = loc;
        }
    }

    /// @brief Alokuje pamet struktury RGB obrazku, pokud neni alokovana
    /// @param img3c Struktura obrazku
    /// @param width Pozadovana sirka obrazku
    /// @param height Pozadovana vyska obrazku
    /// @return
    static bool allocateIfIsNull(ImageRGB::Image *img3c, unsigned int width, unsigned int height)
    {
        if (img3c == nullptr)
            return false;

        // kanaly obrazku jsou jiz alokovany
        if (img3c->red_channel != nullptr &&
            img3c->green_channel != nullptr &&
            img3c->blue_channel != nullptr &&
            img3c->width != 0 &&
            img3c->height != 0)
        {
            return true;
        }

        img3c->width = width;
        img3c->height = height;

        // alokuje pamet pro kanaly obrazku
        size_t size;
        unsigned char *d_r = nullptr;
        unsigned char *d_g = nullptr;
        unsigned char *d_b = nullptr;
        switch (img3c->location)
        {
        case HOST:
            // alokuje na HOST
            size = width * height;
            img3c->red_channel = new unsigned char[size];
            img3c->green_channel = new unsigned char[size];
            img3c->blue_channel = new unsigned char[size];
            break;

        case DEVICE:
            // alokuje na GPU
            size = width * height * sizeof(unsigned char);
            CUDA_CHECK_RETURN(cudaMalloc(&d_r, size));
            CUDA_CHECK_RETURN(cudaMalloc(&d_g, size));
            CUDA_CHECK_RETURN(cudaMalloc(&d_b, size));
            img3c->red_channel = d_r;
            img3c->green_channel = d_g;
            img3c->blue_channel = d_b;
            break;

        default:
            return false;
        }

        return true;
    }

    /// @brief Nacte obrazek se souboru a obrazove data vlozi do pameti alokovane pro CPU "HOST"
    /// @param img3c Pointer na Image strukturu
    /// @param path Cesta k souboru ze ktereho bude nacitat obrazek
    /// @return True v pripade uspesne provedeni operace
    static bool loadImageFromFile(ImageRGB::Image *img3c, char *path)
    {
        if (img3c == nullptr || path == nullptr)
            return false;
        if (img3c->location != HOST)
            return false;

        // otevre obrazek ze specifikovaneho souboru
        png::image<png::rgb_pixel> imgFile(path);
        unsigned int width = imgFile.get_width();
        unsigned int height = imgFile.get_height();

        // obrazek nacte do pameti alokovane na hostu
        img3c->location = HOST;
        if (!ImageRGB::allocateIfIsNull(img3c, width, height))
            return false;
        pvg::pngToRgb3(
            img3c->red_channel,
            img3c->green_channel,
            img3c->blue_channel,
            imgFile);
        img3c->width = width;
        img3c->height = height;

        return true;
    }

    /// @brief Zkopiruje data jednoho obrazku do druheho. Obrazku musi mit rozdilne lokace.
    /// @param img_src Zdrojovy obrazek
    /// @param img_dst Cilovy obrazek
    /// @return True v pripade uspesne provedeni operace
    static bool copy(ImageRGB::Image *img_src, ImageRGB::Image *img_dst)
    {
        if (img_src == nullptr || img_dst == nullptr)
            return false;
        if (img_src->location == img_dst->location)
            return false;
        if (img_src->red_channel == nullptr ||
            img_src->green_channel == nullptr ||
            img_src->blue_channel == nullptr ||
            img_src->width == 0 ||
            img_src->height == 0)
            return false;

        // alokuje cilovy obrazek pokud jeste neni
        if (!ImageRGB::allocateIfIsNull(img_dst, img_src->width, img_src->height))
            return false;

        // u dst nastavi stejnou velikost
        img_dst->width = img_src->width;
        img_dst->height = img_src->height;

        // kopirovani dat
        size_t size = img_src->width * img_src->height * sizeof(unsigned char);
        cudaMemcpyKind memCpyKind = img_src->location == HOST && img_dst->location == DEVICE ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        CUDA_CHECK_RETURN(cudaMemcpy(img_dst->red_channel, img_src->red_channel, size, memCpyKind));
        CUDA_CHECK_RETURN(cudaMemcpy(img_dst->green_channel, img_src->green_channel, size, memCpyKind));
        CUDA_CHECK_RETURN(cudaMemcpy(img_dst->blue_channel, img_src->blue_channel, size, memCpyKind));

        return true;
    }

    /// @brief Ulozi obrazek na disk. Obrazek musi byt alokovan na hostu
    /// @param img3c Obrazek ktery bude ukladan
    /// @param path Adresa souboru do ktereho bude obrazek ulozen
    /// @return True v pripade uspesne provedeni operace
    static bool saveToFile(ImageRGB::Image *img3c, const char *path)
    {
        if (img3c == nullptr)
            return false;
        if (img3c->location != HOST)
            return false;

        png::image<png::rgb_pixel> imgFile(img3c->width, img3c->height);
        pvg::rgb3ToPng(imgFile,
                       img3c->red_channel,
                       img3c->green_channel,
                       img3c->blue_channel);
        imgFile.write(path);
        return true;
    }

    /// @brief Odstrani obrazek z pameti
    /// @param img3c Obrazek ktery bude odstranen
    /// @return True v pripade uspesne provedeni operace
    static bool deleteImage(ImageRGB::Image *img3c)
    {
        if (img3c == nullptr)
            return false;

        switch (img3c->location)
        {
        case HOST:
            if (img3c->red_channel != nullptr)
                delete[] img3c->red_channel;
            if (img3c->green_channel != nullptr)
                delete[] img3c->green_channel;
            if (img3c->blue_channel != nullptr)
                delete[] img3c->blue_channel;
            break;

        case DEVICE:
            if (img3c->red_channel != nullptr)
                CUDA_CHECK_RETURN(cudaFree(img3c->red_channel));
            if (img3c->green_channel != nullptr)
                CUDA_CHECK_RETURN(cudaFree(img3c->green_channel));
            if (img3c->blue_channel != nullptr)
                CUDA_CHECK_RETURN(cudaFree(img3c->blue_channel));
            break;

        default:
            return false;
        }

        img3c->red_channel = nullptr;
        img3c->green_channel = nullptr;
        img3c->blue_channel = nullptr;

        return true;
    }

}

#endif