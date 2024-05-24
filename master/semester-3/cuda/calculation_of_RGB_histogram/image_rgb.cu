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

/// @brief Ulehcuje praci s obrazky a jejich presouvani s hosta na device. Umozunje take kopirovani dat pomoci CUDA streamu
namespace ImageRGB
{

    /// @brief Umisteni alokovane pameti
    enum MemLocation
    {
        HOST,  /** Pamet je alokovana na hostu */
        DEVICE /** Pamet je alokovana na device (GPU) */
    };

    /// @brief Mod zpracovani obrazku
    enum ProcessMode
    {
        DEFAULT, /** Defaultni kopirovani dat bez pouziti streamu */
        STREAM   /** Kopirovani dat z vyuztitim streamu */
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

    static cudaStream_t stream_red;   /** Stream pro kopirovani cerveneho kanalu obrazku*/
    static cudaStream_t stream_green; /** Stream pro kopirovani zeleneho kanalu obrazku*/
    static cudaStream_t stream_blue;  /** Stream pro kopirovani modreho kanalu obrazku*/

    /// @brief Inicializace streamu
    static void initStreams()
    {
        CUDA_CHECK_RETURN(cudaStreamCreate(&stream_red));
        CUDA_CHECK_RETURN(cudaStreamCreate(&stream_green));
        CUDA_CHECK_RETURN(cudaStreamCreate(&stream_blue));
    }

    /// @brief Zavreni streamu
    static void closeStreams()
    {
        CUDA_CHECK_RETURN(cudaStreamDestroy(stream_red));
        CUDA_CHECK_RETURN(cudaStreamDestroy(stream_green));
        CUDA_CHECK_RETURN(cudaStreamDestroy(stream_blue));
    }

    /// @brief Synchronizace vsech streamu (R, G, B)
    static void synchronizeRGBStreams()
    {
        CUDA_CHECK_RETURN(cudaStreamSynchronize(stream_red));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(stream_green));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(stream_blue));
    }

    /// @brief Inicializuje obrazek
    /// @param img3c Pointer na strukturu obrazku
    /// @param loc Umisteni obrazku (alokovane pro HOST nebo DEVICE)
    static void init(ImageRGB::Image *img3c,
                     ImageRGB::MemLocation loc)
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
    /// @param pMode Zpusob zpracovani obrazku (zakladni nebo pomoci streamu)
    /// @return
    static bool allocateIfIsNull(ImageRGB::Image *img3c,
                                 unsigned int width,
                                 unsigned int height,
                                 ProcessMode pMode = DEFAULT)
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
        switch (img3c->location)
        {
        case HOST:
            // alokuje na HOST
            size = width * height * sizeof(unsigned char);
            CUDA_CHECK_RETURN(cudaMallocHost(&img3c->red_channel, size, cudaHostAllocDefault));
            CUDA_CHECK_RETURN(cudaMallocHost(&img3c->green_channel, size, cudaHostAllocDefault));
            CUDA_CHECK_RETURN(cudaMallocHost(&img3c->blue_channel, size, cudaHostAllocDefault));
            break;

        case DEVICE:
            // alokuje na GPU
            size = width * height * sizeof(unsigned char);
            if (pMode == STREAM)
            {
                CUDA_CHECK_RETURN(cudaMallocAsync(&img3c->red_channel, size, stream_red));
                CUDA_CHECK_RETURN(cudaMallocAsync(&img3c->green_channel, size, stream_green));
                CUDA_CHECK_RETURN(cudaMallocAsync(&img3c->blue_channel, size, stream_blue));
            }
            else
            {
                CUDA_CHECK_RETURN(cudaMalloc(&img3c->red_channel, size));
                CUDA_CHECK_RETURN(cudaMalloc(&img3c->green_channel, size));
                CUDA_CHECK_RETURN(cudaMalloc(&img3c->blue_channel, size));
            }
            break;

        default:
            return false;
        }

        return img3c->red_channel != nullptr &&
               img3c->green_channel != nullptr &&
               img3c->blue_channel != nullptr &&
               img3c->width != 0 &&
               img3c->height != 0;
    }

    static bool getImageSize(const char *path,
                             unsigned int *width,
                             unsigned int *height)
    {
        if (width == nullptr || height == nullptr || path == nullptr)
        {
            return false;
        }
        png::image<png::rgb_pixel> imgFile(path);
        *width = imgFile.get_width();
        *height = imgFile.get_height();
        return true;
    }

    /// @brief Nacte obrazek se souboru a obrazove data vlozi do pameti alokovane pro CPU "HOST"
    /// @param img3c Pointer na Image strukturu
    /// @param path Cesta k souboru ze ktereho bude nacitat obrazek
    /// @param pMode Zpusob zpracovani obrazku (zakladni nebo pomoci streamu)
    /// @return True v pripade uspesne provedeni operace
    static bool loadImageFromFile(ImageRGB::Image *img3c,
                                  const char *path,
                                  ProcessMode pMode = DEFAULT)
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
        if (!ImageRGB::allocateIfIsNull(img3c, width, height, pMode))
        {
            return false;
        }
        pvg::pngToRgb3(
            img3c->red_channel,
            img3c->green_channel,
            img3c->blue_channel,
            imgFile);
        img3c->width = width;
        img3c->height = height;

        return true;
    }

    /// @brief Zkopiruje data jednoho obrazku do druheho. Obrazky musi mit rozdilne lokace.
    /// @param img_src Zdrojovy obrazek
    /// @param img_dst Cilovy obrazek
    /// @param pMode Zpusob zpracovani obrazku (zakladni nebo pomoci streamu)
    /// @return True v pripade uspesne provedeni operace
    static bool copy(ImageRGB::Image *img_src,
                     ImageRGB::Image *img_dst,
                     ProcessMode pMode = DEFAULT)
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
        if (!ImageRGB::allocateIfIsNull(img_dst, img_src->width, img_src->height, pMode))
            return false;

        // u dst nastavi stejnou velikost
        img_dst->width = img_src->width;
        img_dst->height = img_src->height;

        // kopirovani dat
        size_t size = img_src->width * img_src->height * sizeof(unsigned char);
        cudaMemcpyKind memCpyKind = img_src->location == HOST && img_dst->location == DEVICE ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        switch (pMode)
        {
        case DEFAULT:
            // bez streamu
            CUDA_CHECK_RETURN(cudaMemcpy(img_dst->red_channel, img_src->red_channel, size, memCpyKind));
            CUDA_CHECK_RETURN(cudaMemcpy(img_dst->green_channel, img_src->green_channel, size, memCpyKind));
            CUDA_CHECK_RETURN(cudaMemcpy(img_dst->blue_channel, img_src->blue_channel, size, memCpyKind));
            break;

        case STREAM:
            // pomoci streamu
            CUDA_CHECK_RETURN(cudaMemcpyAsync(img_dst->red_channel, img_src->red_channel, size, memCpyKind, stream_red));
            CUDA_CHECK_RETURN(cudaMemcpyAsync(img_dst->green_channel, img_src->green_channel, size, memCpyKind, stream_green));
            CUDA_CHECK_RETURN(cudaMemcpyAsync(img_dst->blue_channel, img_src->blue_channel, size, memCpyKind, stream_blue));
            break;
        }

        // ceka na doknoceni jen v pripade pokud je obrazek presouvan z GPU na CPU a jde o typ kopirovani "STREAM"
        if (pMode == STREAM)
        {
            if (img_dst->location == HOST)
            {
                ImageRGB::synchronizeRGBStreams();
            }
        }

        return true;
    }

    /// @brief Ulozi obrazek na disk. Obrazek musi byt alokovan na hostu
    /// @param img3c Obrazek ktery bude ukladan
    /// @param path Adresa souboru do ktereho bude obrazek ulozen
    /// @return True v pripade uspesne provedeni operace
    static bool saveToFile(ImageRGB::Image *img3c,
                           const char *path)
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
    /// @param pMode Zpusob zpracovani obrazku (zakladni nebo pomoci streamu)
    /// @return True v pripade uspesne provedeni operace
    static bool deleteImage(ImageRGB::Image *img3c, ProcessMode pMode = DEFAULT)
    {
        if (img3c == nullptr)
            return false;

        switch (img3c->location)
        {
        case HOST:
            if (img3c->red_channel != nullptr)
                CUDA_CHECK_RETURN(cudaFreeHost(img3c->red_channel));
            if (img3c->green_channel != nullptr)
                CUDA_CHECK_RETURN(cudaFreeHost(img3c->green_channel));
            if (img3c->blue_channel != nullptr)
                CUDA_CHECK_RETURN(cudaFreeHost(img3c->blue_channel));
            break;

        case DEVICE:
            if (pMode == STREAM)
            {
                if (img3c->red_channel != nullptr)
                    CUDA_CHECK_RETURN(cudaFreeAsync(img3c->red_channel, stream_red));
                if (img3c->green_channel != nullptr)
                    CUDA_CHECK_RETURN(cudaFreeAsync(img3c->green_channel, stream_green));
                if (img3c->blue_channel != nullptr)
                    CUDA_CHECK_RETURN(cudaFreeAsync(img3c->blue_channel, stream_blue));
            }
            else
            {
                if (img3c->red_channel != nullptr)
                    CUDA_CHECK_RETURN(cudaFree(img3c->red_channel));
                if (img3c->green_channel != nullptr)
                    CUDA_CHECK_RETURN(cudaFree(img3c->green_channel));
                if (img3c->blue_channel != nullptr)
                    CUDA_CHECK_RETURN(cudaFree(img3c->blue_channel));
            }
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