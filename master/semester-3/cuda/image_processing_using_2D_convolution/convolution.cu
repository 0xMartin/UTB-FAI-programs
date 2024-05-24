#ifndef __CONV_2D_CU__
#define __CONV_2D_CU__

#include <cuda_runtime.h>

#include "mask_utils.h"
#include "image_rgb.cu"

#define FILTER_SIZE (5u)
#define BLOCK_SIZE (16u)
#define TILE_SIZE (BLOCK_SIZE - (FILTER_SIZE / 2) * 2)

namespace Conv2D
{

    /// @brief Aplikuje konvolucni masku 5x5 na jeden kanal obrazeku
    /// @param out_channel  Vystupni kanala obrazku (obrazek po aplikaci konvoluce) [DEVICE]
    /// @param in_channel   Vstupni kanal obrazku (puvodni obrazek) [DEVICE]
    /// @param mask         Konvolucni maska o velikosti 5x5 [DEVICE]
    /// @param width        Sirka obrazku v pixelech
    /// @param height       Vyska obraku v pixelech
    __global__ static void conv2DKernel(unsigned char *out_channel,
                                        unsigned char *in_channel,
                                        float *mask,
                                        unsigned int width,
                                        unsigned int height)
    {
        // pozice pixelu v celem obrazku
        int x = threadIdx.x + blockIdx.x * TILE_SIZE;
        int y = threadIdx.y + blockIdx.y * TILE_SIZE;

        // nacteni casti obrazku do sdilene pameti bloku (s okolnim presahem FILTER_SIZE/2)
        __shared__ unsigned char s_in_channel[BLOCK_SIZE * BLOCK_SIZE];
        int x_sbuf = x - FILTER_SIZE / 2;
        int y_sbuf = y - FILTER_SIZE / 2;
        if (x_sbuf >= 0 && x_sbuf < width && y_sbuf >= 0 && y_sbuf < height)
        {
            s_in_channel[threadIdx.x + threadIdx.y * BLOCK_SIZE] = in_channel[y_sbuf * width + x_sbuf];
        }
        else
        {
            s_in_channel[threadIdx.x + threadIdx.y * BLOCK_SIZE] = 0;
        }

        // nacteni konvolucni masky masky
        __shared__ float s_mask[FILTER_SIZE * FILTER_SIZE];
        if (threadIdx.x < FILTER_SIZE && threadIdx.y < FILTER_SIZE)
        {
            s_mask[threadIdx.x + threadIdx.y * FILTER_SIZE] = mask[threadIdx.x + threadIdx.y * FILTER_SIZE];
        }

        // synchronizace vsech vlaken v bloku
        __syncthreads();

        // provedeni 2D konvoluce
        float sum = 0.0;
        int row, col;
        if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE)
        {
            for (row = 0; row < FILTER_SIZE; row++)
            {
                for (col = 0; col < FILTER_SIZE; col++)
                {
                    sum += s_in_channel[(threadIdx.x + col) + (threadIdx.y + row) * BLOCK_SIZE] * s_mask[col + row * FILTER_SIZE];
                }
            }
            if (x < width && y < height)
            {
                out_channel[y * width + x] = (unsigned char)(sum);
            }
        }
    }

    /// @brief Aplikuje konvolucni masku 5x5 na obrazek
    /// @param img  Obrazek na ktery bude aplikovana konvoluce [DEVICE]
    /// @param mask Konvolucni maska o velikosti 5x5 [HOST]
    static bool applyConvMask(ImageRGB::Image *img,
                              MaskUtils::Mask_t *mask)
    {
        if (img == nullptr || mask == nullptr)
            return false;
        if (img->width == 0 ||
            img->height == 0 ||
            img->red_channel == nullptr ||
            img->green_channel == nullptr ||
            img->blue_channel == nullptr)
            return false;
        if (img->location != ImageRGB::DEVICE)
            return false;

        // vypocet grid size
        dim3 grid_size((img->width + TILE_SIZE - 1) / TILE_SIZE, (img->height + TILE_SIZE - 1) / TILE_SIZE);
        dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

        // presuna masku konvoluce na GPU
        float *d_mask = nullptr;
        CUDA_CHECK_RETURN(cudaMalloc(&d_mask, FILTER_SIZE * FILTER_SIZE * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemcpy(d_mask, mask->values, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Vytvoreni vystupniho obrazku na device
        ImageRGB::Image outImg;
        ImageRGB::init(&outImg, ImageRGB::DEVICE);
        ImageRGB::allocateIfIsNull(&outImg, img->width, img->height);

        // provedeni konvoluce pomoci 3 kernelu
        conv2DKernel<<<grid_size, block_size>>>(outImg.red_channel, img->red_channel, d_mask, img->width, img->height);
        conv2DKernel<<<grid_size, block_size>>>(outImg.green_channel, img->green_channel, d_mask, img->width, img->height);
        conv2DKernel<<<grid_size, block_size>>>(outImg.blue_channel, img->blue_channel, d_mask, img->width, img->height);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // dealokace masky
        CUDA_CHECK_RETURN(cudaFree(d_mask));

        // puvodni vstupni obrazek odstrani "dealokuje z pameti GPU" a do struktury vlozi 
        // pointery na nove alokovanou pamet GPU s vystupnim upravenym obrazkem
        ImageRGB::deleteImage(img);
        img->red_channel = outImg.red_channel;
        img->green_channel = outImg.green_channel;
        img->blue_channel = outImg.blue_channel;

        return true;
    }

}

#endif