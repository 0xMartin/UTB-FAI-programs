#ifndef __MASK_UTILS_H__
#define __MASK_UTILS_H__

#include <stdlib.h>

namespace MaskUtils
{

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Zakladni masky
    ////////////////////////////////////////////////////////////////////////////////////////////////

    static const float blur_mask_5[25u] = {
        1 / 273.0, 4 / 273.0, 7 / 273.0, 4 / 273.0, 1 / 273.0,
        4 / 273.0, 16 / 273.0, 26 / 273.0, 16 / 273.0, 4 / 273.0,
        7 / 273.0, 26 / 273.0, 41 / 273.0, 26 / 273.0, 7 / 273.0,
        4 / 273.0, 16 / 273.0, 26 / 273.0, 16 / 273.0, 4 / 273.0,
        1 / 273.0, 4 / 273.0, 7 / 273.0, 4 / 273.0, 1 / 273.0};

    static const float laplacian_mask_5[25] = {
        0, 0, -1, 0, 0,
        0, -1, -2, -1, 0,
        -1, -2, 16, -2, -1,
        0, -1, -2, -1, 0,
        0, 0, -1, 0, 0};

    static const float sharpen_mask_5[25] = {
        -1, -1, -1, -1, -1,
        -1, 2, 2, 2, -1,
        -1, 2, 8, 2, -1,
        -1, 2, 2, 2, -1,
        -1, -1, -1, -1, -1};

    static const float sobel_mask_x_5[25] = {
        -1, -2, 0, 2, 1,
        -4, -8, 0, 8, 4,
        -6, -12, 0, 12, 6,
        -4, -8, 0, 8, 4,
        -1, -2, 0, 2, 1};

    static const float sobel_mask_y_5[25] = {
        -1, -4, -6, -4, -1,
        -2, -8, -12, -8, -2,
        0, 0, 0, 0, 0,
        2, 8, 12, 8, 2,
        1, 4, 6, 4, 1};

    static const float prewitt_mask_x_5[25] = {
        -1, -1, 0, 1, 1,
        -1, -1, 0, 1, 1,
        -1, -1, 0, 1, 1,
        -1, -1, 0, 1, 1,
        -1, -1, 0, 1, 1};

    static const float prewitt_mask_y_5[25] = {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1};

    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// @brief Struktura masky (hodnoty masky + velikost masky)
    struct Mask_t
    {
        float *values;
        size_t size;
    };

    /// @brief Vytvori masku z templatu
    /// @param template Ukazatel na template
    /// @param size Velikost masky
    /// @return True v pripadae uspesneho provedeni operace
    extern bool createFromTemplate(Mask_t *mask, const float *mask_template, size_t size);

    /// @brief Normalizuje hodnoty v masce. Nebude provedena takova normalizace ktera by vynulovaval masku
    /// @param mask Ukazatel na masku
    extern void normalize(Mask_t *mask);

    /// @brief Odstrani masku
    /// @param mask Pointer na masku ktera bude odstranena
    extern void deleteMask(Mask_t *mask);

}

#endif