#ifndef __CPU_IMAGE_H__
#define __CPU_IMAGE_H__

#include "headers.h"
#include "utils/pngio.h"

/// @brief Nastroje pro generovani obrazu z matice pomoci CPU
namespace CPUImage
{
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
        unsigned char *data_h = new unsigned char[width * height * 3];

        // vygenerovani obrazu z matice
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                // normalizuje aktualni hodnotu matice[x, y]: 0.0 az 1.0 => 0.0 minimum, 1.0 maximum
                float normalized = (matrix_d[i * width + j] - minVal) / (maxVal - minVal);

                // vypocita index pixelu v image bufferu 3b RGB
                int pixel_i = (i * width + j) * 3;

                // vypocita barvy pixelu [minimu = modra, maximu = cervena]
                unsigned char r, g, b;
                rainbowPalette(normalized, &r, &g, &b);
                data_h[pixel_i] = r;
                data_h[pixel_i + 1] = g;
                data_h[pixel_i + 2] = b;
            }
        }

        // ulozi obraz na disk
        png::image<png::rgb_pixel> img(width, height);
        pvg::rgbToPng(img, data_h);
        img.write(fileName);

        // dealokace pameti na CPU
        delete[] data_h;
    }
}

#endif