#ifndef __CPU_GENERATOR_H__
#define __CPU_GENERATOR_H__

#include "headers.h"
#include "opt_function.h"

namespace CPUGenerator
{
    /// @brief Na CPU vygeneruje matici hodnot testovaci funkce v rozsahu (X ∈ [minX, maxX], Y ∈ [minY, maxY]). 
    ///        Pomer (maxX - minX)/(maxY-minY) se musi rovnat (img_width)/(img_height) jinak dojde k nerovnomernemu meritku
    ///
    /// @param matrix Pointer na matici do ktere se budou vysledne hodnoty zapisovat (CUP)
    /// @param width Sirka matice
    /// @param height Vyska matice
    /// @param minX Minimalni hodnota na ose X
    /// @param maxX Maximalni hodnota na ose X
    /// @param minY Minimalni hodnota na ose Y
    /// @param maxY Maximalni hodnota na ose Y
    static void generate(float *matrix_h, int width, int height, float minX, float maxX, float minY, float maxY)
    {
        if (abs(((float)width/height) - ((maxX - minX)/(maxY - minY))) > 0.01) {
            std::cout << "Wrong ratio, the image becomes distorted!!" << std::endl;
        }

        // vypocet kroku v ose X, Y => 1 krok je zmena realne hodnoty v dane dimenzi kdyz je vzdalenost v obraze 1 pixel
        float xStep = (maxX - minX) / (width - 1);
        float yStep = (maxY - minY) / (height - 1);

        unsigned int i, j;
        float x, y;
        for (i = 0; i < height; ++i)
        {
            for (j = 0; j < width; ++j)
            {
                // vypocet realnych pozici X, Y
                x = minX + j * xStep;
                y = minY + i * yStep;
                // vypocet hodnoty funkce a zapis do matice
                matrix_h[i * width + j] = opt(x, y);
            }
        }
    }
}

#endif