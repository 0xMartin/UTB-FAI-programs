#include "mask_utils.h"

namespace MaskUtils
{

    bool createFromTemplate(Mask_t *mask, const float *mask_template, size_t size)
    {
        if (mask == nullptr || mask_template == nullptr || size == 0)
            return false;

        mask->values = new float[size * size];
        mask->size = size;
        for (int i = 0; i < size * size; ++i)
        {
            mask->values[i] = mask_template[i];
        }

        return true;
    }

    void normalize(Mask_t *mask)
    {
        if (mask == nullptr)
            return;
        if (mask->values == nullptr)
            return;
        float sum = 0.0;
        for (int i = 0; i < mask->size * mask->size; ++i)
            sum += mask->values[i];
        if (sum <= 1.0)
        {
            return;
        }
        for (int i = 0; i < mask->size * mask->size; ++i)
            mask->values[i] /= sum;
    }

    void deleteMask(Mask_t *mask)
    {
        if (mask->values != nullptr)
            delete[] mask->values;
        mask->values = nullptr;
    }

}