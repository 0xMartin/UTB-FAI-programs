#ifndef __PNGIO_H_
#define __PNGIO_H_

#include <png++/png.hpp>

typedef png::image<png::rgb_pixel> png_img_t;

namespace pvg {
    void rgbToPng( png_img_t& imgPng,
                   const unsigned char *imgRgb );
                   
    void pngToRgb( unsigned char *imgRgb,
                   const png_img_t& imgPng );
    
    void rgb3ToPng( png_img_t& imgPng,
                    const unsigned char *r,
                    const unsigned char *g,
                    const unsigned char *b );
                    
    void pngToRgb3( unsigned char *r,
                    unsigned char *g,
                    unsigned char *b,
                    const png_img_t& imgPng );
}

#endif //__PNGIO_H_