#include "pngio.h"

namespace pvg {
    void rgbToPng( png_img_t& imgPng, const unsigned char *imgRgb )
    {
        unsigned int width = imgPng.get_width();
        unsigned int height = imgPng.get_height();
        
        for( unsigned int y = 0; y < height; ++y )
            for( unsigned int x = 0; x < width; ++x ) {
			    imgPng.set_pixel( x, y, png::rgb_pixel( *imgRgb,
			                                            *(imgRgb + 1),
			                                            *(imgRgb + 2) ) );
			    imgRgb += 3;
		    }
    }
    
    void pngToRgb( unsigned char *imgRgb, const png_img_t& imgPng )
    {
        unsigned int height = imgPng.get_height();
        
        for( unsigned int y = 0; y < height; ++y ) {
            std::vector<png::rgb_pixel> row = imgPng.get_row( y );
            for(std::vector<png::rgb_pixel>::iterator it = row.begin(); it != row.end(); ++it ) {
                png::rgb_pixel pixel = *it;
                
                *imgRgb++ = pixel.red; 
                *imgRgb++ = pixel.green;
                *imgRgb++ = pixel.blue;
            }
        }
    }
    
    void rgb3ToPng( png_img_t& imgPng,
                    const unsigned char *r,
                    const unsigned char *g,
                    const unsigned char *b )
    {
        unsigned int width = imgPng.get_width();
        unsigned int height = imgPng.get_height();
        
        for( unsigned int y = 0; y < height; ++y )
            for( unsigned int x = 0; x < width; ++x ) {
			    imgPng.set_pixel( x, y, png::rgb_pixel( *r++,
			                                            *g++,
			                                            *b++ ) );
		    }
    }
                    
    void pngToRgb3( unsigned char *r,
                    unsigned char *g,
                    unsigned char *b,
                    const png_img_t& imgPng )
    {
        unsigned int height = imgPng.get_height();
        
        for( unsigned int y = 0; y < height; ++y ) {
            std::vector<png::rgb_pixel> row = imgPng.get_row( y );
            for(std::vector<png::rgb_pixel>::iterator it = row.begin(); it != row.end(); ++it ) {
                png::rgb_pixel pixel = *it;
                
                *r++ = pixel.red; 
                *g++ = pixel.green;
                *b++ = pixel.blue;
            }
        }
    }
}