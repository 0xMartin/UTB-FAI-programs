"""
Simple library for multiple views game aplication with pygame

File:       colors.py
Date:       08.02.2022

Github:     https://github.com/0xMartin
Email:      martin.krcma1@gmail.com
 
Copyright (C) 2022 Martin Krcma
 
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:
 
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

# colors
BLACK = (0, 0, 0)
GRAY = (127, 127, 127)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)


def colorChange(color: tuple, amount: float) -> tuple:
    """
    Change color lightness
    Parameters:
        color -> default color
        amount -> from -2(darker) o 2(lighter)
    """
    rgb = list(color)
    amount = 1.0 + amount / 2.0
    rgb[0] *= amount
    rgb[1] *= amount
    rgb[2] *= amount
    rgb[0] = max(min(rgb[0], 255), 0)
    rgb[1] = max(min(rgb[1], 255), 0)
    rgb[2] = max(min(rgb[2], 255), 0)
    return tuple(rgb)

def colorAdd(color: tuple, amount: int) -> tuple:
    """
    Add number to color
    Parameters:
        color -> default color
        amount -> 
    """
    rgb = list(color)
    rgb[0] += amount
    rgb[0] = max(min(rgb[0], 255), 0)
    rgb[1] += amount
    rgb[1] = max(min(rgb[1], 255), 0)
    rgb[2] += amount
    rgb[2] = max(min(rgb[2], 255), 0)
    return tuple(rgb)


def colorInvert(color: tuple) -> tuple:
    """
    Invert color
    Parameters:
        color -> default color
    """
    rgb = list(color)
    rgb[0] = 255 - rgb[0]
    rgb[1] = 255 - rgb[1]
    rgb[2] = 255 - rgb[2]
    return tuple(rgb)


def createColor(red: int, green: int, blue: int) -> tuple:
    """
    Create color
    Parameters:
        red -> 0-255
        green -> 0-255
        blue -> 0-255
    """
    return tuple(
        max(min(red, 255), 0),
        max(min(green, 255), 0),
        max(min(blue, 255), 0)
    )
