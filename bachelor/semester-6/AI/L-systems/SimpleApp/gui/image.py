"""
Simple library for multiple views game aplication with pygame

File:       image.py
Date:       09.02.2022

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

import pygame
from ..utils import *
from ..colors import *
from ..guielement import *


class Image(GUIElement):
    def __init__(self, view, image_path: str, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create Image element 
        Parameters:
            view -> View where is element
            image_path -> Image path
            width -> Width of Image
            height -> Height of Image
            x -> X position
            y -> Y position
        """
        super().__init__(view, x, y, width, height, None)
        self.image = loadImage(image_path)

    def setImage(self, image_path: str):
        """
        Set new image
        Parameters:
            image_path -> Image path
        """
        self.image = loadImage(image_path) 

    def getImage(self) -> pygame.Surface:
        """
        Get image
        """
        return self.image  

    @overrides(GUIElement)
    def draw(self, view, screen):
        if self.image is not None:
            screen.blit(pygame.transform.scale(self.image, (super().getWidth(
            ), super().getHeight())), (super().getX(), super().getY()))

    @overrides(GUIElement)
    def processEvent(self, view, event):
        pass

    @overrides(GUIElement)
    def update(self, view):
        pass
