"""
Simple library for multiple views game aplication with pygame

File:       button.py
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

import pygame
from ..utils import *
from ..colors import *
from ..guielement import *


class Button(GUIElement):

    def __init__(self, view, style: dict, text: str, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create button
        Parameters:
            view -> View where is element
            style -> More about style for this element in config/styles.json
            text - Text of Button
            width -> Width of Button
            height -> Height of Button
            x -> X position
            y -> Y position
        """
        super().__init__(view, x, y, width, height, style)
        self.text = text
        self.callback = None
        self.hover = False
        self.font = pygame.font.SysFont(
            super().getStyle()["font_name"], super().getStyle()["font_size"], bold=super().getStyle()["font_bold"])

    def setText(self, text: str):
        """
        Set text of Button
        Parameters:
            text -> New text of Button 
        """
        self.text = text

    def getText(self) -> str:
        """
        Get text of Button
        """
        return self.text

    def setClickEvt(self, callback):
        """
        Set button click event callback
        Parameters:
            callback -> callback function
        """
        self.callback = callback

    @overrides(GUIElement)
    def draw(self, view, screen):
        # background
        if self.isSelected():
            c = super().getStyle()["background_color"]
            pygame.draw.rect(screen, colorChange(
                c, -0.2 if c[0] > 128 else 0.6), super().getViewRect(), border_radius=10)
        else:
            pygame.draw.rect(screen, super().getStyle()[
                             "background_color"], super().getViewRect(), border_radius=10)
        # button text
        if len(self.text) != 0:
            text = self.font.render(
                self.text, 1, super().getStyle()["foreground_color"])
            if text.get_height() + 4 > super().getHeight():
                super().setHeight(text.get_height() + 4)
            if text.get_width() + 4 > super().getWidth():
                super().setWidth(text.get_width() + 4)
            screen.blit(text, (super().getX() + (super().getWidth() - text.get_width())/2,
                               super().getY() + (super().getHeight() - text.get_height())/2))
        # outline
        pygame.draw.rect(screen, super().getStyle()[
            "outline_color"], super().getViewRect(), 2, border_radius=10)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                if self.callback is not None:
                    self.callback(self)
        elif event.type == pygame.MOUSEMOTION:
            if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                self.select()
            else:
                self.unSelect()

    @overrides(GUIElement)
    def update(self, view):
        pass
