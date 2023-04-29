"""
Simple library for multiple views game aplication with pygame

File:       canvas.py
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


class Canvas(GUIElement):
    def __init__(self, view, style: dict, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create Canvas
        Parameters:
            view -> View where is element
            style -> More about style for this element in config/styles.json
            width -> Width of Canvas
            height -> Height of Canvas
            x -> X position
            y -> Y position
        """
        super().__init__(view, x, y, width, height, style, pygame.SYSTEM_CURSOR_SIZEALL)
        self.callback = None
        self.control = False
        self.mouse_sensitivity = 2.0
        self.offset = [0, 0]
        self.font = pygame.font.SysFont(
            super().getStyle()["font_name"], super().getStyle()["font_size"], bold=super().getStyle()["font_bold"])

    def enableMouseControl(self):
        """
        Enable mouse countrol of canvas (offset, rotation)
        """
        self.control = True

    def setMouseSensitivity(self, mouse_sensitivity: float):
        """
        Set mouse sensitivity
        Parameters:
            mouse_sensitivity -> Mouse sensitivity
        """
        self.mouse_sensitivity = mouse_sensitivity

    def getMouseSensitivity(self) -> float:
        """
        Get mouse sensitivity
        """
        return self.mouse_sensitivity

    def disableMouseControl(self):
        """
        Disable mouse countrol of canvas (offset, rotation)
        """
        self.control = False

    def setOffset(self, offset: list):
        """
        Set drawing offset
        """
        self.offset = offset

    def getOffset(self) -> list:
        """
        Get drawing offset
        """

    def setPaintEvt(self, callback):
        """
        Set paint event callback
        Parameters:
            callback -> callback function
        """
        self.callback = callback

    @overrides(GUIElement)
    def draw(self, view, screen):
        # background
        pygame.draw.rect(screen, super().getStyle()[
                         "background_color"], super().getViewRect())

        # create subsurface
        surface = screen.subsurface(
            pygame.Rect(
                super().getX(),
                super().getY(),
                min(max(super().getWidth(), 10),
                    screen.get_width() - super().getX()),
                min(max(super().getHeight(), 10),
                    screen.get_height() - super().getY())
            )
        )
        # call paint callback
        if self.callback is not None:
            self.callback(surface, self.offset)

        # info
        text = self.font.render(
            "x: " + str(self.offset[0]) + " y: " + str(self.offset[1]),
            1,
            super().getStyle()["foreground_color"]
        )
        screen.blit(text, (self.getX() + 10, self.getY() + self.getHeight() - 10 - text.get_height()))

        # outline
        pygame.draw.rect(screen, super().getStyle()[
                         "outline_color"], super().getViewRect(), 2)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        if self.control:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                    super().select()
                    self.mouse_motion = True
            elif event.type == pygame.MOUSEBUTTONUP:
                super().unSelect()
            elif event.type == pygame.MOUSEMOTION:
                if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                    if self.isSelected():
                        if self.mouse_motion:
                            self.last_pos = event.pos
                        else:
                            self.offset[0] += (event.pos[0] - self.last_pos[0]) * self.mouse_sensitivity
                            self.offset[1] += (event.pos[1] - self.last_pos[1]) * self.mouse_sensitivity
                        self.mouse_motion = not self.mouse_motion

    @overrides(GUIElement)
    def update(self, view):
        pass
