"""
Simple library for multiple views game aplication with pygame

File:       vertical_scrollbar.py
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


class VerticalScrollbar(GUIElement):
    def __init__(self, view, style: dict, scroller_size: int, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create VerticalScrollbar
        Parameters:
            view -> View where is element
            style -> more about style for this element in config/styles.json
            scroller_size -> Scroller height
            width -> Width of VerticalScrollbar
            height -> Height of VerticalScrollbar
            x -> X position
            y -> Y position
        """
        super().__init__(view, x, y, width, height, style, pygame.SYSTEM_CURSOR_SIZENS)
        self.callback = None
        self.scroller_pos = 0
        self.scroller_size = scroller_size

    def setScrollerSize(self, size: int):
        """
        Set size of scroller
        Parameters:
            size -> Height in pixels
        """
        self.scroller_size = max(size, super().getWidth())

    def setOnScrollEvt(self, callback):
        """
        Set on scroll event callback
        Parameters:
            callback -> callback function
        """
        self.callback = callback

    @overrides(GUIElement)
    def draw(self, view, screen):
        # background
        pygame.draw.rect(screen, super().getStyle()[
                         "background_color"], super().getViewRect())
        # scroller
        pygame.draw.rect(
            screen,
            super().getStyle()["foreground_color"],
            pygame.Rect(
                super().getX(),
                super().getY() + self.scroller_pos,
                super().getWidth(),
                self.scroller_size
            ),
            border_radius=6
        )
        # outline
        pygame.draw.rect(screen, super().getStyle()[
                         "outline_color"], super().getViewRect(), 2)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        if self.scroller_size >= super().getHeight():
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                super().select()
                self.def_scroller_pos = self.scroller_pos
                self.drag_start = event.pos[1]
        elif event.type == pygame.MOUSEBUTTONUP:
            super().unSelect()
        elif event.type == pygame.MOUSEMOTION:
            if super().isSelected():
                self.scroller_pos = self.def_scroller_pos + \
                    (event.pos[1] - self.drag_start)
                self.scroller_pos = min(
                    max(0, self.scroller_pos), super().getHeight() - self.scroller_size)
                if self.callback is not None:
                    self.callback(self.scroller_pos /
                                  (super().getHeight() - self.scroller_size))

    @overrides(GUIElement)
    def update(self, view):
        pass
