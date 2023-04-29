"""
Simple library for multiple views game aplication with pygame

File:       panel.py
Date:       11.02.2022

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
from ..application import *


class Panel(GUIElement, Layout, Container):
    def __init__(self, view, style: dict, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create Panel element 
        Parameters:
            view -> View where is element
            style -> More about style for this element in config/styles.json
            x -> X position
            y -> Y position
            width -> Width of Panel
            height -> Height of Panel
        """
        GUIElement.__init__(self, view, x, y, width, height, style)
        Layout.__init__(self, view)
        self.layoutmanager = None

    def setLayoutManager(self, layoutmanager: Layout):
        """
        Set layout manager
        Parameters:
            layoutmanager -> layout manager
        """
        self.layoutmanager = layoutmanager
        self.getView().unregisterLayoutManager(self.layoutmanager)

    @overrides(GUIElement)
    def draw(self, view, screen):
        # background
        pygame.draw.rect(screen, super().getStyle()[
            "background_color"], super().getViewRect(), border_radius=5)

        # draw elements inside panel
        if len(self.getLayoutElements()) != 0:
            panel_screen = screen.subsurface(
                pygame.Rect(
                    super().getX() + 5,
                    super().getY() + 5,
                    min(max(super().getWidth() - 10, 10), screen.get_width() - super().getX() - 5),
                    min(max(super().getHeight() - 10, 10), screen.get_height() - super().getY() - 5)
                )
            )
            for el in self.getLayoutElements():
                el["element"].draw(view, panel_screen)

        # outline
        pygame.draw.rect(screen, super().getStyle()[
            "outline_color"], super().getViewRect(), 2, border_radius=5)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        if len(self.getLayoutElements()) != 0:
            panel_evt = event

            # offset
            if panel_evt.type == pygame.MOUSEMOTION or \
                    panel_evt.type == pygame.MOUSEBUTTONUP or \
                    panel_evt.type == pygame.MOUSEBUTTONDOWN:
                panel_evt.pos = tuple([
                    panel_evt.pos[0] - super().getX(),
                    panel_evt.pos[1] - super().getY()
                ])

            # for each
            for el in self.getLayoutElements():
                el["element"].processEvent(view, panel_evt)

            # restore
            if panel_evt.type == pygame.MOUSEMOTION or \
                    panel_evt.type == pygame.MOUSEBUTTONUP or \
                    panel_evt.type == pygame.MOUSEBUTTONDOWN:
                panel_evt.pos = tuple([
                    panel_evt.pos[0] + super().getX(),
                    panel_evt.pos[1] + super().getY()
                ])

    @overrides(GUIElement)
    def update(self, view):
        for el in self.getLayoutElements():
            el["element"].update(view)

    @overrides(Layout)
    def updateLayout(self, width, height):
        if self.layoutmanager is not None:
            self.layoutmanager.setElements(self.getLayoutElements())
            self.layoutmanager.updateLayout(
                self.getWidth() - 10, self.getHeight() - 10)

    @overrides(Container)
    def getChilds(self):
        elements = []
        for le in self.getLayoutElements():
            elements.append(le["element"])    
        return elements
