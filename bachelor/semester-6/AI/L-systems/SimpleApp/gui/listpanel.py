"""
Simple library for multiple views game aplication with pygame

File:       listpanel.py
Date:       15.02.2022

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

from SimpleApp.gui.vertical_scrollbar import VerticalScrollbar
import pygame
from ..utils import *
from ..colors import *
from ..guielement import *
from ..application import *


class ListPanel(GUIElement, Container):
    def __init__(self, view, style: dict, data: list, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create ListPanel element 
        Parameters:
            view -> View where is element
            style -> More about style for this element in config/styles.json
            data -> List with strings
            x -> X position
            y -> Y position
            width -> Width of Panel
            height -> Height of Panel
        """
        self.data = data
        self.v_scroll = None
        self.body_offset_y = 0
        super().__init__(view, x, y, width, height, style)
        self.v_scroll = VerticalScrollbar(view, super().getStyle(
        )["scrollbar"], super().getStyle()["scrollbar_width"])
        self.v_scroll.setOnScrollEvt(self.scrollVertical)
        self.layoutmanager = None
        self.callback = None
        self.refreshList()

    @overrides(GUIElement)
    def updateViewRect(self):
        super().updateViewRect()
        self.refreshList()

    def setItemClickEvet(self, callback):
        """
        Set item click event callback
        Parameters:
            callback -> callback function
        """
        self.callback = callback

    def scrollVertical(self, position: float):
        """
        Event for body v_scroll
        Parameters:
            position -> Vertical position of table body (0.0 - 1.0)
        """
        total_body_data_height = 10 + \
            (self.font.get_height() + 10) * len(self.data)
        h = super().getHeight()
        self.body_offset_y = -max(0, (total_body_data_height - h)) * position

    def refreshList(self, new_data: list = None):
        """
        Refresh data in list panel
        Parameters:
            new_data -> New list with string   
        """
        if new_data is not None:
            self.data = new_data

        self.font = pygame.font.SysFont(
            super().getStyle()["font_name"], super().getStyle()["font_size"], bold=super().getStyle()["font_bold"])

        self.height = 10 + (self.font.get_height() + 10) * min(5, len(self.data))

        if self.v_scroll is not None:
            sw = super().getStyle()["scrollbar_width"]
            self.v_scroll.setX(super().getX() + super().getWidth() - sw)
            self.v_scroll.setY(super().getY())
            self.v_scroll.setWidth(sw)
            self.v_scroll.setHeight(super().getHeight())

            height = 10 + (self.font.get_height() + 10) * len(self.data)
            self.v_scroll.setScrollerSize(
                (1.0 - max(0, height - super().getHeight()) / height) * self.v_scroll.getHeight())

    @overrides(GUIElement)
    def draw(self, view, screen):
        # background
        pygame.draw.rect(screen, super().getStyle()[
            "background_color"], super().getViewRect(), border_radius=5)

        # draw elements inside panel
        if len(self.data) != 0:
            screen.set_clip(super().getViewRect())
            offset = super().getY() + 10 + self.body_offset_y
            for line in self.data:
                text = self.font.render(
                    line, 1, super().getStyle()["foreground_color"])
                screen.blit(text, (super().getX() + 10, offset))
                offset += text.get_height() + 10
            screen.set_clip(None)

        # vertical scrollbar
        self.v_scroll.draw(view, screen)

        # outline
        pygame.draw.rect(screen, super().getStyle()[
            "outline_color"], super().getViewRect(), 2, border_radius=5)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        self.v_scroll.processEvent(view, event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            offset = super().getY() + 10 + self.body_offset_y
            for line in self.data:
                if inRect(
                        event.pos[0],
                        event.pos[1],
                        pygame.Rect(
                            super().getX(),
                            offset,
                            super().getWidth() - self.v_scroll.getWidth() - 5,
                            self.font.get_height()
                        )):
                    if self.callback is not None:
                        self.callback(line)
                offset += self.font.get_height() + 10

    @overrides(GUIElement)
    def update(self, view):
        pass

    @overrides(Container)
    def getChilds(self):
        return [self.v_scroll]
