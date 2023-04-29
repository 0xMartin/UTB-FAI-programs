"""
Simple library for multiple views game aplication with pygame

File:       table.py
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
from SimpleApp.gui.vertical_scrollbar import VerticalScrollbar
from SimpleApp.gui.horizontal_scrollbar import HorizontalScrollbar


class Table(GUIElement, Container):
    def __init__(self, view, style: dict, data: dict, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create Table element
        Parameters:
            view -> View where is element
            style -> more about style for this element in config/styles.json
            data -> Data of Table
            width -> Width of Table
            height -> Height of Table
            x -> X position
            y -> Y position
        """
        self.last_data = None
        self.body_offset_x = 0
        self.body_offset_y = 0
        super().__init__(view, x, y, width, height, style)
        # vertical scrollbar
        self.v_scroll = VerticalScrollbar(
            view,
            super().getStyle()["scrollbar"],
            super().getStyle()["body"]["scrollbar_width"]
        )
        self.v_scroll.setOnScrollEvt(self.tableScrollVertical)
        # horizontal scrollbar
        self.h_scroll = HorizontalScrollbar(
            view,
            super().getStyle()["scrollbar"],
            super().getStyle()["body"]["scrollbar_width"]
        )
        self.h_scroll.setOnScrollEvt(self.tableScrollHorizontal)
        # refresh table data
        self.refreshTable(data)

    def tableScrollVertical(self, position: float):
        """
        Event for table body v_scroll
        Parameters:
            position -> Vertical position of table body (0.0 - 1.0)
        """
        header_height = self.header_font.get_height() * 1.8
        total_body_data_height = header_height + self.body_font.get_height() * 1.4 * \
            len(self.body)
        h = super().getHeight() - super().getStyle()["body"]["scrollbar_width"]
        self.body_offset_y = -max(
            0, (total_body_data_height - h)) * position

    def tableScrollHorizontal(self, position: float):
        """
        Event for table body h_scroll
        Parameters:
            position -> Horizontal position of table body (0.0 - 1.0)
        """
        total_body_data_width = sum(self.col_width)
        w = super().getWidth() - super().getStyle()["body"]["scrollbar_width"]
        self.body_offset_x = -max(
            0, (total_body_data_width - w)) * position

    def refreshTable(self, data: dict = None):
        """
        Refresh table data
        Parameters:
            data -> {header: [], body: [[],[],[], ...]}
        """
        if data is None:
            data = self.last_data
        self.last_data = data
        if data is None:
            return

        self.header_font = pygame.font.SysFont(
            super().getStyle()["header"]["font_name"],
            super().getStyle()["header"]["font_size"],
            bold=super().getStyle()["header"]["font_bold"]
        )
        self.body_font = pygame.font.SysFont(
            super().getStyle()["body"]["font_name"],
            super().getStyle()["body"]["font_size"],
            bold=super().getStyle()["body"]["font_bold"]
        )
        # build table header
        self.header = data["header"]
        # build table body
        self.body = data["body"]

        scroll_size = super().getStyle()["body"]["scrollbar_width"]

        # calculate max width for each col of table
        self.col_width = [0] * len(self.header)
        for row in self.body:
            for i, cell in enumerate(row):
                # (super().getWidth() - scroll_size) / len(self.header)
                self.col_width[i] = max(self.body_font.size(
                    cell)[0] + 10, self.col_width[i])
        for i, cell in enumerate(self.header):
            # (super().getWidth() - scroll_size) / len(self.header)
            self.col_width[i] = max(self.header_font.size(
                cell)[0] + 10, self.col_width[i])
        if sum(self.col_width) <= super().getWidth() - scroll_size:
            for i in range(len(self.header)):
                self.col_width[i] = super().getWidth() / len(self.header)

        # vertical scrollbar
        self.v_scroll.setX(
            super().getX() + super().getWidth() - 1 - scroll_size)
        self.v_scroll.setY(super().getY())
        self.v_scroll.setWidth(scroll_size)
        self.v_scroll.setHeight(super().getHeight())
        header_height = self.header_font.get_height() * 1.8
        total_body_data_height = header_height + self.body_font.get_height() * 1.4 * \
            len(self.body)
        self.v_scroll.setScrollerSize(
            (1.0 - max(0, total_body_data_height - super().getHeight()) / total_body_data_height) * self.v_scroll.getHeight())

        # horizontal scrollbar
        self.h_scroll.setX(super().getX())
        self.h_scroll.setY(
            super().getY() + super().getHeight() - 1 - scroll_size)
        self.h_scroll.setWidth(super().getWidth() - scroll_size)
        self.h_scroll.setHeight(scroll_size)
        total_body_data_width = sum(self.col_width)
        self.h_scroll.setScrollerSize(
            (1.0 - max(0, total_body_data_width - super().getWidth()) / total_body_data_width) * self.h_scroll.getWidth())

    @overrides(GUIElement)
    def updateViewRect(self):
        super().updateViewRect()
        # scroller
        self.refreshTable()

    @overrides(GUIElement)
    def draw(self, view, screen):
        # set clip
        screen.set_clip(
            pygame.Rect(
                super().getX(),
                super().getY(),
                super().getWidth() - 1,
                super().getHeight() - 1,
            )
        )

        # size of table body + header
        w = super().getWidth() - super().getStyle()["body"]["scrollbar_width"]
        h = super().getHeight() - super().getStyle()["body"]["scrollbar_width"]
        rect = pygame.Rect(
            super().getX(),
            super().getY(),
            w,
            h
        )
        # draw table body background
        pygame.draw.rect(
            screen,
            super().getStyle()["body"]["background_color"],
            rect
        )
        # draw col lines
        offset = self.body_offset_x
        for i in range(len(self.header)):
            pygame.draw.line(
                screen,
                colorChange(super().getStyle()[
                            "body"]["background_color"], -0.5),
                (super().getX() + offset, super().getY()),
                (super().getX() + offset, super().getY() + h - 4),
                2
            )
            offset += self.col_width[i]

        # draw body data
        for j, row in enumerate(self.body):
            offset = self.body_offset_x
            for i, cell in enumerate(row):
                if len(cell) != 0:
                    text = self.body_font.render(
                        cell, 1, super().getStyle()["body"]["foreground_color"])
                    header_offset = self.header_font.get_height() * 1.8
                    screen.blit(
                        text,
                        (
                            super().getX() + 5 + offset,
                            super().getY() + header_offset +
                            self.body_font.get_height() * 1.4 * j + self.body_offset_y
                        )
                    )
                    offset += self.col_width[i]

        # draw table header
        if self.header is not None:
            pygame.draw.rect(
                screen,
                super().getStyle()["header"]["background_color"],
                pygame.Rect(
                    super().getX(),
                    super().getY(),
                    w,
                    self.header_font.get_height() * 1.8
                )
            )
            offset = self.body_offset_x
            for i, col in enumerate(self.header):
                if len(col) != 0:
                    text = self.header_font.render(
                        col, 1, super().getStyle()["header"]["foreground_color"])
                    screen.blit(
                        text,
                        (
                            super().getX() + 5 + offset,
                            super().getY() + self.header_font.get_height() * 0.4
                        )
                    )
                    offset += self.col_width[i]

        # draw v_scrollbar
        self.v_scroll.draw(view, screen)
        # draw h_scrollbar
        self.h_scroll.draw(view, screen)

        # draw outline
        pygame.draw.rect(
            screen,
            super().getStyle()["header"]["background_color"],
            rect,
            2
        )

        # reset clip
        screen.set_clip(None)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        self.v_scroll.processEvent(view, event)
        self.h_scroll.processEvent(view, event)

    @overrides(GUIElement)
    def update(self, view):
        pass

    @overrides(Container)
    def getChilds(self):
        elements = [self.v_scroll, self.h_scroll]
        return elements
