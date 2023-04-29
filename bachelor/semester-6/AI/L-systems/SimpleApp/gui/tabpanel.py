"""
Simple library for multiple views game aplication with pygame

File:       tabpanel.py
Date:       12.02.2022

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


class TabPanel(GUIElement, Container):
    def __init__(self, view, style: dict, tabs: list, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create TabPanel element
        Parameters:
            view -> View where is element
            style -> More about style for this element in config/styles.json
            tabs -> List of tabs
            x -> X position
            y -> Y position
            width -> Width of Panel
            height -> Height of Panel
        """
        GUIElement.__init__(self, view, x, y, width, height, style)
        self.layoutmanager = None
        self.selected_tab = 0
        self.font = pygame.font.SysFont(
            super().getStyle()["font_name"], super().getStyle()["font_size"], bold=super().getStyle()["font_bold"])
        self.tabs = []
        for t in tabs:
            if isinstance(t, Tab):
                self.addTab(t)

    def setTabs(self, tabs: list):
        """
        Set tab names
        Parameters:
            tabs - List with tab names
        """
        self.tabs = []
        for t in tabs:
            if isinstance(t, Tab):
                self.addTab(t)

    def setSelectedTab(self, index: int):
        """
        Set selected tab
        Parameters:
            index -> Tab index       
        """
        self.selected_tab = index

    def addTab(self, tab):
        """
        Add new tab
        Parameters:
            tab -> New tab
        """
        if isinstance(tab, Tab):
            self.tabs.append(tab)
            self.updateTabSize(tab)

    def updateTabSize(self, tab):
        content = tab.getContent()
        if content is not None:
            tab_header_height = self.font.render(
                "W",
                1,
                super().getStyle()["foreground_color"]
            ).get_height() + 10
            content.setX(0)
            content.setY(0)
            content.setWidth(super().getWidth())
            content.setHeight(super().getHeight() - tab_header_height)
            if isinstance(content, Layout) and content.getWidth() > 0 and content.getHeight() > 0:
                content.updateLayout(0, 0)

    def removeTab(self, tab):
        """
        Remove tab
        Parameters:
            tab -> Some tab 
        """
        self.tabs.remove(tab)

    @overrides(GUIElement)
    def setWidth(self, width):
        super().setWidth(width)
        for tab in self.tabs:
            self.updateTabSize(tab)

    @overrides(GUIElement)
    def setHeight(self, height):
        super().setHeight(height)
        for tab in self.tabs:
            self.updateTabSize(tab)

    @overrides(GUIElement)
    def draw(self, view, screen):
        if len(self.tabs) == 0:
            return

        tab_header_height = 0
        selected_x = [0, 0]
        x_offset = 5 + super().getX()
        # tab names
        for i, tab in enumerate(self.tabs):
            if len(tab.getName()) != 0:
                # tab label (create text bitmap)
                text = self.font.render(
                    tab.getName(),
                    1,
                    super().getStyle()["foreground_color"]
                )
                tab_header_height = max(
                    tab_header_height, text.get_height() + 10)

                # for selected tab label draw bg
                x1 = x_offset
                x2 = x_offset + text.get_width() + 10
                if i == self.selected_tab:
                    # highlight bg for selected tab label
                    pygame.draw.rect(
                        screen,
                        super().getStyle()["background_color"],
                        pygame.Rect(
                            x1,
                            super().getY(),
                            x2 - x1,
                            tab_header_height
                        )
                    )
                    # cords for line that make line under label trasparent
                    selected_x = [x1 + 2, x2 - 1]
                # outlines
                pygame.draw.lines(
                    screen,
                    super().getStyle()["outline_color"],
                    False,
                    [
                        (x1, super().getY() + tab_header_height),
                        (x1, super().getY()),
                        (x2, super().getY()),
                        (x2, super().getY() + tab_header_height)
                    ],
                    2
                )
                # draw label text
                screen.blit(
                    text,
                    (x_offset + 5, 5 + super().getY())
                )
                # change x offset
                x_offset += text.get_width() + 10

        rect = pygame.Rect(
            super().getX(),
            super().getY() + tab_header_height,
            super().getWidth(),
            super().getHeight() - tab_header_height
        )

        # background
        pygame.draw.rect(
            screen,
            super().getStyle()["background_color"],
            rect,
            border_radius=5
        )
        # outline
        pygame.draw.rect(
            screen,
            super().getStyle()["outline_color"],
            rect,
            2,
            border_radius=5
        )
        # content of tab
        if self.selected_tab >= 0 and self.selected_tab < len(self.tabs):
            tab_screen = screen.subsurface(rect)
            content = self.tabs[self.selected_tab].getContent()
            if content is not None:
                content.draw(view, tab_screen)
        # outline 2
        pygame.draw.line(
            screen,
            super().getStyle()["background_color"],
            (selected_x[0], super().getY() + tab_header_height),
            (selected_x[1], super().getY() + tab_header_height),
            2
        )

    @overrides(GUIElement)
    def processEvent(self, view, event):
        # tab selector
        if event.type == pygame.MOUSEBUTTONDOWN:
            x_offset = 5 + super().getX()
            # tab names
            for i, tab in enumerate(self.tabs):
                if len(tab.getName()) != 0:
                    # tab label (create text bitmap)
                    text = self.font.render(
                        tab.getName(),
                        1,
                        super().getStyle()["foreground_color"]
                    )
                    # for selected tab label draw bg
                    x1 = x_offset
                    x2 = x_offset + text.get_width() + 10
                    rect = pygame.Rect(
                        x1,
                        super().getY(),
                        x2 - x1,
                        text.get_height() + 10
                    )
                    # change x offset
                    x_offset += text.get_width() + 10
                    if inRect(event.pos[0], event.pos[1], rect):
                        self.selected_tab = i
                        break

        # offset
        tab_header_height = self.font.render(
            "W",
            1,
            super().getStyle()["foreground_color"]
        ).get_height() + 10
        if event.type == pygame.MOUSEMOTION or \
                event.type == pygame.MOUSEBUTTONUP or \
                event.type == pygame.MOUSEBUTTONDOWN:
            event.pos = tuple([
                event.pos[0] - super().getX(),
                event.pos[1] - super().getY() - tab_header_height
            ])
        # events for tab content
        if self.selected_tab >= 0 and self.selected_tab < len(self.tabs):
            content = self.tabs[self.selected_tab].getContent()
            if content is not None:
                content.processEvent(view, event)
        # restore
        if event.type == pygame.MOUSEMOTION or \
                event.type == pygame.MOUSEBUTTONUP or \
                event.type == pygame.MOUSEBUTTONDOWN:
            event.pos = tuple([
                event.pos[0] + super().getX(),
                event.pos[1] + super().getY() + tab_header_height
            ])

    @overrides(GUIElement)
    def update(self, view):
        for tab in self.tabs:
            if tab.getContent() is not None:
                tab.getContent().update(view)

    @overrides(Container)
    def getChilds(self):
        list = []
        for tab in self.tabs:
            if tab.getContent() is not None:
                list.append(tab.getContent())
        return list


class Tab:
    def __init__(self, name: str, content: GUIElement):
        self.name = name
        if isinstance(content, GUIElement):
            self.content = content
        else:
            self.content = None

    def getName(self):
        return self.name

    def setName(self, name: str):
        self.name = name

    def getContent(self):
        return self.content

    def setContent(self, content: GUIElement):
        if isinstance(content, GUIElement):
            self.content = content
