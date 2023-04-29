"""
Simple library for multiple views game aplication with pygame

File:       gui.py
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
from typing import final
import abc


class GUIElement(metaclass=abc.ABCMeta):
    """
    Base class for GUI elements
    """

    def __init__(
        self,
        view, x: int,
        y: int,
        width: int,
        height: int,
        style: dict,
        selected_cursor=pygame.SYSTEM_CURSOR_HAND
    ):
        """
        Create GUIElement
        Parameters:
            x -> X position of Element
            y -> Y position of Element
            width -> Width of Element
            height -> Height of Element
            style -> Style of Element
            selected_cursor -> The type of cursor that appears when this element is selected
        """
        self.view = view
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.selected_cursor = selected_cursor
        self.visible = True

        sm = view.getApp().getStyleManager()
        if style is None:
            self.style = sm.getStyleWithName(self.__class__.__name__)
        else:
            self.style = style

        self.selected = False
        self.updateViewRect()

    def setVisibility(self, visible: bool):
        """
        Set visibility of element
        Parameters:
            visible -> True/False
        """
        self.visible = visible

    def isVisible(self) -> bool:
        """
        Check if element is visible
        """
        return self.visible

    def setSelectCursor(self, cursor):
        """
        Set cursor type when this element is selected
        Parameters:
            cursor -> Type of cursor that appears when this element is selected
        """
        self.selected_cursor = cursor

    @final
    def getSelectCursor(self):
        """
        Return cursor type when this element is selected
        """
        return self.selected_cursor

    @final
    def getView(self):
        """
        Get view to which the element belongs
        """
        return self.view

    @final
    def getX(self) -> int:
        """
        Get x position of this element
        """
        return self.x

    @final
    def getY(self) -> int:
        """
        Get y position of this element
        """
        return self.y

    @final
    def getWidth(self) -> int:
        """
        Get width of this element
        """
        return self.width

    @final
    def getHeight(self) -> int:
        """
        Get height of this element
        """
        return self.height

    @final
    def getStyle(self) -> dict:
        """
        Get style of this element
        """
        return self.style

    def setX(self, x: int):
        """
        Set x position of this element
        Parameters:
            x -> New X position
        """
        self.x = x
        self.updateViewRect()

    def setY(self, y: int):
        """
        Set y position of this element
        Parameters:
            y -> New Y position
        """
        self.y = y
        self.updateViewRect()

    def setWidth(self, width: int):
        """
        Set width of this element
        Parameters:
            width -> New width
        """
        if width >= 0:
            self.width = width
            self.updateViewRect()

    def setHeight(self, height: int):
        """
        Set height of this element
        Parameters:
            height -> New height
        """
        if height >= 0:
            self.height = height
            self.updateViewRect()

    def setStyle(self, style: dict):
        """
        Set style of this element
        Parameters:
            style -> New style of element
        """
        self.style = style

    def updateViewRect(self):
        """
        Update view rect of this element
        """
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    @final
    def getViewRect(self) -> pygame.Rect:
        """
        Get view rect of this element
        """
        return self.rect

    @final
    def select(self):
        """
        Select this element
        """
        self.selected = True

    @final
    def unSelect(self):
        """
        Unselect this element
        """
        self.selected = False

    @final
    def isSelected(self) -> bool:
        """
        Check if element is selected
        """
        return self.selected

    @abc.abstractmethod
    def draw(self, view, screen: pygame.Surface):
        """
        Draw element on screen
        Parameters:
            view -> View which is rendering this element
            screen -> Screen where element is rendered 
        """
        pass

    @abc.abstractmethod
    def processEvent(self, view, event):
        """
        Process event from view
        Parameters:
            view -> View which is sending event
            event -> Pygame event
        """
        pass

    @abc.abstractmethod
    def update(self, view):
        """
        Update element
        Parameters:
            view -> View which is updating this element
        """
        pass


class Container(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getChilds(self) -> list:
        """
        Get child elements of Container object
        """
        pass
