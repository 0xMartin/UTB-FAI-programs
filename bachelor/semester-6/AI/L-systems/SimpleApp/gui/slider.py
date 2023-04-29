"""
Simple library for multiple views game aplication with pygame

File:       slider.py
Date:       10.02.2022

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
from SimpleApp.gui import Label


class Slider(GUIElement):
    def __init__(self, view, style: dict, number: float, min: float, max: float, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create Slider
        Parameters:
            view -> View where is element
            style -> More about style for this element in config/styles.json
            number -> Number of Slider
            width -> Width of Slider
            height -> Height of Slider
            x -> X position
            y -> Y position
        """
        self.label = None
        super().__init__(view, x, y, width, height, style, pygame.SYSTEM_CURSOR_SIZEWE)
        self.label = Label(view, super().getStyle()["label"], " ", False, True)
        self.callback = None
        self.format = "@"
        self.min = min
        self.max = max
        self.setNumber(number)

    def setMin(self, val: int):
        """
        Set minimum value of slider
        Parameters:
            val -> new value
        """
        self.min = val

    def setMax(self, val: int):
        """
        Set maximum value of slider
        Parameters:
            val -> new value
        """
        self.max = val

    def setOnValueChange(self, callback):
        """
        Set on value change event callback
        Parameters:
            callback -> callback function
        """
        self.callback = callback

    def getValue(self) -> int:
        """
        Get % value of slider
        """
        dot_radius = super().getHeight() / 2
        return (self.position - dot_radius) / (super().getWidth() - dot_radius * 2) * 100

    def getNumber(self) -> int:
        """
        Get current number (min <-> max) of slider 
        """
        return self.getValue() / 100.0 * (self.max - self.min) + self.min

    def setValue(self, value: int):
        """
        Set % value 
        Parameters:
            value -> Value of slider 0 - 100    
        """
        if value is None:
            value = self.last_set_value
        if value is None:
            return
        if value < 0 or value > 100:
            return
        self.last_set_value = value
        dot_radius = super().getHeight() / 2
        # set position
        self.position = dot_radius + value / 100.0 * \
            (super().getWidth() - dot_radius * 2)

    def setNumber(self, value: int):
        """
        Set number value
        Parameters:
            value -> from range min<->max 
        """
        if value <= self.max and value >= self.min:
            value = (value - self.min) / (self.max - self.min) * 100
            self.setValue(value)

    def setLabelFormat(self, format: str):
        """
        Set label format
        Parameters:
            format -> string, symbol '#' replace by % value and '@' replace by numerical value (min <-> max)
        """
        self.format = format

    def refreshLabel(self):
        """
        Refresh slider value label
        """
        if len(self.format) != 0:
            txt = self.format
            txt = txt.replace("#", '%.2f' % self.getValue())
            txt = txt.replace("@", '%.2f' % self.getNumber())
            self.label.setText(txt)

    @overrides(GUIElement)
    def updateViewRect(self):
        super().updateViewRect()
        if self.label is not None:
            self.label.setX(super().getX() + super().getWidth() + 20)
            self.label.setY(super().getY() + super().getHeight() / 2)

    @overrides(GUIElement)
    def setWidth(self, width):
        super().setWidth(width)
        self.setValue(None)
        self.refreshLabel()

    @overrides(GUIElement)
    def setHeight(self, height):
        super().setHeight(height)
        self.setValue(None)
        self.refreshLabel()

    @overrides(GUIElement)
    def draw(self, view, screen):
        # background
        pygame.draw.rect(screen, super().getStyle()[
                         "background_color"], super().getViewRect(), border_radius=10)
        # slider bar
        pygame.draw.rect(
            screen,
            colorChange(super().getStyle()["foreground_color"], 0.8),
            pygame.Rect(
                super().getX(),
                super().getY(),
                self.position,
                super().getHeight()
            ),
            border_radius=10
        )
        # outline
        pygame.draw.rect(screen, super().getStyle()[
                         "outline_color"], super().getViewRect(), 2, border_radius=10)
        # slider
        pygame.draw.circle(
            screen,
            super().getStyle()["foreground_color"],
            (super().getX() + self.position,
             super().getY() + super().getHeight() / 2),
            super().getHeight() * 0.8
        )
        # label with current value
        self.label.draw(view, screen)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if math.dist(
                (event.pos[0], event.pos[1]),
                (super().getX() + self.position,
                 super().getY() + super().getHeight() / 2)
            ) <= super().getHeight() * 0.8:
                super().select()
                self.def_position = self.position
                self.drag_start = event.pos[0]
        elif event.type == pygame.MOUSEBUTTONUP:
            super().unSelect()
            self.setValue(self.getValue())
        elif event.type == pygame.MOUSEMOTION:
            if super().isSelected():
                self.position = self.def_position + \
                    (event.pos[0] - self.drag_start)
                dot_radius = super().getHeight() / 2
                self.position = min(
                    max(dot_radius, self.position), super().getWidth() - dot_radius)
                self.refreshLabel()
                if self.callback is not None:
                    self.callback(self.getNumber())

    @overrides(GUIElement)
    def update(self, view):
        pass
