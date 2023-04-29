"""
Simple library for multiple views game aplication with pygame

File:       togglebutton.py
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
from SimpleApp.gui.label import Label


class ToggleButton(GUIElement):

    def __init__(self, view, style: dict, text: str, status: bool = False, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create ToggleButton
        Parameters:
            view -> View where is element
            style -> More about style for this element in config/styles.json
            text -> Text of ToggleButton
            status -> True/False
            width -> Width of ToggleButton
            height -> Height of ToggleButton
            x -> X position
            y -> Y position
        """
        super().__init__(view, x, y, width, height, style)
        self.label = Label(view, super().getStyle()[
                           "label"], text, False, True)
        self.callback = None
        self.hover = False
        self.status = status

    def setText(self, text: str):
        """
        Set text of label
        Parameters:
            text -> New text
        """
        if self.label is not None:
            self.label.setText(text)

    def getStatus(self) -> bool:
        """
        Get status of toggle button (True/False)
        """
        return self.status

    def getLabel(self) -> Label:
        """
        Get label
        """
        return self.label

    def setValueChangedEvt(self, callback):
        """
        Set value changed event callback
        Parameters:
            callback -> callback function
        """
        self.callback = callback

    @overrides(GUIElement)
    def draw(self, view, screen):
        # bg and outline
        if self.status:
            bg_color = colorChange(super().getStyle()["foreground_color"], 0.8)
        else:
            bg_color = super().getStyle()["background_color"]
        pygame.draw.rect(
            screen,
            bg_color,
            super().getViewRect(),
            border_radius=int(super().getHeight() / 2)
        )
        pygame.draw.rect(
            screen,
            super().getStyle()["outline_color"],
            super().getViewRect(),
            2,
            border_radius=int(super().getHeight() / 2)
        )
        # toggle switch
        if self.status:
            pos = super().getWidth() - super().getHeight() / 2
            pygame.draw.circle(
                screen,
                super().getStyle()["foreground_color"],
                (super().getX() + pos, super().getY() + super().getHeight() / 2),
                super().getHeight() / 2
            )
        else:
            pygame.draw.circle(
                screen,
                super().getStyle()["foreground_color"],
                (super().getX() + super().getHeight() / 2,
                 super().getY() + super().getHeight() / 2),
                super().getHeight() / 2
            )
        # lable
        if self.label is not None:
            self.label.setX(super().getX() + super().getWidth() + 5)
            self.label.setY(super().getY() + super().getHeight() / 2)
            self.label.draw(view, screen)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                self.status = not self.status
                if self.callback is not None:
                    self.callback(self.status)
        elif event.type == pygame.MOUSEMOTION:
            if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                self.select()
            else:
                self.unSelect()

    @overrides(GUIElement)
    def update(self, view):
        pass
