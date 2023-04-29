"""
Simple library for multiple views game aplication with pygame

File:       checkbox.py
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
from SimpleApp.gui.label import Label


class CheckBox(GUIElement):
    def __init__(self, view, style: dict, text: str, checked: bool, size: int = 20, x: int = 0, y: int = 0):
        """
        Create CheckBox element 
        Parameters:
            view -> View where is element
            style -> More about style for this element in config/styles.json
            text -> Text of TextInput
            checked -> Is checked?
            width -> Width of CheckBox
            height -> Height of CheckBox
            x -> X position
            y -> Y position
        """
        super().__init__(view, x, y, size, size, style)
        self.label = Label(view, super().getStyle()[
                           "label"], text, False, True, x, y)
        self.checked = checked
        self.callback = None

    def setText(self, text: str):
        """
        Set text of label
        Parameters:
            text -> New text
        """
        if self.label is not None:
            self.label.setText(text)

    def getLabel(self) -> Label:
        """
        Get label
        """
        return self.label

    def setCheckedEvt(self, callback):
        """
        Set checkbox Checked event
        Parameters:
            callback -> callback function
        """
        self.callback = callback

    def setChecked(self, checked: bool):
        """
        Set checked state of this check box
        Parameters:
            checked -> True = Is checked    
        """
        self.checked = checked

    def isChecked(self) -> Label:
        """
        Return if this check box is checked
        """
        return self.checked

    @overrides(GUIElement)
    def draw(self, view, screen):
        # lable
        if self.label is not None:
            self.label.setX(super().getX() + super().getWidth() + 5)
            self.label.setY(super().getY() + super().getHeight() / 2)
            self.label.draw(view, screen)
        # check box
        if super().isSelected():
            c = super().getStyle()["background_color"]
            pygame.draw.rect(screen, colorChange(
                c, -0.2 if c[0] > 128 else 0.6), super().getViewRect(), border_radius=6)
        else:
            pygame.draw.rect(screen, super().getStyle()[
                "background_color"], super().getViewRect(), border_radius=5)
        pygame.draw.rect(screen, super().getStyle()[
            "outline_color"], super().getViewRect(), 2, border_radius=5)
        # check
        if self.checked:
            pts = [
                (super().getX() + super().getWidth() * 0.2,
                 super().getY() + super().getWidth() * 0.5),
                (super().getX() + super().getWidth() * 0.4,
                 super().getY() + super().getWidth() * 0.75),
                (super().getX() + super().getWidth() * 0.8,
                 super().getY() + super().getWidth() * 0.2)
            ]
            pygame.draw.lines(screen, super().getStyle()
                              ["foreground_color"], False, pts, round(7 * super().getWidth() / 40))

    @overrides(GUIElement)
    def processEvent(self, view, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                if self.callback is not None:
                    self.callback(self)
                self.checked = not self.checked
        elif event.type == pygame.MOUSEMOTION:
            if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                super().select()
            else:
                super().unSelect()

    @overrides(GUIElement)
    def update(self, view):
        pass
