"""
Simple library for multiple views game aplication with pygame

File:       radiobutton.py
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


class RadioButton(GUIElement):
    def __init__(self, view, style: dict, text: str, group, size=20, x: int = 0, y: int = 0):
        """
        Create RadioButton element 
        Parameters:
            view -> View where is element
            style -> More about style for this element in config/styles.json
            text -> Text of RadioButton
            size -> Size of radio button (circe diameter)
            x -> X position
            y -> Y position
        """
        super().__init__(view, x, y, size, size, style)
        self.label = Label(view, super().getStyle()["label"], text, False, True, x, y)
        self.group = group
        group.addRadioButton(self)
        self.checked = False
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
        Set radiobutton Checked event
        Parameters:
            callback -> callback function
        """
        self.callback = callback

    def setChecked(self, checked: bool):
        """
        Set checked state of this radio button
        Parameters:
            checked -> True = Is checked    
        """
        self.checked = checked

    def isChecked(self) -> bool:
        """
        Return if is checked
        """
        return self.checked

    @overrides(GUIElement)
    def draw(self, view, screen):
        # lable
        if self.label is not None:
            self.label.setX(super().getX() + super().getWidth() + 5)
            self.label.setY(super().getY() + super().getHeight() / 2)
            self.label.draw(view, screen)
        # radio box
        center = (
            super().getX() + super().getWidth()/2,
            super().getY() + super().getWidth()/2
        )
        if super().isSelected():
            c = super().getStyle()["background_color"]
            pygame.draw.circle(screen, colorChange(c, -0.2 if c[0] > 128 else 0.6), center, super().getWidth() / 2)
        else:
            pygame.draw.circle(screen, super().getStyle()[
                "background_color"], center, super().getWidth() / 2)
        pygame.draw.circle(screen, super().getStyle()[
            "outline_color"], center, super().getWidth() / 2, 2)
        # check
        if self.checked:
            pygame.draw.circle(screen, super().getStyle()[
                "foreground_color"], center, super().getWidth() / 4)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                if self.callback is not None:
                    self.callback(self)
                self.group.checkRadioButton(self)
        elif event.type == pygame.MOUSEMOTION:
            if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                super().select()
            else:
                super().unSelect()

    @overrides(GUIElement)
    def update(self, view):
        pass


class RadioButtonGroup:
    def __init__(self, radiobtns: list):
        """
        Create RadioButton group
        Parameters:
            radiobtns -> list with radio buttons
        """
        self.radiobtns = []
        for r in radiobtns:
            if isinstance(r, RadioButton):
                self.radiobtns.append(r)

    def addRadioButton(self, radiobtn: RadioButton):
        """
        Add radio button to this group
        Parameters:
            radiobtn -> Combo box    
        """
        if isinstance(radiobtn, RadioButton):
            self.radiobtns.append(radiobtn)

    def removeRadioButton(self, radiobtn: RadioButton):
        """
        Remove radio button from this group
        Parameters:
            radiobtn -> Combo box    
        """
        self.radiobtns.remove(radiobtn)

    def getRadioButton(self):
        """
        Return checked radio button from group
        """
        for r in self.radiobtns:
            if r.isChecked():
                return r

    def checkRadioButton(self, radiobtn: RadioButton):
        """
        Check one radio button from this group
        Parameters:
            radiobtn -> combo box    
        """
        if isinstance(radiobtn, RadioButton):
            for r in self.radiobtns:
                if r != radiobtn:
                    r.setChecked(False)
                else:
                    r.setChecked(True)
