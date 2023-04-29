"""
Simple library for multiple views game aplication with pygame

File:       combobox.py
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

from SimpleApp.gui.button import Button
from SimpleApp.gui.listpanel import ListPanel
import pygame
from ..utils import *
from ..colors import *
from ..guielement import *


class ComboBox(GUIElement, Container):

    def __init__(self, view, style: dict, values: list, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create button
        Parameters:
            view -> View where is element
            style -> More about style for this element in config/styles.json
            values - List with texts
            width -> Width of Button
            height -> Height of Button
            x -> X position
            y -> Y position
        """
        self.button = None
        self.listpanel = None
        super().__init__(view, x, y, width, height, style)
        self.values = values
        self.callback = None
        self.hover = False
        self.selected_item = values[0]
        # popup panel
        self.listpanel = ListPanel(
            view, super().getStyle()["listpanel"], values)
        self.listpanel.setVisibility(False)
        self.listpanel.setItemClickEvet(lambda p: self.setSelectedItem(p))
        # button for open/close popup panel
        self.button = Button(view, super().getStyle()["button"], "↓")
        self.button.setClickEvt(
            lambda x: self.setPopupPanelVisibility(not self.listpanel.isVisible()))
        # font
        self.font = pygame.font.SysFont(
            super().getStyle()["font_name"], super().getStyle()["font_size"], bold=super().getStyle()["font_bold"])
        super().updateViewRect()

    @overrides(GUIElement)
    def updateViewRect(self):
        super().updateViewRect()
        if self.button is not None:
            self.button.setWidth(super().getHeight())
            self.button.setHeight(super().getHeight())
            self.button.setX(super().getX() +
                             super().getWidth() - self.button.getWidth())
            self.button.setY(super().getY())
        if self.listpanel is not None:
            self.listpanel.setWidth(super().getWidth())
            self.listpanel.setX(super().getX())
            self.listpanel.setY(super().getY() + super().getHeight())

    def setPopupPanelVisibility(self, visibility):
        """
        Set visibility of popup panel
        Parameters:
            visibility -> Visibility of panel
        """
        self.listpanel.setVisibility(visibility)
        if self.listpanel.isVisible():
            self.getView().setFilter_processOnly(self)
            self.button.setText("↑")
        else:
            self.button.setText("↓")
            self.getView().clearFilter()

    def setValues(self, values: list):
        """
        Set text of Button
        Parameters:
            text -> New text of Button 
        """
        self.values = values

    def getValues(self) -> list:
        """
        Get text of Button
        """
        return self.values

    def getSelectedItem(self) -> str:
        """
        Get selected item
        """
        return self.selected_item

    def setSelectedItem(self, item_name: str):
        """
        Set selected item
        Parameters:
            item_name -> Selected item name
        """
        self.selected_item = item_name
        self.setPopupPanelVisibility(False)
        if self.callback is not None:
            self.callback(self.selected_item)

    def setValueChangeEvt(self, callback):
        """
        Set on value change event callback
        Parameters:
            callback -> callback function
        """
        self.callback = callback

    @overrides(GUIElement)
    def draw(self, view, screen):
        # background
        if self.isSelected():
            c = super().getStyle()["background_color"]
            pygame.draw.rect(screen, colorChange(
                c, -0.2 if c[0] > 128 else 0.6), super().getViewRect(), border_radius=10)
        else:
            pygame.draw.rect(screen, super().getStyle()[
                             "background_color"], super().getViewRect(), border_radius=10)
        # button text
        if len(self.values[0]) != 0:
            screen.set_clip(super().getViewRect())
            text = self.font.render(
                self.selected_item,
                1,
                super().getStyle()["foreground_color"]
            )
            screen.blit(text, (super().getX() + (super().getWidth() - text.get_width())/2,
                               super().getY() + (super().getHeight() - text.get_height())/2))
            screen.set_clip(None)
        # outline
        pygame.draw.rect(screen, super().getStyle()[
            "outline_color"], super().getViewRect(), 2, border_radius=10)
        # button
        self.button.draw(view, screen)
        # draw panel
        if self.listpanel.isVisible():
            self.getView().getApp().drawLater(1000, self.listpanel.draw)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        self.button.processEvent(view, event)
        if self.listpanel.isVisible():
            self.listpanel.processEvent(view, event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if not inRect(
                event.pos[0],
                event.pos[1],
                pygame.Rect(
                    super().getX(),
                    super().getY(),
                    super().getWidth(),
                    super().getHeight() + self.listpanel.getHeight() + 5
                )
            ):
                self.setPopupPanelVisibility(False)

    @overrides(GUIElement)
    def update(self, view):
        pass

    @overrides(Container)
    def getChilds(self):
        return [self.button]
