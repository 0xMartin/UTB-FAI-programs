"""
Simple library for multiple views game aplication with pygame

File:       textinput.py
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
import re
from ..utils import *
from ..colors import *
from ..guielement import *


class TextInput(GUIElement):
    def __init__(self, view, style: dict, text: str, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create TextInput element 
        Parameters:
            view -> View where is element
            style -> more about style for this element in config/styles.json
            text -> Text of TextInput
            width -> Width of TextInput
            height -> Height of TextInput
            x -> X position
            y -> Y position
        """
        super().__init__(view, x, y, width, height, style, pygame.SYSTEM_CURSOR_IBEAM)
        self.callback = None
        self.filter_pattern = None
        self.text = text
        self.caret_position = 0
        self.font = pygame.font.SysFont(
            super().getStyle()["font_name"], super().getStyle()["font_size"], bold=super().getStyle()["font_bold"])

    def setText(self, text: str):
        """
        Set text of TextInput
        Parameters:
            text -> New text
        """
        self.text = text

    def getText(self):
        """
        Get text of TextInput
        """
        return self.text

    def setTextChangedEvt(self, callback):
        """
        Set text changed event
        Parameters:
            callback -> Event callback    
        """
        self.callback = callback

    def setFilterPattern(self, pattern: str):
        """
        Set filter pattern
        Parameters:
            pattern -> pattern for text in this text input
        """
        # "^([A-Z][0-9]+)+$"
        self.filter_pattern = re.compile(pattern)

    @overrides(GUIElement)
    def draw(self, view, screen):
        # background
        if super().isSelected():
            c = super().getStyle()["background_color"]
            pygame.draw.rect(screen, colorChange(
                c, 0.4 if c[0] > 128 else 0.7), super().getViewRect(), border_radius=5)
        else:
            pygame.draw.rect(screen, super().getStyle()[
                             "background_color"], super().getViewRect(), border_radius=5)

        # create subsurface
        surface = screen.subsurface(super().getViewRect())
        text_offset = 0
        caret_offset = 0
        if len(self.text) != 0:
            text = self.font.render(
                self.text,
                1,
                super().getStyle()["foreground_color"]
            )
            # calculate carret offset
            caret_offset = self.font.size(self.text[0: self.caret_position])[0]
            # offset for text
            text_offset = max(caret_offset + 20 - super().getWidth(), 0)
            if not super().isSelected():
                text_offset = 0
            # draw text
            surface.blit(
                text, (5 - text_offset, (super().getHeight() - text.get_height())/2))

        # caret
        if super().isSelected() and generateSignal(400):
            # caret position
            x = 5 - text_offset + caret_offset
            y = surface.get_height() * 0.2
            pygame.draw.line(surface, super().getStyle()[
                             "foreground_color"], (x, y), (x, surface.get_height() - y), 2)

        # outline
        pygame.draw.rect(screen, super().getStyle()[
                         "outline_color"], super().getViewRect(), 2, border_radius=5)

    @overrides(GUIElement)
    def processEvent(self, view, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # select textinput
            if inRect(event.pos[0], event.pos[1], super().getViewRect()):
                super().select()
                self.caret_position = len(self.text)
            else:
                self.unselectTI()
        elif event.type == pygame.KEYDOWN:
            # text writing
            if super().isSelected():
                if event.key == pygame.K_RETURN:
                    self.unselectTI()
                elif event.key == pygame.K_BACKSPACE:
                    # delate last char
                    i = self.caret_position
                    if i >= len(self.text):
                        self.text = self.text[:i-1]
                        self.caret_position = max(0, self.caret_position - 1)
                    elif i != 0:
                        self.text = self.text[:i-1] + self.text[i:]
                        self.caret_position = max(0, self.caret_position - 1)

                elif event.key == pygame.K_LEFT:
                    self.caret_position = max(0, self.caret_position - 1)
                elif event.key == pygame.K_RIGHT:
                    self.caret_position = min(
                        len(self.text), self.caret_position + 1)
                else:
                    # new char
                    if event.unicode in string.printable and event.unicode != '':
                        # add char to text buffer
                        i = self.caret_position
                        if i < len(self.text):
                            self.text = self.text[:i] + \
                                event.unicode + self.text[i:]
                        elif i == 0:
                            self.text = event.unicode + self.text
                        else:
                            self.text += event.unicode
                        # increment caret position
                        self.caret_position += 1

    @overrides(GUIElement)
    def update(self, view):
        pass

    def unselectTI(self):
        # call event
        if super().isSelected():
            # text filter
            if self.filter_pattern is not None:
                if not self.filter_pattern.match(self.text):
                    # delate text
                    self.text = ""
            if self.callback is not None:
                self.callback(self.text)
        # unselectd TI
        super().unSelect()
