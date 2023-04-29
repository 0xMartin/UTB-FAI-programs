"""
Simple library for multiple views game aplication with pygame

File:       stylemanager.py
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

from .utils import *


class StyleManager:
    """
    Provides style for each GUI element. Loading and preserves all application styles.
    """

    def __init__(self, styles_path):
        """
        Create style manager
        Parameters:
            styles_path -> Path where is file with styles for all guil elements    
        """
        self.styles_path = styles_path

    def init(self):
        """
        Init style manager
        """
        self.loadStyleSheet(self.styles_path)

    def loadStyleSheet(self, styles_path):
        """
        Load stylesheet from file
        Parameters:
            styles_path -> Path where is file with styles for all guil elements   
        """
        self.styles = loadConfig(styles_path)

    def getStyleWithName(self, name) -> dict:
        """
        Get style with specific name from stylsheet 
        Parameters:
            name -> Name of style
        """
        if name not in self.styles.keys():
            return None
        else:
            return self.processStyle(self.styles[name])

    def processStyle(self, style) -> dict:
        """
        Some string values are replaced by an object if necessary
        Parameters:
            style -> Some style    
        """
        # colors
        new_style = style.copy()
        for tag in new_style.keys():
            if "color" in tag:
                rgb = new_style[tag].split(",")
                new_style[tag] = tuple([int(rgb[0]), int(rgb[1]), int(rgb[2])])
            elif isinstance(new_style[tag], dict):
                new_style[tag] = self.processStyle(new_style[tag])
        return new_style
