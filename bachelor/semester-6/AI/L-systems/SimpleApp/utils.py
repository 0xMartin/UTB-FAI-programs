"""
Simple library for multiple views game aplication with pygame

File:       utils.py
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

import os.path
import json
import math
import copy
import string
import threading
import numpy as np
from time import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pylab


def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


def inRect(x: int, y: int, rect: pygame.Rect) -> bool:
    """
    Check if x, y is in rect
    Parameters:
        x -> X position
        y -> Y position
        rect -> rectangle {left, top, width, height}
    """
    if x >= rect.left and y >= rect.top and x <= rect.left + rect.width and y <= rect.top + rect.height:
        return True
    else:
        return False


def generateSignal(ms_periode: int) -> bool:
    """
    Genereta pariodic signal -> (ms_periode) True -> (ms_periode) False -> ...
    Parameters:
        ms_periode -> Half periode of signal in ms
    """
    return round((time() * 1000) / ms_periode) % 2 == 0


def loadImage(img_path: str) -> pygame.Surface:
    """
    Load image from File system
    Parameters:
        img_path -> Path of image
    """
    if os.path.isfile(img_path):
        return pygame.image.load(img_path)
    else:
        return None

def drawGraph(fig: matplotlib.figure, dark: bool = False):
    """
    Draw graph and print it to the image
    Parameters:
        width -> Width of graph
        height -> Height of graph
        fig -> Data of graph
        dark -> True = dark mode
    """

    matplotlib.use("Agg")
    if dark == "dark":
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    plt.close()

    return pygame.image.frombuffer(raw_data, canvas.get_width_height(), "RGBA")


def loadConfig(path: str) -> str:
    """
    Load config
    Parameters:
        path -> path to the file
    """
    if not os.path.isfile(path):
        return None
    f = open(path)
    data = json.load(f)
    f.close()
    return data


def getDisplayWidth() -> int:
    """
    Get width of display
    """
    return pygame.display.get_surface().get_size().get_width()

def getDisplayHeight() -> int:
    """
    Get height of display
    """
    return pygame.display.get_surface().get_size().get_height()

def runTaskAsync(task):
    task = threading.Thread(target=task, args=(1,))
    task.start()
