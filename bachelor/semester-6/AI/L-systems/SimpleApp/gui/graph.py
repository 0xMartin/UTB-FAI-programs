"""
Simple library for multiple views game aplication with pygame

File:       graph.py
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

import pygame
from ..utils import *
from ..colors import *
from ..guielement import *


class Graph(GUIElement):
    def __init__(self, view, style: dict, width: int = 0, height: int = 0, x: int = 0, y: int = 0):
        """
        Create Graph element
        Parameters:
            view -> View where is element
            width -> Width of Graph
            height -> Height of Graph
            x -> X position
            y -> Y position
        """
        super().__init__(view, x, y, width, height, style)
        self.graph = None
        self.fig_builder = None

    def setFigureBuilderFunc(self, func):
        """
        Set figure builder function
        Parameters:
            func -> builder function -> def __name__(fig) : fig - matplotlib.figure from graph
                * in builder function using matplotlib make your own graph
        """
        self.fig_builder = func

    @overrides(GUIElement)
    def setWidth(self, width):
        super().setWidth(width)
        self.refreshGraph()

    @overrides(GUIElement)
    def setHeight(self, height):
        super().setHeight(height)
        self.refreshGraph()

    def refreshGraph(self):
        if self.fig_builder is not None and super().getWidth() > 50 and super().getHeight() > 50:
            fig = pylab.figure(
                figsize=[self.getWidth()/100, self.getHeight()/100], dpi=100)
            fig.patch.set_alpha(0.0)
            self.fig_builder(fig)
            self.graph = drawGraph(
                fig,
                super().getStyle()["theme"]
            )

    @overrides(GUIElement)
    def draw(self, view, screen):
        if self.graph is not None:
            screen.blit(pygame.transform.scale(self.graph, (super().getWidth(
            ), super().getHeight())), (super().getX(), super().getY()))

    @overrides(GUIElement)
    def processEvent(self, view, event):
        pass

    @overrides(GUIElement)
    def update(self, view):
        pass

    @staticmethod
    def builderFunc_lineGraph(fig, x_label, y_label, values, legend=None):
        """
        Builder function for Graph: Plot line graph
        Parameters:
            x_label -> Label for X axis
            y_label -> Label for Y axis
            values -> List with collections for each line of graph [[0, ...], ...]
            legend -> Legend of graph: List of strings ['str', ...]
        Example: 
        Graph.builderFunc_lineGraph(
            f,
            "X axis",
            "Y axis",
            [[1, 2, 3, 4], [2, 4, 10, 8], [3, 7, 17, 12]],
            ['A', 'B', 'C']
        )
        """
        ax = fig.gca()
        for i, line in enumerate(values):
            line, = ax.plot(line)
            if i < len(legend):
                line.set_label(legend[i])
                ax.legend()

    @staticmethod
    def builderFunc_pieGraph(fig: matplotlib.figure, labels: list, values: list, explode: list = None):
        """
        Builder function for Graph: Plot pie graph
        Parameters:
            labels -> Labels for parts of pie graph (list of strings) ['str', ...]
            values -> Values for parts of pie graph (list of numbers) [1, ...]
            explode -> Offsets from center of pie graph for each part (list of numbers) [0.1, ...]
        Example:
        Graph.builderFunc_pieGraph(
            f,
            ['A', 'B', 'C', 'D'],
            [1, 2, 3, 5],
            (0, 0.2, 0, 0)
        )
        """
        ax = fig.gca()
        ax.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.axis('equal')

    @staticmethod
    def builderFunc_dotGraph(fig: matplotlib.figure, x_label: str, y_label: str, values: list, legend: list = None):
        """
        Builder function for Graph: Dot graph
        Parameters:
            x_label -> Label for X axis
            y_label -> Label for Y axis
            values -> List with collections for each line of graph [[0, ...], ...]
            legend -> Legend of graph: List of strings ['str', ...]
        Example: 
        Graph.builderFunc_dotGraph(
            f,
            "X axis",
            "Y axis",
            [[1, 2, 3, 4], [2, 4, 10, 8], [3, 7, 17, 12]],
            ['A', 'B', 'C']
        )
        """
        ax = fig.gca()
        for i, line in enumerate(values):
            line, = ax.plot(line, '.')
            if i < len(legend):
                line.set_label(legend[i])
                ax.legend()

    @staticmethod
    def builderFunc_barGraph(fig: matplotlib.figure, labels: list, values: list):
        """
        Builder function for Graph: Bar graph
        Parameters:
            labels -> Labels for bars ['str', ...]
            values -> Values for each bar [1, ...]
        Example: 
        Graph.builderFunc_barGraph(
            f,
            ['A', 'B', 'C', 'D'],
            [2, 4, 10, 8]
        )
        """
        ax = fig.gca()
        ax.bar(labels, values, width=1, edgecolor="white", linewidth=0.7)

    @staticmethod
    def builderFunc_scatterGraph(fig: matplotlib.figure, values: list, xlim: tuple, ylim: tuple):
        """
        Builder function for Graph: Bar graph
        Parameters:
            values -> List with values for each point [(1, 2), ...]
            xlim -> Limit for x axis
            ylim -> Limit for y axis
        Example: 
        Graph.builderFunc_barGraph(
            f,
            [(1, 2), (4, 5), (4, 7), (6, 1), (4, 3)],
            (0, 8),
            (0, 8)
        )
        """
        ax = fig.gca()
        x = []
        y = []
        for pt in values:
            x.append(pt[0])
            y.append(pt[1])
        sizes = np.random.uniform(15, 80, len(x))
        colors = np.random.uniform(15, 80, len(x))
        ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
        ax.set(xlim=xlim, ylim=ylim)
