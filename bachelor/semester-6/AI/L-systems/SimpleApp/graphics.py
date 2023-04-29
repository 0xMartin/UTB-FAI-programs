"""
Simple library for multiple views game aplication with pygame

File:       graphics.py
Date:       17.02.2022

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
import math


class Vertex:
    def __init__(self, coordinates: list):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]

    def setX(self, x: float):
        self.x = x

    def setY(self, y: float):
        self.y = y

    def setZ(self, z: float):
        self.z = z

    def getX(self) -> float:
        return self.x

    def getY(self) -> float:
        return self.y

    def getZ(self) -> float:
        return self.z

    def rotateX(self, center, angle):
        y = self.y - center.y
        z = self.z - center.z
        d = math.hypot(y, z)
        theta = math.atan2(y, z) + angle
        self.z = center.z + d * math.cos(theta)
        self.y = center.y + d * math.sin(theta)

    def rotateY(self, center, angle):
        x = self.x - center.x
        z = self.z - center.z
        d = math.hypot(x, z)
        theta = math.atan2(x, z) + angle
        self.z = center.z + d * math.cos(theta)
        self.x = center.x + d * math.sin(theta)

    def rotateZ(self, center, angle):
        x = self.x - center.x
        y = self.y - center.y
        d = math.hypot(y, x)
        theta = math.atan2(y, x) + angle
        self.x = center.x + d * math.cos(theta)
        self.y = center.y + d * math.sin(theta)


class Edge:
    def __init__(self, start: Vertex, end: Vertex):
        self.start = start
        self.end = end

    def setStart(self, start: Vertex):
        self.start = start

    def setEnd(self, end: Vertex):
        self.end = end

    def getStart(self) -> Vertex:
        return self.start

    def getEnd(self) -> Vertex:
        return self.end


class Wireframe:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.vertexColor = (255, 255, 255)
        self.edgeColor = (200, 200, 200)
        self.vertexSize = 3

    def setVertexColor(self, color):
        self.vertexColor = color

    def setEdgeColor(self, color):
        self.edgeColor = color

    def getVertexCount(self) -> int:
        return len(self.vertices)

    def getEdgeCount(self) -> int:
        return len(self.edges)

    def addVertex(self, coordinates: list):
        self.vertices.append(Vertex(coordinates))

    def addVertices(self, vertexList: list):
        for v in vertexList:
            self.vertices.append(Vertex(v))

    def addEdge(self, start_index: int, end_index: int):
        if start_index >= 0 and end_index >= 0 and start_index < len(self.vertices) and end_index < len(self.vertices):
            self.edges.append(
                Edge(self.vertices[start_index], self.vertices[end_index]))

    def addEdges(self, edgeList: list):
        for (start, end) in edgeList:
            self.edges.append(Edge(self.vertices[start], self.vertices[end]))

    def draw(self, screen, offset, displayVertex=True, displayEdge=True):
        if displayEdge:
            for edge in self.edges:
                pygame.draw.line(
                    screen,
                    self.edgeColor,
                    (edge.start.x + offset[0], edge.start.y + offset[1]),
                    (edge.end.x + offset[0], edge.end.y + offset[1]),
                    2)
        if displayVertex:
            for vertex in self.vertices:
                pygame.draw.circle(
                    screen,
                    self.vertexColor,
                    (int(vertex.x + offset[0]), int(vertex.y + offset[1])),
                    self.vertexSize, 0)

    def translate(self, axis, d):
        if axis in ['x', 'y', 'z']:
            for vertex in self.vertices:
                setattr(vertex, axis, getattr(vertex, axis) + d)

    def scale(self, center, scale):
        for vertex in self.vertices:
            vertex.x = center[0] + scale * (vertex.x - center[0])
            vertex.y = center[1] + scale * (vertex.y - center[1])
            vertex.z *= scale

    def computeCenter(self):
        cnt = len(self.vertices)
        avg_x = sum([v.x for v in self.vertices]) / cnt
        avg_y = sum([v.y for v in self.vertices]) / cnt
        avg_z = sum([v.z for v in self.vertices]) / cnt
        return (avg_x, avg_y, avg_z)

    def rotateX(self, center, angle):
        for vertex in self.vertices:
            y = vertex.y - center[1]
            z = vertex.z - center[2]
            d = math.hypot(y, z)
            theta = math.atan2(y, z) + angle
            vertex.z = center[2] + d * math.cos(theta)
            vertex.y = center[1] + d * math.sin(theta)

    def rotateY(self, center, angle):
        for vertex in self.vertices:
            x = vertex.x - center[0]
            z = vertex.z - center[2]
            d = math.hypot(x, z)
            theta = math.atan2(x, z) + angle
            vertex.z = center[2] + d * math.cos(theta)
            vertex.x = center[0] + d * math.sin(theta)

    def rotateZ(self, center, angle):
        for vertex in self.vertices:
            x = vertex.x - center[0]
            y = vertex.y - center[1]
            d = math.hypot(y, x)
            theta = math.atan2(y, x) + angle
            vertex.x = center[0] + d * math.cos(theta)
            vertex.y = center[1] + d * math.sin(theta)
