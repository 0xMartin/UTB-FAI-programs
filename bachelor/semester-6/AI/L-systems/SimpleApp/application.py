"""
Simple library for multiple views game aplication with pygame

File:       application.py
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
import abc
import threading
from typing import final

from .guielement import *
from .colors import *
from .utils import *
from .stylemanager import *


# Application class, provides work with views
class Application:
    """
    The main component of the framework. Provides switched view work
    with them (rendering, events, updates, ...). Here I set the 
    styles, icon, application name, window size, ...
    """

    def __init__(self, views, fps=60, ups=60, dark=False):
        """
        Create Application
        Parameters:
            views -> List with views
            fps -> Frame per second
            ups -> Update per second
        """
        self.fps = fps
        self.ups = ups
        self.views = []
        self.draw_queue = []
        self.visible_view = None
        self.inited = False
        self.running = False
        if dark:
            self.stylemanager = StyleManager(
                "SimpleApp/config/styles_dark.json")
        else:
            self.stylemanager = StyleManager(
                "SimpleApp/config/styles_light.json")
        self.setFillColor(WHITE)
        for v in views:
            if isinstance(v, View):
                v.setApplication(self)
                self.views.append(v)

    def setFillColor(self, color: tuple):
        """
        Set default fill color for views of application
        Parameters:
            color -> Fill color
        """
        self.fill_color = color

    def addView(self, view) -> bool:
        """
        Add new view to application
        Parameters:
            view -> New view
        Returns False in case of fail
        """
        if(isinstance(view, View)):
            view.setApplication(self)
            self.views.append(view)
            # call create event (only if app is running)
            if self.run:
                view.createEvt_base(self.screen.get_width(),
                                    self.screen.get_height())
            return True
        else:
            return False

    def getStyleManager(self) -> StyleManager:
        """
        Get application style manager
        """
        return self.stylemanager

    def reloadStyleSheet(self, styles_path: str):
        """
        Reload stylesheet
        Parameters:
            styles_path -> Path where is file with styles for all GUI elements  
        """
        self.stylemanager.loadStyleSheet(styles_path)

    def reloadElementStyles(self):
        """
        Reload style of all elements (from all views of application)
        """
        fill_color = self.stylemanager.getStyleWithName("default")[
            "fill_color"]
        for view in self.views:
            view.setFillColor(fill_color)
            view.reloadElementStyle()

    def removeView(self, view) -> bool:
        """
        Remove view from application
        Parameters:
            view -> View to be removed
        Returns False in case of fail
        """
        if(isinstance(view, View)):
            self.views.remove(view)
            return True
        else:
            return False

    def getScreen(self) -> pygame.Surface:
        """
        Get screen of application
        """
        return self.screen

    def init(self, width: int, height: int, name: str, icon: str):
        """
        Init application
        Parameters:
            width -> Width of application window
            height -> Height of application window
            name -> Name of application (window title)
            icon -> Icon of application 
        """
        self.width = max(width, 50)
        self.height = max(height, 50)
        self.name = name
        self.icon = icon

        self.stylemanager.init()

        pygame.init()
        self.default_font = pygame.font.SysFont("Verdana", 35, bold=True)
        pygame.display.set_caption(name)
        img = loadImage(self.icon)
        if img is None:
            img = loadImage("./SimpleApp/src/icon.png")    
        if img is not None:
            pygame.display.set_icon(img)
        self.screen = pygame.display.set_mode((width, height))
        self.inited = True

    def render_loop(self, arg):
        """
        Render loop
        """
        clock = pygame.time.Clock()
        while self.running:
            # clear screen
            if self.visible_view is not None:
                if self.visible_view.getFillColor() is None:
                    self.screen.fill(self.fill_color)
                else:
                    self.screen.fill(self.visible_view.getFillColor())
            # visible view proccess render
            if self.visible_view is not None:
                self.visible_view.render(self.screen)
            # draw later
            if len(self.draw_queue) != 0:
                draw_elements = sorted(
                    self.draw_queue, reverse=True, key=lambda x: x["Z-INDEX"])
                for el in draw_elements:
                    el["CALLBACK"](self.visible_view, self.screen)
                self.draw_queue.clear()
            # update display
            pygame.display.flip()
            clock.tick(self.fps)

    def update_loop(self, arg):
        """
        Update loop
        """
        clock = pygame.time.Clock()
        while self.running:
            if self.visible_view is not None:
                self.visible_view.update()
            clock.tick(self.ups)

    def run(self, start_view=None) -> bool:
        """
        Run application loop
        Parameters:
            start_view -> View that will open first    
        Returns False in case of fail
        """
        if not self.inited:
            return False

        self.running = True

        # call start event for each view
        for view in self.views:
            view.createEvt_base(self.screen.get_width(),
                                self.screen.get_height())

        # threads for render and update loops
        render_thread = threading.Thread(target=self.render_loop, args=(1,))
        render_thread.start()
        update_thread = threading.Thread(target=self.update_loop, args=(1,))
        update_thread.start()

        if start_view is not None:
            self.showView(start_view)

        # event loop
        while self.running:
            # quit event
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                self.running = False
            # visible view proccess app events and render
            if self.visible_view is not None:
                self.visible_view.processEvt(event)

        # close
        pygame.quit()

        return True

    def close(self):
        """
        Close application
        """
        self.running = False
        # call close event for each view
        for view in self.views:
            view.closeEvt()
        self.views = []

    def showView(self, view) -> bool:
        """
        Show view of aplication
        Parameters:
            view -> View to be displayed in application
        Return True in success
        """
        if not self.running:
            return False

        if view in self.views:
            # hide current visible view
            if self.visible_view is not None:
                self.visible_view.setVisibility(False)
                view.hideEvt()
            # show new view
            view.setVisibility(True)
            view.openEvt_base(self.screen.get_width(),
                              self.screen.get_height())
            self.visible_view = view
            # change window title
            if len(view.name) == 0:
                pygame.display.set_caption(self.name)
            else:
                pygame.display.set_caption(self.name + " - " + view.name)
            return True
        else:
            return False

    def showViewWithName(self, name: str) -> bool:
        """
        Show view with specific name
        Parameters:
            name -> Name of view
        """
        for view in self.views:
            if view.name == name:
                return self.showView(view)

    def showViewWithID(self, id: int) -> bool:
        """
        Show view with specif ID
        Parameters:
            id -> ID of view
        """
        for view in self.views:
            if view.ID == id:
                return self.showView(view)

    def drawLater(self, z_index, draw_callback):
        """
        Add some function to queue for later drawing (after rendering of all elements)
        Parameters:
            z_index -> Z-INDEX of drawing (higher value means that it will be drawed over others with a lower value)
            draw_callback -> Draw callback (def draw(self, view, screen))
        """
        self.draw_queue.append({"Z-INDEX": z_index, "CALLBACK": draw_callback})


# View class
class View(metaclass=abc.ABCMeta):
    """
    View represents the content/page of the application window 
    that the user sees and with which he can interact.
    """

    def __init__(self, name: str, id: int):
        """
        Create view
        Parameters:
            name -> Name of view (the name will be visible in the window title)
            id -> ID of view (can be used for navigation)
        """
        self.name = name
        self.ID = id
        self.visible = False
        self.fill_color = None
        self.filter = None
        self.GUIElements = []
        self.layout_manager_list = []
        self.setDefaultCursor()

    def setID(self, id: int):
        """
        Set view ID
        Parameters:
            id -> New ID for view
        """
        self.ID = id

    def addGUIElements(self, elements):
        """
        Add GUI elements to this view
        Parameters:
            elements -> List with GUI elements
        """
        for el in elements:
            if isinstance(el, GUIElement):
                self.GUIElements.append(el)

    def removeGUIElement(self, element):
        """
        Remove GUI element from view
        Parameters:
            element -> GUI element to be removed
        """
        self.GUIElements.remove(element)

    @final
    def getApp(self):
        """
        Get reference on app
        """
        return self.app

    def registerLayoutManager(self, layoutManager) -> bool:
        """
        Register new layout manager
        Parameters:
            layoutManager -> New layout manager    
        """
        if isinstance(layoutManager, Layout):
            self.layout_manager_list.append(layoutManager)
            return True
        else:
            return False

    def unregisterLayoutManager(self, layoutManager):
        """
        Unregister layout manager
        Parameters:
            layoutManager -> Layout manager    
        """
        self.layout_manager_list.remove(layoutManager)

    @final
    def getGUIElements(self) -> list:
        """
        Get list of GUIElements
        """
        return self.GUIElements

    def setDefaultCursor(self, cursor=pygame.SYSTEM_CURSOR_ARROW):
        """
        Set default cursor for view
        Parameters:
            cursor -> Default cursor
        """
        self.cursor = cursor

    def setFillColor(self, color: tuple):
        """
        Set view fill color
        Parameters:
            color -> View fill color
        """
        self.fill_color = color

    @final
    def getFillColor(self) -> tuple:
        """
        Get view fill color
        """
        return self.fill_color

    def setVisibility(self, visible: bool):
        """
        Set visibility of view
        Parameters:
            visible -> True=view is visible
        """
        self.visible = visible

    def setApplication(self, app: Application):
        """
        Assigns an application to this view
        Parameters:
            app -> Application
        """
        if(isinstance(app, Application)):
            self.app = app
            return True
        else:
            return False

    def setFilter_processOnly(self, element):
        """
        Set process only filter -> call process only for specified GUI element
        Parameters:
            element -> GUI element which will be processed
        """
        if element is not None:
            self.filter = {"type": "process_only", "element": element}

    def clearFilter(self):
        """
        Clear filter
        """
        self.filter = None

    @final
    def reloadElementStyle(self, list=None):
        """
        Reload style of all GUI elements from list (do not set "list" if you want all view elements)
        Parameters:
            list -> List with GUI elements
        """
        if list is None:
            list = self.GUIElements
        for el in list:
            if el is None:
                continue
            style = self.app.getStyleManager().getStyleWithName(el.__class__.__name__)
            if style is not None:
                el.setStyle(style)
            if isinstance(el, Container):
                self.reloadElementStyle(el.getChilds())
        self.reloadStyleEvt()

    @final
    def createEvt_base(self, width: int, height: int):
        """
        Create event + layout update
        """
        # call abstract def
        self.createEvt()
        # update style
        if self.fill_color is None:
            self.fill_color = self.getApp().getStyleManager(
            ).getStyleWithName("default")["fill_color"]
        # update layout managers
        for lm in self.layout_manager_list:
            lm.updateLayout(width, height)

    @abc.abstractmethod
    def createEvt(self):
        """
        Create event - when the application starting
        """
        pass

    @abc.abstractmethod
    def closeEvt(self):
        """
        Close event - when the application closing
        """
        pass

    @final
    def openEvt_base(self, width: int, height: int):
        """
        Open event + layout update + unselect all
        """
        # update layout managers
        for lm in self.layout_manager_list:
            lm.updateLayout(width, height)
        # unselect all elements
        for el in self.GUIElements:
            el.unSelect()
        # call abstract def
        self.openEvt()

    @abc.abstractmethod
    def openEvt(self):
        """
        Open event - when the application show this view
        """
        pass

    @abc.abstractmethod
    def hideEvt(self):
        """
        Hide event - when the application hide this view
        """
        pass

    @abc.abstractmethod
    def reloadStyleEvt(self):
        """
        Reload style event - when the application reloading styles of view
        """
        pass

    @final
    def processEvt(self, event):
        """
        Process event from application
        Parameters:
            event -> Application event
        """
        # all events send to view elements
        if self.app is not None:
            if self.filter is None:
                for el in self.GUIElements:
                    el.processEvent(self, event)
            else:
                self.filter["element"].processEvent(self, event)
        # change cursor
        selected = self.findElement(
            self.GUIElements, lambda el: el.isSelected())
        if selected is not None:
            pygame.mouse.set_cursor(selected.getSelectCursor())
        else:
            pygame.mouse.set_cursor(self.cursor)

    def findElement(self, list, procces_function=None):
        """
        Find element in "list of GUI elements" for which procces function return True
        Parameters:
            list -> List with GUI elements
            procces_function -> True/False function, return first element for which return True
        """
        if procces_function is None or list is None:
            return None
        ret = None
        for el in list:
            if isinstance(el, Container):
                # container object (panel, table, ...) -> complex objects
                ret_container = self.findElement(
                    el.getChilds(), procces_function)
                if ret_container is not None:
                    ret = ret_container
                    break
            else:
                # simple object
                if procces_function(el):
                    ret = el
                    break
        return ret

    def render(self, screen: pygame.Surface):
        """
        Render view
        """
        if self.app is not None:
            for el in self.GUIElements:
                if el.isVisible():
                    el.draw(self, screen)

    def update(self):
        """
        Update view
        """
        if self.app is not None:
            for el in self.GUIElements:
                if el.isVisible():
                    el.update(self)

# base layout manager for view


class Layout(metaclass=abc.ABCMeta):
    """
    Base layout class mainly consists from list of layout elements and an abstract
    updateLayout function. The layout element consists of a reference to the 
    GUI element and properies for a specific layout manager.
    Layout element structure: {"element": value1, "propt": value2}
    """

    def __init__(self, view: View, register: bool = True):
        """
        Base layout class, automatically register layout manager to view
        Parameters:
            screen -> Pygame surface
            view -> View for which the layout manager will register
            register -> False: disable registration of this layout manager
        """
        if isinstance(view, View):
            self.view = view
        self.layoutElements = []
        # register
        if register:
            view.registerLayoutManager(self)

    @final
    def getLayoutElements(self) -> list:
        """
        Return all elements from layout
        """
        return self.layoutElements

    def setElements(self, layoutElements):
        """
        Set new layout element list
        Parameters:
            layoutElements -> Layout element list
        """
        self.layoutElements = layoutElements

    def addElement(self, element: GUIElement, propt: bool = None):
        """
        Add new layout element to layout manager
        Parameters:
            element -> GUIElement
            propt -> Property of element for layout manager (LEFT, RIGHT, CENTER, ...) values denpends on manager
        """
        if isinstance(element, GUIElement):
            self.layoutElements.append({"element": element, "propt": propt})

    @abc.abstractmethod
    def updateLayout(self, width: int, height: int):
        """
        Update layout of all GUI elements
        Parameters:
            width -> Width of view screen  
            height -> Height of view screen   
        """
        pass

    @final
    def getView(self):
        return self.view
