import math
from SimpleApp import *


class LSysConfig(metaclass=abc.ABCMeta):
    """
    Abstraktni trida reprezentujici konfiguraci L-Systemu
    """

    def __init__(self, alphabet, axiom, rules, angle, start_angle, display_mode):
        self.alphabet = alphabet
        self.axiom = axiom
        self.rules = rules
        self.angle = angle
        self.start_angle = start_angle
        self.display_mode = display_mode

    @final
    def getAlphabet(self):
        return self.alphabet

    @final
    def getAxiom(self):
        return self.axiom

    @final
    def getRules(self):
        return self.rules

    @final
    def getAngle(self):
        return self.angle

    @final
    def getStartAngle(self):
        return self.start_angle

    @final
    def getDisplayMode(self):
        return self.display_mode


class Tree(LSysConfig):
    def __init__(self):
        super().__init__(
            ["M", "S", "[", "]", "+", "-"],
            "M",
            {
                "M": "S[+M][-M]SM",
                "S": "SS"
            },
            45,
            90,
            "2D"
        )


class KochsSnowflake(LSysConfig):
    def __init__(self):
        super().__init__(
            ["F", "-", "+"],
            "F",
            {
                "F": "F+F--F+F"
            },
            45,
            0,
            "2D"
        )


class Tree3D(LSysConfig):
    def __init__(self):
        super().__init__(
            ["F", "A", "[", "]", "+", "-", "^", "&", "/"],
            "F",
            {
                "F": "A[-&AF][/AF][+^AF]",
                "A": "AA"
            },
            30,
            0,
            "3D"
        )


class SquareSierpinski(LSysConfig):
    def __init__(self):
        super().__init__(
            ["F", "f", "+", "-"],
            "F+XF+F+XF",
            {
                "X": "XF-F+F-XF+F+XF-F+F-X"
            },
            90,
            0,
            "2D"
        )


class LSystem():
    def __init__(self, canvas, word_list, config=None):
        """
        L-system
        Parametry:
            canvas -> Canvas do ktereho se bude vykreslovat (SimpleApp.gui.canvas)
            word_list -> Vystupni list pro slova (prazdny list)
            config -> Configurace L-Systemu (LSysConfig)
        """
        self.canvas = canvas
        self.word_list = word_list
        self.config = config

        self.canvas.setPaintEvt(self.draw)
        self.iterations = 1
        self.scale = 1.0
        self.repaint = True
        self.image = None

    def getConfig(self):
        """
        Nastaveni L-Systemu
        """
        return self.config

    def setConfig(self, config):
        """
        Zmeni konfiguraci L-Systemu
        """
        self.config = config
        self.axiom = config.getAxiom()

    def getIterations(self):
        """
        Aktualne nastaveni pocet iteraci
        """
        return self.iterations

    def setIterations(self, iterations):
        """
        Nastaveni poctu iteraci
        Parametry:
            iterations -> Pocet iteraci
        """
        self.iterations = max(1, iterations)

    def setAxiom(self, axiom):
        """
        Nastavi axiom
        Parametry:
            axiom -> Axiom
        """
        if self.config is not None:
            self.config.axiom = axiom

    def setScale(self, scale):
        """
        Nastaveni meritka
        """
        self.scale = scale

    def process(self):
        runTaskAsync(self.__process)

    def __process(self, args):
        if self.config is not None:
            self.word_list.clear()
            word = self.config.getAxiom()
            rules = self.config.getRules()
            for i in range(self.iterations):
                # aplikovani pravidel
                for key in rules.keys():
                    word = word.replace(key, rules[key])
                self.word_list.append([str(i + 1), word])

            # repaint
            self.repaint = True

    def draw(self, surface, offset):
        if self.config is None:
            return
        if self.config.getDisplayMode() == "2D":
            # 2D display mode ##########################################################
            if self.repaint and len(self.word_list) != 0:
                self.repaint = False
                # slovo z posledni iterace
                word = self.word_list[-1][1]
                # vytvori image
                size = self.calculateSizeOfImage_2D(
                    word,
                    self.config.getStartAngle(),
                    (0, 0),
                    self.config.getAngle()
                )
                self.image = pygame.Surface(size["size"])
                self.image.fill(self.canvas.getStyle()["background_color"])
                # pocatetecni pozice
                pos = size["pos"]
                # vykresli obrazec
                self.turtleGraphics_2D(
                    self.image,
                    colorInvert(self.canvas.getStyle()["background_color"]),
                    word,
                    self.config.getStartAngle(),
                    pos,
                    self.config.getAngle()
                )
            # image vykresli do canvasu
            if self.image is not None:
                surface.blit(
                    self.image,
                    (
                        offset[0] + (surface.get_width() -
                                     self.image.get_width()) / 2,
                        offset[1] + (surface.get_height() -
                                     self.image.get_height()) / 2
                    )
                )
        else:
            # 3D display mode ##########################################################
            if self.repaint and len(self.word_list) != 0:
                self.repaint = False
                self.wireframe = Wireframe()
                self.wireframe.setEdgeColor(colorInvert(self.canvas.getStyle()["background_color"]))
                # slovo z posledni iterace
                word = self.word_list[-1][1]
                # vytvori wireframe
                self.turtleGraphics_3D(
                    self.wireframe,
                    word,
                    self.config.getStartAngle(),
                    (surface.get_width()/2, surface.get_height() * 0.8, 0),
                    self.config.getAngle()
                )
            self.wireframe.rotateY(self.wireframe.computeCenter(), 0.03)
            self.wireframe.draw(surface, offset, False, True)

    # 2D section #######################################################################################

    def turtleGraphics_2D(self, surface, color, word, start_angle, start_position, change_angle):
        angle = math.radians(start_angle)
        position = start_position
        stack = []
        for char in word:
            if ord(char) >= ord('A') and ord(char) <= ord('U'):
                # „Nakresli úsečku.“, úsečka je nakreslena ve směru orientace štětce
                x = position[0] + math.cos(angle) * 10 * self.scale
                y = position[1] - math.sin(angle) * 10 * self.scale
                next_position = tuple([x, y])
                pygame.draw.line(surface, color, position, next_position, 2)
                position = next_position
            elif ord(char) >= ord('a') and ord(char) <= ord('u'):
                # „Pohni se vpřed (bez kreslení).“
                x = position[0] + math.cos(angle) * 10 * self.scale
                y = position[1] - math.sin(angle) * 10 * self.scale
                position = tuple([x, y])
            elif char == '|':
                # „Otoč se čelem vzad.“ (rotace o 180°)
                angle += math.radians(180)
            elif char == '[':
                # „Zapamatuj si aktuální stav.“, přesněji řečeno „Ulož aktuální stav na zásobník.“
                # (stavem se myslí aktuální poloha a orientace v rovině)
                stack.append({"pos": position, "angle": angle})
            elif char == ']':
                # „Přesuň se na naposledy zapamatovaný stav.“, přesněji řečeno „Přesuň se na pozici,
                # kterou určuje stav na vrcholu zásobníku a ten z vrcholu odstraň“
                state = stack.pop()
                position = state["pos"]
                angle = state["angle"]
            elif char == '+':
                # „Otoč se doleva (resp. doprava) o předem stanovený úhel.“ (o úhel otočení)
                angle += math.radians(change_angle)
            elif char == '-':
                # „Otoč se doleva (resp. doprava) o předem stanovený úhel.“ (o úhel otočení)
                angle -= math.radians(change_angle)

    def calculateSizeOfImage_2D(self, word, start_angle, start_position, change_angle):
        """
        Vypocita potrebnou velikost image do kteterho se pak jednorazove vykresli
        obrazec + pozice na ktere se ma zacit kreslit
        Pouze pro 2D
        """
        min_size = [0, 0]
        max_size = [0, 0]
        angle = math.radians(start_angle)
        position = start_position
        stack = []
        for char in word:
            if ord(char.lower()) >= ord('a') and ord(char) <= ord('u'):
                x = position[0] + math.cos(angle) * 10 * self.scale
                y = position[1] - math.sin(angle) * 10 * self.scale
                position = tuple([x, y])
                min_size[0] = min(min_size[0], x)
                min_size[1] = min(min_size[1], y)
                max_size[0] = max(max_size[0], x)
                max_size[1] = max(max_size[1], y)
            elif char == '|':
                angle += math.radians(180)
            elif char == '[':
                stack.append({"pos": position, "angle": angle})
            elif char == ']':
                state = stack.pop()
                position = state["pos"]
                angle = state["angle"]
            elif char == '+':
                angle += math.radians(change_angle)
            elif char == '-':
                angle -= math.radians(change_angle)
        size = tuple([
            int(max_size[0] - min_size[0]) + 10,
            int(max_size[1] - min_size[1]) + 10
        ])
        start_pos = tuple([
            start_position[0] - min_size[0] + 5,
            start_position[1] - min_size[1] + 5
        ])
        return {"size": size, "pos": start_pos}

    # 3D section ####################################################################

    def turtleGraphics_3D(self, wireframe, word, start_angle, start_position, change_angle):
        """
        Vykreslovani ve 3D modu
        Parametry:
            wireframe -> Wireframe 3D objekt
            start_angle -> pocatecni uhel pro osu Z
            start_position -> pocatecni pozice x,y,z
            change_angle -> uhel zmeni
        """
        yaw, pitch, roll = math.radians(start_angle), 0, 0
        position = start_position
        stack = []
        for char in word:
            if ord(char) >= ord('A') and ord(char) <= ord('U') or ord(char) >= ord('a') and ord(char) <= ord('u'):
                if char.isupper():
                    wireframe.addVertex(position)

                # vypocet pozice dalsiho vertexu
                origin = Vertex(position)
                next = Vertex(position)
                next.y -= 5 * self.scale
                next.rotateX(origin, yaw)
                next.rotateY(origin, pitch)
                next.rotateZ(origin, roll)
                position = tuple([next.x, next.y, next.z])

                if char.isupper():
                    wireframe.addVertex(position)
                    wireframe.addEdge(wireframe.getVertexCount(
                    ) - 2, wireframe.getVertexCount() - 1)
            elif char == '|':
                yaw += math.radians(180)
            elif char == '[':
                stack.append({"pos": position, "yaw": yaw,
                             "pitch": pitch, "roll": roll})
            elif char == ']':
                state = stack.pop()
                position = state["pos"]
                yaw = state["yaw"]
                pitch = state["pitch"]
                roll = state["roll"]
            elif char == '+':
                yaw += math.radians(change_angle)
            elif char == '-':
                yaw -= math.radians(change_angle)
            elif char == '^':
                pitch += math.radians(change_angle)
            elif char == '&':
                pitch -= math.radians(change_angle)
            elif char == '\\':
                roll += math.radians(change_angle)
            elif char == '/':
                roll -= math.radians(change_angle)
