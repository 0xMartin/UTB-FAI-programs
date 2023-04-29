import sys
import threading
import time
import math
import pygame
from pygame.locals import *
from game_api import *
from rules import *


class Game2048:
    def __init__(self, _fps, _rps, _size):
        self.running = False
        self.fps = _fps
        self.rps = _rps
        self.size = _size
        self.AI = None
        self.endless = False
        self.stats = GameStats()

    def init(self):
        # init pygame and window
        pygame.init()
        self.surface = pygame.display.set_mode((self.size, self.size))
        pygame.display.set_caption('2048')
        print("Game window opened")

        # init font
        self.tile_font = pygame.font.SysFont('Comic Sans MS', 70)
        self.text_font_small = pygame.font.SysFont('Comic Sans MS', 23)

        # create game handler
        self.handler = GameHandler()
        self.handler.addActionCallback(self.gameHandlerCallback)

    def setHandler(self, _handler):
        self.handler = _handler
        self.handler.setCallBack(self.gameHandlerCallback)

    def setAI(self, ai, games_count):
        self.AI = ai
        self.ai_games_count = games_count
        self.AI.init(self.stats, self.handler)

    def setEndlessMode(self, mode: bool):
        self.endless = mode

    def start(self):
        if not self.running:
            # create game
            self.newGame()
            # run
            self.running = True
            render_thread = threading.Thread(
                target=self.render_loop, args=(1,))
            render_thread.start()
            update_thread = threading.Thread(
                target=self.update_loop, args=(1,))
            update_thread.start()

    def render_loop(self, arg):
        clock = pygame.time.Clock()
        while self.running:
            # bg fill
            self.surface.fill((0, 0, 0))
            # field out line
            pygame.draw.rect(
                self.surface,
                (44, 43, 49),
                pygame.Rect(
                    self.size * 0.1,
                    self.size * 0.18,
                    self.size * 0.8,
                    self.size * 0.8
                )
            )
            # fild tiles
            step = self.size * 0.2
            for x in range(4):
                for y in range(4):
                    # position of tile
                    x_pos, y_pos = self.size * 0.11 + x * step, self.size * 0.19 + y * step
                    # draw tile
                    value = self.game_data[x + y * 4]
                    if value != 0:
                        # with value
                        pygame.draw.rect(
                            self.surface,
                            getColor(self.game_data, value),
                            pygame.Rect(
                                x_pos,
                                y_pos,
                                self.size * 0.18,
                                self.size * 0.18
                            )
                        )
                        txtSurf = self.tile_font.render(
                            str(value), True, (20, 20, 20))
                        self.surface.blit(
                            txtSurf,
                            (x_pos + (step - txtSurf.get_width())/2,
                             y_pos + (step - txtSurf.get_height())/2)
                        )
                    else:
                        # empty
                        pygame.draw.rect(
                            self.surface,
                            (94, 87, 88),
                            pygame.Rect(
                                x_pos,
                                y_pos,
                                self.size * 0.18,
                                self.size * 0.18
                            )
                        )

            # statistics
            txtSurf = self.text_font_small.render(
                "Moves: " + str(self.stats.getMoveCount()), True, (100, 100, 100))
            self.surface.blit(
                txtSurf,
                (self.size * 0.1, self.size * 0.03)
            )
            txtSurf = self.text_font_small.render(
                "Max value: " + str(self.stats.getMaxValue()), True, (100, 100, 100))
            self.surface.blit(
                txtSurf,
                (self.size * 0.5 - txtSurf.get_width() / 2, self.size * 0.03)
            )
            txtSurf = self.text_font_small.render(
                "Score: " + str(self.stats.getScore()), True, (100, 100, 100))
            self.surface.blit(
                txtSurf,
                (self.size * 0.9 - txtSurf.get_width(), self.size * 0.03)
            )

            txtSurf = self.text_font_small.render(
                "Best Score: " + str(self.stats.getBestScore()), True, (100, 100, 100))
            self.surface.blit(
                txtSurf,
                (self.size * 0.1, self.size * 0.07)
            )
            txtSurf = self.text_font_small.render(
                "Worst Score: " + str(self.stats.getWorstScore()), True, (100, 100, 100))
            self.surface.blit(
                txtSurf,
                (self.size * 0.5 - txtSurf.get_width() / 2, self.size * 0.07)
            )
            txtSurf = self.text_font_small.render(
                "Avg Score: " + '%.2f' % self.stats.getAvgScore(), True, (100, 100, 100))
            self.surface.blit(
                txtSurf,
                (self.size * 0.9 - txtSurf.get_width(), self.size * 0.07)
            )

            txtSurf = self.text_font_small.render(
                "Victory: " + str(self.stats.getVictories()), True, (100, 100, 100))
            self.surface.blit(
                txtSurf,
                (self.size * 0.1, self.size * 0.11)
            )
            txtSurf = self.text_font_small.render(
                "Defeat: " + str(self.stats.getDefeats()), True, (100, 100, 100))
            self.surface.blit(
                txtSurf,
                (self.size * 0.5 - txtSurf.get_width() / 2, self.size * 0.11)
            )
            txtSurf = self.text_font_small.render(
                "Avg Max Value: " + '%.2f' % self.stats.getAvgMaxValue(), True, (100, 100, 100))
            self.surface.blit(
                txtSurf,
                (self.size * 0.9 - txtSurf.get_width(), self.size * 0.11)
            )

            # game status
            if self.stats.getGameStatus() == GameStatus.VICTORY:
                txtSurf = self.text_font_small.render(
                    "VICTORY (next game in 5s)", True, (100, 190, 100))
                self.surface.blit(
                    txtSurf,
                    (self.size * 0.5 - txtSurf.get_width()/2, self.size * 0.14)
                )
            elif self.stats.getGameStatus() == GameStatus.DEFEAT:
                txtSurf = self.text_font_small.render(
                    "DEFEAT (next game in 5s)", True, (210, 100, 100))
                self.surface.blit(
                    txtSurf,
                    (self.size * 0.5 - txtSurf.get_width()/2, self.size * 0.14)
                )

            # update display
            pygame.display.flip()
            clock.tick(self.fps)

    def update_loop(self, arg):
        clock = pygame.time.Clock()
        while self.running:
            if self.AI is not None:
                if self.stats.getVictories() + self.stats.getDefeats() < self.ai_games_count:
                    self.AI.doMove(self.game_data)
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                    time.sleep(0.1)
                    pygame.quit()
                    sys.exit()
                if self.stats.getGameStatus() == GameStatus.PLAYING:
                    if event.type == pygame.KEYDOWN and self.AI is None:
                        if event.key == pygame.K_LEFT:
                            self.handler.left(self)
                        if event.key == pygame.K_RIGHT:
                            self.handler.right(self)
                        if event.key == pygame.K_UP:
                            self.handler.up(self)
                        if event.key == pygame.K_DOWN:
                            self.handler.down(self)
                if self.AI is None:
                    clock.tick(self.rps)

    def gameHandlerCallback(self, action, sender):
        changed = False
        if action == GameHandelerAction.UP:
            changed, score = moveUP(self.game_data)
        elif action == GameHandelerAction.DOWN:
            changed, score = moveDOWN(self.game_data)
        elif action == GameHandelerAction.LEFT:
            changed, score = moveLEFT(self.game_data)
        elif action == GameHandelerAction.RIGHT:
            changed, score = moveRIGHT(self.game_data)

        if action != GameHandelerAction.NONE:
            if changed:
                # victory
                if not self.endless:
                    for i in self.game_data:
                        if i == 2048:
                            self.victory()
                # add new number
                if not spawnRandom(self.game_data):
                    self.defeat()
                    return
                # defeat check => no possible move
                if noMoves(self.game_data):
                    self.defeat()
                    return
                # statisctics
                self.statisctics()

    def statisctics(self):
        # moves
        self.stats.addMove()
        # max value
        max_val = 2
        for i in self.game_data:
            max_val = max(max_val, i)
        self.stats.setMaxValue(max_val)
        # score
        self.stats.setScore(sum(self.game_data))

    def victory(self):
        self.stats.setGameStatus(GameStatus.VICTORY)
        if self.AI is None:
            time.sleep(5)
            pygame.event.clear()
        self.stats.writeToGlobalStats(self.game_data)
        self.newGame()

    def defeat(self):
        self.stats.setGameStatus(GameStatus.DEFEAT)
        if self.AI is None:
            time.sleep(5)
            pygame.event.clear()
        self.stats.writeToGlobalStats(self.game_data)
        self.newGame()

    def newGame(self):
        # AI games limit
        if self.AI is not None:
            if self.stats.getDefeats() + self.stats.getVictories() >= self.ai_games_count:
                return
        # game data
        self.game_data = [0] * 16
        spawnRandom(self.game_data)
        spawnRandom(self.game_data)
        # statistics
        self.stats.clear()
        self.stats.setGameStatus(GameStatus.PLAYING)
        self.statisctics()


# <===========utils=================================================================================>


def getColor(game_data, current):
    max_val = 2048
    for i in game_data:
        max_val = max(i, max_val)
    wl = 750 - math.log2(current) / math.log2(max_val) * (750 - 380)
    return wavelengthToRGB(wl)


def wavelengthToRGB(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B))
