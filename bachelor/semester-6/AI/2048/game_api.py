from enum import Enum
from abc import ABC, abstractmethod


class GameHandelerAction(Enum):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class GameStatus(Enum):
    PLAYING = 0
    VICTORY = 1
    DEFEAT = 2


class GameStats:
    def __init__(self):
        self.best_score = None
        self.worst_score = None
        self.total_score = 0
        self.victories = 0
        self.defeats = 0
        self.avg_max_value_total = 0
        self.clear()

    def clear(self):
        self.move_counter = 0
        self.max_value = 0
        self.score = 0
        self.game_status = GameStatus.PLAYING

    def addMove(self):
        self.move_counter += 1

    def setMaxValue(self, _value):
        self.max_value = _value

    def setScore(self, _score):
        self.score = _score

    def setGameStatus(self, _status: GameStatus):
        self.game_status = _status

    def getMaxValue(self):
        return self.max_value

    def getScore(self):
        return self.score

    def getMoveCount(self):
        return self.move_counter

    def getGameStatus(self) -> GameStatus:
        return self.game_status

    def writeToGlobalStats(self, game_data: list):
        if self.game_status == GameStatus.VICTORY:
            self.victories += 1
        elif self.game_status == GameStatus.DEFEAT:
            self.defeats += 1
        if self.best_score == None:
            self.best_score = self.score
        if self.worst_score == None:
            self.worst_score = self.score
        self.best_score = max(self.best_score, self.score)
        self.worst_score = min(self.worst_score, self.score)
        self.total_score += self.score
        self.avg_max_value_total += max(game_data)

    def getVictories(self):
        return self.victories

    def getDefeats(self):
        return self.defeats

    def getAvgMaxValue(self):
        if self.victories + self.defeats == 0:
            return 0.0
        else:
            return self.avg_max_value_total / (self.victories + self.defeats)

    def getBestScore(self):
        return self.best_score

    def getWorstScore(self):
        return self.worst_score

    def getAvgScore(self):
        if self.victories + self.defeats == 0:
            return 0.0
        else:
            return self.total_score / (self.victories + self.defeats)


class GameHandler:
    def __init__(self):
        self.actionCallbacks = []
        self.action = GameHandelerAction.NONE

    def up(self, sender):
        self.action = GameHandelerAction.UP
        self.callCallback(sender)

    def down(self, sender):
        self.action = GameHandelerAction.DOWN
        self.callCallback(sender)

    def left(self, sender):
        self.action = GameHandelerAction.LEFT
        self.callCallback(sender)

    def right(self, sender):
        self.action = GameHandelerAction.RIGHT
        self.callCallback(sender)

    def setAction(self, sender, _action : GameHandelerAction):
        self.action = _action
        self.callCallback(sender)

    def callCallback(self, sender):
        for c in self.actionCallbacks:
            c(self.action, sender)

    def getAction(self) -> GameHandelerAction:
        return self.action

    def addActionCallback(self, _callback):
        self.actionCallbacks.append(_callback)


class AI(ABC):

    @abstractmethod
    def init(self, game_data: list, stats: GameStats, handler: GameHandler):
        pass

    @abstractmethod
    def doMove(self) -> GameHandelerAction:
        pass
