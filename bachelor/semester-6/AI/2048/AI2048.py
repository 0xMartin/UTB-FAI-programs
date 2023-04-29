from time import sleep
from game_api import *
from rules import *
import random
import multiprocessing
from threading import Lock
import time


class AI2048(AI):
    def __init__(self, _iterations):
        self.iterations = _iterations
        self.current_iterations = _iterations / 2
        self.moves = [
            (moveDOWN, GameHandelerAction.DOWN),
            (moveUP, GameHandelerAction.UP),
            (moveLEFT, GameHandelerAction.LEFT),
            (moveRIGHT, GameHandelerAction.RIGHT)
        ]
        self.pool = multiprocessing.Pool(processes=6)
        self.mutex = Lock()
        self.total = 0
        self.cnt = 0

    def init(self, _stats: GameStats, _handler: GameHandler):
        self.handler = _handler

    def doMove(self, game_data: list) -> GameHandelerAction:
        self.game_data = game_data

        # hledani prumerenho skore => pro kazdy ze 4 smeru vytvori vlastni vlakno
        results = []
        for move in self.moves:
            result = self.pool.apply_async(
                getAvgScoreForMove, (move, self.moves, self.game_data, int(self.current_iterations)))
            results.append(result)

        res_list = [result.get() for result in results]

        # najde nejlepsi tah a nejmensi prumerny pocet tahu v prohledavani
        best_score = -1
        best_move = None
        min_avg_move = 99999
        for score, action, avg_moves in res_list:
            if score > best_score:
                best_score = score
                best_move = action
            if score > 0:
                min_avg_move = min(min_avg_move, avg_moves)

        # dynamicky meni pocet itaracu v urcitem rozmezi v zavislosti na prumernem poctu tahu
        if min_avg_move < 10 and self.current_iterations < self.iterations * 3.0:
            self.current_iterations += 10
        elif min_avg_move < 40 and self.current_iterations < self.iterations * 2.0:
            self.current_iterations += 10
        elif min_avg_move > 60 and self.current_iterations > self.iterations * 0.7:
            self.current_iterations -= 10

        # game handleru preda akci
        self.handler.setAction(self, best_move)


def getAvgScoreForMove(move: tuple, moves: list, game_data: list, iterations: int):
    total_score = 0

    last = []
    move_cnt = 0

    for _ in range(iterations):
        new_game_data = game_data.copy()
        changed, s = move[0](new_game_data)
        if not changed:
            return [-1.0, GameHandelerAction.NONE, 0]
        total_score += s
        spawnRandom(new_game_data)

        while True:
            if noMoves(new_game_data):
                break
            move_cnt += 1
            last.clear()
            while True:
                changed, s = moves[nextRandomMove(last)][0](new_game_data)
                if changed:
                    break
            total_score += s
            spawnRandom(new_game_data)

    return [total_score / iterations, move[1], move_cnt / iterations]


def nextRandomMove(last: list):
    rnd = 0
    while True:
        rnd = random.randint(0, 3)
        if rnd not in last:
            break
    last.append(rnd)
    return rnd
