import random


def reverseCol(game_data: list):
    for x in range(4):
        for y in range(2):
            tmp = game_data[x + y * 4]
            game_data[x + y * 4] = game_data[x + (3 - y) * 4]
            game_data[x + (3 - y) * 4] = tmp


def transpose(game_data: list):
    for x in range(1, 4):
        for y in range(3):
            if y < x:
                tmp = game_data[x + y * 4]
                game_data[x + y * 4] = game_data[y + x * 4]
                game_data[y + x * 4] = tmp


def moveUP(game_data: list) -> tuple:
    changed = False
    score = 0

    for x in range(4):
        pos = 0
        for y in range(4):
            if(game_data[x + y * 4] != 0):
                if pos != y:
                    changed = True
                if pos > 0:
                    if game_data[x + (pos - 1) * 4] == game_data[x + y * 4]:
                        game_data[x + (pos - 1) * 4] += game_data[x + y * 4]
                        game_data[x + y * 4] = 0
                        score += game_data[x + (pos - 1) * 4]
                        pos -= 1
                    else:
                        game_data[x + pos * 4] = game_data[x + y * 4]
                        if pos != y:
                            game_data[x + y * 4] = 0
                else:
                    game_data[x + pos * 4] = game_data[x + y * 4]
                    if pos != y:
                        game_data[x + y * 4] = 0
                pos += 1

    return changed, score


def moveDOWN(game_data: list) -> tuple:
    reverseCol(game_data)
    changed, score = moveUP(game_data)
    reverseCol(game_data)
    return changed, score


def moveLEFT(game_data: list) -> tuple:
    transpose(game_data)
    changed, score = moveUP(game_data)
    transpose(game_data)
    return changed, score


def moveRIGHT(game_data: list) -> tuple:
    transpose(game_data)
    reverseCol(game_data)
    changed, score = moveUP(game_data)
    reverseCol(game_data)
    transpose(game_data)
    return changed, score


def spawnRandom(game_data: list) -> bool:
    # field is full
    for i in range(16):
        if game_data[i] != 0:
            if i == 15:
                return False
        else:
            break

    # add new number
    number = 2 if random.randint(1,100) < 90 else 4
    while True:
        pos = random.randint(0, 15)
        if game_data[pos] == 0:
            game_data[pos] = number
            return True

def noMoves(game_data: list):
    for i in range(16):
        if game_data[i] == 0:
            return False
    for x in range(1, 3):
        for y in range(4):
            a = game_data[x + y * 4]
            if a == game_data[x + y * 4 - 1]:
                return False
            if a == game_data[x + y * 4 + 1]:
                return False
    for x in range(4):
        for y in range(1, 3):
            a = game_data[x + y * 4]
            if a == game_data[x + (y - 1) * 4]:
                return False
            if a == game_data[x + (y + 1) * 4]:
                return False
    return True