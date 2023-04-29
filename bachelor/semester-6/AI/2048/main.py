from game import *
from AI2048 import *

if __name__ == "__main__":
    print("[2048 konfigurace]")
    endless = input("Nekonecny mod (hra nekonci pri dosazeni 2048) [y, N]:")
    endless = True if endless == 'y' else False

    ai_enabled = input("Povolit umelou inteligenci [Y, n]:")
    ai_enabled = False if ai_enabled == 'n' else True

    game = Game2048(20, 50, 700)
    game.init()
    if ai_enabled:
        ai = AI2048(60)
        game.setAI(ai, 30)
    game.setEndlessMode(endless)
    game.start()
