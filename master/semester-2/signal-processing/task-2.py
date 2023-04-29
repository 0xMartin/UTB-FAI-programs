import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

def aperiodicSignal(id: int, ax: plt.Axes, start: int, end: 
                    int, pulse_start: int, pulse_max: int, pulse_end: int):
    """
    Vygeneruje trojhuhelnikovy aperiodicky signal
        id -> grafu
        ax -> plt.Axes
        start -> zacatek casoveho useku
        end -> konec casoveho useku
        pulse_start -> cas kde trojhuhelnikovy aperiodicky signal zacina
        pulse_max -> cas ve kterem trojhuhelnikovy aperiodicky signal dosahuje maxima
        pulse_end -> cas kde trojhuhelnikovy aperiodicky signal konci
    """
    if pulse_max < pulse_start or pulse_max > pulse_end:
        return
    
    # vygeneruje casovou osu prubehu (x)
    t = np.linspace(start, end, end - start + 1)
    # stred casoveho intervalu
    mid = int((end - start) / 2)

    # vygeneruje trojhuhelnikovy aperiodicky signal podle definovanych paramtetru
    y = np.zeros_like(t)
    dy1 = 1.0 / (pulse_max - pulse_start)
    dy2 = 1.0 / (pulse_max - pulse_end)
    for i in range(pulse_start, pulse_end):
        y[i + mid + 1] += (dy1 if i < pulse_max else dy2) + y[i + mid]

    # vykresli singnal do grafu
    ax.plot(t, y, 'bo-')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('y%d [-]' % id)
    ax.set_title('Aperiod triangle pulse y%d' % id)


def unitImpulse(id: int, ax: plt.Axes, impluse_n: int, start: int, end: int):
    """
    Vygeneruje jednotkovy impuls
        id -> grafu
        ax -> plt.Axes
        impluse_n -> n pri ktetem nastane impuls
        start -> zacatek casoveho useku
        end -> konec casoveho useku
    """

    # vygeneruje osu (n) a hodnoty signalu (y)
    x = np.arange(start, end + 1, 1)
    y = np.zeros_like(x)
    y[np.where(x == impluse_n)] = 1

    # vykresli singnal do grafu
    ax.set_title("Unit impulse δ%d" % id, fontdict={'fontsize': 16})
    ax.set_xlabel("n [-]")
    ax.set_ylabel("δ%d [-]" % id)
    ax.set_ylim(-1, 2)
    markerline, _, _ = ax.stem(x, y, linefmt='r-', markerfmt='ro', basefmt='k-')
    markerline.set_markerfacecolor('none')


def unitStep(id: int, ax: plt.Axes, impluse_n: int, start: int, end: int):
    """
    Vygeneruje jednotkovy skok
        id -> grafu
        ax -> plt.Axes
        impluse_n -> n pri kterem dochazi ke skokove zmene signalu
        start -> zacatek casoveho useku
        end -> konec casoveho useku
    """

    # vygeneruje osu (n) a hodnoty signalu (y)
    x = np.arange(start, end + 1, 1)
    y = np.zeros_like(x)
    y[np.where(x >= impluse_n)] = 1

    # vykresli singnal do grafu
    ax.set_title("Unit impulse u%d" % id, fontdict={'fontsize': 16})
    ax.set_xlabel("n [-]")
    ax.set_ylabel("u%d [-]" % id)
    ax.set_ylim(-1, 2)
    markerline, _, _ = ax.stem(x, y, linefmt='b-', markerfmt='bo', basefmt='k-')
    markerline.set_markerfacecolor('none')

def functionSinc(start: int, end: int):
    """
    Vygeneruje prubeh funkce sinc(t)
        start -> zacatek generovaneho casoveho useku
        end -> konec generovaneho casoveho useku
    """

    # vygeneruje casovou osu t a pro ni vygeneruje funkcni hodnoty funkce sinc(t)
    t = np.linspace(start, end, 400)
    y = np.sinc(t)

    # vykresli singnal do grafu
    plt.figure(figsize=(12, 5))
    plt.scatter(t, y, facecolors='none', edgecolors='black')
    plt.title("Function Sinc(t)", fontdict={'fontsize': 16})
    plt.xlabel("t [s]")
    plt.ylabel("y [-]")
    plt.ylim(-1, 1.5)
    plt.show()

if __name__ == "__main__": 
    _, axs = plt.subplots(3, 1, figsize=(12, 10)) 
    aperiodicSignal(1, axs[0], -20, 20, -5, 0, 5)
    aperiodicSignal(2, axs[1], -20, 20, 0, 10, 11)
    aperiodicSignal(3, axs[2], -20, 20, -11, -10, 0)
    plt.show()

    _, axs = plt.subplots(1, 2, figsize=(12, 5))
    unitImpulse(1, axs[0], 0, -10, 10)
    unitImpulse(2, axs[1], 5, -10, 10)
    plt.show()

    _, axs = plt.subplots(1, 2, figsize=(12, 5))    
    unitStep(1, axs[0], 0, -10, 10)
    unitStep(1, axs[1], -4, -10, 10)
    plt.show()

    functionSinc(-10, 10)

"""
[[Zaver]]
----------------------------------------------------------------
V této úloze byl cílem vygenerovat několik různých typů signálů a následně je zobrazit 
v grafech. 

Bod zadání č. 1 bylo vygenerování aperiodického trojúhelníkového signálu. 
V zadání bylo uvedeno, že se má signál generovat pomocí funkce "tripuls", ta se ale nachází
v programu Matlab. Vygenerování tohoto signálu jsem tedy vyřešil pomocí jednoduchého
přičítání vypočtených konstant dy1 (pro vzestupnou hranu trojúhelníkového signálu) a dy2
(pro sestupnou hranu signálu). Takto vznikne požadovaný trojúhelníkový aperiodický signál.

Body zadání č. 2 a č. 3 jsou si velmi podobné, liší se pouze tím, že u zadání č. 2 se má
vygenerovat krátký impuls pro jedno definované "n", zatímco u úlohy č. 3 se má signál
skokově změnit od "n". 

Posledním bodem zadání bylo vygenerovat a zobrazit funkci sinc(t).
Stačilo jen vygenerovat časovou osu a pro jednotlivé časy (t) vypočítat hodnotu funkce sinc(t).
"""
