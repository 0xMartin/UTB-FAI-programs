import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def generateSignal(type: str, color: str, dc: float, 
                   amplitude: float, freq: float, phase: float, 
                   sample_rate: float, duration: float):
    """
    
    """

    # vygeneruje cas (x)
    time = np.arange(0, duration, 1/sample_rate)
    # vygeneruje signal (y)
    if type == "Sinusoidal":
        gs = dc + amplitude * np.sin(2 * np.pi * freq * time + phase)
    elif type == "Squared":
        gs = dc + amplitude * signal.square(2 * np.pi * freq * time + phase)
    elif type == "Saw-toothed":
        gs = dc + amplitude * signal.sawtooth(2 * np.pi * freq * time + phase)
    else:
        return

    # nastaveni figure + title
    plt.figure(figsize=(10, 6))
    title = type + " sampled signal\n"
    title += "Amp={:.1f} V".format(amplitude)
    title += ", DC={:.1f} V".format(dc)
    title += ", fi0={:.1f} rad".format(phase)
    title += ", Fsig={:.1f} Hz".format(freq)
    title += ", Fs={:.1f} Hz".format(sample_rate)
    plt.title(title, fontdict={'fontsize': 16})

    # vykresli graf
    markerline, _, _ = plt.stem(time, gs, linefmt=(color+'-'), markerfmt=(color+'o'), basefmt='k-')
    markerline.set_markerfacecolor('none')
    plt.grid(True, axis="y", linestyle=':')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.show()

if __name__ == "__main__":  
    generateSignal("Sinusoidal", "r", 1, 2, 0.2, 0, 5, 10)
    generateSignal("Squared", "b", 1, 2, 0.2, 0, 5, 10)
    generateSignal("Saw-toothed", "k", 1, 2, 0.2, 0, 5, 10)

"""
[[Zaver]]
----------------------------------------------------------------
Zadáním této úlohy bylo vygenerovaní tři různých signálu se shodnými vlastnostmi. 
Signály jsou generovaný pomocí python knihoven numpy a scipy. Pro generovaní 
všech signálu je v programu definovaná jedná procedura, u které volíme typ generovaného
signálu a následně její vlastnosti. Výsledný vygenerovaný signál je vykreslen do 
grafu pomocí knihovny matplotlib. Jednotlivé signály v rámci tohoto řešení úkolu 
jsou vykreslovaný vždy separátně v jednom okně aby na monitoru bylo mozne si je 
zobraziv ve vetsi velikosti.
"""