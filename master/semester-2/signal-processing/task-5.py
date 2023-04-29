import numpy as np
import matplotlib.pyplot as plt


def convolve(u: np.array, g: np.array):
    """
    Vlastni implementace algoritmu pro konvoluci

    Parametry:
        x - vstupni signal
        g - impluzni odezva 
    """
    M = len(u)
    N = len(g)
    y = np.zeros(M+N-1)
    # konvolutarni suma
    for n in range(M+N-1):
        y[n] = 0
        for k in range(max(0, n-N+1), min(M, n+1)):
            y[n] += u[k]*g[n-k]
    return y


def drawSignal(g: np.array,
               u: np.array,
               conv_own: np.array,
               conv: np.array,
               n_max: int):
    """
    Vykresli signaly

    Parametry:
        g - impulzni odezva
        u - vstupni signal
        conv_own - vypoctena odezva systemu (vlastni algoritmus pro vypocet konvoluce)
        conv - vypoctena odezva systemu (numpy algoritmus)
        n_max - pocet vzorku
    """

    # osa X
    x = np.linspace(0, n_max, n_max)

    # konfigurace plt
    _, axs = plt.subplots(4, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.5)

    # Impulse response
    array_pad = np.pad(g, (0, n_max - len(g)), 'constant', constant_values=0)
    _, _, _ = axs[0].stem(x, array_pad, linefmt=('r-'),
                          markerfmt=('bo'), basefmt='k-')
    axs[0].set_title("Impulse response g(n) = [%s]" %
                     ';'.join([str(i) for i in g]))
    axs[0].grid(True, axis="y", linestyle=':')
    axs[0].set_xlabel('n [-]')
    axs[0].set_ylabel('g (n)')

    # Impulse signal
    array_pad = np.pad(u, (0, n_max - len(u)), 'constant', constant_values=0)
    _, _, _ = axs[1].stem(x, array_pad, linefmt=('r-'),
                          markerfmt=('bo'), basefmt='k-')
    axs[1].set_title("Impulse signal u(n) = [%s]" %
                     ';'.join([str(i) for i in u]))
    axs[1].grid(True, axis="y", linestyle=':')
    axs[1].set_xlabel('n [-]')
    axs[1].set_ylabel('u (n)')

    # Funciton CONV (numpy)
    array_pad = np.pad(conv, (0, n_max - len(conv)),
                       'constant', constant_values=0)
    _, _, _ = axs[2].stem(x, array_pad, linefmt=('r-'),
                          markerfmt=('bo'), basefmt='k-')
    axs[2].set_title("Funciton CONV: Output singal y(n) = [%s]" %
                     '; '.join([str(i) for i in conv]))
    axs[2].grid(True, axis="y", linestyle=':')
    axs[2].set_xlabel('n [-]')
    axs[2].set_ylabel('y (n)')

    # OWN algorithm
    array_pad = np.pad(conv_own, (0, n_max - len(conv_own)),
                       'constant', constant_values=0)
    _, _, _ = axs[3].stem(x, array_pad, linefmt=('r-'),
                          markerfmt=('bo'), basefmt='k-')
    axs[3].set_title("OWN algorithm: Output singal y(n) = [%s]" %
                     '; '.join([str(i) for i in conv_own]))
    axs[3].grid(True, axis="y", linestyle=':')
    axs[3].set_xlabel('n [-]')
    axs[3].set_ylabel('y (n)')

    plt.show()


if __name__ == "__main__":
    u = np.array([2, 3, 3, 2, 1])
    g = np.array([1, 1, 2])

    conv_own = convolve(u, g)
    conv = np.convolve(u, g)

    drawSignal(g, u, conv_own, conv, 15)


"""
[[Zaver]]
----------------------------------------------------------------
Cílem této úlohy bylo implementovat vlastní algoritmus pro vypočet
odezvy systému pomocí konvolutární sumy. Algoritmus jsem implementoval
a porovnal s algoritmem, který je součástí knihovny numpy. Výsledne 
signal jsem zobrazil do grafů v jedno okně.

Grafy ukazují, že obě metody výpočtu konvoluce dávají stejné výsledky
a výstupní signál y(n) vypočtený vlastní implementovanou funkcí 
vykazuje požadované vlastnosti.
"""
