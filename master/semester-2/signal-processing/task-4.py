import numpy as np
import matplotlib.pyplot as plt


def generateSignal(P: float,
                   Fsig: int,
                   Fs: int,
                   Amp: float,
                   DC: float,
                   SNR: float) -> tuple([np.array, np.array, np.array, np.array]):
    """
    Vygeneruje sadu singnalu

    Parametry:
        P - pocet vygenerovanych period signalu
        Fsig - frekvence signalu
        Fs - frekvence vzorkovani
        Amp - amplituda signalu
        DC - stejnosmerna slozka signalu
        SNR - pomer urovne signalu a sumu

    Return:
        casova osa, puvodni signal, vygenerovany sum, vysledny signal + sum
    """

    # vygenerovani casove osy (T - doba trvani, N - pocet vzorku)
    T = P / Fsig
    N = int(T * Fs)
    t = np.linspace(0, T, N)

    # vygenerovani puvodniho signalu
    signal = Amp * np.sin(2 * np.pi * Fsig * t) + DC
    # vypoceteni stredniho vykonu puvodniho signalu
    signal_power = np.sum(signal**2) / len(signal)

    # vygeneruje sum tak aby SNR odpovidalo definovane hodnoty argumentu funkce
    noise_power = signal_power / (10**(SNR/10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # vysledny signal (puvodni + sum)
    result = signal + noise

    return t, signal, noise, result


def plotSignals(snr: float,
                t: np.array,
                signal: np.array,
                noise: np.array,
                final_signal: np.array):
    """
    Vykresli vsechny signali do grafu

    Parametry:
        snr - SNR
        t - casova osa
        signal - puvodni signal
        noise - vygenerovany sum 
        final_signal - vysledny signal + sum
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].set_title("Signal and noise separately shown (desired: SNR %f)" % snr)
    ax[0].plot(t, noise, label='Noise')
    ax[0].plot(t, signal, label='Signal')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Voltage [V]')
    ax[0].legend()

    # vypocet stredniho vykonu puvodniho signalu a sumu
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))
    # vypocet SNR pomoci vykonu
    snr = 10 * np.log10(signal_power / noise_power)

    ax[1].plot(t, final_signal, label='Superimposed (Real SNR: %f)' % snr)
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Voltage [V]')
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    # a (10 snr)
    t, signal, noise, final_signal = generateSignal(2, 1, 100, 1, 2, 10)
    plotSignals(10, t, signal, noise, final_signal)
    # b (30 snr)
    t, signal, noise, final_signal = generateSignal(2, 1, 100, 1, 2, 30)
    plotSignals(30, t, signal, noise, final_signal)


"""
[[Zaver]]
----------------------------------------------------------------
Cílem této úlohy bylo prace se signálem a šumem a vypočet SNR. 
K tomuto účelu jsem použil funkci generateSignal, která generuje signál
a sum s definovanými vlastnostmi, jako jsou frekvence signálu, amplituda,
stejnosměrná složka, vzorkovací frekvence a SNR. Pomocí této funkce
jsem vygeneroval dva signály s hodnotami SNR (10 dB a 30 dB) a vykreslil jsem 
je pomocí funkce plotSignals.

Výsledky ukázaly, že s rostoucím SNR se původní signál a signál se superponovaným
šumem stávají stále méně odlišitelné. Naopak, s klesajícím SNR je výsledek opačný. 
"""