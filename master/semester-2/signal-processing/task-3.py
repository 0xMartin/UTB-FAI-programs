import numpy as np
import matplotlib.pyplot as plt
import math

class SignalCharacteristics:

    def __init__(self, signal: np.array) -> None:
        self.signal = signal

    def mean(self) -> float:
        sum_y = sum(y for y in self.signal)
        return sum_y / len(self.signal)  

    def effectiveValue(self) -> float:
        sum_yy = sum(y**2 for y in self.signal)
        return math.sqrt(sum_yy / len(self.signal))

    def variance(self) -> float:
        m = self.mean()
        sum_r = sum((y - m) ** 2 for y in self.signal)
        return sum_r / len(self.signal)

    def standardDeviation(self) -> float:
        return math.sqrt(self.variance())

    def median(self) -> float:
        n = len(self.signal)
        s = sorted(self.signal)
        if n % 2 == 0:
            i = int(n / 2)
            median = (s[i - 1] + s[i]) / 2
        else:
            median = s[int((n - 1) / 2)]
        return median

    def instantPower(self, time) -> float:
        if time < 0 or time > len(self.signal):
            return 0.0
        return self.signal[time] ** 2

    def mediumPower(self) -> float:
        sum_yy = sum(y**2 for y in self.signal)
        return sum_yy / len(self.signal)

    def energy(self) -> float:
        return sum(y**2 for y in self.signal)
    
    def showTable(self):
        """
        Zobrazi tabulku s vypoctenyma charakteristikama napetoveho signalu
        """
        data = [
            ['Mean value', self.mean(), np.mean(self.signal), 'V'],
            ['RMS value', self.effectiveValue(), np.sqrt(np.mean(signal ** 2)), 'V'],
            ['Variance', self.variance(), np.var(signal), 'V^2'],
            ['Standard deviation', self.standardDeviation(), np.std(signal), 'V'],
            ['Median', self.median(), np.median(signal), 'V'],
            ['Medium power', self.mediumPower(), None, 'V'],
            ['Energy', self.energy(), None, 'J']
        ]

        for row in data:
            print(row[0], row[1], row[2], row[3])
        
        fig, ax = plt.subplots(figsize=(14,10))
        ax.axis('off')
        ax.axis('tight')
        t = ax.table(cellText=data, colLabels=['Name', 'Own Function', 'Numpy Function', 'Unit'], loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(11)
        plt.show()


def generateVoltageSignal(f1: float, f2: float, fs: float, time: float) -> tuple([np.array, np.array]):
    """
    Vygeneruje a napetovy signal

    Signal:
        y=5*cos(2*pi*f1*t)-2*sin(2*pi*f2*t)

    Parametry:
        f1 - Frekvence signalu f1 (zakladni frekvence) [v Hz]
        f2 - Frekvence signalu f2 (harmonicka frekvence) [v Hz]
        fs - Vzorkovaci frekvence [v Hz]
        time - Cas trvani signalu [v sekundach]
    """
    samples = int(fs * time)

    # vytvoreni casove osy
    t = np.arange(samples) / fs
    # vypocet napetoveho signalu
    y = 5 * np.cos(2 * np.pi * f1 * t) - 2 * np.sin(2 * np.pi * f2 * t)

    return t, y


def plotSignal(time: np.array, signal: np.array, label_x: str, label_y: str):
    """
    Vykresli napetovy signal
    
    Parametry:
        time    - Casova osa signalu
        signal  - Napetovy signal
        label_x - Label pro osu X
        label_y - Label pro osu Y
    """
    fig, ax = plt.subplots()
    ax.plot(time, signal, '-k', linewidth=1.5, label='analog signal')
    ax.plot(time, signal, 'o', markersize=7, markerfacecolor='none', color='b', markeredgecolor='b', label='sampled signal')
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    time, signal = generateVoltageSignal(2, 8, 200, 0.5)
    plotSignal(time, signal, "t [s]", "Voltage [V]")

    sigChar = SignalCharacteristics(signal)
    sigChar.showTable()

    plotSignal(time, signal ** 2, "t [s]", "Instant Power [V^2]")

"""
[[Zaver]]
----------------------------------------------------------------
Cílem této úlohy bylo vykreslit + navzorkovat napěťový signál a určit 
jeho charakteristiky. Signál jsme dle jeho definice ze zadání vygeneroval
a vykreslil do grafu s využitím knihovný matplotlib. Následně jsme implementoval
vlastní funkce pro výpočet charakteristik napěťového signálu (střední hodnota,
efektivní hodnota, rozptyl, směrodatná odchylka, median, okamžitý výkon, 
střední hodnota energie). Pomocí těchto funkcí jsem vypočetl charakteristiky 
signálu a jejich hodnoty porovnal v zobrazené tabulce s hodnotami funkcí 
knihovny numpy. Nakonec jsme vykreslil signál okamžitého výkonu.
"""