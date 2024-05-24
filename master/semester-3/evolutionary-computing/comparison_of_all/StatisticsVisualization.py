from AlgorithmStatistics import *
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np

class StatisticsVisualization:
    def __init__(self, stat: AlgorithmStatistics) -> None:
        self.stat = stat

    def showTableFor(self, names: [str]):
        """
        Vygeneruje a zobrazi tabulku statistickych charakteristik pro vybrane algoritmy

        Parametry:
            names - List jmen algoritmu ktere budou zobrazeny v tabulce
        """
        table = PrettyTable()
        table.field_names = ["Name", "Average", "Std.Dev.", "Median", "Min", "Max"]

        for name in names:
            worker = self.stat.findWorkerByName(name)
            if worker is None:
                continue
            res = worker.getResults()
            if res is None:
                continue
            if len(res.yBestList) == 0:
                print(f"Warning!! YBEST is empty for [{name}]")
                continue
            avg = np.mean(res.yBestList)
            std_dev = np.std(res.yBestList)
            median = np.median(res.yBestList)
            minimum = np.min(res.yBestList)
            maximum = np.max(res.yBestList)

            table.add_row([name, avg, std_dev, median, minimum, maximum])

        print(table)
            

    def showGraphFor(self, title: str, names: [str], max_y_value: float = None):
        """
        Vygeneruje a zobrazi konvergencni graf pro vybrane algoritmy

        Parametry:
            names - List jmen algoritmu ktere budou zobrazeny v grafu
            max_y_value - Maximální hodnota na ose y (volitelný parametr)
        """
        plt.figure(figsize=(16, 10)) 

        for name in names:
            worker = self.stat.findWorkerByName(name)
            if worker is None:
                continue
            res = worker.getResults()
            avgHist = res.avgHistory()
            if len(avgHist) != 0:
                plt.plot(avgHist, label=name)

        plt.title(f"Konvergenční graf pro {title}")
        plt.xlabel("Generace") 
        plt.ylabel("Fitness") 

        if max_y_value is not None:
            plt.ylim(0, max_y_value)

        plt.legend()
        plt.show()
        print("changed")