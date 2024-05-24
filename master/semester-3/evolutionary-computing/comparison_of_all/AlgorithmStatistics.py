import time
import csv
import os
import sys
import numpy as np
import ast
import multiprocessing
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

csv.field_size_limit(sys.maxsize)

class Args:
    def __init__(self, D, func, bounds):
        self.D = D
        self.func = func
        self.bounds =bounds
        

class WorkerResults:
    def __init__(self):
        """
        Vysledky
        xBestList - hodnoty xBest ze vsech opakovani algoritmu
        yBestList - hodnoty yBest ze vsech opakovani algoritmu
        historyList - vyvoj hodnoty fitness ze vsech opakovani algoritmu
        """
        self.xBestList = []
        self.yBestList = []
        self.historyList = []
        self.total_time = 0.0

    def avgHistory(self):
        """
        Navrati zprumerovany vyvoje konvergencni historie
        """
        if len(self.historyList) == 0:
            return []
        historyLen = len(self.historyList[0])
        repeat_count = len(self.historyList)
        prumery = []
        for i in range(historyLen):
            prumery.append(
                sum(history[i] for history in self.historyList) / repeat_count)
        return prumery
    
    def save_to_csv(self, file_path):
        """
        Ulozi vsechny parametry do CSV souboru.

        Parametry:
            file_path - Cesta CSV souboru.
        """
        data = list(zip(self.xBestList, self.yBestList, self.historyList))
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['xBest', 'yBest', 'history'])
            writer.writerows(data)

    def load_from_csv(self, file_path):
        """
        Nacte vsechny parametry z CSV souboru.

        Parametry:
            file_path - Cesta k CSV souboru.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            # Skip the header row
            next(reader, None)

            # Clear existing data
            self.xBestList.clear()
            self.yBestList.clear()
            self.historyList.clear()

            # Read data from CSV and populate class attributes
            for row in reader:
                x_best_str, y_best_str, history_str = row
                x_best = np.fromstring(x_best_str, sep=' ')
                y_best = float(y_best_str)

                self.xBestList.append(x_best)
                self.yBestList.append(y_best)
                self.historyList.append(ast.literal_eval(history_str))


class Worker:
    def __init__(self, repeat_count: int, name: str, callback, args: Args):
        """
        Vytvori worker, ktery umoznu spusteni libovolneho algoritmu v samostatnem vlakne

        Parametry:
            repeat_count - Pocet opakovani algoritmu
            name - Nazev algoritmu
            callback- callback ktery musi obsahovat samotny algoritmus (navraci: best X, best Y, fitness history)
        """
        self.repeat_count = repeat_count
        self.name = name
        self.callback = callback
        self.res = WorkerResults()
        self.args = args

    def runWorker(self, queue):
        self.res.total_time = 0.0
        queue.put(f"Worker [{self.name}] is running now!")

        try:
            for iteration in range(self.repeat_count):

                # provedeni algoritmu
                start_time = time.time()
                bestX, bestY, history = self.callback(self.args.D, self.args.func, self.args.bounds)
                end_time = time.time()
                
                # vypocet casu a zbyvajiciho casu
                duration = end_time - start_time
                self.res.total_time += duration
                remaining = (self.repeat_count - iteration - 1) * \
                    duration * 1.05
                
                # vypis progresu (zapise do fronty zprav)
                queue.put(f"Worker [{self.name}] progress {iteration + 1}/{self.repeat_count} [Time: {duration:.1f} s | Remaining: {remaining:.1f} s]")

                # zapis do vysledku
                self.res.xBestList.append(bestX)
                self.res.yBestList.append(bestY)
                self.res.historyList.append(history)

        except Exception as e:
            self._status = f"Error: {str(e)}"

    def getResults(self) -> WorkerResults:
        return self.res


def workerMethod(worker: Worker, queue):
    """
    Hlavni metoda pro spousteni workeru
    """
    worker.runWorker(queue)
    return worker.getResults()


class AlgorithmStatistics:
    def __init__(self, repeat_count: int):
        """
        Vytvori tridu pro statisticke vyhodnoceni algoritmu

        Parametry:
            repeat_count - Pocet opakovani jednotlivych algoritmu
        """
        self.repeat_count = repeat_count
        self.workers = []

    def getWorkers(self) -> [Worker]:
        return self.workers

    def addWorker(self, name: str, callback, args: Args):
        """
        Prida worker do statisticke tridy

        Parametry:
            name - Nazev algoritmu/workeru (musi byt unikatni)
            calback - Callback ktery se zavola vzdy a provede pozadovane vykonani nejakeho evolucniho algoritmu (navraci: best X, best Y, fitness history)
        """
        if name == None:
            raise RuntimeError("Name is not defined")
        if self.findWorkerByName(name) != None:
            raise RuntimeError("Name is not unique")
        if callback == None:
            raise RuntimeError("Callback is not defined")
        self.workers.append(
            Worker(repeat_count=self.repeat_count, name=name, callback=callback, args=args))

    def runAllWorkers(self):
        """
        Spusti statisticke testy pro vsechny pridane algoritmy
        """
        print("[Starting all workers]")

        with multiprocessing.Manager() as manager:
            # Vytvoření fronty pro sdílení dat
            queue = manager.Queue()

            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {executor.submit(workerMethod, worker, queue): worker for worker in self.getWorkers()}

                while any(not future.done() for future in futures):
                    while not queue.empty():
                        print(queue.get())
                    time.sleep(1)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        worker = futures[future]
                        worker.res = future.result()
                        print(f"Worker [{worker.name}] done")
                    except Exception as e:
                        print(f"Error: {e}")

        print("[All workers finished their work]")

    def findWorkerByName(self, name: str) -> Worker:
        """
        Najde worker podle jmena

        Parametry:
            name - Jmeno workeru
        """
        for w in self.getWorkers():
            if w.name == name:
                return w
        return None

    def showTime(self):
        """
        Graficky znazorni a porovnani casovych narocnosti jednotlivych algoritmu
        """
        worker_names = []
        worker_times = []

        for w in self.getWorkers():
            worker_names.append(w.name)
            worker_times.append(w.res.total_time)

        plt.figure(figsize=(15, 17))
        plt.barh(worker_names, worker_times, color='green')  # Vodorovný sloupcový graf
        plt.xlabel('Total time [s]')
        plt.title('Times')

        # Nastavení formátu osy x na jedno desetinné místo
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # Přidání hodnot k sloupcům
        for index, value in enumerate(worker_times):
            plt.text(value, index, str(round(value, 1)))

        plt.show()