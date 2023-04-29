import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================================
# DEFINICE TESTOVACICH FUNKCI
# ===========================================================================================


def sphere(x: list) -> float:
    x = np.array(x)
    return np.sum(x**2)


def rosenbrock(x: list) -> float:
    x = np.array(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin(x: list) -> float:
    x = np.array(x)
    d = x.shape[0]
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley(x: list) -> float:
    x = np.array(x)
    d = x.shape[0]
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq/d)) - np.exp(sum_cos/d) + 20 + np.exp(1)


def boothFunction(x: list) -> float:
    x = np.array(x)
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


# ===========================================================================================
# OPTIMALIZACNI ALGORITMY
# ===========================================================================================
def shc(function,
        dim: int, x_min: float, x_max: float,
        max_iter: int,
        neighbor_size: float,
        num_neighbors: int) -> tuple([list, float, np.array]):
    """
    Implementace Stochastic Hill Climber

    Paramtery:
        dim           - velikost dimenze (pocet paramteru funkce)
        x_min         - minimalni hodnota parametru
        x_max         - maximalni hodnota parametru
        neighbor_size - velikost sousedstvi
        num_neighbors - pocet sousedu
    """
    # historie
    history = []

    # nahodna inicializace
    x_best = np.random.uniform(-5, 5, dim)
    y_best = function(x_best)
    history.append([x_best, y_best])

    # Iterativní vylepšování řešení
    for _ in range(max_iter):
        # vygeneruje nahodneho souseda
        x_new = x_best + np.random.normal(0, neighbor_size, dim)
        x_new = np.clip(x_new, x_min, x_max)
        y_new = function(x_new)

        # pokud je reseni lepsi zaznamena si ho jako nejlepsi reseni
        if y_new < y_best:
            x_best = x_new
            y_best = y_new

        # zaznamenani nejlepsi hodnoty do historie
        history.append([x_best, y_best])

    return tuple([x_best, y_best, history])


def ls(function,
       dim: int, x_min: float, x_max: float,
       max_iter: int,
       neighbor_size: float,
       num_neighbors: int) -> tuple([list, float, np.array]):
    """
    Implementace Local Search

    Paramtery:
        dim           - velikost dimenze (pocet paramteru funkce)
        x_min         - minimalni hodnota parametru
        x_max         - maximalni hodnota parametru
        neighbor_size - velikost sousedstvi
        num_neighbors - pocet sousedu
    """
    # historie
    history = []

    # nahodna inicializace
    x_best = np.random.uniform(x_min, x_max, dim)
    y_best = function(x_best)
    history.append([x_best, y_best])

    # iterace => hledani lepsiho reseni
    for _ in range(max_iter):
        # vygenerovani nahodnych sousedu
        neighbors = x_best + \
            np.random.normal(0, neighbor_size, size=(num_neighbors, dim))
        neighbors = np.clip(neighbors, x_min, x_max)
        ys = [function(x) for x in neighbors]

        # nejde nejlepsiho souseda
        idx_best = np.argmin(ys)
        y_new = ys[idx_best]
        x_new = neighbors[idx_best]

        # pokud je reseni lepsi zaznamena si ho jako nejlepsi reseni
        if y_new < y_best:
            x_best = x_new
            y_best = y_new

        # zaznamenani nejlepsi hodnoty do historie
        history.append([x_best, y_best])

    return tuple([x_best, y_best, history])


# ===========================================================================================
# POMOCNE FUNKCE
# ===========================================================================================
def plotFunctionWithPts(func,
                        points: np.array,
                        pt_best: list,
                        x_start: float,
                        x_end: float):
    """
    Vizualizace funkce (pro 2 paramtery)

    Parametry:
        func     - reference na testovaci funkce
        points   - historie vyvoje nejlepsi nalezene hodnoty ucelove funkce (sklada se z bodu [x parametry, y funkci hodnota])
        pt_best  - x parametry pro nejlepsi nalezenou hodnotu ucelove funkce (zobrazeni v grafu zelenou teckou)
        x_start  - pocatecni hodnota pro pro osy (parametr x1 & x2)
        x_end    - koncova hodnota pro pro osy (parametr x1 & x2)
    """
    # vytvori grid
    x = np.linspace(x_start, x_end, 200)
    y = np.linspace(x_start, x_end, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # vypocte hodnoty pro kazdy bod gridu
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = func([X[i, j], Y[i, j]])

    # zobrazi graf funkce
    c = plt.contourf(X, Y, Z, levels=50, cmap='plasma')
    plt.colorbar(c, format='%.3f')

    # body
    pts_x, pts_y = [], []
    pt_start = points[0][0]
    for pt in points:
        pts_x.append(pt[0][0])
        pts_y.append(pt[0][1])
    plt.scatter(pts_x, pts_y, color='black')
    plt.scatter(pt_start[0], pt_start[1], color='red')
    plt.scatter(pt_best[0], pt_best[1], color='green')

    plt.show()


def plot1DFunctionWithPts(func,
                          points: np.array,
                          pt_best: list,
                          x_start: float,
                          x_end: float):
    """
    Vizualizace funkce (pro 1 paramter)

    Parametry:
        func     - reference na testovaci funkce
        points   - historie vyvoje nejlepsi nalezene hodnoty ucelove funkce (sklada se z bodu [x parametry, y funkci hodnota])
        pt_best  - x parametry pro nejlepsi nalezenou hodnotu ucelove funkce (zobrazeni v grafu zelenou teckou)
        x_start  - pocatecni hodnota pro pro osy (parametr x1 & x2)
        x_end    - koncova hodnota pro pro osy (parametr x1 & x2)
    """
    # zobrazi graf funkce
    x = np.linspace(x_start, x_end, 200)
    y = [ func(_x) for _x in x.tolist() ]
    plt.plot(x, y)

    # body
    pts_x, pts_y = [], []
    pt_start = points[0]
    for pt in points:
        pts_x.append(pt[0])
        pts_y.append(pt[1])
    plt.scatter(pts_x, pts_y, color='black')
    plt.scatter(pt_start[0], pt_start[1], color='red')
    plt.scatter(pt_best, func(pt_best), color='green')

    plt.show()



def plotHistory(points: np.array):
    """
    Vykresli graf historie vyvoje nejlepsi nalezene hodnoty ucelove funkce

    Parametry:
        points - historie vyvoje nejlepsi nalezene hodnoty ucelove funkce (sklada se z bodu [x parametry, y funkci hodnota])
    """
    x_list = []
    y_list = []
    for i, pt in enumerate(points):
        x_list.append(i)
        y_list.append(pt[1])
    plt.plot(x_list, y_list, '-o')
    plt.xlabel("iterace")
    plt.ylabel("f(x)")
    plt.show()


# ===========================================================================================
# MAIN
# ===========================================================================================
if __name__ == "__main__":
    # nastaveni parametru

    # vyber funkce
    _func = sphere
    val = input(
        "Vyber funkci (1):\n1 - sphere\n2 - rosenbrock\n3 - rastrigin\n4 - ackley\n5 - boothFunction\n")
    if len(val) != 0:
        val = int(val)
        if val == 1:
            _func = sphere
        elif val == 2:
            _func = rosenbrock
        elif val == 3:
            _func = rastrigin
        elif val == 4:
            _func = ackley
        elif val == 5:
            _func = boothFunction

    # vyber algoritmu
    alg = "Local Search"
    val = input(
        "Vyber optimalizacniho algoritmu (1):\n1 - Local Search\n2 - Stochastic Hill Climber\n")
    if len(val) != 0:
        val = int(val)
        if val == 1:
            alg = "Local Search"
        elif val == 2:
            alg = "Stochastic Hill Climber"

    # pocet iteraci
    max_iter = 30
    val = input("Pocet iteraci (30):")
    if len(val) != 0:
        max_iter = int(val)

    # dimenze
    dim = 2
    val = input("Dimenze (2):")
    if len(val) != 0:
        dim = int(val)

    # pocet sousedu
    num_neighbors = 4
    val = input("Populace (4):")
    if len(val) != 0:
        num_neighbors = int(val)

    # maximalni odchylka mezi sousedy
    neighbor_size = 0.4
    val = input("Velikost sousedstvi (0.4):")
    if len(val) != 0:
        neighbor_size = float(val)

    # minimalni hodnata parametru
    x_min = -5.0
    val = input("Minimalni hodnoty vstupniho parametru funkce (-5.0):")
    if len(val) != 0:
        x_min = float(val)

    # minimalni hodnata parametru
    x_max = 5.0
    val = input("Maximalni hodnoty vstupniho parametru funkce (5.0):")
    if len(val) != 0:
        x_max = float(val)

    # << VYPOCET >>

    # uptimalizacni algoritmus
    if alg == "Local Search":
        x_best, y_best, history = ls(
            _func, dim, x_min, x_max, max_iter, neighbor_size, num_neighbors)
    else:
        x_best, y_best, history = shc(
            _func, dim, x_min, x_max, max_iter, neighbor_size, num_neighbors)

    # vypis nejlepsiho reseni
    print(alg)
    print("Best solution:")
    print("x_best:", x_best)
    print("y_best:", y_best)

    # vykresleni reseni
    plt.title(alg)
    if dim >= 2:
        plotFunctionWithPts(_func, history, x_best, x_min, x_max)
    else:
        plot1DFunctionWithPts(_func, history, x_best, x_min, x_max)

    # zobrazi historii vyvoje nejlepsi nalezene hodnoty ucelove funkce
    plt.title(alg)
    plotHistory(history)
