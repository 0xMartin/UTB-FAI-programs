import numpy as np
import matplotlib.pyplot as plt


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


def randomSearch(
        func, 
        dim: int, 
        maxFES: int) -> tuple([np.array, float, list]): 
    """
    Random Search algoritmus
        func -> reference na optimalizovanou funkci
        dim -> dimenze ve ktere hleda reseni
        maxFES -> maximali pocet ohodnoceni ucelove funkce
    """
    # prni ohodnoceni ucelove funkce (nastavi na best)
    x_best = np.random.uniform(-5, 5, dim)
    f_best = func(x_best)
    f_evals = [f_best]

    # iterativni generovani nahodnych vstupnich parametru a vypocet ucelove funkce
    for _ in range(maxFES - 1):
        x_rand = np.random.uniform(-5, 5, dim)
        f_rand = func(x_rand)
        # pokud je reseni lepsi ...
        if f_rand < f_best:
            x_best = x_rand
            f_best = f_rand
        f_evals.append(f_best)

    return tuple([x_best, f_best, f_evals])


def plotFunction(func, 
                 tree_dim: bool,
                 title: str,
                 x1_start: float, 
                 x1_end: float, 
                 x2_start: float, 
                 x2_end: float):
    """
    Vizualizace funkce (pro 2 paramtery)
        func     - reference na testovaci funkce
        tree_d   - true: 3D zobrazeni / false: 2D zobrazeni
        x1_start - pocatecni hodnota parametru 1 (x1)
        x1_end   - koncova hodnota parametru 1 (x1)
        x2_start - pocatecni hodnota parametru 2 (x2)
        x2_end   - koncova hodnota parametru 2 (x2)
    """
    # vytvori grid
    x = np.linspace(x1_start, x1_end, 200)
    y = np.linspace(x2_start, x2_end, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # vypocte hodnoty pro kazdy bod gridu
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = func([X[i, j], Y[i, j]])

    # zobrazi graf funkce
    if tree_dim:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.plot_surface(X, Y, Z)
    else:
        plt.title(title)
        plt.contourf(X, Y, Z, levels=50, cmap='plasma')
        plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # vizualizace testovacich funkci
    plotFunction(sphere, True, "Sphere", -2, 2, -2, 2)
    plotFunction(rosenbrock, False, "Rosenbrock", -2, 2, -1, 3)
    plotFunction(rastrigin, False, "Rastrigin", -4.5, 4.5, -4.5, 4.5)
    plotFunction(ackley, False, "Ackley", -4.5, 4.5, -4.5, 4.5)
    plotFunction(boothFunction, False, "Booth Function", -10, 10, -10, 10)

    # randomSearch pro Sphere
    dims = [5, 10, 20]
    for dim in dims:
        plt.figure(figsize=(12, 10))
        
        # volani randomSearch
        x_best, f_best, f_evals = randomSearch(sphere, dim, 10000 * dim)

        # zobrazi konvergencni graf 
        x_best_str = ""
        for i, x in enumerate(x_best):
            if i != 0:
                x_best_str += ","    
            if i % 5 == 0 and i != 0:
                x_best_str += '\n'
            x_best_str += str(x)
        plt.title("%s in %dD\nBest solution: %.8f in\n [%s]" % ("Sphere", dim, f_best, x_best_str))
        plt.xlabel("FES")
        plt.ylabel("f(x)")
        plt.plot(f_evals, label="Sphere")
        plt.show()