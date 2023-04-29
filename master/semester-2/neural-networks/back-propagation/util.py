import matplotlib.pyplot as plt
import pandas as pd

from neural_network import *


def showLerningGraph(lerning: list):
    """
    Zobrazi graf vyvoje globalni chyby neuronove size

    Parametery:
        lerning - hodnoty vyvoje globalni chyby    
    """
    epochs = range(1, len(lerning) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lerning)
    plt.xlabel('Epocha')
    plt.ylabel('Globalni chyba')
    plt.show()


def showNeuralNetwork3DFunction(nn: NeuralNetwork, x1: list, x2: list):
    """
    Zobrazi funkce neuronove site (jen pro 3D)

    Parametery:
        nn - NeuralNetwork
    """
    # Vytvoření mřížky vstupních hodnot
    X1 = np.arange(min(x1), max(x1), 0.1)
    X2 = np.arange(min(x2), max(x2), 0.1)
    X1, X2 = np.meshgrid(X1, X2)

    # Aplikace neuronové sítě na vstupní hodnoty
    Y = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            Y[i, j] = nn.predict([X1[i, j], X2[i, j]])[0]
  
    # Vytvoření 3D grafu
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X1, X2, Y, cmap='coolwarm',
                        linewidth=0, antialiased=False)

    # Přidání os a popisků grafu
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('Funkce neuronové sítě')

    plt.show()
