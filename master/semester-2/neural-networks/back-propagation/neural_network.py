import numpy as np
import random

class Neuron:
    def __init__(self, inputs) -> None:
        """
        Vytvori neuron s pozadovanym poctem vstupu

        Parametry:
            inputs - pocet vstupu (pocet vah = pocet vstupu + 1 -> 1 = bias)
        """
        self.X = np.array([])
        self.Y: float = 0.0
        self.W = np.random.randn(inputs + 1)
        self.connected_neurons: list = []
        self.delta = 0.0
        self.deltaW = np.zeros_like(self.W)

    def connect(self, neurons: list):
        """
        Pripoji neuron k tomu neuronu

        Parametry:
            neuron - reference na neuron
        """
        for neuron in neurons:
            self.connected_neurons.append(neuron)

    def update(self):
        """
        Obnoveni hodnoty na vystupu. Zavisi na vystupnich hodnotach pripojenych neuronu.
        """
        a = 0.0
        # neurony
        self.X = []
        for i, neuron in enumerate(self.connected_neurons):
            self.X.append(neuron.Y)
            a += neuron.Y * self.W[i]
        self.X.append(1.0)
        self.X = np.array(self.X)
        # bias
        a += 1.0 * self.W[-1]
        self.Y = self.__sigmoid(a)

    def __sigmoid(self, x) -> float:
        """
        Logisticka sigmoida

        Parametry:
            x - Hodnota aktivacni funkce
        """
        return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Vytvori neuronovou sit

        Parametry:
            input_size - pocet vstupu neuronove site
            hidden_size - pocet neuronu na skryte vrstve
            output_size - pocet vystupu neuronove site
        """
        self.i_layer: list[Neuron] = []
        self.h_layer: list[Neuron] = []
        self.o_layer: list[Neuron] = []

        # sestaveni site neuronu podle parametru
        # input layer (neurone jen pro primi prenos signalu, nepocita se prenosova funkce)
        for _ in range(input_size):
            self.i_layer.append(Neuron(1))
        # hidden layer
        for _ in range(hidden_size):
            n = Neuron(input_size)
            n.connect(self.i_layer)
            self.h_layer.append(n)
        # output layer
        for _ in range(output_size):
            n = Neuron(hidden_size)
            n.connect(self.h_layer)
            self.o_layer.append(n)
        print("Input: %d, Hidden: %d, Output: %d" %
              (len(self.i_layer), len(self.h_layer), len(self.o_layer)))

    def predict(self, X: list) -> list:
        """
        Predukuje vystup dle naucenych vah site

        Parametry:
            X - Vstupni hodnoty neuronove site
        """
        # preda hodnoty do site
        for i, input_neuron in enumerate(self.i_layer):
            input_neuron.Y = X[i]
        # update cele site
        for n in self.h_layer:
            n.update()
        for n in self.o_layer:
            n.update()
        return [n.Y for n in self.o_layer]

    def train(self, X, T, learning_rate=0.5, momentum=0.1, epochs=100):
        """
        Nauci neuronovou sit s vyuzitim algoritmu back propagation

        Parametry:
            X - Vstupy neuronove site
            T - Pozadovany vystup neuronove site
            learning_rate - koeficient uceni neuronove site  
            momentum - setrvacnost prirustku vah 
        """
        global_err_history = []

        for _ in range(epochs):
   

            # globalni chyba
            error = 0

            for i in range(len(X)):
                # predikce
                y = self.predict(X[i])
                #print(X[i], "->", y, T[i])
                error += sum((T[i] - y) ** 2)

                # vystupni vrstva (vypocet delta W)
                for n in self.o_layer:
                    # delta
                    n.delta = (T[i] - n.Y) * n.Y * (1 - n.Y)
                    # prirustky vah pro neurony
                    n.deltaW = n.X * n.delta * learning_rate + momentum * n.deltaW

                # hidden vrstva (vypocet delta W)
                for i, n in enumerate(self.h_layer):
                    # delta * sum(T[i] - y)
                    s = 0.0
                    for n2 in self.o_layer:
                        s += n2.delta * n2.W[i]
                    n.delta = n.Y * (1 - n.Y) * s
                    # prirustky vah pro neurony
                    n.deltaW = n.X * n.delta * learning_rate + momentum * n.deltaW

                # uprava vaha pro vsechny neurony
                for n in self.h_layer:
                    n.W += n.deltaW
                for n in self.o_layer:
                    n.W += n.deltaW
            global_err_history.append(error)

        return global_err_history

    def printW(self):
        print("Hidden layer:")
        for i, n in enumerate(self.h_layer):
            w = ", ".join([str(i) for i in np.array(n.W)])
            print("%d - [%s]" % (i, w))
        print("Ouput layer:")
        for i, n in enumerate(self.o_layer):
            w = ", ".join([str(i) for i in np.array(n.W)])
            print("%d - [%s]" % (i, w))