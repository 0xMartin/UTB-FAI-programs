import pandas as pd

from neural_network import *
from util import *

def testNetwork(inputs: pd.DataFrame,
                outputs: pd.DataFrame,
                epochs: int):
    """
    Otestune neuronovou sit na predenem datasetu

    Parametry:
        inputs - Vstupni hodnoty neuronove site datasetu
        outputs - Vystupni hodnoty neuronove site datasetu    
    """

    # vytvori neuronovou sit (2 vstupy, 2 skryta, 1 vystup)
    nn = NeuralNetwork(2, 2, 1)
    # nauci neuronovou sit na datasetu
    global_err_history = nn.train(inputs, outputs, 0.5, 0.1, epochs)

    # zobrazi vyhy
    nn.printW()

    # zobrazi graf vyvoje globalni chyby
    showLerningGraph(global_err_history)

    # zobrazi 3D funkce neuronove site
    showNeuralNetwork3DFunction(nn, inputs[:, 0], inputs[:, 1])

if __name__ == "__main__":
    print("XOR")
    df = pd.read_csv("xor.csv")
    testNetwork(df.drop("Y", axis=1).values, df["Y"].values, 4000)

    print("DATA")
    df = pd.read_csv("data.csv")
    testNetwork(df.drop("Y", axis=1).values, df["Y"].values, 500)
