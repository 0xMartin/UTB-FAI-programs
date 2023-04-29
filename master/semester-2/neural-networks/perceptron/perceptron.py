import random


class Perceptron:

    w_list: list = None

    bias_x: float = 0.0
    bias_w: float = 1.0

    output: float = 0.0

    def __init__(self, x_size) -> None:
        """
        x_size -> pocet vstupu
        """
        self.w_list = list()
        for _ in range(x_size):
            self.w_list.append(1.0)  
        print("Perceptron created - number of inputs:", x_size, ", ID:", self) 


    def predict(self, x_list) -> float:
        sum = self.bias_x * self.bias_w
        for i in range(len(x_list)):
            sum += x_list[i] * self.w_list[i]
        return 1.0 if (sum >= 0) else 0.0


    def train(self, x, t, c, epochs) -> list:  
        """
        x -> vstupni data
        y -> vystupni data
        c -> koeficient uceni
        epochs -> pocet epoch

        retunr: prubeh uceni v ramci jednotlivych epoch (globalni chyby)
        """        
        lerning = list()

        for _ in range(epochs):
            global_error = 0.0

            # uprava vah a zpracovani lokalnich chyb
            for k in range(len(x)):
                y = self.predict(x[k]) 
                if y != t[k]:
                    global_error += 1.0
                    # lokalni chyba -> uprava vah
                    for input_index in range(len(self.w_list)):
                        self.w_list[input_index] = self.w_list[input_index] + c * (t[k] - y) * x[k][input_index]    

            # zapis globalni chyby do historie
            lerning.append(global_error)

            # pokud je globalni chyba nulova ukonci proces uceni
            if global_error == 0.0:
                break

            # nahodne zamichani
            data = list(zip(x, t))
            random.shuffle(data)
            x, t = zip(*data)

        print("Lerning done. Number of epochs:", _)
        print("W =", list(map(str, self.w_list)))

        return lerning
