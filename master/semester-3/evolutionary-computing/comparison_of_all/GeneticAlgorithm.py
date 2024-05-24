import random
import matplotlib.pyplot as plt
import numpy as np
import struct

REPRESENTATION_TYPE_DEFAULT = 'default'
REPRESENTATION_TYPE_IEEE754 = 'IEEE754'
REPRESENTATION_TYPE_FIXED_POINT = 'FIXED_POINT'
REPRESENTATION_TYPE_BCD = 'BCD'


class BinRepresentation:
    @staticmethod
    def real_to_binary_ieee754(number: float) -> list:
        binary = format(struct.unpack(
            'I', struct.pack('f', number))[0], '032b')
        binary_list = [int(bit) for bit in binary]
        return binary_list

    @staticmethod
    def binary_to_real_ieee754(binary: list) -> float:
        binary_str = ''.join(str(bit) for bit in binary)
        integer_value = int(binary_str, 2)
        real_value = struct.unpack('f', struct.pack('I', integer_value))[0]
        return float(real_value)

    @staticmethod
    def real_to_binary_fixed_point(number: float) -> list:
        sign_bit = 1 if number < 0 else 0
        number = abs(number)
        int_part = int(number)
        frac_part = int((number - int_part) * (2 ** 16))
        int_bin = format(int_part, '015b')  # 15 bitů pro celou část
        frac_bin = format(frac_part, '017b')  # 17 bitů pro desetinnou část
        bit_list = [int(sign_bit)] + [int(bit)
                                      for bit in int_bin] + [int(bit) for bit in frac_bin]
        return bit_list

    @staticmethod
    def binary_to_real_fixed_point(binary: list) -> float:
        # Znaménko: 1 pro záporné, 0 pro kladné číslo
        sign = -1 if binary[0] else 1
        int_part = int(''.join(str(bit) for bit in binary[1:16]), 2)
        frac_part = int(''.join(str(bit)
                        for bit in binary[16:]), 2) / (2 ** 16)
        return sign * (int_part + frac_part)

    @staticmethod
    def real_to_binary_bcd(number: float) -> list:
        binary = []
        # cislice v bcd
        int_part = int(abs(number))
        decimal_part = (abs(number) - int_part) * 1e5
        total = int(int_part * 1e5 + decimal_part)
        for _ in range(8):
            digit = total % 10
            for bit in reversed(format(int(digit), '04b')):
                binary.append(int(bit))
            total //= 10
        binary.reverse()
        # znameko cisla
        binary[0] = int(number > 0)
        return binary

    @staticmethod
    def binary_to_real_bcd(bcd_list):
        bcd_digits = [bcd_list[i:i + 4] for i in range(0, len(bcd_list), 4)]
        bcd_digits[0] = bcd_digits[0][1:]
        # cislice v bcd
        result = 0
        for digit in bcd_digits:
            digit_str = ''.join(str(bit) for bit in digit)
            result = result * 10 + int(digit_str, 2)
        # znameko cisla
        if bcd_list[0] == 0:
            result *= -1
        return float(result / 1e5)


class Bin_Individual:
    def __init__(self, representation_type: str, chromosomes: list[list], fitness_function):
        """
        Vytvori jedince populace

        Parametry:
            representation_type - typ reprezentace chromozomu
            chromosomes - chromozom jedince (v tomto pripade binarni retezec) multi-D 
            fitness_function - reference na ucelovou funkci pro hodnoceni kvality jedince
        """
        self.fitness_function = fitness_function
        self.fitness = 0.0

        self.representation_type = representation_type
        self.chromosomes = chromosomes
        self.bounds = None

    def set_bounds(self, bounds: list):
        """
        Nastaveni mezi chromozomu
        """
        self.bounds = bounds

    def get_representation_type(self):
        """
        Navrati typ reprezentace chromozomu (jeho format)
        """
        return self.representation_type

    def get_chromosome(self):
        """
        Navrati chromozom jedince
        """
        return self.chromosomes

    def get_fitness(self):
        """
        Navrati hodnotu kvality uzivatele
        """
        return self.fitness

    def set_chromosome_value(self, value, index):
        """
        Nastavi hodnotu chromozomu (pokud ma jedinec zvolenou nejakou
        reprezentaci, je binarni chromozom vypocitan podle transformacni 
        funkce dane reprezentace binarniho chromozomu)
        """
        if self.representation_type == REPRESENTATION_TYPE_IEEE754:
            self.chromosomes[index] = BinRepresentation.real_to_binary_ieee754(
                value)

        elif self.representation_type == REPRESENTATION_TYPE_FIXED_POINT:
            self.chromosomes[index] = BinRepresentation.real_to_binary_fixed_point(
                value)
        elif self.representation_type == REPRESENTATION_TYPE_BCD:
            self.chromosomes[index] = BinRepresentation.real_to_binary_bcd(
                value)
        else:
            self.chromosomes[index] = (value)

    def get_value_of_chromosome(self):
        """
        Navrati hodnotu chromozomu po aplikaci prevodni trasformace (puvodni binarni 
        chromozom bude navracen jen v pripade ze zvoleny typ reprezentace je 'default')
        """
        x_values = []

        for chromosome in self.chromosomes:
            if self.representation_type == REPRESENTATION_TYPE_IEEE754:
                x_values.append(
                    BinRepresentation.binary_to_real_ieee754(chromosome))
            elif self.representation_type == REPRESENTATION_TYPE_FIXED_POINT:
                x_values.append(BinRepresentation.binary_to_real_fixed_point(
                    chromosome))
            elif self.representation_type == REPRESENTATION_TYPE_BCD:
                x_values.append(
                    BinRepresentation.binary_to_real_bcd(chromosome))
            else:
                x_values.append(chromosome)

        return x_values

    def evaluate_fitness(self):
        """
        Ohodnoti kvalitu jedince (podte typu reprezentace chromozomu)
        """
        x_values = np.array(self.get_value_of_chromosome())
        self.fitness = self.fitness_function(x_values)
        self.check_bounds()

    def check_bounds(self):
        """
        Overi hranice chromozomu a podu je prekrocena tak ji nastavi na limitni hodnotu 
        (aktivni jen pokud nejde o 'default' reprezentaci)
        """
        x_values = self.get_value_of_chromosome()
        for i, x_value in enumerate(x_values):
            if self.representation_type != REPRESENTATION_TYPE_DEFAULT:
                if self.bounds is not None:
                    if x_value < self.bounds[0]:
                        self.set_chromosome_value(self.bounds[0], i)
                        self.fitness = self.fitness_function(
                            np.array(self.get_value_of_chromosome()))
                    elif x_value > self.bounds[1]:
                        self.set_chromosome_value(self.bounds[1], i)
                        self.fitness = self.fitness_function(
                            np.array(self.get_value_of_chromosome()))

    def mutate(self, probability: float):
        """
        Mutace, zmutuju chromozom jedince

        Paramtery:
            x - Pole bitu 0/1
            probability - Pravdepodobnost mutace jednoho bitu retezce
        """
        for i, chromosome in enumerate(self.chromosomes):
            mutated = []
            for bit in chromosome:
                if random.random() < probability:
                    mutated.append(1 - bit)
                else:
                    mutated.append(bit)
            self.chromosomes[i] = mutated

    def crossover(self, other):
        """
        Jednobodove krizeni

        Paramtery:
            other - Reference na druheho rodice "jedince populace"

        Return: Potomek 1, Potomek 2
        """
        rep_type = other.get_representation_type()

        # reprezentace typy obou rodicu musi byt shodne (pripad ze by se neshodovali nenastave pokud jsou v populaci vsichni jedinci stejneho typu reprezentace)
        if rep_type != self.representation_type:
            return None

        chromosome1 = []
        chromosome2 = []
        for i in range(len(self.chromosomes)):
            point = random.randint(1, len(self.chromosomes[i]) - 1)
            chromosome1.append(
                self.chromosomes[i][:point] + other.get_chromosome()[i][point:])
            chromosome2.append(other.get_chromosome()[
                               i][:point] + self.chromosomes[i][point:])

        child1 = Bin_Individual(rep_type, chromosome1, self.fitness_function)
        child2 = Bin_Individual(rep_type, chromosome2, self.fitness_function)
        child1.set_bounds(self.bounds)
        child2.set_bounds(self.bounds)
        return child1, child2


class Bin_Population:
    def __init__(self, individuals: list[Bin_Individual]) -> None:
        """
        Vytvori populaci jedincu

        Paramtery:
            individuals - inicializacni list s jedinci populace
        """
        self.individuals = individuals
        pass

    def random_init(self, population_size: int, chromosome_lengths: int, Dim: int, fitness_function, representation_type: str = "default", bounds: list = [-1, 1]):
        """
        Nahodna inicializace populace

        Parametry:
            population_size - Velikost populace
            chromosome_lengths - Delka chromozomu jedince  
            fitness_function - Reference na ucelovou funkci hodnotici kvalitu jedincu v populaci
        """
        self.Dim = Dim
        self.individuals.clear()

        for _ in range(population_size):

            # nahodne generovani hodnoty chromozomu
            chromosomes = []

            for _ in range(self.Dim):
                if representation_type == 'ieee754':
                    chromosomes.append(BinRepresentation.real_to_binary_ieee754(
                        random.uniform(bounds[0], bounds[1])))
                elif representation_type == 'fixed_point':
                    chromosomes.append(BinRepresentation.real_to_binary_fixed_point(
                        random.uniform(bounds[0], bounds[1])))
                elif representation_type == 'bcd':
                    chromosomes.append(BinRepresentation.real_to_binary_bcd(
                        random.uniform(bounds[0], bounds[1])))
                else:
                    chromosomes.append([random.randint(0, 1)
                                        for _ in range(chromosome_lengths)])

            individual = Bin_Individual(
                representation_type, chromosomes, fitness_function)
            individual.set_bounds(bounds)
            self.individuals.append(individual)

    def add_individual(self, individual: Bin_Individual):
        """
        Prida jedince do populace

        Paramtery:
            individual - Jedinec populace

        Return: 
        """
        self.individuals.append(individual)

    def select_roulette(self) -> Bin_Individual:
        """
        Rulotova selekce

        Return: Vybrany potomek populace
        """
        total_fitness = sum(i.fitness for i in self.individuals)
        if total_fitness == 0:
            return None

        r = random.uniform(0, total_fitness)
        current_fitness = 0

        for i in self.individuals:
            current_fitness += i.get_fitness()
            if current_fitness >= r:
                return i
        return None

    def select_rank(self) -> Bin_Individual:
        """
        Poradova selekce

        Return: Vybrany potomek populace
        """
        ranks = list(range(1, len(self.individuals) + 1))
        total_rank = sum(ranks)
        probabilities = [rank / total_rank for rank in ranks]
        selected_index = random.choices(
            range(len(self.individuals)), probabilities)[0]
        return self.individuals[selected_index]

    def evaluate_population(self):
        """
        Ohodnoti celou populaci jedincu
        """
        for i in self.individuals:
            i.evaluate_fitness()

    def calculate_stats(self) -> tuple([float, float, float, float, float]):
        """
        Vypocet statistickych udaju populace

        Return: min. hodnota fitness, max. hodnota fitness, prumer, median, smerodatna odchyka
        """
        self.evaluate_population()
        fitness_values = [i.get_fitness() for i in self.individuals]
        if not fitness_values:
            return None, None, None, None

        fitness_values = np.array(fitness_values)
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        mean_fitness = np.mean(fitness_values)
        median_fitness = np.median(fitness_values)
        std_deviation_fitness = np.std(fitness_values)

        return min_fitness, max_fitness, mean_fitness, median_fitness, std_deviation_fitness


class GeneticAlgorithm:
    def __init__(self, population: Bin_Population, elitism_ratio: float, selection_type: str, mutation_probability: float, num_generations: int):
        self.population = population
        self.elitism_ratio = elitism_ratio
        self.selection_type = selection_type
        self.mutation_probability = mutation_probability
        self.num_generations = num_generations

    def __select_individual(self):
        """
        Vyber jedince podle zvoleneho typu vyberu (roulette nebo rank)

        Return: Vybrany jedinec
        """
        if self.selection_type == 'roulette':
            return self.population.select_roulette()
        elif self.selection_type == 'rank':
            return self.population.select_rank()
        else:
            raise ValueError("Undefined selector type")

    def __evolve_population(self):
        """
        Vytvori novou generaci potomku a ta nahradi tu aktualni
        """
        new_population = list()

        elite_count = int(self.elitism_ratio *
                          len(self.population.individuals))
        elites = sorted(self.population.individuals,
                        key=lambda x: x.get_fitness())[:elite_count]
        new_population.extend(elites)

        while len(new_population) < len(self.population.individuals):
            parent1 = self.__select_individual()
            parent2 = self.__select_individual()
            if parent1 is None or parent2 is None:
                continue
            child1, child2 = parent1.crossover(parent2)
            child1.mutate(self.mutation_probability)
            child2.mutate(self.mutation_probability)
            new_population.extend([child1, child2])

        for i in new_population:
            i.check_bounds()

        self.population.individuals = new_population

    def getBest(self, ignore_x = False):
        best_x = []
        best_y = float('inf')
        for individual in self.population.individuals:
            f = individual.get_fitness()
            if f < best_y:
                best_y = f
                if not ignore_x:
                    best_x = individual.get_value_of_chromosome()

        return best_x, best_y

    def run(self) -> list[float]:
        """
        Spusti geneticky algoritmus

        Return: Prubeh hodnoty ucelove funkce nejlepsiho jedince v ramci vsech chromosomeraci
        """
        fitness_history = []
        for _ in range(self.num_generations):
            self.population.evaluate_population()
            # nejlepsi hodna fitness v populaci
            _, best_fitness = self.getBest(True)
            fitness_history.append(best_fitness)
            # vyvinuti populace
            self.__evolve_population()

        # ziskani nejlepsiho reseni
        self.population.evaluate_population()
        best_x, best_y = self.getBest()
        return best_x, best_y, fitness_history


def sphere_function(x):
    return np.sum(x**2)


def rastrigins_function(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    sum_term = 0
    for i in range(len(x) - 1):
        sum_term += 100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2
    return sum_term



#pop = Bin_Population([])
#ga = GeneticAlgorithm(pop, 0.1, 'rank', 0.01, 100) 
#pop.random_init(20, 32, 10, rastrigins_function,REPRESENTATION_TYPE_BCD, [-10.0, 10.0])
#x, y, h = ga.run()
#print(x, y)
#print(h)
