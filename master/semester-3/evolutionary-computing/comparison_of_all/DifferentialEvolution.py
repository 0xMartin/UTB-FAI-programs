import numpy as np


def rand_1_mutation(population, current_index, F):
    """
    Rand/1 mutacni strategie

    Parametry:
        population - populace    
        current_index - aktualni index, aktualne zvoleny jedinec
        F - mutacni konstanta
    """
    idlist = [idx for idx in range(len(population)) if idx != current_index]
    r1, r2, r3 = np.random.choice(idlist, 3, replace=False)
    return population[r1].X + F * (population[r2].X - population[r3].X)


def best_1_mutation(population, current_index, F, best_index):
    """
    Best/1 mutacni strategie

    Parametry:
        population - populace
        current_index - aktualni index, aktualne zvoleny jedinec
        F - mutacni konstanta
        best_index - index nejlepsiho jedince
    """
    idlist = [idx for idx in range(len(population)) if idx != current_index]
    r1, r2 = np.random.choice(idlist, 2, replace=False)
    return population[best_index].X + F * (population[r1].X - population[r2].X)


def current_to_rand_1_mutation(population, current_index, F):
    """
    Current-to-rand/1 mutacni strategie

    Parametry:
        population - populace
        current_index - aktualni index, aktualne zvoleny jedinec
        F - mutacni konstanta
    """
    r1, r2, r3 = np.random.choice(len(population), 3, replace=False)
    K = 0.5 * (F + 1)
    return population[current_index].X + K * (population[r1].X - population[current_index].X) + F * (population[r2].X - population[r3].X)


class DE_Individual:
    def __init__(self, X, F=0.5, CR=0.8, mutation_strategy_id=None):
        self.X = X
        self.F = F
        self.CR = CR
        self.mutation_strategy_id = mutation_strategy_id


# type evolucnich algoritmu
DE_TYPE_RAND = 0
DE_TYPE_BEST = 1
DE_TYPE_JDE = 2
DE_TYPE_EPSDE = 3


class DE_Population:

    def __init__(self, type, dimension, CR, F):
        """
        Vytvori tridu pro diferencialni evoluci

        Parametry:
            type - typy DE algoritmu (DE_TYPE_RAND, DE_TYPE_BEST, DE_TYPE_JDE, DE_TYPE_EPSDE)
            dimension - Dimenze reseni
            CR - prah krizeni
            F - mutacni konstanta
        """
        self.type = type
        self.dimension = dimension
        self.CR = CR
        self.F = F
        self.fitness_function = None
        self.population_size = 0
        self.mutation_strategy = [rand_1_mutation, best_1_mutation, current_to_rand_1_mutation]

    def create_population(self, population_size, bounds):
        """
        Vytvori populaci

        Parametry:
            population_size - velikost populace
            bounds - inicializacni rozsah populace   
        """
        self.population_size = population_size
        self.population = []
        for _ in range(self.population_size):
            self.population.append(
                DE_Individual(
                    X=np.random.uniform(bounds[0], bounds[1], size=(self.dimension)), 
                    F=self.F, 
                    CR=self.CR, 
                    mutation_strategy_id=np.random.randint(0, len(self.mutation_strategy))
                )
            )
        self.population = np.array(self.population)


    def set_fitness_function(self, func):
        """
        Priradi fitness funkci evolucnimu algoritmu

        Parametry:
            func - fitness funkce  
        """
        self.fitness_function = func

    def calculate_population_statistics(self):
        """
        Vypocita statistiky pro aktuálni populaci

        Returns:
            min_value: Minimalni hodnota fitness v populaci
            max_value: Maximalni hodnota fitness v populaci
            avg_value: Průměrná hodnota fitness v populaci
            std_value: Smerodatna odchylka hodnot fitness v populaci
            median_value: Median hodnot fitness v populaci
        """
        fitness_values = [self.fitness_function(individual.X) for individual in self.population]
        min_value = np.min(fitness_values)
        max_value = np.max(fitness_values)
        avg_value = np.mean(fitness_values)
        std_value = np.std(fitness_values)
        median_value = np.median(fitness_values)
        return min_value, max_value, avg_value, std_value, median_value

    def run_evolution(self, generations):
        if self.fitness_function is None:
            print("Fitness function is not defined!!!!")
            return

        convergence_curve = []
        best_index = 0

        # pocatecni nejlepsi jedinec (index + fitness)
        best_index, best_fitness = self.__find_best()

        for _ in range(generations):
            new_population = []

            # generovani nove populace
            for i in range(self.population_size):
                # zpracovani jedne iterace evolucniho algoritmu v zavyslosti na jeho typu
                if self.type == DE_TYPE_EPSDE:
                    new_individual = self.__epsde_evolution(i, best_index)
                elif self.type == DE_TYPE_JDE:
                    new_individual = self.__jde_evolution(i)
                elif self.type == DE_TYPE_BEST:
                    new_individual = self.__best_evolution(i, best_index)
                else:
                    new_individual = self.__rand_evolution(i)

                # selekce 
                if self.fitness_function(new_individual.X) < self.fitness_function(self.population[i].X):
                    # nove vytvoreny jedinec
                    new_population.append(new_individual)
                else:
                    # nahodna reinicializace 
                    if self.type == DE_TYPE_EPSDE:
                        self.__reinit_epsde_F_CR_M(self.population[i])

                    # rodic (predchozi jedinec bez zmeny)
                    new_population.append(self.population[i])
            
            # predani nove populace
            self.population = np.array(new_population)

            # nalezeni nejlepsiho jedince (index + fitness)
            best_index, best_fitness = self.__find_best()

            # zapis do grafu
            convergence_curve.append(best_fitness)

        best_index, best_fitness = self.__find_best()
        return self.population[best_index].X, best_fitness, convergence_curve

    def __rand_evolution(self, current_index):
        """
        Metoda pro RAND1
        """
        # aplikace mutacni strategie (na indexu 0 je rand/1 mutacni strategie)
        mutation_vector = self.mutation_strategy[0](self.population, current_index, self.F)
        # krizeni
        new_individual = DE_Individual(
            X=self.__binomial_crossover(self.population[current_index].X, mutation_vector, self.CR)
        )
        return new_individual

    def __best_evolution(self, current_index, best_index):
        """
        Metoda pro BEST/1
        """
        # aplikace mutacni strategie (na indexu 1 je best/1 mutacni strategie)
        mutation_vector = self.mutation_strategy[1](self.population, current_index, self.F, best_index)
        # krizeni
        new_individual = DE_Individual(
            X=self.__binomial_crossover(self.population[current_index].X, mutation_vector, self.CR)
        )
        return new_individual

    def __jde_evolution(self, current_index):
        """
        Metoda pro jDE
        """
        # re-inicializace hodnot parametru CR a F s pravděpodobnosti 0.1
        current_cr = self.population[current_index].CR
        current_f = self.population[current_index].F
        if np.random.rand() < 0.1:
            current_cr = np.random.uniform(0, 1)
        if np.random.rand() < 0.1:
            current_f = np.random.uniform(0, 1)

        # CR a F je pouzite to ktere ma jedinec zakodovano v sobe
        # aplikace mutacni strategie
        mutation_vector = self.mutation_strategy[0](self.population, current_index, current_f)
        # krizeni
        new_individual = DE_Individual(
            X=self.__binomial_crossover(self.population[current_index].X, mutation_vector, current_cr),
            F=current_f,
            CR=current_cr
        )

        return new_individual


    def __epsde_evolution(self, current_index, best_index):
        """
        Metoda pro EPSDE
        """

        # {strategie, F, CR} aktualniho jedince
        current_cr = self.population[current_index].CR
        current_f = self.population[current_index].F
        ms_func_id = self.population[current_index].mutation_strategy_id
        ms_func = self.mutation_strategy[int(ms_func_id)]

        # aplikace mutacni strategie
        if ms_func == best_1_mutation:
            mutation_vector = ms_func(self.population, current_index, current_f, best_index)
        else:
            mutation_vector = ms_func(self.population, current_index, current_f)
            
        # krizeni
        new_individual = DE_Individual(
            X=self.__binomial_crossover(self.population[current_index].X, mutation_vector, current_cr),
            F=current_f,
            CR=current_cr,
            mutation_strategy_id=ms_func_id
        )

        return new_individual
    
    def __reinit_epsde_F_CR_M(self, new_individual):
        """
        Nahodna reinicialize F, CR a vybrane mutacni strategie
        """
        new_individual.F = np.random.choice([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        new_individual.CR = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        new_individual.mutation_strategy = np.random.randint(0, len(self.mutation_strategy))

    def __binomial_crossover(self, target_individual, mutant, crossover_rate):
        """
        Binarni krizeni X parametru jedincu

        Parametry:
            target - X vektor aktivniho jedinece (Pouze X, nejde o Individual!!!)
            mutant - mutacni vektor (Pouze X, nejde o Individual!!!)
            crossover_rate - prah krizeni
        """
        crossover_mask = np.random.rand(len(target_individual)) < crossover_rate
        return np.where(crossover_mask, mutant, target_individual)
    
    def __find_best(self):
        """
        V populaci najde jedince s nejlepsi hodnotou fitness funkce a navrati jeho index a hodnotu fitness
        """
        best_fitness = self.fitness_function(self.population[0].X)
        best_index = 0
        for i, individual in enumerate(self.population):
            current_fitness = self.fitness_function(individual.X)
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_index = i
        return best_index, best_fitness