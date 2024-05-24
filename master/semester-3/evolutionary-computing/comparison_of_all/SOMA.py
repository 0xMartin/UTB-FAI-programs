import numpy as np
import math
import random
import SharedUtils

class SOMA_Individual:
    def __init__(self, dim: int, range: SharedUtils.Range):
        """
        Trida pro jedince

        Parametry:
            dim - dimenze jedince
            range - rozsah hodnot reseni problemu
        """
        self.position = np.random.uniform(range.min, range.max, size=dim)
        self.value = float('inf')
        self.personal_best_pos = self.position.copy()
        self.personal_best_val = float('inf')
        self.range = range

    def get_position(self):
        """
        Navrati aktualni pozici jedince
        """
        return self.position
    
    def get_value(self):
        """
        Navrati aktualni hodnotu jedince
        """
        return self.value
    
    def get_personal_best_position(self):
        """
        Navrati nejlepsi nalezenou pozici jedince (pBest - position)
        """
        return self.personal_best_pos

    def get_personal_best_value(self):
        """
        Navrati nejlepsi nalezenou hodnotu jedince (pBest - fitness)
        """
        return self.personal_best_val

    def set_position(self, new_pos):
        """
        Nastavi jedinci novou pozici, nova pozice je hodnotou automaticky omezena zvolenym rozsahem.

        Parametry:
            position - nova pozice jedince
        """
        new_pos = np.clip(new_pos, self.range.min, self.range.max)
        self.position = new_pos

    def reset_personal_best(self):
        """
        Resetuje nejlepsi nalezenou pocizi
        """
        self.personal_best_pos = self.position
        self.personal_best_val = self.value

    def set_position_to_personal_best(self):
        """
        Aktualni pozici a hodnote priradi hodnoty "personal best" drive nejlepsi nalezene pozice
        """
        self.position = self.personal_best_pos
        self.value = self.personal_best_val   

    def update_fitness(self, obj_func):
        """
        Aktualizuje nejlepsi pozici jedince. Pokud je aktualni pozice lepe ohodnocena nez dosavadni nejlepsi pozice, tak je prepsana novou.

        Parametry:
            obj_func - funkce optimalizaniho problemu (urcuje kvalitu aktualni pozice)
        """
        self.value = obj_func(self.position)
        if self.value < self.personal_best_val:
            self.personal_best_val = self.value
            self.personal_best_pos = self.position.copy()

class SOMA_Population:
    def __init__(self, dim: int, nindividual: int, range: SharedUtils.Range) -> None:
        """
        Trida pro populaci jedincu

        Parametry:
            dim - dimenze castic
            nparticles - pocet jedincu
            range - rozsah hodnot reseni
            topology - topologie 
        """
        self.dim = dim
        self.nindividual = nindividual
        self.range = range
        self.propts = {}
        self.random_init(None)

    def setProperty(self, name: str, value):
        """
        Nastavi libovolnou konfigurace pro roj castic
        """
        self.propts[name] = value

    def random_init(self, obj_func):
        """
        Nahodna inicializace/reinicializace roje castic

        Parametry:
            obj_func - funkce optimalizovaneho problemu (urcuje kvalitu aktualni pozice)
        """
        self.global_best_individual = None
        self.individuals = [SOMA_Individual(self.dim, self.range)
                          for _ in range(self.nindividual)]
        if obj_func is not None:
            for individual in self.individuals:
                individual.update_fitness(obj_func)
                self.update_global_best_with_one_indivial(individual)

    def get_individuals(self):
        """
        Navrati vsechny jedince populace
        """
        return self.individuals

    def get_range(self):
        """
        Navrati rozsah hodnot reseni
        """
        return self.range

    def get_dim(self):
        """
        Navrati dimenzi reseneho problemu
        """
        return self.dim

    def get_global_best(self):
        """
        Navrati nejlepsiho jedince populace
        """
        return self.global_best_individual

    def update_global_best_with_one_indivial(self, individual: SOMA_Individual):
        """
        Obnovi novou globalne nejlepsi hodnotu

        Parametry:
            individual - jedinec jejiz personalni nejlepsi hodnota muze byt nastavena jako globalne nejlepsi
        """
        if self.global_best_individual == None:
            self.global_best_individual = individual   
        elif individual.get_value() < self.global_best_individual.get_value():
            self.global_best_individual = individual


MIGRATION_ALL_TO_ONE = 'all-to-one'
MIGRATION_ALL_TO_ALL = 'all-to-all'

class SOMA:
    def __init__(self, population: SOMA_Population):
        super().__init__()
        """
        Třída pro SOMA algoritmus

        Parametry:
            population - Reference na popilaci
        """
        # reference na populaci
        self.population = population
        self.evaluation_cnt = 0


    def set_obj_func(self, obj_func):
        """
        Priradi algoritmu optimalizovanou funkci

        Parametry:
            obj_func - Funkce ktera bude optimalizovana
        """
        self.obj_func = obj_func


    def get_population(self) -> SOMA_Population:
        """
        Navrati populaci
        """
        return self.population
    

    def __invokeResetCallback(self):
        for callback in self.reset_callback:
            callback(self)


    def __invokeIterationCallback(self):
        for callback in self.iteration_callback:
            gBest = self.population.get_global_best()
            callback(self, self.population.get_individuals(), gBest.get_position().copy(), gBest.get_value())

    def __generatePRTVector(self, prt: float, dim: int):
        # nahodna generace podle zakladniho pravidla
        prt_vector = np.array([1.0 if random.random() < prt else 0.0 for _ in range(self.population.get_dim())])
        # pokud je vektor cely 0, tak nahodne vlozi 1
        if np.sum(prt_vector) == 0.0:
            index_to_set_one = random.randint(0, len(prt_vector) - 1)
            prt_vector[index_to_set_one] = 1.0
        return prt_vector

    def __migrate_all_to_one(self, prt: float, step: float, path_len: float):
        """
        Provadi migraci vsech jedincu k nejlepsimu jedinci (All-To-One).

        Parametry:
            prt - prah pro mutaci (migrace), doporucene: 0.3
            step - velikost kroku, doporucene: 0.11, 0.22, 0.33, ..
            path_len - delke cesty (prozkoumavane oblasti), doporucene: 2.0 - 3.0
        """
        gBest = self.population.get_global_best()

        for individual in self.population.get_individuals():
            # napocita se pro leadera (nejlepsi aktualni reseni)
            if individual == gBest:
                continue

            # zalohuje si pocatecni pozici jedince + resetuje personal best
            base_position = individual.get_position().copy()
            individual.reset_personal_best()

            # vypocet ucelove funkce v nahodnem smeru ve vsech krocich
            t = step
            while t <= path_len:
                # nahodna volba smeru (perturbacni vektor)
                prt_vector = self.__generatePRTVector(prt, self.population.get_dim())
                # vypocet pozice
                individual.set_position(base_position + (gBest.get_position() - individual.get_position()) * t * prt_vector)
                individual.update_fitness(self.obj_func)
                self.evaluation_cnt += 1 # pro diagnostiku
                t += step
            
            # nastavi jedinci nejlepsi pozici ze vsech kroku rozsahu (provadi se za behu)
            individual.set_position_to_personal_best()


    def __migrate_all_to_all(self, prt: float, step: float, path_len: float):
        """
        Provadi migraci vsech jedincu ke všem ostatnim navzajem (All-To-All).

        Parametry:
            prt - prah pro mutaci (migrace), doporucene: 0.3
            step - velikost kroku, doporucene: 0.11, 0.22, 0.33, ..
            path_len - delke cesty (prozkoumavane oblasti), doporucene: 2.0 - 3.0
        """

        # reset personal best pro vsechny jedince
        for individual in self.population.get_individuals():
            individual.reset_personal_best()

        # migracni algoritmus all-to-all
        for i1, individual_1 in enumerate(self.population.get_individuals()):
            for i2, individual_2 in enumerate(self.population.get_individuals()):
                # jedinec 1 a 2 musi byt jini, nesmi jit o jednoho a toho sameho
                if i1 == i2:
                    continue

                # zalohuje si pocatecni pozici jedince 1
                base_position = individual_1.get_position().copy()

                # vypocet ucelove funkce v nahodnem smeru ve vsech krocich
                t = step
                while t <= path_len:
                    # nahodna volba smeru (perturbacni vektor)
                    prt_vector = self.__generatePRTVector(prt, self.population.get_dim())
                    # vypocet pozice
                    individual_1.set_position(base_position + (individual_2.get_position() - individual_1.get_position()) * t * prt_vector)
                    individual_1.update_fitness(self.obj_func)
                    self.evaluation_cnt += 1 # pro diagnostiku
                    t += step
                
                # obnoveni puvodni pozice jedince 1
                individual_1.set_position(base_position)
        
        # vsem jedinucem nastavi nejlepsi nalezene pozice (provede se az uplne na konci)
        for individual in self.population.get_individuals():
            individual.set_position_to_personal_best()


    def optimize(self, max_migrations: int, prt: float, step: float, path_len: float, migration_type=MIGRATION_ALL_TO_ONE):
        """
        Optimalizuje problem pomoci SOMA algoritmu

        Parametry:
            max_migrations - Maximalni pocet migraci populace
            prt - prah pro mutaci (migrace), doporucene: 0.3
            step - velikost kroku, doporucene: 0.11, 0.22, 0.33, ..
            path_len - delke cesty (prozkoumavane oblasti), doporucene: 2.0 - 3.0
            migration_type - typ migrace (MIGRATION_ALL_TO_ONE nebo MIGRATION_ALL_TO_ALL)
        """
 
        # nahodna inicializace
        self.population.random_init(self.obj_func)

        fitness_history = []

        # hlavni cast algoritmu
        for _ in range(max_migrations):
            # aktualni nejlepsi hodnotu fitnes v populaci zapise do historie vyvoje fitness
            gBest = self.population.get_global_best()
            fitness_history.append(gBest.get_value())

            # Aktualizace globalne nejlepsiho jedince (jen pokud je to vyzadovano)
            if migration_type == MIGRATION_ALL_TO_ONE:
                for individual in self.population.get_individuals():
                    self.population.update_global_best_with_one_indivial(individual)

            # Migrace populace jedincu
            if migration_type == MIGRATION_ALL_TO_ONE:
                self.__migrate_all_to_one(prt=prt, step=step, path_len=path_len)
            elif migration_type == MIGRATION_ALL_TO_ALL:
                self.__migrate_all_to_all(prt=prt, step=step, path_len=path_len)
            else:
                raise ValueError("Unsupported migration type!")


        # Aktualizace globalne nejlepsiho jedince (jen pokud je to vyzadovano)
        if migration_type == MIGRATION_ALL_TO_ALL:
            for individual in self.population.get_individuals():
                self.population.update_global_best_with_one_indivial(individual)

        # navrati gBest
        gBest = self.population.get_global_best()
        return gBest.get_position().copy(), gBest.get_value(), fitness_history