import numpy as np
import math
import SharedUtils

def w_func_linear(iteration: int, max_iterations: int):
    return (0.8 - ((0.8 - 0.3) * iteration) / max_iterations)

def w_func_constant_07(iteration: int, max_iterations: int):
    return 0.7

def w_func_constant_06(iteration: int, max_iterations: int):
    return 0.6

class Particle:
    def __init__(self, dim: int, vmax: float, range: SharedUtils.Range):
        """
        Trida pro particle

        Parametry:
            dim - dimenze castice
            vmax - maximalni rychlost pohybu castice
            range - rozsah hodnot reseni
        """
        self.position = np.random.uniform(range.min, range.max, size=dim)
        self.velocity = np.random.uniform(-vmax, vmax, size=dim)
        self.personal_best_pos = self.position.copy()
        self.personal_best_val = float('inf')

    def get_position(self):
        """
        Navrati aktualni pozici castice
        """
        return self.position

    def get_velocity(self):
        """
        Navrati aktualni rychlost castice
        """
        return self.velocity

    def get_personal_best_position(self):
        """
        Navrati pozici nejlepsi nalazenou hodnotu castice
        """
        return self.personal_best_pos

    def get_personal_best_value(self):
        """
        Navrati nejlepsi nalazenou hodnotu castice
        """
        return self.personal_best_val

    def set_position(self, position):
        """
        Nastavi castici novou pozici

        Parametry:
            position - nova pozice castice     
        """
        self.position = position

    def update_personal_best(self, obj_func):
        """
        Aktualizuje nejlepsi pozicizi castice. Pokud je aktualni pozice lepe ohodnoceny nez drive personalne nejlepe ohodnocena pozice tak je prepsana

        Parametry:
            obj_func - funkce optimalizovaneho problemu (urcuje kvalitu aktualni pozice)
        """
        new_value = obj_func(self.position)
        if new_value < self.personal_best_val:
            self.personal_best_val = new_value
            self.personal_best_pos = self.position.copy()


PS_TOPOLOGY_GLOBAL = 'global'
PS_TOPOLOGY_RING = 'ring'


class ParticleSwarm:
    def __init__(self, dim: int, nparticles: int, vmax: float, range: SharedUtils.Range, topology=PS_TOPOLOGY_GLOBAL) -> None:
        """
        Trida pro Particle swarm

        Parametry:
            dim - dimenze castic
            nparticles - pocet castic v roji
            vmax - maximalni rychlost pohybu castic
            range - rozsah hodnot reseni
            topology - topologie 
        """
        self.dim = dim
        self.nparticles = nparticles
        self.vmax = vmax
        self.topology = topology
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
        self.global_best_pos = None
        self.global_best_val = float('inf')
        self.particles = [Particle(self.dim, self.vmax, self.range)
                          for _ in range(self.nparticles)]
        if obj_func is not None:
            for p in self.particles:
                p.update_personal_best(obj_func)
                self.update_global_best_with_one_particle(p)

    def get_particles(self):
        """
        Navrati vsechny castice
        """
        return self.particles

    def get_range(self):
        """
        Navrati rozsaho hodnot reseni
        """
        return self.range

    def get_dim(self):
        """
        Navrati dimenzi castic
        """
        return self.dim

    def get_vmax(self):
        """
        Navrati maximalni rychlost pohybu castic
        """
        return self.vmax

    def get_global_best(self):
        """
        Navrati nejlepsi nalezenou hodnotu

        Return:
            pozice nejlepsiho reseni, hodnota nejelepsiho reseni      
        """
        return self.global_best_pos, self.global_best_val

    def update_global_best_with_one_particle(self, particle: Particle):
        """
        Update novou globalne nejlepsi hodnotu.

        Parametry:
            particle - castice jejiz personalni nejlepsi hodnota muze byt nastavena jako globalne nejlepsi   
        """
        if particle.get_personal_best_value() < self.global_best_val:
            self.global_best_val = particle.get_personal_best_value()
            self.global_best_pos = particle.get_personal_best_position().copy()

    def get_neighbors_best(self, particle_index: int):
        """
        Navrati nejlepsi nalezenou hodnotu v sousedstvi v zavyslosti na zvolene topologii roje castic

        Parametry:
            particle_index - index vybrana castice

        Return:
             pozice nejlepsiho reseni, hodnota nejelepsiho reseni    
        """
        n_best_pos = None
        n_best_value = float('inf')

        if self.topology == PS_TOPOLOGY_GLOBAL:
            n_best_pos = self.global_best_pos
            n_best_value = self.global_best_val
        elif self.topology == PS_TOPOLOGY_RING:
            n_best_pos, n_best_value = self.get_ring_neighbors_best(
                particle_index, int(self.propts["RING_SIZE"]))
        else:
            raise ValueError("Unsupported topology!!")

        return n_best_pos, n_best_value

    def get_ring_neighbors_best(self, particle_index, size: int):
        """
        Z kruhoveho sousedstvi navrati nejlepsi nelazenou hodnotu

        Parametry:
            particle_index - index castice ktera je stredem sousedstvi
            size - velikost sousedstvi

        Return:
            pozice nejlepsiho reseni, hodnota nejelepsiho reseni   
        """
        neighbor_best_pos = None
        neighbor_best_value = float('inf')
        for i in range(particle_index - size, particle_index + size + 1):
            neighbor_index = i % self.nparticles
            if self.particles[neighbor_index].get_personal_best_value() < neighbor_best_value:
                neighbor_best_value = self.particles[neighbor_index].get_personal_best_value(
                )
                neighbor_best_pos = self.particles[neighbor_index].get_personal_best_position(
                )

        return neighbor_best_pos, neighbor_best_value
    

    
PSO_TOPOLOGY_GLOBAL = 'global'
PSO_TOPOLOGY_RING = 'ring'

class PSO:
    def __init__(self, p_swarm: ParticleSwarm, obj_func, max_iter: int, w_func: float, c1: float, c2: float):
        """
        Trida implementujici PSO optimalizacni algoritmus

        Parametry:
            p_swarm - Reference na roj castic
            obj_func - funkce optimalizovaneho problemu   
            max_iter - maximalni pocet iteraci
            w - funkce vypoctu setrvacnosti. Predpist w_func(iterace, pocet iteraci)
            c1 - ucici faktor 1
            c2 - ucici faktor 2
        """
        self.p_swarm = p_swarm
        self.obj_func = obj_func
        self.max_iter = max_iter
        self.w_func = w_func
        self.c1 = c1
        self.c2 = c2

    def optimize(self):
        # nahodna inicializace
        self.p_swarm.random_init(self.obj_func)

        fitness_history = []

        for _ in range(self.max_iter):
            # aktualni nejlepsi hodnotu fitnes v populaci zapise do historie vyvoje fitness
            _, v = self.p_swarm.get_global_best()
            fitness_history.append(v)

            for i, particle in enumerate(self.p_swarm.get_particles()):
                # nahodna generovani cisel pro vypocet rychlosti
                r1, r2 = np.random.rand(self.p_swarm.get_dim()), np.random.rand(
                    self.p_swarm.get_dim())

                # nalezeni nejlepsi pozice ze sousedstvi (hodnota je zavysla na typu topologie ve p_swam objektu)
                neighbors_best_position, _ = self.p_swarm.get_neighbors_best(
                    i)
                if neighbors_best_position is None:
                    raise ValueError("Failed to get neighbors best!")

                # vypocet nove pozice aktualne zpracovavane castice
                particle_velocity = (
                    self.w_func(i, self.max_iter) * particle.get_velocity() +
                    self.c1 * r1 * (particle.get_personal_best_position() - particle.get_position()) +
                    self.c2 * r2 * (neighbors_best_position -
                                    particle.get_position())
                )
                particle_position = particle.get_position() + particle_velocity
                particle_position = np.clip(
                    particle_position, self.p_swarm.get_range().min, self.p_swarm.get_range().max)

                # nastaveni nove vypoctene pozice
                particle.set_position(particle_position)

                # update personalni nejlepsi pozice castice
                particle.update_personal_best(self.obj_func)

                # update globalni nejlepsi pozice
                self.p_swarm.update_global_best_with_one_particle(particle)

        gBest_p, gBest_v = self.p_swarm.get_global_best()
        return gBest_p, gBest_v, fitness_history