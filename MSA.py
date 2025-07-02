import random, math
import numpy as np

class Problem:
    def __init__(self):
        self.dim = 15
        self.domains = {
            'X': [(0, 15), (0, 10), (0, 25), (0, 4), (0, 30)],
            'C': [(160, 200), (300, 350), (40, 80), (100, 120), (10, 20)],
            'V': [(65, 85), (90, 95), (40, 60), (60, 80), (20, 30)]
        }
        self.budget_limits = {
            'tv': 3800, 'diario_rev': 2800, 'diario_rad': 3500
        }

    def get_domain_limits(self, index):
        if index < 5:
            return self.domains['X'][index]
        elif index < 10:
            return self.domains['C'][index - 5]
        else:
            return self.domains['V'][index - 10]

    def update_C_from_V(self, x):
        for i in range(5):
            v = x[10 + i]
            if i == 0:
                x[5 + i] = 2 * v + 30
            elif i == 1:
                x[5 + i] = 10 * v - 600
            elif i == 2:
                x[5 + i] = 2 * v - 40
            elif i == 3:
                x[5 + i] = v + 40
            elif i == 4:
                x[5 + i] = v - 10

    def check(self, x):
        X, C = x[0:5], x[5:10]
        for i in range(15):
            min_val, max_val = self.get_domain_limits(i)
            if x[i] < min_val or x[i] > max_val:
                return False
        if X[0]*C[0] + X[1]*C[1] > self.budget_limits['tv']:
            return False
        if X[2]*C[2] + X[3]*C[3] > self.budget_limits['diario_rev']:
            return False
        if X[2]*C[2] + X[4]*C[4] > self.budget_limits['diario_rad']:
            return False
        return True

    def fit(self, x):
        X, C, V = x[0:5], x[5:10], x[10:15]
        valorization = sum(X[i] * V[i] for i in range(5))
        cost = sum(X[i] * C[i] for i in range(5))
        return (valorization, -cost)

    def levy_flight(self, beta=1.5):
        num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (num / den) ** (1 / beta)
        u = random.gauss(0, sigma_u)
        v = random.gauss(0, 1)
        return u / (abs(v) ** (1 / beta))

class Individual:
    def __init__(self):
        self.p = Problem()
        self.dimension = self.p.dim
        self.x = []
        self.fitness_values = None

        for i in range(5):
            min_val, max_val = self.p.domains['X'][i]
            self.x.append(random.randint(min_val, max_val))

        V = []
        for i in range(5):
            v_min, v_max = self.p.domains['V'][i]
            valid = False
            while not valid:
                v = random.randint(v_min, v_max)
                if i == 0:
                    c = 2 * v + 30
                elif i == 1:
                    c = 10 * v - 600
                elif i == 2:
                    c = 2 * v - 40
                elif i == 3:
                    c = v + 40
                else:
                    c = v - 10
                c_min, c_max = self.p.domains['C'][i]
                if c_min <= c <= c_max:
                    V.append(v)
                    self.x.append(c)
                    valid = True
        self.x.extend(V)

    def is_feasible(self):
        return self.p.check(self.x)

    def fitness(self):
        if self.fitness_values is None:
            self.fitness_values = self.p.fit(self.x)
        return self.fitness_values

    def dominates(self, other):
        my_fit = self.fitness()
        other_fit = other.fitness()
        return all(m >= o for m, o in zip(my_fit, other_fit)) and any(m > o for m, o in zip(my_fit, other_fit))

    def is_better_than(self, other):
        my_score = self.fitness()[0] + 0.1 * self.fitness()[1]
        other_score = other.fitness()[0] + 0.1 * other.fitness()[1]
        return my_score > other_score

    def mantis_move(self, global_best, archive, t, max_iter):
        p = 0.5
        F = 1 - (t % (max_iter // 10)) / (max_iter // 10)
        m = 1 - t / max_iter
        new_x = self.x.copy()
        if random.random() < p:
            if random.random() < F:
                for j in range(self.dimension):
                    levy_step = self.p.levy_flight()
                    min_val, max_val = self.p.get_domain_limits(j)
                    new_x[j] = self.x[j] + levy_step * (max_val - min_val) * 0.1
                    new_x[j] = max(min_val, min(max_val, int(new_x[j])))
            elif archive:
                target = random.choice(archive)
                for j in range(self.dimension):
                    alpha = math.cos(math.pi * random.random()) * m
                    new_x[j] = self.x[j] + alpha * (target.x[j] - self.x[j])
                    min_val, max_val = self.p.get_domain_limits(j)
                    new_x[j] = max(min_val, min(max_val, int(new_x[j])))
        elif global_best:
            for j in range(self.dimension):
                alp = 2 * random.random() - 1
                vs = 1 / (1 + math.exp(alp))
                distance = global_best.x[j] - self.x[j]
                new_x[j] = (self.x[j] + global_best.x[j]) / 2.0 + vs * distance
                min_val, max_val = self.p.get_domain_limits(j)
                new_x[j] = max(min_val, min(max_val, int(new_x[j])))
        if random.random() < 0.2:
            for j in range(self.dimension):
                r3 = random.uniform(-1, 1)
                perturbation = r3 * random.uniform(-0.1, 0.1) * self.x[j]
                new_x[j] += perturbation
                min_val, max_val = self.p.get_domain_limits(j)
                new_x[j] = max(min_val, min(max_val, int(new_x[j])))
        self.p.update_C_from_V(new_x)
        self.x = new_x
        self.fitness_values = None

    def copy(self, other):
        if isinstance(other, Individual):
            self.x = other.x.copy()
            self.fitness_values = other.fitness_values

    def __str__(self):
        X, C, V = self.x[0:5], self.x[5:10], self.x[10:15]
        fit = self.fitness()
        return f"X={X}, C={C}, V={V}, fitness={fit}"

class MSA_Swarm:
    def __init__(self):
        self.max_iter = 50
        self.n_individual = 25
        self.swarm = []
        self.g = None
        self.archive = []
        self.archive_size = 5

    def random(self):
        attempts = 0
        while len(self.swarm) < self.n_individual and attempts < 1000:
            ind = Individual()
            if ind.is_feasible():
                self.swarm.append(ind)
                self.update_archive(ind)
            attempts += 1
        if not self.swarm:
            print("No se pudieron generar soluciones factibles")
            return
        self.g = max(self.swarm, key=lambda x: x.fitness()[0] + 0.1 * x.fitness()[1])
        print(f"Inicialización MSA: {len(self.swarm)} mantis factibles")
        self.show_results(0)

    def update_archive(self, individual):
        if len(self.archive) < self.archive_size:
            new_ind = Individual()
            new_ind.copy(individual)
            self.archive.append(new_ind)
        else:
            worst_idx = min(range(len(self.archive)), key=lambda i: self.archive[i].fitness()[0] + 0.1 * self.archive[i].fitness()[1])
            if individual.is_better_than(self.archive[worst_idx]):
                self.archive[worst_idx].copy(individual)

    def evolve(self):
        for t in range(1, self.max_iter + 1):
            for i in range(len(self.swarm)):
                temp_mantis = Individual()
                temp_mantis.copy(self.swarm[i])
                for _ in range(10):
                    temp_mantis.mantis_move(self.g, self.archive, t, self.max_iter)
                    if temp_mantis.is_feasible():
                        break
                if temp_mantis.is_feasible():
                    if temp_mantis.is_better_than(self.swarm[i]):
                        self.swarm[i].copy(temp_mantis)
                        self.update_archive(temp_mantis)
                    if temp_mantis.is_better_than(self.g):
                        self.g.copy(temp_mantis)
            self.show_results(t)

    def show_results(self, t):
        if self.g:
            fit = self.g.fitness()
            print(f"t: {t}, best_global: valorización={fit[0]}, costo={-fit[1]}")
        else:
            print(f"t: {t}, no solution found")

    def optimizer(self):
        print("=== MANTIS SEARCH ALGORITHM - PROBLEMA ANUNCIOS ===")
        self.random()
        if self.swarm:
            self.evolve()
            print(f"\n=== RESULTADO FINAL MSA ===")
            if self.g:
                print(f"Mejor solución: {self.g}")
                X, C, V = self.g.x[0:5], self.g.x[5:10], self.g.x[10:15]
                fit = self.g.fitness()
                print(f"Valorización total: {fit[0]}")
                print(f"Costo total: {-fit[1]}")
                print(f"Tamaño del archivo: {len(self.archive)}")

if __name__ == "__main__":
    MSA_Swarm().optimizer()
