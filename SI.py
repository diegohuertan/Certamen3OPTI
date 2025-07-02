import random, math

class Problem:
    def __init__(self):
        # Definir el problema
        self.dim = 20
        self.min_value = 0 # ejemplo de un problema binario
        self.max_value = 1 # ejemplo de un problema binario
        
    def check(self, x):
        # Definir las restricciones
        return True

    def fit(self, x):
        # Definir la o las funciones objetivos
        return 0

    # Metodo para manten el valor, en el dominio
    def keep_domain(self, x):
        probs = []
        for _ in range(self.min_value, self.max_value):
            probs.append(1 / (1 + math.exp(-(x + random.gauss(0, 0.5)))))
        total = sum(probs)
        normalized = [p / total for p in probs]
        acc = 0
        for i, p in enumerate(normalized):
            acc += p
            if random.random() <= acc:
                return i
        return self.max_value - 1

class Individual:
    def __init__(self):
        self.p = Problem()
        self.dimension = self.p.dim
        self.x = [] # solucion
        
        # Crear una solución inicial sin validar
        for _ in range(self.dimension):
            self.x.append(random.randint(self.p.min_value, self.p.max_value))

    def is_feasible(self):
        return self.p.check(self.x)

    def fitness(self):
        return self.p.fit(self.x)

    def is_better_than(self, g):
        # > para problemas de maximización
        return self.fitness() > g.fitness()

    def move(self, g): # Por parametro recibe los parametros del algoritmo
        for j in range(self.dimension):
            self.x[j] = self.p.keep_domain(self.x[j]) # Operadores de movimiento

    def copy(self, other):
        if isinstance(other, Individual):
            self.x = other.x.copy()

    def __str__(self):
        return f"x: {self.x}, fitness {self.fitness()}"

class Swarm:
    def __init__(self):
        self.max_iter = 25
        self.n_individual = 10
        # Definir los parametros del algoritmo
        self.swarm = []
        self.g = None

    def random(self):
        for _ in range(self.n_individual):
            feasible = False
            while not feasible:
                ind = Individual()
                feasible = ind.is_feasible() 
            self.swarm.append(ind)
        
        self.g = self.swarm[0]
        for i in range(1, self.n_individual):
            if self.swarm[i].is_better_than(self.g):
                self.g.copy(self.swarm[i])

        self.show_results(0)

    def evolve(self):
        t = 1
        ind = Individual()
        while t <= self.max_iter:
            for i in range(1, self.n_individual):
                feasible = False
                while not feasible:
                    ind.copy(self.swarm[i])                    
                    # Por argumento se envian los parametros del algoritmo
                    ind.move(self.g)
                    feasible = ind.is_feasible()
                self.swarm[i].copy(ind)

                if self.swarm[i].is_better_than(self.g):
                    self.g.copy(self.swarm[i])

            self.show_results(t)
            t += 1
        
    def show_results(self, t):
        print(f"t: {t}, best_global: {self.g}")

    def optimizer(self):
        self.random()
        self.evolve()

# Ejecutar
Swarm().optimizer()