import random, math
import numpy as np
import matplotlib.pyplot as plt

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

    def fit(self, x, w_valor=0.6, w_costo=0.4):
        X, C, V = x[0:5], x[5:10], x[10:15]
        valorization = sum(X[i] * V[i] for i in range(5))
        cost = sum(X[i] * C[i] for i in range(5))
        
        # Valores de referencia para normalización
        # Mejor valorización posible (todos los máximos)
        valorization_best = (9*78 + 6*90 + 25*60 + 3*68 + 30*30)  # 3846

        # Peor costo factible observado
        cost_worst = (9*186 + 6*300 + 25*80 + 3*108 + 30*20)      # 6398

        cost_best = (0*160 + 0*300 + 0*40 + 0*100 + 0*10)       # 0 (mejor caso teórico)
        
        # Escalarización académica normalizada
        # fp(X) / ep(X^best) para maximización
        term1 = w_valor * (valorization / valorization_best)
        
        # (C - fq(X)) / (C - eq(X^best)) para minimización
        if cost_worst > cost_best:
            term2 = w_costo * ((cost_worst - cost) / (cost_worst - cost_best))
        else:
            term2 = w_costo * (1 - cost / cost_worst)  # Fallback si denominador es 0
        
        return term1 + term2  # Escalarización académica
    
    def fit_multiobj(self, x):
        """Función para mantener capacidad de análisis multiobjetivo"""
        X, C, V = x[0:5], x[5:10], x[10:15]
        valorization = sum(X[i] * V[i] for i in range(5))
        cost = sum(X[i] * C[i] for i in range(5))
        return (valorization, -cost)  # Tupla para análisis

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
            self.fitness_values = self.p.fit(self.x)  # Usa escalarización
        return self.fitness_values

    def fitness_multiobj(self):
        """Método para obtener fitness multiobjetivo para análisis"""
        return self.p.fit_multiobj(self.x)

    def dominates(self, other):
        """Dominancia basada en fitness multiobjetivo para análisis de Pareto"""
        my_fit = self.fitness_multiobj()
        other_fit = other.fitness_multiobj()
        return all(m >= o for m, o in zip(my_fit, other_fit)) and any(m > o for m, o in zip(my_fit, other_fit))

    def is_better_than(self, other):
        """Comparación directa basada en fitness escalarizado"""
        return self.fitness() > other.fitness()

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
        # ✅ AGREGADO: Almacenar todas las soluciones para Pareto
        self.all_solutions = []

    def random(self):
        attempts = 0
        while len(self.swarm) < self.n_individual and attempts < 1000:
            ind = Individual()
            if ind.is_feasible():
                self.swarm.append(ind)
                self.update_archive(ind)
                # ✅ AGREGADO: Guardar todas las soluciones
                self.all_solutions.append(ind)
            attempts += 1
        if not self.swarm:
            print("No se pudieron generar soluciones factibles")
            return
        self.g = max(self.swarm, key=lambda x: x.fitness())
        print(f"Inicialización MSA: {len(self.swarm)} mantis factibles")
        self.show_results(0)

    def update_archive(self, individual):
        if len(self.archive) < self.archive_size:
            new_ind = Individual()
            new_ind.copy(individual)
            self.archive.append(new_ind)
        else:
            worst_idx = min(range(len(self.archive)), key=lambda i: self.archive[i].fitness())
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
                    # ✅ AGREGADO: Guardar todas las soluciones factibles
                    new_sol = Individual()
                    new_sol.copy(temp_mantis)
                    self.all_solutions.append(new_sol)
            self.show_results(t)

    def show_results(self, t):
        if self.g:
            fit = self.g.fitness()
            fit_multi = self.g.fitness_multiobj()
            print(f"t: {t}, best_global: fitness_escalar={fit:.1f}, valorización={fit_multi[0]}, costo={-fit_multi[1]}")
        else:
            print(f"t: {t}, no solution found")

    # ✅ AGREGADO: Función para calcular Frente de Pareto
    def calculate_pareto_front(self):
        """Calcula el frente de Pareto de todas las soluciones"""
        if not self.all_solutions:
            return []
        
        pareto_front = []
        
        for candidate in self.all_solutions:
            is_dominated = False
            
            # Verificar si el candidato es dominado por alguna otra solución
            for other in self.all_solutions:
                if other != candidate and other.dominates(candidate):
                    is_dominated = True
                    break
            
            # Si no es dominado, pertenece al frente de Pareto
            if not is_dominated:
                # Verificar si ya existe una solución similar
                is_duplicate = False
                for existing in pareto_front:
                    if (existing.fitness_multiobj()[0] == candidate.fitness_multiobj()[0] and 
                        existing.fitness_multiobj()[1] == candidate.fitness_multiobj()[1]):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    pareto_front.append(candidate)
        
        return pareto_front

    # ✅ AGREGADO: Función para generar gráficos
    def plot_pareto_analysis(self):
        """Genera gráficos de análisis de Pareto"""
        if not self.all_solutions:
            print("No hay soluciones para graficar")
            return
        
        # Extraer valores de fitness
        valorizations = [sol.fitness_multiobj()[0] for sol in self.all_solutions]
        costs = [-sol.fitness_multiobj()[1] for sol in self.all_solutions]  # Convertir a positivo
        
        # Calcular frente de Pareto
        pareto_front = self.calculate_pareto_front()
        pareto_valorizations = [sol.fitness_multiobj()[0] for sol in pareto_front]
        pareto_costs = [-sol.fitness_multiobj()[1] for sol in pareto_front]
        
        # Crear figura con múltiples subgráficos
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis MSA - Problema de Anuncios Multiobjetivo', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Frente de Pareto
        ax1.scatter(valorizations, costs, alpha=0.6, c='lightblue', s=30, label='Todas las soluciones')
        ax1.scatter(pareto_valorizations, pareto_costs, alpha=0.9, c='red', s=80, 
                   label=f'Frente de Pareto ({len(pareto_front)} soluciones)', marker='D')
        ax1.set_xlabel('Valorización Total')
        ax1.set_ylabel('Costo Total')
        ax1.set_title('Frente de Pareto - Valorización vs Costo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Distribución de Valorización
        ax2.hist(valorizations, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(valorizations), color='red', linestyle='--', 
                   label=f'Media: {np.mean(valorizations):.1f}')
        ax2.axvline(np.median(valorizations), color='blue', linestyle='--', 
                   label=f'Mediana: {np.median(valorizations):.1f}')
        ax2.set_xlabel('Valorización Total')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Valorización')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Distribución de Costo
        ax3.hist(costs, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(np.mean(costs), color='red', linestyle='--', 
                   label=f'Media: {np.mean(costs):.1f}')
        ax3.axvline(np.median(costs), color='blue', linestyle='--', 
                   label=f'Mediana: {np.median(costs):.1f}')
        ax3.set_xlabel('Costo Total')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Costo')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Análisis de trade-offs
        if len(pareto_front) > 1:
            # Ordenar frente de Pareto por valorización
            pareto_sorted = sorted(pareto_front, key=lambda x: x.fitness_multiobj()[0])
            pareto_val_sorted = [sol.fitness_multiobj()[0] for sol in pareto_sorted]
            pareto_cost_sorted = [-sol.fitness_multiobj()[1] for sol in pareto_sorted]
            
            ax4.plot(pareto_val_sorted, pareto_cost_sorted, 'ro-', linewidth=2, 
                    markersize=8, label='Frente de Pareto')
            ax4.fill_between(pareto_val_sorted, pareto_cost_sorted, alpha=0.3, color='red')
            ax4.set_xlabel('Valorización Total')
            ax4.set_ylabel('Costo Total')
            ax4.set_title('Trade-off: Valorización vs Costo (Frente de Pareto)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Frente de Pareto\ninsuficiente para\ntrade-off analysis', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Trade-off Analysis')
        
        plt.tight_layout()
        plt.savefig('pareto_analysis_msa.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Imprimir estadísticas
        print("\n" + "="*60)
        print("ANÁLISIS DE FRENTE DE PARETO")
        print("="*60)
        print(f"Total de soluciones evaluadas: {len(self.all_solutions)}")
        print(f"Soluciones en frente de Pareto: {len(pareto_front)}")
        print(f"Porcentaje de soluciones Pareto-óptimas: {len(pareto_front)/len(self.all_solutions)*100:.2f}%")
        
        print(f"\nEstadísticas de Valorización:")
        print(f"  Rango: [{min(valorizations):.1f}, {max(valorizations):.1f}]")
        print(f"  Media: {np.mean(valorizations):.1f}")
        print(f"  Desviación estándar: {np.std(valorizations):.1f}")
        
        print(f"\nEstadísticas de Costo:")
        print(f"  Rango: [{min(costs):.1f}, {max(costs):.1f}]")
        print(f"  Media: {np.mean(costs):.1f}")
        print(f"  Desviación estándar: {np.std(costs):.1f}")
        
        print(f"\nMejores 5 soluciones del Frente de Pareto:")
        pareto_sorted = sorted(pareto_front, key=lambda x: x.fitness(), reverse=True)
        for i, sol in enumerate(pareto_sorted[:5]):
            fit_multi = sol.fitness_multiobj()
            print(f"  {i+1}. Valorización: {fit_multi[0]:.1f}, Costo: {-fit_multi[1]:.1f}, Fitness: {sol.fitness():.1f}")

        best_val_solution = max(self.all_solutions, key=lambda ind: ind.fitness_multiobj()[0])
        val, cost = best_val_solution.fitness_multiobj()
        print("\n== MEJOR VALORIZACIÓN FACTIBLE ENCONTRADA ==")
        print(f"Valorización: {val}")
        print(f"Costo: {-cost}")
        print(f"Configuración:")
        print(best_val_solution)


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
                fit_multi = self.g.fitness_multiobj()
                print(f"Fitness escalarizado: {fit:.1f}")
                print(f"Valorización total: {fit_multi[0]}")
                print(f"Costo total: {-fit_multi[1]}")
                print(f"Tamaño del archivo: {len(self.archive)}")
            
            # ✅ AGREGADO: Generar gráficos de Pareto
            print("\nGenerando análisis de Frente de Pareto...")
            self.plot_pareto_analysis()

if __name__ == "__main__":
    MSA_Swarm().optimizer()