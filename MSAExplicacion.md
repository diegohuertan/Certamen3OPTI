# Algoritmo MSA (Mantis Search Algorithm) - Explicación Completa

## Introducción

El **Mantis Search Algorithm (MSA)** es un algoritmo metaheurístico bioinspirado que simula el comportamiento de caza de las mantis religiosas. Este algoritmo se caracteriza por su capacidad para explorar y explotar eficientemente el espacio de búsqueda, combinando estrategias de búsqueda local y global.

## Problema de Optimización Multiobjetivo

### Descripción del Problema
Estamos resolviendo un problema de optimización de campaña publicitaria con **dos objetivos conflictivos**:

1. **Maximizar la valorización total** (f1)
2. **Minimizar el costo total** (f2)

### Variables de Decisión
- **x₁, x₂, x₃, x₄, x₅**: Cantidad de anuncios de cada tipo (enteros)
- **v₁, v₂, v₃, v₄, v₅**: Valorización de cada anuncio (enteros)
- **c₁, c₂, c₃, c₄, c₅**: Costo de cada anuncio (calculados automáticamente)

### Dominios y Restricciones

#### Dominios de las Variables:
```
x₁ ∈ [0, 15]    # Cantidad de anuncios tipo 1
x₂ ∈ [0, 10]    # Cantidad de anuncios tipo 2  
x₃ ∈ [0, 25]    # Cantidad de anuncios tipo 3
x₄ ∈ [0, 4]     # Cantidad de anuncios tipo 4
x₅ ∈ [0, 30]    # Cantidad de anuncios tipo 5

c₁ ∈ [160, 200] # Costo anuncio tipo 1
c₂ ∈ [300, 350] # Costo anuncio tipo 2
c₃ ∈ [40, 80]   # Costo anuncio tipo 3
c₄ ∈ [100, 120] # Costo anuncio tipo 4
c₅ ∈ [10, 20]   # Costo anuncio tipo 5

v₁ ∈ [65, 85]   # Valorización anuncio tipo 1
v₂ ∈ [90, 95]   # Valorización anuncio tipo 2
v₃ ∈ [40, 60]   # Valorización anuncio tipo 3
v₄ ∈ [60, 80]   # Valorización anuncio tipo 4
v₅ ∈ [20, 30]   # Valorización anuncio tipo 5
```

#### Relaciones Costo-Valorización:
```
c₁ = 2v₁ + 30   # Costo anuncio tipo 1
c₂ = 10v₂ - 600 # Costo anuncio tipo 2
c₃ = 2v₃ - 40   # Costo anuncio tipo 3
c₄ = v₄ + 40    # Costo anuncio tipo 4
c₅ = v₅ - 10    # Costo anuncio tipo 5
```

#### Restricciones de Presupuesto:
1. **TV**: x₁c₁ + x₂c₂ ≤ 3800
2. **Diario/Revista**: x₃c₃ + x₄c₄ ≤ 2800
3. **Diario/Radio**: x₃c₃ + x₅c₅ ≤ 3500

## Algoritmo MSA - Fundamentos

### Inspiración Biológica
El MSA se inspira en el comportamiento de caza de las mantis religiosas:
- **Fase de Exploración**: Movimiento amplio para encontrar presas
- **Fase de Explotación**: Concentración en áreas prometedoras
- **Estrategia de Acecho**: Espera paciente y ataque preciso

### Diferencias con PSO
| Aspecto | PSO | MSA |
|---------|-----|-----|
| Inspiración | Bandadas de aves | Mantis religiosas |
| Movimiento | Continuo, basado en velocidad | Discreto, basado en saltos |
| Estrategia | Social (mejor global/personal) | Individual (acecho y ataque) |
| Exploración | Mediante inercia | Mediante búsqueda aleatoria |
| Explotación | Atracción a mejores posiciones | Refinamiento local |

## Estructura del Código MSA.py

### 1. Importación de Librerías
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

### 2. Función de Evaluación Multiobjetivo
```python
def fit(self, x):
    X, C, V = x[0:5], x[5:10], x[10:15]
    
    # Calcula objetivos
    valorization = sum(X[i] * V[i] for i in range(5))  # Maximizar valorización
    cost = sum(X[i] * C[i] for i in range(5))          # Minimizar costo
    
    return (valorization, -cost)  # Retorna (-costo) para maximización
```

### 3. Verificación de Factibilidad
```python
def check(self, x):
    X, C = x[0:5], x[5:10]
    
    # Verificar dominios
    for i in range(15):
        min_val, max_val = self.get_domain_limits(i)
        if x[i] < min_val or x[i] > max_val:
            return False
    
    # Verificar restricciones de presupuesto
    if X[0]*C[0] + X[1]*C[1] > 3800:      # TV
        return False
    if X[2]*C[2] + X[3]*C[3] > 2800:      # Diario/Revista
        return False
    if X[2]*C[2] + X[4]*C[4] > 3500:      # Diario/Radio
        return False
    
    return True
```

### 4. Generación de Soluciones Iniciales
```python
def __init__(self):
    # ... inicialización de parámetros ...
    
    # Generar cantidades X aleatoriamente
    for i in range(5):
        min_val, max_val = self.p.domains['X'][i]
        self.x.append(random.randint(min_val, max_val))
    
    # Generar valorizaciones V que produzcan costos C válidos
    V = []
    for i in range(5):
        v_min, v_max = self.p.domains['V'][i]
        valid = False
        while not valid:
            v = random.randint(v_min, v_max)
            # Calcular costo según la relación C-V
            c = self.calculate_cost_from_v(i, v)
            c_min, c_max = self.p.domains['C'][i]
            if c_min <= c <= c_max:
                V.append(v)
                self.x.append(c)
                valid = True
    self.x.extend(V)
```

### 5. Conceptos de Dominancia de Pareto
```python
def dominates(self, other):
    # Una solución domina a otra si:
    # - Es mejor en al menos un objetivo
    # - No es peor en ningún objetivo
    my_fit = self.fitness()
    other_fit = other.fitness()
    
    # Para maximizar valorización y minimizar costo
    return (all(m >= o for m, o in zip(my_fit, other_fit)) and 
            any(m > o for m, o in zip(my_fit, other_fit)))
```

### 6. Algoritmo MSA Principal
```python
def optimizer(self):
    print("=== MANTIS SEARCH ALGORITHM - PROBLEMA ANUNCIOS ===")
    
    # Inicialización de la población
    self.random()  # Genera población inicial factible
    
    if self.swarm:
        # Evolución del algoritmo
        self.evolve()
        
        # Mostrar resultados finales
        print(f"\n=== RESULTADO FINAL MSA ===")
        if self.g:  # Mejor solución global
            print(f"Mejor solución: {self.g}")
            X, C, V = self.g.x[0:5], self.g.x[5:10], self.g.x[10:15]
            fit = self.g.fitness()
            print(f"Valorización total: {fit[0]}")
            print(f"Costo total: {-fit[1]}")
        
        # Generar análisis de Pareto
        print("\nGenerando análisis de Frente de Pareto...")
        self.plot_pareto_analysis()

def evolve(self):
    for t in range(1, self.max_iter + 1):
        for i in range(len(self.swarm)):
            # Crear copia temporal de la mantis
            temp_mantis = Individual()
            temp_mantis.copy(self.swarm[i])
            
            # Aplicar movimiento de mantis (hasta 10 intentos)
            for _ in range(10):
                temp_mantis.mantis_move(self.g, self.archive, t, self.max_iter)
                if temp_mantis.is_feasible():
                    break
            
            # Actualizar si la nueva solución es mejor
            if temp_mantis.is_feasible():
                if temp_mantis.is_better_than(self.swarm[i]):
                    self.swarm[i].copy(temp_mantis)
                    self.update_archive(temp_mantis)
                if temp_mantis.is_better_than(self.g):
                    self.g.copy(temp_mantis)
                
                # Almacenar para análisis de Pareto
                new_sol = Individual()
                new_sol.copy(temp_mantis)
                self.all_solutions.append(new_sol)
        
        self.show_results(t)
```

## Flujo del Algoritmo

### 1. Inicialización
- Genera población inicial de soluciones factibles
- Cada solución respeta dominios y restricciones

### 2. Iteración Principal
Para cada iteración:
- **Selección de estrategia**: Exploración vs Explotación
- **Generación de nueva solución**: Según estrategia seleccionada
- **Evaluación**: Calcula objetivos f1 y f2
- **Actualización**: Reemplaza si la nueva solución es mejor

### 3. Estrategias de Movimiento

#### Movimiento de Mantis (mantis_move):
El movimiento de mantis combina diferentes estrategias según probabilidades:

```python
def mantis_move(self, global_best, archive, t, max_iter):
    p = 0.5  # Probabilidad de exploración
    F = 1 - (t % (max_iter // 10)) / (max_iter // 10)  # Factor de intensidad
    m = 1 - t / max_iter  # Factor de momentum
    
    if random.random() < p:
        # Exploración
        if random.random() < F:
            # Movimiento Lévy Flight (búsqueda amplia)
            levy_step = self.p.levy_flight()
            # Aplicar paso con perturbación
        else:
            # Movimiento hacia solución del archivo
            target = random.choice(archive)
            # Movimiento dirigido con momentum
    else:
        # Explotación hacia mejor global
        # Movimiento sigmoideal hacia global_best
    
    # Perturbación adicional (20% probabilidad)
    if random.random() < 0.2:
        # Añadir ruido aleatorio pequeño
```

#### Tipos de Movimiento:

1. **Lévy Flight**: Movimiento de búsqueda amplia inspirado en vuelos de insectos
2. **Movimiento hacia Archivo**: Dirección hacia soluciones élite almacenadas  
3. **Movimiento hacia Global**: Atracción hacia la mejor solución encontrada
4. **Perturbación**: Ruido aleatorio para evitar estancamiento

#### Exploración vs Explotación:
- **Exploración**: Movimientos amplios para encontrar nuevas regiones
- **Explotación**: Movimientos locales para refinar soluciones
- **Balance dinámico**: La probabilidad cambia según la iteración

### 4. Almacenamiento y Análisis
- Guarda todas las soluciones generadas
- Calcula frente de Pareto óptimo
- Genera visualizaciones y estadísticas

## Análisis del Frente de Pareto

### ¿Qué es el Frente de Pareto?
El frente de Pareto es el conjunto de soluciones **no dominadas**, es decir, soluciones donde no se puede mejorar un objetivo sin empeorar el otro.

### Interpretación de Resultados
- **Soluciones en el frente**: Representan compromisos óptimos
- **Extremos del frente**: Optimización de un solo objetivo
- **Puntos intermedios**: Balances entre objetivos

### Gráficas Generadas
1. **Scatter Plot**: Muestra todas las soluciones y destaca el frente de Pareto
2. **Estadísticas**: Rangos de valores para cada objetivo
3. **Análisis de trade-off**: Relación entre valorización y costo

## Ventajas del MSA

### Fortalezas del Algoritmo:
1. **Simplicidad**: Implementación directa y comprensible
2. **Eficiencia**: Buen balance exploración-explotación
3. **Robustez**: Maneja bien restricciones complejas
4. **Flexibilidad**: Adaptable a diferentes problemas

### Aplicaciones:
- Optimización de campañas publicitarias
- Problemas de asignación de recursos
- Diseño de productos con múltiples criterios
- Planificación de proyectos

## Parámetros del Algoritmo

### Configuración Utilizada:
- **Tamaño de población**: 25 individuos (mantis)
- **Iteraciones máximas**: 50
- **Tamaño del archivo**: 5 mejores soluciones
- **Dimensión del problema**: 15 variables (5 X, 5 C, 5 V)
- **Intentos de reparación**: 10 por movimiento

### Parámetros del Movimiento de Mantis:
- **Probabilidad de exploración**: p = 0.5
- **Factor de intensidad**: F = 1 - (t % (max_iter // 10)) / (max_iter // 10)
- **Factor de momentum**: m = 1 - t / max_iter
- **Probabilidad de perturbación**: 0.2

### Recomendaciones de Ajuste:
- **Población pequeña**: Convergencia rápida, menor diversidad
- **Población grande**: Mayor diversidad, convergencia lenta
- **Más iteraciones**: Mejor convergencia, mayor tiempo de cómputo
- **Archivo más grande**: Mejor memoria de soluciones élite
- **Más intentos de reparación**: Mayor probabilidad de factibilidad

## Conclusiones

El algoritmo MSA implementado resuelve efectivamente el problema de optimización multiobjetivo de campaña publicitaria, generando un frente de Pareto que permite al tomador de decisiones seleccionar la solución que mejor se adapte a sus preferencias entre valorización y costo.

La implementación demuestra la capacidad del MSA para:
- Manejar restricciones complejas
- Generar soluciones diversas y de calidad
- Proporcionar análisis visual comprensible
- Adaptarse a problemas multiobjetivo

Este enfoque puede extenderse a problemas similares modificando las funciones de evaluación, restricciones y dominios según las necesidades específicas del problema.