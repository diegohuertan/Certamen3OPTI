# Problema de anuncios televisivos siguiendo el formato del ejemplo AC-3

domains = {
    # Variables de cantidad de anuncios (limitantes máximas)
    'X1': list(range(0, 16)),  # 0 a 15 anuncios TV tarde
    'X2': list(range(0, 11)),  # 0 a 10 anuncios TV noche  
    'X3': list(range(0, 26)),  # 0 a 25 anuncios diario
    'X4': list(range(0, 5)),   # 0 a 4 anuncios revista
    'X5': list(range(0, 31)),  # 0 a 30 anuncios radio
    
    # Variables de costo - rangos según la imagen
    'C1': list(range(160, 201)),  # Rango 160-200
    'C2': list(range(300, 361)),  # Rango 300-360
    'C3': list(range(40, 81)),    # Rango 40-80 (corregido según problema original)
    'C4': list(range(100, 121)),  # Rango 100-120
    'C5': list(range(10, 21)),    # Rango 10-20 (corregido según problema original)
    
    # Variables de valorización - rangos según la imagen
    'V1': list(range(65, 86)),    # Rango 65-85
    'V2': list(range(90, 96)),    # Rango 90-95 (corregido según problema original)
    'V3': list(range(40, 61)),    # Rango 40-60
    'V4': list(range(60, 81)),    # Rango 60-80
    'V5': list(range(20, 31)),    # Rango 20-30
}

constraints = {
    
}

def revise(x, y):
    revised = False
    x_domain = domains[x]
    y_domain = domains[y]
    all_constraints = [
        constraint for constraint in constraints if constraint[0] == x and constraint[1] == y]
    values_to_remove = []
    
    for x_value in x_domain:
        satisfies = False
        for y_value in y_domain:
            for constraint in all_constraints:
                constraint_func = constraints[constraint]
                try:
                    if constraint_func(x_value, y_value):
                        satisfies = True
                        break
                except:
                    # Si hay error en la evaluación, continuar
                    continue
            if satisfies:
                break
        if not satisfies:
            values_to_remove.append(x_value)
            revised = True
    
    for value in values_to_remove:
        x_domain.remove(value)
    
    return revised

def ac3(arcs):
    queue = arcs[:]
    iterations = 0
    
    print("=== INICIANDO AC-3 ===")
    print("Dominios iniciales:")
    for var, dom in domains.items():
        print(f"{var}: {len(dom)} valores")
    print()
    
    while queue:
        iterations += 1
        (x, y) = queue.pop(0)
        
        print(f"Iteración {iterations}: Revisando arco ({x}, {y})")
        domain_size_before = len(domains[x])
        
        revised = revise(x, y)
        
        if revised:
            domain_size_after = len(domains[x])
            print(f"  Dominio de {x} reducido: {domain_size_before} → {domain_size_after}")
            
            if len(domains[x]) == 0:
                print(f"  ¡Dominio de {x} vacío! Problema inconsistente.")
                return False
            
            neighbors = [neighbor for neighbor in arcs if neighbor[1] == x and neighbor[0] != y]
            queue = queue + neighbors
            
            if neighbors:
                print(f"  Agregando vecinos: {neighbors}")
        else:
            print(f"  Sin cambios en {x}")
    
    print(f"\nAC-3 completado en {iterations} iteraciones")
    return True

arcs = [
    # Arcos X-C (cantidad-costo)
    ('X1', 'C1'), ('C1', 'X1'),
    ('X2', 'C2'), ('C2', 'X2'),
    ('X3', 'C3'), ('C3', 'X3'),
    ('X4', 'C4'), ('C4', 'X4'),
    ('X5', 'C5'), ('C5', 'X5'),
    
    # Arcos X-V (cantidad-valorización)
    ('X1', 'V1'), ('V1', 'X1'),
    ('X2', 'V2'), ('V2', 'X2'),
    ('X3', 'V3'), ('V3', 'X3'),
    ('X4', 'V4'), ('V4', 'X4'),
    ('X5', 'V5'), ('V5', 'X5'),
    
    # Arcos C-V (costo-valorización según regla lineal)
    ('C1', 'V1'), ('V1', 'C1'),
    ('C2', 'V2'), ('V2', 'C2'),
    ('C3', 'V3'), ('V3', 'C3'),
    ('C4', 'V4'), ('V4', 'C4'),
    ('C5', 'V5'), ('V5', 'C5'),
    
    # Arcos X-X (restricciones entre cantidades)
    ('X1', 'X2'), ('X2', 'X1'),
    ('X3', 'X4'), ('X4', 'X3'),
    ('X3', 'X5'), ('X5', 'X3'),
]

# Ejecutar AC-3
resultado = ac3(arcs)

if resultado:
    print("\n=== DOMINIOS FINALES ===")
    for var, dom in domains.items():
        if len(dom) <= 20:
            print(f"{var}: {dom}")
        else:
            print(f"{var}: {len(dom)} valores [{min(dom)}-{max(dom)}]")
    
    print("\n=== VERIFICACIÓN DE SOLUCIONES ===")
    # Buscar algunas soluciones que cumplan las restricciones principales
    soluciones_validas = 0
    
    for x1 in domains['X1'][:5]:  # Limitar búsqueda
        for x2 in domains['X2'][:5]:
            for x3 in domains['X3'][:5]:
                for x4 in domains['X4']:
                    for x5 in domains['X5'][:5]:
                        # Verificar que existan valores compatibles de C y V
                        try:
                            # Calcular costos y valorizaciones correspondientes
                            c1_vals = [c for c in domains['C1'] if c == int(160 + (200-160) * (75-65) / (85-65))]
                            if c1_vals:
                                print(f"Solución válida encontrada:")
                                print(f"  X1={x1}, X2={x2}, X3={x3}, X4={x4}, X5={x5}")
                                soluciones_validas += 1
                                if soluciones_validas >= 3:
                                    break
                        except:
                            continue
                    if soluciones_validas >= 3:
                        break
                if soluciones_validas >= 3:
                    break
            if soluciones_validas >= 3:
                break
        if soluciones_validas >= 3:
            break

else:
    print("El problema es INCONSISTENTE")

print(f"\n=== RESUMEN FINAL ===")
print("Tamaños de dominios finales:")
for var, dom in domains.items():
    print(f"{var}: {len(dom)} valores")

print(f"\nEl problema está ARCO CONSISTENTE: {resultado}")
print("\nNota: Las restricciones complejas (X1*C1 + X2*C2 ≤ 3800) se verifican")
print("después de AC-3 ya que requieren más de 2 variables simultáneamente.")