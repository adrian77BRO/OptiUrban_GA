import random
import numpy as np
import matplotlib.pyplot as plt
from utils import calcular_estadisticas_ruta

# Parámetros del AG
TAM_POBLACION = 100
NUM_GENERACIONES = 50
TASA_CRUZAMIENTO = 0.8
TASA_MUTACION = 0.4
TORNEO_SIZE = 3

# Pesos del costo
w_t = 0.2
w_c = 0.7
w_p = 0.3
w_s = 0.2

CONSUMO_PROMEDIO_L_100KM = 8.0

# Matrices globales
matriz_tiempo = None
matriz_consumo = None
matriz_seguridad = None


def calcular_costo_ruta(ruta):
    global matriz_tiempo, matriz_consumo, matriz_seguridad
    tiempo_total = 0.0
    consumo_total = 0.0
    inseguridad_total = 0.0

    for i in range(len(ruta) - 1):
        o = ruta[i]
        d = ruta[i + 1]
        tiempo_total += matriz_tiempo[o, d]
        consumo_total += matriz_consumo[o, d]
        inseguridad_total += (1 - matriz_seguridad[o, d])

    tiempo_h = tiempo_total / 60.0
    dist_km = consumo_total / 1000.0
    consumo_litros = (dist_km / 100.0) * CONSUMO_PROMEDIO_L_100KM
    num_paradas = len(ruta) - 1

    costo = (w_t * tiempo_h + w_c * consumo_litros + w_p * num_paradas + w_s * inseguridad_total)
    return costo


def crear_ruta_aleatoria_fijando_extremos(num_nodos):
    if num_nodos <= 2:
        return list(range(num_nodos))
    indices = list(range(num_nodos))
    intermedios = indices[1:-1]
    random.shuffle(intermedios)
    return [0] + intermedios + [num_nodos - 1]


def crear_poblacion_inicial(tam_poblacion, num_nodos):
    return [crear_ruta_aleatoria_fijando_extremos(num_nodos) for _ in range(tam_poblacion)]


def seleccion_por_torneo(poblacion, k=TORNEO_SIZE):
    elegidos = random.sample(poblacion, k)
    return min(elegidos, key=calcular_costo_ruta)


def order_crossover(padre1, padre2):
    size = len(padre1)
    hijo1 = [None] * size
    hijo2 = [None] * size

    if size <= 2:
        return padre1[:], padre2[:]

    inicio, fin = sorted(random.sample(range(1, size - 1), 2))

    hijo1[0], hijo1[-1] = padre1[0], padre1[-1]
    hijo2[0], hijo2[-1] = padre2[0], padre2[-1]

    for i in range(inicio, fin + 1):
        hijo1[i] = padre1[i]
        hijo2[i] = padre2[i]

    def rellenar(hijo, padre):
        pos = (fin + 1) % size
        while pos in [0, size - 1]:
            pos = (pos + 1) % size
        for i in range(size):
            idx = (fin + 1 + i) % size
            gene = padre[idx]
            if gene not in hijo:
                hijo[pos] = gene
                pos = (pos + 1) % size
                while pos in [0, size - 1]:
                    pos = (pos + 1) % size

    rellenar(hijo1, padre2)
    rellenar(hijo2, padre1)

    return hijo1, hijo2


def mutacion(ruta):
    size = len(ruta)
    if size <= 2:
        return ruta
    idx1, idx2 = random.sample(range(1, size - 1), 2)
    ruta[idx1], ruta[idx2] = ruta[idx2], ruta[idx1]
    return ruta


def algoritmo_genetico(num_nodos):
    global matriz_tiempo, matriz_consumo, matriz_seguridad

    poblacion = crear_poblacion_inicial(TAM_POBLACION, num_nodos)
    mejor_global = None
    costo_mejor_global = float("inf")
    historial_mejor_aptitud, historial_promedio, historial_max, historial_min = [], [], [], []
    distancias, tiempos, inseguridades, consumos_litros = [], [], [], []

    for gen in range(NUM_GENERACIONES):
        nueva_poblacion = []

        while len(nueva_poblacion) < TAM_POBLACION:
            padre1 = seleccion_por_torneo(poblacion)
            padre2 = seleccion_por_torneo(poblacion)

            if random.random() < TASA_CRUZAMIENTO:
                hijo1, hijo2 = order_crossover(padre1, padre2)
            else:
                hijo1, hijo2 = padre1[:], padre2[:]

            if random.random() < TASA_MUTACION:
                hijo1 = mutacion(hijo1)
            if random.random() < TASA_MUTACION:
                hijo2 = mutacion(hijo2)

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < TAM_POBLACION:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion
        costos = [calcular_costo_ruta(ind) for ind in poblacion]
        mejor_local = poblacion[np.argmin(costos)]
        costo_mejor_local = min(costos)

        if costo_mejor_local < costo_mejor_global:
            mejor_global = mejor_local
            costo_mejor_global = costo_mejor_local

        historial_mejor_aptitud.append(costo_mejor_global)
        historial_promedio.append(np.mean(costos))
        historial_min.append(np.min(costos))
        historial_max.append(np.max(costos))

        dist_km, tiempo_h, inseg = calcular_estadisticas_ruta(mejor_local)
        consumo_l = (dist_km / 100.0) * CONSUMO_PROMEDIO_L_100KM

        distancias.append(dist_km)
        tiempos.append(tiempo_h)
        inseguridades.append(inseg)
        consumos_litros.append(consumo_l)

    fig, ax1 = plt.subplots()
    ax1.plot(historial_mejor_aptitud, label='Fitness mínimo', color='green', marker='o')
    ax1.plot(historial_promedio, label='Fitness promedio', linestyle='--', color='blue')
    ax1.plot(historial_max, label='Fitness máximo', linestyle='--', color='red')
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Generación')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(distancias, label='Distancia (km)', linestyle=':', color='orange')
    ax2.plot(tiempos, label='Tiempo (h)', linestyle=':', color='purple')
    ax2.plot(inseguridades, label='Inseguridad', linestyle=':', color='brown')
    ax2.plot(consumos_litros, label='Consumo (L)', linestyle='-.', color='cyan')
    ax2.set_ylabel('Parámetros individuales')
    ax2.legend(loc='upper right')

    plt.title('Evolución del AG: fitness y métricas')
    plt.tight_layout()
    plt.show()

    return mejor_global, costo_mejor_global, historial_mejor_aptitud, poblacion


def set_matrices(tiempo, consumo, seguridad):
    global matriz_tiempo, matriz_consumo, matriz_seguridad
    matriz_tiempo = tiempo
    matriz_consumo = consumo
    matriz_seguridad = seguridad