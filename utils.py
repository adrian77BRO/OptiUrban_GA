matriz_tiempo = None
matriz_consumo = None
matriz_seguridad = None

def calcular_estadisticas_ruta(ruta):
    dist_m = 0.0
    tiempo_min = 0.0
    inseg_total = 0.0

    for i in range(len(ruta) - 1):
        o = ruta[i]
        d = ruta[i + 1]
        dist_m += matriz_consumo[o, d]
        tiempo_min += matriz_tiempo[o, d]
        inseg_total += (1 - matriz_seguridad[o, d])

    dist_km = dist_m / 1000.0
    tiempo_h = tiempo_min / 60.0
    return dist_km, tiempo_h, inseg_total

def set_matrices(tiempo, consumo, seguridad):
    global matriz_tiempo, matriz_consumo, matriz_seguridad
    matriz_tiempo = tiempo
    matriz_consumo = consumo
    matriz_seguridad = seguridad