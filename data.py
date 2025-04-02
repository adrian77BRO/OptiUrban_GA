import osmnx as ox
import networkx as nx
import random
import numpy as np
from heapq import nsmallest
from osmnx import distance

def construir_matrices_origen_destino(lugar, orig_lat, orig_lon, dest_lat, dest_lon, num_total_nodos=8):
    G_local = ox.graph_from_place(lugar, network_type='drive')
    G_local = G_local.to_undirected()

    for u, v, key, data in G_local.edges(keys=True, data=True):
        dist_m = data.get("length", 1.0)
        vel_kmh = 30.0
        vel_m_min = (vel_kmh * 1000) / 60.0
        t_min = dist_m / vel_m_min
        data["tiempo"] = t_min

    origen_node = distance.nearest_nodes(G_local, orig_lon, orig_lat)
    destino_node = distance.nearest_nodes(G_local, dest_lon, dest_lat)

    n_intermedios = num_total_nodos - 2
    if n_intermedios < 0:
        n_intermedios = 0

    origen_x = G_local.nodes[origen_node]['x']
    origen_y = G_local.nodes[origen_node]['y']
    destino_x = G_local.nodes[destino_node]['x']
    destino_y = G_local.nodes[destino_node]['y']

    eucl_candidatos = []
    for n in G_local.nodes:
        if n in (origen_node, destino_node):
            continue
        x = G_local.nodes[n]['x']
        y = G_local.nodes[n]['y']
        d_origen = ((x - origen_x)**2 + (y - origen_y)**2)**0.5
        d_destino = ((x - destino_x)**2 + (y - destino_y)**2)**0.5
        eucl_candidatos.append((n, d_origen, d_destino))

    cerca_origen = [n for n, _, _ in nsmallest(300, eucl_candidatos, key=lambda x: x[1])]
    cerca_destino = [n for n, _, _ in nsmallest(300, eucl_candidatos, key=lambda x: x[2])]
    nodos_candidatos = list(set(cerca_origen + cerca_destino))

    distancias_conjuntas = []
    for n in nodos_candidatos:
        try:
            d_origen = nx.shortest_path_length(G_local, origen_node, n, weight='length')
            d_destino = nx.shortest_path_length(G_local, n, destino_node, weight='length')
            suma_distancias = d_origen + d_destino
            distancias_conjuntas.append((n, suma_distancias))
        except nx.NetworkXNoPath:
            continue

    porc_cercanos = 0.6
    num_cercanos = int(n_intermedios * porc_cercanos)
    num_random = n_intermedios - num_cercanos

    nodos_cercanos = [n for n, _ in nsmallest(num_cercanos, distancias_conjuntas, key=lambda x: x[1])]
    restantes = [n for n, _ in distancias_conjuntas if n not in nodos_cercanos]
    nodos_random = random.sample(restantes, min(num_random, len(restantes)))

    intermedios = nodos_cercanos + nodos_random
    random.shuffle(intermedios)

    nodos = [origen_node] + intermedios + [destino_node]
    n = len(nodos)

    mt = np.zeros((n, n))
    mc = np.zeros((n, n))
    ms = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                mt[i, j] = 0
                mc[i, j] = 0
                ms[i, j] = 1.0
            else:
                try:
                    r_corta = nx.shortest_path(G_local, source=nodos[i], target=nodos[j], weight='tiempo')
                    t_total = 0.0
                    d_total = 0.0
                    for k in range(len(r_corta) - 1):
                        u = r_corta[k]
                        v = r_corta[k + 1]
                        edge_data = G_local[u][v][0]
                        t_total += edge_data.get("tiempo", 1)
                        d_total += edge_data.get("length", 1)

                    mt[i, j] = t_total
                    mc[i, j] = d_total
                    ms[i, j] = random.uniform(0.5, 1.0)

                except nx.NetworkXNoPath:
                    mt[i, j] = 999999
                    mc[i, j] = 999999
                    ms[i, j] = 0.0

    return mt, mc, ms, nodos, G_local