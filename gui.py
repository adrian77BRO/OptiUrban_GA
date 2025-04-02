import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import pandas as pd

from geal import set_matrices as set_algo_matrices, algoritmo_genetico, calcular_estadisticas_ruta, calcular_costo_ruta
from data import construir_matrices_origen_destino
from utils import set_matrices as set_util_matrices

matriz_tiempo = None
matriz_consumo = None
matriz_seguridad = None
nodos_relevantes = None
G = None
NUM_NODOS = 12

def ejecutar_ag_gui(lugar_str, orig_lat_str, orig_lon_str, dest_lat_str, dest_lon_str, num_nodos_str):
    global matriz_tiempo, matriz_consumo, matriz_seguridad, nodos_relevantes, G
    global NUM_NODOS

    lugar = lugar_str.strip()
    if not lugar:
        messagebox.showwarning("Aviso", "Por favor ingresa la ciudad o región (lugar).")
        return

    try:
        orig_lat = float(orig_lat_str)
        orig_lon = float(orig_lon_str)
        dest_lat = float(dest_lat_str)
        dest_lon = float(dest_lon_str)
        num_nodos = int(num_nodos_str)
    except ValueError:
        messagebox.showwarning("Aviso", "Coordenadas o número de nodos inválidos.")
        return

    if num_nodos < 2:
        messagebox.showwarning("Aviso", "El número de nodos debe ser >= 2.")
        return

    try:
        mt, mc, ms, nds, grafo = construir_matrices_origen_destino(
            lugar, orig_lat, orig_lon, dest_lat, dest_lon, num_nodos
        )
    except Exception as e:
        messagebox.showerror("Error", f"Error al construir matrices: {e}")
        return

    matriz_tiempo = mt
    matriz_consumo = mc
    matriz_seguridad = ms
    nodos_relevantes = nds
    G = grafo

    set_algo_matrices(mt, mc, ms)
    set_util_matrices(mt, mc, ms)

    NUM_NODOS = len(nodos_relevantes)
    if NUM_NODOS <= 1:
        messagebox.showwarning("Aviso", "No se encontraron nodos suficientes.")
        return

    mejor_ruta, mejor_costo, historial, poblacion_final = algoritmo_genetico(NUM_NODOS)

    print("\nMejor ruta (índices):", mejor_ruta)
    print("Costo de la mejor ruta:", mejor_costo)

    dist_km, tiempo_h, inseg_total = calcular_estadisticas_ruta(mejor_ruta)
    consumo_litros = (dist_km / 100.0) * 8.0

    messagebox.showinfo(
        "Resultados de la mejor ruta",
        f"Ruta (índices): {mejor_ruta}\n"
        f"Distancia total: {dist_km:.2f} km\n"
        f"Tiempo total: {tiempo_h:.2f} horas\n"
        f"Suma de inseguridad: {inseg_total:.2f}\n"
        f"Consumo estimado: {consumo_litros:.2f} L\n"
        f"Costo de la mejor ruta: {mejor_costo:.2f}"
    )

    plt.figure()
    plt.plot(historial, marker='o')
    plt.title('Evolución del fitness')
    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    plt.show()

    datos = []
    for r in poblacion_final:
        datos.append({"Ruta": r, "Costo": calcular_costo_ruta(r)})
    df = pd.DataFrame(datos)
    df_ordenado = df.sort_values(by='Costo', ascending=True).reset_index(drop=True)
    top5 = df_ordenado.head(5)
    pd.set_option('display.max_colwidth', None)
    print("\nTop 5 rutas:")
    print(top5)

    top5.to_csv("top5_rutas.csv", index=False)
    print("\nLas 5 mejores rutas se guardaron en el archivo 'top5_rutas.csv'")

    if len(mejor_ruta) > 1:
        import networkx as nx
        import osmnx as ox
        best_ruta_osm = []
        for i in range(len(mejor_ruta) - 1):
            s = nodos_relevantes[mejor_ruta[i]]
            t = nodos_relevantes[mejor_ruta[i+1]]
            sub_path = nx.shortest_path(G, source=s, target=t, weight='tiempo')
            best_ruta_osm.extend(sub_path[:-1])
        best_ruta_osm.append(nodos_relevantes[mejor_ruta[-1]])

        print("\nGraficando la mejor ruta en el mapa...")
        fig, ax = ox.plot_graph_route(G, best_ruta_osm, route_linewidth=4, node_size=0)
        plt.show()

def configurar_gui():
    ventana = tk.Tk()
    ventana.title("OPTIURBAN - AG con Origen/Destino en GUI")
    ventana.configure(bg="light green")
    ventana.geometry("700x500")

    font_labels = ("Helvetica", 14, "bold")
    font_entries = ("Helvetica", 12)

    frame_principal = tk.Frame(ventana, bg="light green")
    frame_principal.pack(expand=True, fill="both")

    lbl_lugar = tk.Label(frame_principal, text="Ciudad / Región (OSM):", bg="light green", font=font_labels)
    lbl_lugar.pack(pady=10)

    entry_lugar = tk.Entry(frame_principal, width=50, font=font_entries)
    entry_lugar.insert(0, "Tuxtla Gutiérrez, Chiapas, México")
    entry_lugar.pack(pady=5)

    frame_origen = tk.Frame(frame_principal, bg="light green")
    frame_origen.pack(pady=10)

    lbl_origen_lat = tk.Label(frame_origen, text="Origen lat:", bg="light green", font=font_labels)
    lbl_origen_lat.grid(row=0, column=0, padx=5, pady=5)
    entry_orig_lat = tk.Entry(frame_origen, width=12, font=font_entries)
    entry_orig_lat.insert(0, "16.70248")
    entry_orig_lat.grid(row=0, column=1, padx=5, pady=5)

    lbl_origen_lon = tk.Label(frame_origen, text="Origen lon:", bg="light green", font=font_labels)
    lbl_origen_lon.grid(row=0, column=2, padx=5, pady=5)
    entry_orig_lon = tk.Entry(frame_origen, width=12, font=font_entries)
    entry_orig_lon.insert(0, "-93.20500")
    entry_orig_lon.grid(row=0, column=3, padx=5, pady=5)

    frame_dest = tk.Frame(frame_principal, bg="light green")
    frame_dest.pack(pady=10)

    lbl_dest_lat = tk.Label(frame_dest, text="Destino lat:", bg="light green", font=font_labels)
    lbl_dest_lat.grid(row=0, column=0, padx=5, pady=5)
    entry_dest_lat = tk.Entry(frame_dest, width=12, font=font_entries)
    entry_dest_lat.insert(0, "16.80199")
    entry_dest_lat.grid(row=0, column=1, padx=5, pady=5)

    lbl_dest_lon = tk.Label(frame_dest, text="Destino lon:", bg="light green", font=font_labels)
    lbl_dest_lon.grid(row=0, column=2, padx=5, pady=5)
    entry_dest_lon = tk.Entry(frame_dest, width=12, font=font_entries)
    entry_dest_lon.insert(0, "-93.11892")
    entry_dest_lon.grid(row=0, column=3, padx=5, pady=5)

    lbl_num_nodos = tk.Label(frame_principal, text="Número de paradas totales:", bg="light green", font=font_labels)
    lbl_num_nodos.pack(pady=10)

    entry_num_nodos = tk.Entry(frame_principal, width=5, font=font_entries)
    entry_num_nodos.insert(0, "12")
    entry_num_nodos.pack(pady=5)

    btn_ejecutar = tk.Button(frame_principal, text="Generar la mejor ruta",
                             font=("Helvetica", 14, "bold"),
                             bg="#CCCCCC",
                             command=lambda: ejecutar_ag_gui(
                                 entry_lugar.get(),
                                 entry_orig_lat.get(),
                                 entry_orig_lon.get(),
                                 entry_dest_lat.get(),
                                 entry_dest_lon.get(),
                                 entry_num_nodos.get()
                             ))
    btn_ejecutar.pack(pady=20)

    return ventana