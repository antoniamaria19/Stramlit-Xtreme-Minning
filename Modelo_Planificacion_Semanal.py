# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math 
from contextlib import redirect_stdout
from tabulate import tabulate
from contextlib import redirect_stdout
from IPython.display import display, HTML
import time
import Funciones as f
import streamlit as st
#from gurobipy import *


# In[2]:

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ### Funciones

# #### Cargar Planificación Semanal

# In[3]:


def procesar_planificacion(file_path, file_name, df_tiempos, df_familia, df_prioridad, fechas, turnos):
    """
    Procesa un archivo de planificación, duplicando filas según la cantidad de camiones y calculando campos necesarios.

    Parámetros:
        file_path (str): Ruta del directorio donde se encuentra el archivo.
        file_name (str): Nombre del archivo de planificación.
        fechas (list): Lista de fechas a incluir en el DataFrame.
        turnos (list): Lista de turnos a incluir (e.g., ['TA', 'TB']).
        tabla_auxiliar_path (str): Ruta del archivo Excel con la tabla auxiliar de tiempos.

    Retorna:
        pd.DataFrame: DataFrame procesado con las columnas ajustadas y duplicadas según la cantidad de camiones.
    """
    # Construir la ruta completa del archivo
    archivo_completo = os.path.join(file_path, file_name)

    # Cargar la planificación y filtrar por las fechas y turnos especificados
    df_std_pl = pd.read_excel(archivo_completo, sheet_name='Sheet1')
    # Asegurarte de que 'Fecha' y 'fechas' sean del mismo tipo (datetime, por ejemplo)
    df_std_pl['Fecha'] = pd.to_datetime(df_std_pl['Fecha'], errors='coerce')  # Convertir 'Fecha' en el DataFrame
    fechas = pd.to_datetime(fechas, errors='coerce')  # Convertir 'fechas' a datetime

    # Asegurarte de que 'Turno' y 'turnos' sean del mismo tipo (string, por ejemplo)
    df_std_pl['Turno'] = df_std_pl['Turno'].astype(str)  # Convertir 'Turno' en el DataFrame a string
    turnos = [str(turno) for turno in turnos]  # Convertir cada elemento de 'turnos' a string

    # Aplicar el filtro con 'isin' después de la conversión de tipos
    df_std_pl = df_std_pl[
        df_std_pl['Fecha'].isin(fechas) & df_std_pl['Turno'].isin(turnos)
    ].sort_values(by=['Fecha', 'Turno'], ascending=[True, True])

    #df_std_pl = df_std_pl[df_std_pl['Fecha'].isin(fechas) & df_std_pl['Turno'].isin(turnos)].sort_values(by=['Fecha', 'Turno'], ascending=[True, True])
    
    # Realizar cálculos y transformaciones 
    # Realizar cálculos y transformaciones 
    df_std_pl['Volumen_m3'] = df_std_pl['Volumen Total [m3]']  # Transformar "M3" a "Volumen_m3"
    df_std_pl['Codigo_producto'] = df_std_pl['Codigo Producto']  # Nombre columna "Codigo"
    df_std_pl['Descripcion'] = df_std_pl['Descripcion del Producto']  # Nombre columna "Descripción"
    df_std_pl['Frecuencia'] = df_std_pl['Frecuencia (minutos)']  # Nombre de columna "frecuencia"
    df_std_pl['Camiones_pedido'] = df_std_pl['Volumen_m3'].apply(lambda x: math.ceil(x / 7))  # Calcular "Camiones_pedido"
    df_std_pl['Llegada_obra_min'] = 63  # Igual para todas las filas (valor original = 55)
    df_std_pl['Atencion_min'] = 33  # Igual para todas las filas (valor original = 22)
    df_std_pl['Retorno_min'] = 42  # Igual para todas las filas (valor original = 38)
    df_std_pl['Lavado_min'] = 30  # Igual para todas las filas (valor original = 15)
    df_std_pl['Punto_entrega'] = df_std_pl['Punto de Entrega']  # Nombre columna "Punto de entrega"
    df_std_pl['Destino_final'] = df_std_pl['Destino Final']  # Nombre columna "Destino final"
    df_std_pl['Total_vuelta_min'] = df_std_pl['Llegada_obra_min'] + df_std_pl['Atencion_min'] + df_std_pl['Retorno_min'] + df_std_pl['Lavado_min']  # Calcular total
    
    # Crear columna "Pedido_ID" con números ascendentes
    df_std_pl['Pedido_ID'] = range(1, len(df_std_pl) + 1)

    # Obtener la fecha inicial como referencia con formato explícito
    fecha_inicial = pd.to_datetime(fechas[0], format='%d-%m-%Y', dayfirst=True)


    # Cargar la tabla auxiliar de tiempos
    #df_tiempos = pd.read_excel(archivo_completo, sheet_name='Tiempos')

    #MERGE CON TABLAS AUXILIARES

    #1. Tiempos
    # Realizar el merge con la tabla auxiliar para obtener tiempos específicos
    df_std_pl = df_std_pl.merge(
        df_tiempos[['Empresa', 'Punto_entrega', 'Llegada_obra_min', 'Atencion_min', 'Retorno_min', 'Lavado_min', 'Total_vuelta_min']],
        on=['Empresa', 'Punto_entrega'],
        how='left',
        suffixes=('', '_ref')
    )
    
    # Crear series de valores por defecto del mismo tamaño que el DataFrame
    default_llegada_obra = pd.Series(63, index=df_std_pl.index)
    default_atencion_min = pd.Series(33, index=df_std_pl.index)
    default_retorno_min = pd.Series(42, index=df_std_pl.index)
    default_lavado_min = pd.Series(15, index=df_std_pl.index)

    # Usar valores por defecto si no hay match en la tabla auxiliar
    df_std_pl['Llegada_obra_min'] = df_std_pl['Llegada_obra_min_ref'].combine_first(default_llegada_obra).round(0).astype(int)
    df_std_pl['Atencion_min'] = df_std_pl['Atencion_min_ref'].combine_first(default_atencion_min).round(0).astype(int)
    df_std_pl['Retorno_min'] = df_std_pl['Retorno_min_ref'].combine_first(default_retorno_min).round(0).astype(int)
    df_std_pl['Lavado_min'] = df_std_pl['Lavado_min_ref'].combine_first(default_lavado_min).round(0).astype(int)
    df_std_pl['Total_vuelta_min'] = df_std_pl['Total_vuelta_min_ref'].combine_first(
        df_std_pl['Llegada_obra_min'] + df_std_pl['Atencion_min'] + df_std_pl['Retorno_min'] + df_std_pl['Lavado_min']
    ).round(0).astype(int)

    # Eliminar columnas auxiliares con el sufijo '_ref'
    df_std_pl.drop(columns=['Llegada_obra_min_ref', 'Atencion_min_ref', 'Retorno_min_ref', 'Lavado_min_ref', 'Total_vuelta_min_ref'], inplace=True)
    
    # Añadir columnas con tiempos de pre y post descarga
    df_std_pl['Tiempo_pre_descarga'] = df_std_pl['Llegada_obra_min']
    df_std_pl['Tiempo_post_descarga'] = df_std_pl['Atencion_min'] + df_std_pl['Retorno_min'] 
    # Función para duplicar las filas por la cantidad de camiones

    # Cargar datos auxiliares
    #df_familia = pd.read_excel(archivo_completo, sheet_name='Productos')
    #df_prioridad = pd.read_excel(archivo_completo, sheet_name='Prioridades')

    #2. Familia de prododucto 
    # Merge con la tabla de productos para obtener la familia
    df_std_pl = df_std_pl.merge(df_familia[['Codigo_producto', 'Familia']], on='Codigo_producto', how='left')
    
    #3. Prioridades 
    # Merge con la tabla de prioridades, asignando prioridad BAJA cuando no hay match
    df_std_pl = df_std_pl.merge(
        df_prioridad[['Empresa', 'Familia', 'Prioridad']],
        on=['Empresa', 'Familia'],
        how='left'
    )

    df_std_pl['Prioridad'] = df_std_pl['Prioridad'].fillna('BAJA')

    def duplicar_filas_por_camiones(df):
        # Repetir las filas según el valor en 'Camiones_pedido'
        df_duplicado = df.loc[df.index.repeat(df['Camiones_pedido'])].copy()
        # Asignar un número secuencial por camión a cada pedido (1, 2, 3, ...)
        df_duplicado['Camion_secuencia'] = df_duplicado.groupby('Pedido_ID').cumcount() + 1
        return df_duplicado

    # Duplicar las filas por la cantidad de camiones
    df_resultado = duplicar_filas_por_camiones(df_std_pl)

    # Añadir una columna 'Viaje_ID' con un valor secuencial único para cada fila
    df_resultado['Viaje_ID'] = range(1, len(df_resultado) + 1)

    # Función para calcular el volumen transportado, considerando que los primeros camiones llevan 7 m³ y el último el remanente
    def calcular_volumen_transportado_max_capacidad(df):
        df['Volumen_transportado'] = np.where(df['Camion_secuencia'] < df['Camiones_pedido'], 7, df['Volumen_m3'] - 7 * (df['Camiones_pedido'] - 1))
        return df

    # Aplicar la función para calcular el volumen transportado por cada camión
    df_resultado = calcular_volumen_transportado_max_capacidad(df_resultado)
    

    # Inicializar columnas adicionales necesarias
    df_resultado['Camion_ID'] = np.nan
    df_resultado['Tiempo_carga'] = np.nan
    df_resultado['Tiempo_llegada'] = np.nan
    df_resultado['Retorno_depo'] = np.nan
    df_resultado['Hora_llegada_obra'] = np.nan
    df_resultado['Hora_fin_atencion'] = np.nan
    df_resultado['Hora_retorno'] = np.nan
    df_resultado['Hora_fin_lavado'] = np.nan
    df_resultado['Planta_salida'] = np.nan

    
    # Reordenar columnas en el orden solicitado
    columnas_finales = ['Fecha',
        'Viaje_ID', 'Pedido_ID', 'Empresa', 'Codigo_producto','Familia', 'Descripcion', 'Volumen_m3', 
        'Turno', 'Camiones_pedido', 'Camion_secuencia', 'Camion_ID', 'Volumen_transportado', 'Prioridad',
        'Frecuencia', 'Tiempo_carga', 'Tiempo_llegada', 'Retorno_depo', 'Llegada_obra_min', 
        'Atencion_min', 'Retorno_min', 'Lavado_min', 'Total_vuelta_min', 'Tiempo_pre_descarga',
        'Tiempo_post_descarga', 'Punto_entrega', 'Destino_final', 'Obra', 'Uso', 
        'Hora_llegada_obra', 'Hora_fin_atencion', 'Hora_retorno', 'Hora_fin_lavado', 
        'Planta_salida'
    ]

    df_resultado = df_resultado[columnas_finales]

    return df_resultado


# #### Crear y Ejecutar Modelo de Optimización

# In[16]:
# Función para optimizar la planificación semanal
#MODELO ORTOOLS
def optimizar_planificacion_semanal_ortools(df_resultado, time_limit, plants, K = 20):

    # Crear el modelo
    model = pywraplp.Solver.CreateSolver('SCIP')

    # Conjuntos
    N = df_resultado.shape[0] # Nodos a atender en función de la cantidad de pedidos
    A = list(range(N, N + K))  # Índices de nodos de almuerzo (uno por camión)
    nodos = list(range(N + K))  # Se redefine para incluir los nodos de almuerzo asociados a cada camión
    arcos = [(i, j) for i in nodos for j in nodos if i != j]
    K = 20
    vehiculos = list(range(K))
    nodos_almuerzo = {N + i: {'t0i': 0, 'p': 60, 'ti0': 0, 'p0': 0}
                  for i in range(K)} # Definir nodos de almuerzo como un diccionario
    
    # Parámetros
    Li = 0
    Ls = 720
    t0i = np.append(df_resultado['Llegada_obra_min'].values, [nodos_almuerzo[i]['t0i'] for i in A])  # El almuerzo no tiene tiempo de llegada
    p = np.append(df_resultado['Atencion_min'].values, [nodos_almuerzo[i]['p'] for i in A])  # El almuerzo dura 60 minutos
    ti0 = np.append(df_resultado['Retorno_min'].values, [nodos_almuerzo[i]['ti0'] for i in A])  # No hay retorno asociado al almuerzo
    p0 = np.append(df_resultado['Lavado_min'].fillna(0).values, [nodos_almuerzo[i]['p0'] for i in A])  # Asegura que el almuerzo dura 60 min
    M = 10000

    # Variables de decisión
    x = {(i, k): model.BoolVar(f'x_{i}_{k}') for i in nodos for k in vehiculos}
    z = {(i, j, k): model.BoolVar(f'z_{i}_{j}_{k}') for i, j in arcos for k in vehiculos}
    d = {k: model.BoolVar(f'd_{k}') for k in vehiculos}
    C = {i: model.NumVar(0.0, model.infinity(), f'C_{i}') for i in nodos}
    phi = {i: model.NumVar(0.0, model.infinity(), f'phi_{i}') for i in nodos}
    w = {(i, j): model.BoolVar(f'w_{i}_{j}') for i, j in arcos}

    # Función objetivo: minimizar la distancia total recorrida y los costos asociados
    objective = model.Sum(d[k] for k in vehiculos) + 0.000000001* model.Sum(phi[i] for i in nodos)
    model.Minimize(objective)

    #Restricciones
    # Ventana de almuerzo (de 240 a 360 minutos)
    for almuerzo_k in A:
        model.Add(C[almuerzo_k] >= 240)  # Límite inferior
        model.Add(C[almuerzo_k] <= 360)  # Límite superior

    # Atender todos los pedidos i
    for i in nodos:
        if i not in A:
            model.Add(model.Sum(x[i, k] for k in vehiculos) == 1)

    # Todos los camiones utilizados almuerzan
    for k in vehiculos:
        model.Add(model.Sum(x[almuerzo_k, k] for almuerzo_k in A) == d[k])

    # 2. Restricción: z[i,j,k] <= x[i,k] // predecisión supeditada a atención
    for i, j in arcos:
        for k in vehiculos:
            model.Add(z[i, j, k] <= x[i, k])  # Redundante, pero se mantiene por consistencia

    # 3. Restricción: z[i,j,k] <= x[j,k] // predecisión supeditada a atención
    for i, j in arcos:
        for k in vehiculos:
            model.Add(z[i, j, k] <= x[j, k])  # Redundante, pero se mantiene por consistencia

    # 4. Restricción: z[i,j,k] + z[j,i,k] >= x[i,k] + x[j,k] - 1
    for i in nodos:
        for j in nodos:
            if i != j:
                for k in vehiculos:
                    model.Add(z[i, j, k] + z[j, i, k] >= x[i, k] + x[j, k] - 1)

    # 5. Restricción: Atención pedido i con camión k supeditada a utilización de camión k
    for i in nodos:
        for k in vehiculos:
            model.Add(x[i, k] <= d[k])


    # 6. Restricción: MTZ
    for i, j in arcos:
        model.Add(
            C[j] >= phi[i] + t0i[j] + p0[i].item() - M * (1 - sum(z[i, j, k] for k in vehiculos))
        )


    # Restricción: Li[i] <= C[i] <= Ls
    for i in nodos:
        model.Add(C[i] >= Li)  # Límite inferior
        model.Add(C[i] <= Ls)  # Límite superior

    # Break symmetry
    for k in vehiculos:
        if k >= 1:
            model.Add(d[k] <= d[k - 1])

    # 8. Restricciones de no negatividad: C[i] >= 0
    for i in nodos:
        model.Add(C[i] >= 0)


    # Restricciones TAP
    for i in nodos:
        model.Add(C[i] >= 61 + Li)  # Atención después de apertura de TAP
        model.Add(C[i] <= Ls - 61)  # Atención antes del cierre del TAP
        model.Add(C[i] - t0i[i] >= 45)  # Carga comienza 15 min antes de apertura del TAP


    # Vuelta en horario (Llegada a depósito 1 hr antes del fin de turno)
    for i in nodos:
        model.Add(phi[i] <= Ls - 61)

    # Restricción de número máximo de viajes
    for k in vehiculos:
        model.Add(model.Sum(x[i, k] for i in nodos if i not in A) <= 3)

    #for i in nodos:
    #    model.Add(C[i] - t0i[i] >= 0)

    if plants == 1:
        # Restricción de exclusividad con Big-M + definición de phi con igualdad
        for i in nodos:
            model.Add(phi[i] == C[i] + p[i] + ti0[i])

        for (i, j) in arcos:
            if i not in A and j not in A:
                model.Add(C[i] - t0i[i] >= C[j] - t0i[j] + p0[i] - M * (1 - w[i, j]))
                model.Add(C[i] - t0i[i] <= C[j] - t0i[j] - p0[i] + M * w[i, j])

    else:
        # Horarios de salida espaciados por tiempo de carga
        for i in nodos:
            model.Add(phi[i] >= C[i] + p[i] + ti0[i])

        for (i, j) in arcos:
            if i not in A and j not in A:
                model.Add(C[i] - t0i[i] <= C[j] - t0i[j] - p0[i] + M * (1 - w[i, j]))
                model.Add(C[j] - t0i[j] <= C[i] - t0i[i] + p0[i] + M * w[i, j])

        for i in nodos:
            if i not in A:
                model.Add(model.Sum(w[i, j] + w[j, i] for j in nodos if j != i and j not in A) >= N - 2)


    # Establecer parámetros de tiempo y MIPGap
    model.SetTimeLimit(time_limit * 1000)  # OR-Tools usa milisegundos
    model.SetSolverSpecificParametersAsString('limits/gap = 0.02')

    # Resolver el modelo
    # Medir tiempo de resolución
    start_time = time.time()
    status = model.Solve()
    end_time = time.time()
    solve_time = end_time - start_time

    # Verificar el estado de la solución
    if status == pywraplp.Solver.INFEASIBLE:
        print("El modelo es infactible.")
        return False, model, pd.DataFrame()

    elif status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        print("Solución Óptima Encontrada")

        # Extraer resultados
        resultados = {
            'Pedido_ID': [],
            'Camion_ID': [],
            'Tiempo_carga': [],
            'Tiempo_llegada': [],
            'Retorno_depo': [],
        }

        # Definir Tiempo_carga en el DataFrame antes de la optimización
        if 'Tiempo_carga' not in df_resultado.columns:
            df_resultado['Tiempo_carga'] = np.nan

        # Extraer las variables de decisión
        for i in nodos:
            for k in vehiculos:
                if x[i, k].solution_value() > 0.9:
                    tiempo_carga = C[i].solution_value() - t0i[i]
                    resultados['Pedido_ID'].append(i)
                    resultados['Camion_ID'].append(k)
                    resultados['Tiempo_carga'].append(round(tiempo_carga))
                    resultados['Tiempo_llegada'].append(C[i].solution_value())
                    resultados['Retorno_depo'].append(phi[i].solution_value())

        # Convertir resultados a DataFrame
        df_resultados_camiones = pd.DataFrame(resultados)

        return True, model, df_resultados_camiones,solve_time

    else:
        print("No se encontró solución óptima.")
        return False, model, pd.DataFrame(),solve_time

def crear_dataframe_resultados(df_resultado, df_resultados_camiones):
    """
    Procesa los resultados de la optimización para generar un DataFrame detallado.

    Parámetros:
        df_resultado (pd.DataFrame): DataFrame original con los datos base.
        df_resultados_camiones (pd.DataFrame): Resultados del modelo optimizado.
        tabla_auxiliar_path (str): Ruta del archivo Excel con tablas auxiliares ('Productos' y 'Prioridades').

    Retorna:
        pd.DataFrame: DataFrame con los resultados procesados y enriquecidos.
    """
    # Ajustar Pedido_ID para realizar el merge
    df_resultados_camiones['Pedido_ID_Ajustado'] = df_resultados_camiones['Pedido_ID'] + 1

    # Realizar el merge con los resultados de la optimización
    # Realizar el merge con los resultados de la optimización
    df_merged = pd.merge(
        df_resultado,
        df_resultados_camiones[['Pedido_ID_Ajustado', 'Camion_ID', 'Tiempo_carga','Tiempo_llegada', 'Retorno_depo']].rename(
            columns={'Pedido_ID_Ajustado': 'Viaje_ID'}
        ),
        on='Viaje_ID',
        how='inner'
    )

    # Renombrar columnas relevantes
    df_merged = df_merged.rename(columns={
        'Camion_ID_y': 'Camion_ID',
        'Tiempo_llegada_y': 'Tiempo_llegada',
        'Tiempo_carga_y': 'Tiempo_carga',
        'Retorno_depo_y': 'Retorno_depo'
    })

    # Dropear columnas antiguas e irrelevantes
    columnas_a_eliminar = [
        'Camion_ID_x', 'Tiempo_llegada_x','Tiempo_carga_x', 'Retorno_depo_x'
    ]
    df_merged = df_merged.drop(columns=columnas_a_eliminar, errors='ignore')

    def convertir_a_hhmm(minutos_totales, turno):
        """
        Converts elapsed minutes from a reference point to HH:MM format.

        - If the shift is TA (08:00 to 20:00), the reference is 08:00.
        - If the shift is TB (20:00 to 08:00), the reference is 20:00.

        Parameters:
            minutos_totales (int or float): Minutes elapsed from the shift reference.
            turno (str): 'TA' for the daytime shift (08:00 - 20:00) or 'TB' for the nighttime shift (20:00 - 08:00).

        Returns:
            str: Time in "HH:MM" format or None if the value is invalid.
        """
        if pd.isna(minutos_totales) or not isinstance(minutos_totales, (int, float)):
            return None  # Handle null or non-numeric values

        minutos_totales = int(minutos_totales)  # Ensure integers
        horas = minutos_totales // 60
        minutos = minutos_totales % 60

        if turno == "TA":
            hora_resultado = (horas + 8) % 24  # Reference 08:00
        elif turno == "TB":
            hora_resultado = (horas + 20) % 24  # Reference 20:00
        else:
            return None  # Return None if the shift is invalid

        return f"{hora_resultado:02d}:{minutos:02d}"
               
    # Aplicar la función al DataFrame considerando la columna 'Turno'
    df_merged['Hora_llegada_obra'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Tiempo_llegada'], row['Turno']),
        axis=1
    )

    df_merged['Hora_carga'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Tiempo_carga'], row['Turno']),
        axis=1
    )

    df_merged['Hora_fin_atencion'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Tiempo_llegada'] + row['Atencion_min'], row['Turno']),
        axis=1
    )

    df_merged['Hora_salida'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Retorno_depo'], row['Turno']),
        axis=1
    )

    df_merged['Hora_retorno'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Retorno_depo'], row['Turno']), axis=1
    )

    df_merged['Hora_fin_lavado'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Retorno_depo'] + row['Lavado_min'], row['Turno']),
        axis=1
    )

    # Ordenar por tiempo de carga en orden ascendente
    df_merged = df_merged.sort_values(by='Tiempo_carga', ascending=True)


    # Detectar si se necesita una segunda planta
    necesita_segunda_planta = False
    max_salida_anterior = float('-inf')


    for index, row in df_merged.iterrows():
        if row['Tiempo_carga'] - max_salida_anterior < 15:
            necesita_segunda_planta = True
            break  # Se confirma la necesidad y se sale del bucle
        max_salida_anterior = max(max_salida_anterior, row['Tiempo_carga'])


    # Asignar plantas de forma alternada si se necesita la segunda planta
    if necesita_segunda_planta:
        df_merged['Planta_salida'] = ['Planta 1' if i % 2 == 0 else 'Planta 2' for i in range(len(df_merged))]
    else:
        df_merged['Planta_salida'] = 'Planta 1'

    
    # Reordenar columnas en el orden especificado
    columnas_finales = [
        'Fecha', 'Viaje_ID', 'Pedido_ID', 'Empresa', 'Codigo_producto', 'Familia', 'Descripcion', 'Volumen_m3', 'Turno',
        'Camiones_pedido', 'Camion_secuencia', 'Camion_ID', 'Volumen_transportado', 'Frecuencia', 'Tiempo_carga', 'Tiempo_llegada',
        'Retorno_depo', 'Llegada_obra_min', 'Atencion_min', 'Retorno_min', 'Lavado_min', 'Total_vuelta_min', 'Tiempo_pre_descarga',
        'Tiempo_post_descarga', 'Punto_entrega', 'Destino_final', 'Obra', 'Uso', 'Hora_carga', 'Hora_llegada_obra', 'Hora_fin_atencion', 'Hora_retorno', 
        'Hora_fin_lavado', 'Prioridad', 'Planta_salida'
    ]

    # Filtrar columnas en el orden deseado
    df_merged = df_merged[columnas_finales]

    return df_merged

def ejecutar_proceso_planificacion(file_path, file_name, df_tiempos, df_familia, df_prioridad, fechas, turnos,plants,time):
    """
    Función que ejecuta todo el flujo de planificación: cargar datos, procesar, optimizar
    y obtener los resultados.

    Parámetros:
    - file_path (str): Ruta del directorio donde están los archivos.
    - file_name (str): Nombre del archivo de planificación semanal.
    - fechas (list): Lista de fechas a considerar en el análisis.
    - turnos (list): Lista de turnos a incluir en el procesamiento.

    Retorno:
    - df_merged (pd.DataFrame): DataFrame final con los resultados procesados y optimizados.
    """
    # Paso 1: Cargar archivos de planificación
    df_resultado = procesar_planificacion(file_path, file_name, df_tiempos, df_familia, df_prioridad, fechas, turnos)

    #status, model, df_resultados_camiones = optimizar_planificacion_semanal(df_resultado, time, plants, K=20)
    status, model, df_resultados_camiones,solve_time = optimizar_planificacion_semanal_ortools(df_resultado, time, plants, K=20)


    if not status:
        st.error("El modelo es infactible. No se puede continuar.")
        #return pd.DataFrame()  # Retornar un DataFrame vacío en caso de fallo

    # Paso 3: Obtener solución del modelo
    df_merged = crear_dataframe_resultados(df_resultado, df_resultados_camiones)

    #Paso 4 graficar
    referencia_fecha = pd.to_datetime(fechas[0], format='%d-%m-%Y')
    df_merged = f.calcular_horas_ps(df_merged,referencia_fecha)


    return df_merged, solve_time,referencia_fecha

'''
    df_merged = f.calcular_horas_ps(df_merged,referencia_fecha)
    f.graficar_cronograma4(df_merged)
    df_merged["Fecha"] = pd.to_datetime(df_merged["Fecha"])  # Asegurarte de que sea un objeto datetime
    df_merged["Fecha"] = df_merged["Fecha"].dt.strftime('%d-%m-%Y')  # Convertir al formato deseado

    df_merged["INICIO CARGA"] = df_merged["INICIO CARGA"].dt.strftime('%H:%M')
    df_merged["LLEGADA OBRA"] = df_merged["LLEGADA OBRA"].dt.strftime('%H:%M')
    df_merged["RETORNO PLANTA"] = df_merged["RETORNO PLANTA"].dt.strftime('%H:%M')

# Función para optimizar la planificación semanal
def optimizar_planificacion_semanal_gurobi(df_resultado, time_limit, plants, K = 20):
    """
    Crea y resuelve un modelo de secuenciamiento de camiones basado en los datos del DataFrame de entrada.

    Parámetros:
        df_resultado (pd.DataFrame): DataFrame que contiene los datos necesarios para el modelo.

    Retorna:
        model: Modelo de optimización resuelto.
        dict: Diccionario con los resultados del modelo.
    """

    # Crear el modelo
    model = Model("Secuenciamiento_camiones")

    # Conjuntos
    N = df_resultado.shape[0] # Nodos a atender en función de la cantidad de pedidos
    A = list(range(N, N + K))  # Índices de nodos de almuerzo (uno por camión)
    nodos = list(range(N + K))  # Se redefine para incluir los nodos de almuerzo asociados a cada camión
    arcos = [(i, j) for i in nodos for j in nodos if i != j]
    K = 20
    vehiculos = list(range(K))
    nodos_almuerzo = {N + i: {'t0i': 0, 'p': 60, 'ti0': 0, 'p0': 0}
                  for i in range(K)} # Definir nodos de almuerzo como un diccionario


    # Parámetros
    Li = 0
    Ls = 720
    t0i = np.append(df_resultado['Llegada_obra_min'].values, [nodos_almuerzo[i]['t0i'] for i in A])  # El almuerzo no tiene tiempo de llegada
    p = np.append(df_resultado['Atencion_min'].values, [nodos_almuerzo[i]['p'] for i in A])  # El almuerzo dura 60 minutos
    ti0 = np.append(df_resultado['Retorno_min'].values, [nodos_almuerzo[i]['ti0'] for i in A])  # No hay retorno asociado al almuerzo
    p0 = np.append(df_resultado['Lavado_min'].fillna(0).values, [nodos_almuerzo[i]['p0'] for i in A])  # Asegura que el almuerzo dura 60 min
    M = 10000

    # Variables de decisión
    x = model.addVars(nodos, vehiculos, vtype=GRB.BINARY, name="x")
    z = model.addVars(arcos, vehiculos, vtype=GRB.BINARY, name="z")
    d = model.addVars(vehiculos, vtype=GRB.BINARY, name="d")
    C = model.addVars(nodos, lb=0.0, vtype=GRB.CONTINUOUS, name="C")  # Tiempo en que el vehículo llega al punto i
    phi = model.addVars(nodos, lb=0.0, vtype=GRB.CONTINUOUS, name="Phi")  # Tiempo en que el vehículo llega al depo después de atender a i
    w = model.addVars(arcos, vtype=GRB.BINARY, name="w")

    # Función objetivo: minimizar la distancia total recorrida y los costos asociados
    model.setObjective(quicksum(d[k] for k in vehiculos) + 0.000000001 * quicksum(phi[i] for i in nodos), GRB.MINIMIZE)

    # Restricciones

    # Ventana de almuerzo (de 240 a 360 minutos)
    model.addConstrs((C[almuerzo_k] >= 240 for almuerzo_k in A), name="Ventana_inferior_almuerzo")    
    model.addConstrs((C[almuerzo_k] <= 360 for almuerzo_k in A), name="Ventana_superior_almuerzo")

    # Atender todos los pedidos i
    model.addConstrs((quicksum(x[i, k] for k in vehiculos) == 1 for i in nodos if i not in A), name="Atencion_demanda")

    # Todos los camiones utilizados almuerzan
    model.addConstrs((quicksum(x[almuerzo_k, k] for almuerzo_k in A) == d[k] 
                       for k in vehiculos), name="Todos_almuerzan")
    
    # Restricción: z[i,j,k] <= x[i,k] // predecisión supeditada a atención
    model.addConstrs(z[i, j, k] <= x[i, k] for i, j in arcos for k in vehiculos)  
    model.addConstrs(z[i, j, k] <= x[j, k] for i, j in arcos for k in vehiculos)  
    
    # Restricción: z[i,j,k] + z[j,i,k] >= x[i,k] + x[j,k] - 1
    model.addConstrs(z[i, j, k] + z[j, i, k] >= x[i, k] + x[j, k] - 1 
                     for i in nodos for j in nodos for k in vehiculos if i != j)

    # Restricción: Atención pedido i con camión k supeditada a utilización de camión k
    model.addConstrs(x[i, k] <= d[k] for i in nodos if i not in A for k in vehiculos)

    # Restricción: MTZ
    model.addConstrs(C[j] >= phi[i] + t0i[j] + p0[i] - M * (1 - quicksum(z[i, j, k] for k in vehiculos)) for i, j in arcos)

    # Restricción: Li[i] <= sum(C[i,k]) <= Ls
    model.addConstrs(C[i] >= Li for i in nodos)
    model.addConstrs(C[i] <= Ls for i in nodos)

    # Break simetry
    model.addConstrs(d[k] <= d[k - 1] for k in vehiculos if k >= 1)

    # Restricciones TAP
    model.addConstrs(C[i] >= 61 + Li for i in nodos) # Atención después de apertura de TAP (9:00 TA y 21:00 TB)
    model.addConstrs(C[i] <= Ls - 61 for i in nodos) # Atención antes del cierre del TAP (19:00 TA y 07:00 TB) 
    model.addConstrs(C[i] - t0i[i] >= 45 for i in nodos) # Carga comienza 15 min antes de apertura del TAP

    # Vuelta en horario
    model.addConstrs(phi[i] <= Ls - 61 for i in nodos) # Llegada a depo 1 hr antes del fin de turno

    # Restricción de número máximo de viajes
    model.addConstrs(quicksum(x[i, k] for i in nodos if i not in A) <= 3 for k in vehiculos)
        
    # Restricciones asociadas a cantidad de plantas.
        
    if plants == 1:
        # Restricción de exclusividad con Big-M + def phi con igualdad
        model.addConstrs(phi[i] == C[i] + p[i] + ti0[i] for i in nodos)
        model.addConstrs(C[i] - t0i[i] >= C[j] - t0i[j] + p0[i] - M * (1 - w[i,j]) 
                         for (i, j) in arcos if i not in A if j not in A)
        model.addConstrs(C[i] - t0i[i] <= C[j] - t0i[i] - p0[i] + M * w[i,j]
                         for (i, j) in arcos if i not in A if j not in A)
    else:
        # Horarios de salida espaciados por tiempo de carga
        model.addConstrs(phi[i] >= C[i] + p[i] + ti0[i] for i in nodos)
        model.addConstrs(C[i] - t0i[i] <= C[j] - t0i[j] - p0[i] + M * (1 - w[i,j]) 
                         for (i, j) in arcos if i not in A if j not in A)
        model.addConstrs(C[j] - t0i[j] <= C[i] - t0i[i] + p0[i] + M * w[i,j] 
                         for (i, j) in arcos if i not in A if j not in A)
        model.addConstrs(quicksum(w[i, j] + w[j, i] for j in nodos if j != i if j not in A) >= N - 2 
                         for i in nodos if i not in A)

    # Parámetros de tiempo y MIPGap
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.MIPGap, 0.02)
    
    model.update()
    start_time = time.time()
    # Optimizar el modelo
    model.optimize()
    end_time = time.time()
    solve_time = end_time - start_time

    # Verificar estado de la solución
    if model.status == GRB.INFEASIBLE:
        st.error("El modelo es infactible.")
        return False, model, pd.DataFrame(), solve_time
    
    elif model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
        st.info("Solución Óptima Encontrada")
    
        # Extraer resultados
        resultados = {
            'Pedido_ID': [],
            'Camion_ID': [],
            'Tiempo_carga': [],
            'Tiempo_llegada': [],
            'Retorno_depo': [],
            }
        
        # Definir Tiempo_carga en el DataFrame antes de la optimización
        if 'Tiempo_carga' not in df_resultado.columns:
            df_resultado['Tiempo_carga'] = np.nan  # O calcularlo si aplica

        for v in model.getVars():
            if v.x > 0.9 and v.VarName.startswith('x'):
                var_name = v.VarName.split('[')[1].split(']')[0]
                i, k = map(int, var_name.split(','))
                if i not in A:
                    resultados['Pedido_ID'].append(i)
                    resultados['Camion_ID'].append(k)
                    resultados['Tiempo_carga'].append(round(C[i].x - t0i[i]))
                    resultados['Tiempo_llegada'].append(C[i].x)
                    resultados['Retorno_depo'].append(phi[i].x)
        
        return True, model, pd.DataFrame(resultados),solve_time

        # Convertir resultados a DataFrame
        df_resultados_camiones = pd.DataFrame(resultados)
    
    else:
        print("No se encontró solución óptima.")
        return False, model, pd.DataFrame()
'''
