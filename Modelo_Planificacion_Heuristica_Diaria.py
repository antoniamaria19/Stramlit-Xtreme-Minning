# -*- coding: utf-8 -*-
#!/usr/bin/env python

# ### Importar Librerías

# In[1]:

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
#from gurobipy import *
import streamlit as st 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ### Funciones

# #### Crear Planificación Inicial para Heurística

# In[3]:


def procesar_planificacion_inicial(df_std_pl, df_tiempos, df_familia, df_prioridad, fechas, turnos):
    """
    Procesa un archivo de planificación, duplicando filas según la cantidad de camiones y calculando campos necesarios.

    Parámetros:
        file_path (str): Ruta del directorio donde se encuentra el archivo.
        file_name (str): Nombre del archivo de planificación.
        fechas (list): Lista de fechas a incluir en el DataFrame.
        turnos (list): Lista de turnos a incluir (e.g., ['TA', 'TB']).
 (str): Ruta del archivo Excel con la tabla auxiliar de tiempos.

    Retorna:
        pd.DataFrame: DataFrame procesado con las columnas ajustadas y duplicadas según la cantidad de camiones.
    """
    # Construir la ruta completa del archivo
    #archivo_completo = os.path.join(file_path, file_name)

    # Cargar la planificación y filtrar por las fechas y turnos especificados
    #df_std_pl = pd.read_excel(archivo_completo, sheet_name='BD_Programa')
     # Asegurarte de trabajar con una copia para evitar SettingWithCopyWarning
    df_std_pl = df_std_pl.copy()
    df_std_pl = df_std_pl.dropna(how='all')

    # Convertir 'Turno' a string para asegurar consistencia de tipos
    df_std_pl['Turno'] = df_std_pl['Turno'].astype(str)  # Convertir columna 'Turno' a string
    turnos = [str(turno) for turno in turnos]  # Convertir cada elemento de 'turnos' a string

    # Convertir 'Fecha' y 'fechas' al mismo tipo si es necesario
    df_std_pl['Fecha'] = pd.to_datetime(df_std_pl['Fecha'], errors='coerce')  # Convertir 'Fecha' a tipo datetime
    fechas = pd.to_datetime(fechas, errors='coerce')  # Convertir 'fechas' a tipo datetime


    # Aplicar el filtro con 'isin' después de la conversión de tipos
    df_std_pl = df_std_pl[
        df_std_pl['Fecha'].isin(fechas) & df_std_pl['Turno'].isin(turnos)
    ].sort_values(by=['Fecha', 'Turno'], ascending=[True, True])

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
    df_std_pl['Lavado_min'] = 15  # Igual para todas las filas (valor original = 15)
    df_std_pl['Punto_entrega'] = df_std_pl['Punto de Entrega']  # Nombre columna "Punto de entrega"
    df_std_pl['Destino_final'] = df_std_pl['Destino Final']  # Nombre columna "Destino final"
    df_std_pl['Total_vuelta_min'] = df_std_pl['Llegada_obra_min'] + df_std_pl['Atencion_min'] + df_std_pl['Retorno_min'] + df_std_pl['Lavado_min']  # Calcular total
    
    # Crear columna "Pedido_ID" con números ascendentes
    df_std_pl['Pedido_ID'] = range(1, len(df_std_pl) + 1)

    # Obtener la fecha inicial como referencia con formato explícito
    fecha_inicial = pd.to_datetime(fechas[0], format='%d-%m-%Y', dayfirst=True)

    #MERGE CON TABLAS AUXILIARES
    #1. Tiempos
    # Realizar el merge con la tabla auxiliar para obtener tiempos específicos
    # Verificamos si df_tiempos tiene columnas requeridas
    tiempos_cols = ['Empresa', 'Punto_entrega', 'Llegada_obra_min', 'Atencion_min', 'Retorno_min', 'Lavado_min', 'Total_vuelta_min']
    try:
        if isinstance(df_tiempos, pd.DataFrame) and all(col in df_tiempos.columns for col in tiempos_cols):
            df_std_pl = df_std_pl.merge(
                df_tiempos[tiempos_cols],
                on=['Empresa', 'Punto_entrega'],
                how='left',
                suffixes=('', '_ref')
            )
        else:
            raise ValueError("df_tiempos no tiene las columnas necesarias o no es un DataFrame")
    except Exception as e:
        st.error(f"Error al procesar los tiempos: {e}")
        for col in ['Llegada_obra_min_ref', 'Atencion_min_ref', 'Retorno_min_ref', 'Lavado_min_ref', 'Total_vuelta_min_ref']:
            df_std_pl[col] = np.nan
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
    
    #2. Familia de producto 
    # Merge con la tabla de productos para obtener la familia
    if isinstance(df_familia, pd.DataFrame) and {'Codigo_producto', 'Familia'}.issubset(df_familia.columns):
        df_std_pl = df_std_pl.merge(df_familia[['Codigo_producto', 'Familia']], on='Codigo_producto', how='left')
    else:
        df_std_pl['Familia'] = 'Sin asignar'

    #3. Prioridades 
    # Merge con la tabla de prioridades, asignando prioridad BAJA cuando no hay match
    df_std_pl = df_std_pl.merge(
        df_prioridad[['Empresa', 'Familia', 'Prioridad']],
        on=['Empresa', 'Familia'],
        how='left'
    )
    df_std_pl['Prioridad'] = df_std_pl['Prioridad'].fillna('BAJA')
    
    # Añadir columnas con tiempos de pre y post descarga
    df_std_pl['Tiempo_pre_descarga'] = df_std_pl['Llegada_obra_min']
    df_std_pl['Tiempo_post_descarga'] = df_std_pl['Atencion_min'] + df_std_pl['Retorno_min'] + df_std_pl['Lavado_min']
 
    # Función para duplicar las filas por la cantidad de camiones
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
    df_resultado['Hora_llamado'] = "00:00"
    df_resultado['Camion_ID'] = np.nan
    df_resultado['Tiempo_llegada_fijo'] = np.nan
    df_resultado['Tiempo_carga'] = np.nan
    df_resultado['Tiempo_llegada_var'] = np.nan
    df_resultado['ru'] = np.nan
    df_resultado['rl'] = np.nan
    df_resultado['Hora_requerida'] = "00:00"
    df_resultado['Volumen_confirmado'] = 0.0
    df_resultado['Retorno_depo'] = np.nan
    df_resultado['Retorno_depo_proyectado'] = ""
    df_resultado['Hora_carga'] = np.nan
    df_resultado['Hora_llegada_obra_fijo'] = np.nan
    df_resultado['Hora_llegada_obra_var'] = np.nan
    df_resultado['Hora_fin_atencion'] = np.nan
    df_resultado['Hora_retorno'] = np.nan
    df_resultado['Hora_retorno_proyectado'] = ""
    df_resultado['Hora_fin_lavado'] = np.nan
    df_resultado['Planta_salida'] = np.nan
    df_resultado['Observaciones'] = ""


    cols_to_int = ['Llegada_obra_min', 'Atencion_min', 'Retorno_min', 'Lavado_min', 'Total_vuelta_min']
    cols_to_int = [col for col in cols_to_int if not df_resultado[col].astype(str).str.contains(":").any()]
    df_resultado[cols_to_int] = df_resultado[cols_to_int].astype(int)

    # Reordenar columnas en el orden solicitado
    columnas_finales = [
        'Fecha', 'Viaje_ID', 'Pedido_ID', 'Empresa', 'Codigo_producto', 'Familia','Descripcion', 
        'Volumen_m3', 'Turno', 'Camion_ID','Volumen_transportado', 'Tiempo_carga', 'Tiempo_llegada_fijo', 
        'Tiempo_llegada_var', 'Retorno_depo','Retorno_depo_proyectado', 'Llegada_obra_min', 'Atencion_min', 
        'Retorno_min', 'Lavado_min', 'Total_vuelta_min', 'Tiempo_pre_descarga', 
        'Tiempo_post_descarga', 'Punto_entrega', 'Destino_final', 'Obra', 'Uso', 
        'Hora_carga', 'Hora_llegada_obra_fijo','Hora_llegada_obra_var', 'ru', 'rl', 'Hora_fin_atencion', 
        'Hora_retorno','Hora_retorno_proyectado', 'Hora_fin_lavado', 'Prioridad', 'Hora_llamado', 
        'Hora_requerida', 'Volumen_confirmado', 'Planta_salida','Observaciones'
    ]

    df_resultado = df_resultado[columnas_finales]

    return df_resultado

def procesar_tabla_editada(df_resultado):
    # Inicializar columnas adicionales necesarias
    df_resultado["Camion_ID"] = np.nan
    df_resultado["Tiempo_llegada_fijo"] = np.nan
    df_resultado["Tiempo_carga"] = np.nan
    df_resultado["Tiempo_llegada_var"] = np.nan
    df_resultado["ru"] = np.nan
    df_resultado["rl"] = np.nan
    df_resultado["Retorno_depo"] = np.nan
    df_resultado['Retorno_depo_proyectado'] = np.nan
    df_resultado["Hora_carga"] = np.nan
    df_resultado["Hora_llegada_obra_fijo"] = np.nan
    df_resultado["Hora_llegada_obra_var"] = np.nan
    df_resultado["Hora_fin_atencion"] = np.nan
    df_resultado["Hora_retorno"] = np.nan
    df_resultado["Hora_fin_lavado"] = np.nan
    df_resultado["Planta_salida"] = np.nan

    
    # Reordenar columnas en el orden solicitado
    columnas_finales = [
        'Fecha', 'Pedido_ID', 'Empresa', 'Codigo_producto', 'Familia','Descripcion', 
        'Volumen_m3', 'Turno', 'Camion_ID','Volumen_transportado', 'Tiempo_carga', 'Tiempo_llegada_fijo', 
        'Tiempo_llegada_var', 'Retorno_depo','Retorno_depo_proyectado', 'Llegada_obra_min', 'Atencion_min', 
        'Retorno_min', 'Lavado_min', 'Punto_entrega', 'Destino_final', 'Obra', 'Uso', 
        'Hora_carga', 'Hora_llegada_obra_fijo','Hora_llegada_obra_var', 'ru', 'rl', 'Hora_fin_atencion', 
        'Hora_retorno','Hora_retorno_proyectado', 'Hora_fin_lavado', 'Prioridad', 'Hora_llamado', 
        'Hora_requerida', 'Volumen_confirmado', 'Planta_salida','Observaciones'
    ]


    df_resultado = df_resultado[columnas_finales]

    return df_resultado

# #### Procesamiento Planificación con Datos Cargados (Iterada)

# In[4]:
def procesar_planificacion_iterada(df_resultado):
    """
    Procesa un archivo de planificación, eliminando filas canceladas y convirtiendo horas a minutos.

    Parámetros:
        file_path (str): Ruta del archivo de entrada.
        file_name (str): Nombre del archivo de entrada.

    Retorna:
        pd.DataFrame: DataFrame procesado con las columnas ajustadas.
    """
    
    #archivo_input = f"{file_path}/{file_name}"
    
    # Cargar la planificación y ordenar por 'Viaje_ID'
    df_heur = df_resultado.sort_values(by=['Viaje_ID'], ascending=True)
    df_iterado = df_resultado.sort_values(by=['Viaje_ID'], ascending=True)
    #turno = df_resultado['Turno'][0]

    # Filtrar filas donde 'Hora_llamado' no es nulo y 'Volumen_confirmado' es igual a 0
    df_heur = df_heur[(df_heur['Hora_llamado'].notna() & ~(df_heur['Volumen_confirmado'] == 0))]

    # Restablecer índices del DataFrame
    df_heur = df_heur.reset_index(drop=True)

    # Asegurar que 'Hora_requerido_heuristica' sea cadena, limpiar espacios y reemplazar NaN por cadena vacía
    df_heur['Hora_requerida'] = df_heur['Hora_requerida'].astype(str).str.strip().replace("nan", "")
    df_iterado['Hora_requerida'] = df_iterado['Hora_requerida'].astype(str).str.strip().replace("nan", "")
    df_heur['Hora_retorno_proyectado'] = df_heur['Hora_retorno_proyectado'].astype(str).str.strip().replace("nan", "")
    df_iterado['Hora_retorno_proyectado'] = df_iterado['Hora_retorno_proyectado'].astype(str).str.strip().replace("nan", "")


        # Función para convertir horas a minutos considerando turnos
    def convertir_a_minutos_turno(hora, turno):
        """
        Convierte una hora en formato HH:MM o datetime.time a minutos desde un punto de referencia según el turno.
        - Turno TA: Referencia en 08:00.
        - Turno TB: Referencia en 20:00.

        Parámetros:
            hora (str o datetime.time): Hora en formato "HH:MM" o `datetime.time`.
            turno (str): Turno de trabajo, "TA" (08:00 - 19:59) o "TB" (20:00 - 07:59).

        Retorna:
            int: Minutos transcurridos desde la referencia correspondiente o None si el formato es inválido.
        """
        if pd.isna(hora) or hora == "":
            return None  # En lugar de NaN, devolver None para evitar valores numéricos no válidos

        try:
            # Si el valor ya es un objeto datetime.time, extraer horas y minutos directamente
            if isinstance(hora, pd.Timestamp):
                horas, minutos = hora.hour, hora.minute
            elif isinstance(hora, str):
                partes = list(map(int, hora.split(":")))
                if len(partes) >= 2:
                    horas, minutos = partes[:2]
                else:
                    return None
            else:
                return None
            
            # Conversión según turno
            if turno == 'TA':  # Turno de 08:00 a 19:59
                return ((horas - 8) * 60) + minutos if horas >= 8 else ((horas * 60) + minutos - (8 * 60))
            
            elif turno == 'TB':  # Turno de 20:00 a 07:59
                if horas < 20 and horas > 8:
                    return (horas * 60) + minutos - (20 * 60)
                if horas >= 20:  # Entre 20:00 y 23:59
                    return ((horas - 20) * 60) + minutos
                else:  # Entre 00:00 y 07:59
                    return ((horas + 4) * 60) + minutos  
            
            return None
        except (ValueError, IndexError):
            return None


    # Aplicar conversión de hora a minutos con el turno de cada fila (hora requerido y hora retorno proyectado)
    df_heur['Hora_requerido_heuristica_min'] = df_heur.apply(
        lambda row: convertir_a_minutos_turno(row['Hora_requerida'], row['Turno']), axis=1
    ).replace(np.nan, None)  # Evita mostrar NaN en la columna

    df_iterado['Hora_requerido_heuristica_min'] = df_iterado.apply(
        lambda row: convertir_a_minutos_turno(row['Hora_requerida'], row['Turno']), axis=1
    ).replace(np.nan, None)

    df_heur['Hora_retorno_proyectado_min'] = df_iterado.apply(
        lambda row: convertir_a_minutos_turno(row['Hora_retorno_proyectado'], row['Turno']), axis=1
    )


    df_iterado['Hora_retorno_proyectado_min'] = df_iterado.apply(
        lambda row: convertir_a_minutos_turno(row['Hora_retorno_proyectado'], row['Turno']), axis=1
    )

    # Convertir decimales ingresados a enteros
    cols = ['Llegada_obra_min', 'Atencion_min', 'Retorno_min', 'Lavado_min']
    df_heur[cols] = df_heur[cols].apply(lambda x: x.fillna(0).round(0).astype(int) if x.notna().any() else x)
    df_iterado[cols] = df_iterado[cols].apply(lambda x: x.fillna(0).round(0).astype(int) if x.notna().any() else x)
    
    # Añadir cálculo de Tiempo_carga en la planificación iterada considerando turnos
    df_heur['Tiempo_carga'] = df_heur.apply(
        lambda row: convertir_a_minutos_turno(row['Hora_carga'], row['Turno']), axis=1
    ).replace(np.nan, None)
    
    df_iterado['Tiempo_carga'] = df_iterado.apply(
        lambda row: convertir_a_minutos_turno(row['Hora_carga'], row['Turno']), axis=1
    ).replace(np.nan, None)
    
    return df_heur, df_iterado  


# #### Optimización

# In[40]:

def convertir_a_minutos(hora):
    """
    Convierte una hora en formato HH:MM a minutos desde un punto de referencia:
    - Si la hora está entre 08:00 y 19:59, se toma como referencia las 08:00.
    - Si la hora está entre 20:00 y 07:59, se toma como referencia las 20:00.

    Parámetros:
        hora (str): Hora en formato "HH:MM".

    Retorna:
        int: Minutos transcurridos desde la referencia correspondiente o NaN si el formato es inválido.
    """
    if pd.isna(hora) or hora.strip() == "":
        return np.nan
    try:
        partes = list(map(int, hora.split(":")))
        if len(partes) == 2:
            horas, minutos = partes
        elif len(partes) == 3:
            horas, minutos, _ = partes
        else:
            return np.nan
        
        # Caso turno diurno: 08:00 a 19:59 (referencia: 08:00)
        if 8 <= horas < 20:
            return ((horas - 8) * 60) + minutos
        
        # Caso turno nocturno: 20:00 a 07:59 (referencia: 20:00)
        elif horas >= 20 or horas < 8:
            if horas >= 20:  # Entre 20:00 y 23:59
                return ((horas - 20) * 60) + minutos
            else:  # Entre 00:00 y 07:59
                return ((horas + 4) * 60) + minutos  # 00:00 es 240 min desde 20:00
        
        else:
            return np.nan
    except (ValueError, IndexError):
        return np.nan

     
#MODELO DE OPTIMIZACION ORTOOLS------------------------------------------------------------------------------------------------------------------
def optimizar_secuenciamiento_camiones_ortools(df_resultado, tiempo_actual_str, time_limit, plants, K):
    """
    Crea y resuelve un modelo de secuenciamiento de camiones basado en los datos del DataFrame de entrada.

    Parámetros:
        df_resultado (pd.DataFrame): DataFrame con los datos del modelo.
        tiempo_actual_str (str): Tiempo actual en formato "HH:MM".
        Turno (str): Turno de trabajo ("TA" o "TB").
        time_limit (int): Límite de tiempo para la optimización.
        plants (int): Número de plantas.
        K (int): Número de camiones disponibles.

    Retorna:
        model: Modelo de optimización resuelto.
        pd.DataFrame: DataFrame con los resultados obtenidos.
    """

    # Convertir tiempo actual a minutos
    tiempo_actual = convertir_a_minutos(tiempo_actual_str)
    
    # Crear el modelo
    model_h = pywraplp.Solver.CreateSolver('SCIP')

    # Conjuntos
    N = df_resultado.shape[0]
    A = list(range(N, N + K))  # Índices de nodos de almuerzo (uno por camión)
    nodos = list(range(N + K))  # Se redefine para incluir los nodos de almuerzo asociados a cada camión
    arcos = [(i, j) for i in nodos for j in nodos if i != j]
    arcos = [(i, j) for i in nodos for j in nodos if i != j]
    vehiculos = list(range(K))
    nodos_almuerzo = {N + i: {'t0i': 0, 'p': 60, 'ti0': 0, 'p0': 0, 'Prioridad': 'BAJA', 'hora_referencia': 300}
                  for i in range(K)} # Definir nodos de almuerzo como un diccionario

    # Parámetros y variables auxiliares
    Li = 0
    Ls = 720
    t0i = np.append(df_resultado['Llegada_obra_min'].values, [nodos_almuerzo[i]['t0i'] for i in A])  # El almuerzo no tiene tiempo de llegada
    p = np.append(df_resultado['Atencion_min'].values, [nodos_almuerzo[i]['p'] for i in A])  # El almuerzo dura 60 minutos
    ti0 = np.append(df_resultado['Retorno_min'].values, [nodos_almuerzo[i]['ti0'] for i in A])  # No hay retorno asociado al almuerzo
    p0 = np.append(df_resultado['Lavado_min'].fillna(0).values, [nodos_almuerzo[i]['p0'] for i in A])  # Asegura que el almuerzo dura 60 min
    Viaje = np.append(df_resultado['Viaje_ID'].values, [f"ALMUERZO_{i}" for i in A])  # Identificación del almuerzo
    Prioridad = np.append(df_resultado['Prioridad'].values, [nodos_almuerzo[i]['Prioridad'] for i in A])  # Prioridad baja para el almuerzo
    C_t_old = np.append(df_resultado['Tiempo_llegada_var'].values, [np.nan for i in A])
    lamb = {i: 1000 if Prioridad[i] == 'ALTA' else 200 if Prioridad[i] == 'MEDIA' else 1 for i in range(len(Prioridad))}
    k_old = np.append(df_resultado['Camion_ID'].values, [np.nan for i in A])
    M = 10000
    C = np.append(df_resultado['Hora_requerido_heuristica_min'].values, 
                  [nodos_almuerzo[i]['hora_referencia'] for i in A])  # Almuerzo centrado en 300 minutos
    col = df_resultado['Hora_retorno_proyectado_min'].fillna(0).infer_objects(copy=False).astype(int)
    #col = col.infer_objects(copy=False)  # infiere el tipo antes de convertir
    R = np.append(col.astype(int).values, [0 for i in A]) # No tiene retorno proyectado
    b = {i: 0 for i in nodos} # Variable auxiliar para desactivar restricción phi[i] cuando se ingresan llegadas proyectadas  
    

    # Variables de decisión
    x = {(i, k): model_h.BoolVar(f'x_{i}_{k}') for i in nodos for k in vehiculos}
    z = {(i, j, k): model_h.BoolVar(f'z_{i}_{j}_{k}') for i, j in arcos for k in vehiculos}
    C_t = {i: model_h.NumVar(0, model_h.infinity(), f'C_t_{i}') for i in nodos}
    phi = {i: model_h.NumVar(0, model_h.infinity(), f'phi_{i}') for i in nodos}
    w = {(i, j): model_h.BoolVar(f'w_{i}_{j}') for i, j in arcos}
    ru = {i: model_h.NumVar(0, model_h.infinity(), f'ru_{i}') for i in nodos}
    rl = {i: model_h.NumVar(0, model_h.infinity(), f'rl_{i}') for i in nodos}

    # -------- FUNCIÓN OBJETIVO ----------
    # Minimizar las desviaciones sin afectar el almuerzo
    objective = model_h.Sum(lamb[i] * (ru[i] + rl[i]) for i in nodos if i not in A) + \
                0.000000001 * model_h.Sum(phi[i] for i in nodos if i not in A)

    model_h.Minimize(objective)  # Minimizar la función objetivo

    # -------- RESTRICCIONES --------

    # Ventana de almuerzo (de 240 a 360 minutos)
    for almuerzo_k in A:
        model_h.Add(C_t[almuerzo_k] >= 240, f"Ventana_inferior_almuerzo_{almuerzo_k}")
        model_h.Add(C_t[almuerzo_k] <= 360, f"Ventana_superior_almuerzo_{almuerzo_k}")

    # Todos los pedidos deben ser atendidos
    for i in nodos:
        model_h.Add(model_h.Sum(x[i, k] for k in vehiculos) == 1, f"Atencion_demanda_{i}")

    # Todos los vehículos deben almorzar en algún nodo de A
    for k in vehiculos:
        model_h.Add(model_h.Sum(x[almuerzo_k, k] for almuerzo_k in A) == 1, f"Todos_almuerzan_{k}")
    
    # Restricción: z[i,j,k] <= x[i,k] // predecisión supeditada a atención
    for i, j in arcos:
        for k in vehiculos:
            model_h.Add(z[i, j, k] <= x[i, k])
    
    # Restricción: z[i,j,k] <= x[j,k] // predecisión supeditada a atención
    for i, j in arcos:
        for k in vehiculos:
            model_h.Add(z[i, j, k] <= x[j, k])

    #  Restricción: z[i,j,k] + z[j,i,k] >= x[i,k] + x[j,k] - 1
    for i in nodos:
        for j in nodos:
            if i != j:
                for k in vehiculos:
                    model_h.Add(z[i, j, k] + z[j, i, k] >= x[i, k] + x[j, k] - 1)

    # Restricción MTZ (Miller-Tucker-Zemlin)
    for i, j in arcos:
        model_h.Add(
            C_t[j] >= phi[i] + t0i[j] + p0[i] - M * (1 - model_h.Sum(z[i, j, k] for k in vehiculos))
            )

    # Restricción: Li[i] <= C_t[i] <= Ls[i]
    for i in nodos:
        model_h.Add(C_t[i] >= Li)
        model_h.Add(C_t[i] <= Ls)

    # Apertura TAP
    for i in nodos:
        model_h.Add(C_t[i] >= 61 + Li)  # Atención después de apertura de TAP
        model_h.Add(C_t[i] <= Ls - 61)  # Atención antes del cierre del TAP
        model_h.Add(C_t[i] - t0i[i] >= 45)  # Carga comienza 15 min antes de apertura del TAP
    
    # Vuelta en horario
    for i in nodos:
        model_h.Add(phi[i] <= Ls - 61)  # Llegada a depósito 1 hr antes del fin de turno
    
    # Restricción de número máximo de viajes
    for k in vehiculos:
        model_h.Add(model_h.Sum(x[i, k] for i in nodos if i not in A) <= 3)

    # Definición de C_t
    for i in nodos:
        model_h.Add(C_t[i] == C[i] + ru[i] - rl[i])

    for i in nodos:
        if Prioridad[i] == 'ALTA':
            model_h.Add(ru[i] == 0)
            model_h.Add(rl[i] == 0)
        elif Prioridad[i] == 'MEDIA':
            model_h.Add(ru[i] <= 20)
            model_h.Add(rl[i] <= 45)

        if not np.isnan(C_t_old[i]) and C_t_old[i] <= tiempo_actual + 45:
            model_h.Add(C_t[i] == C_t_old[i])
            model_h.Add(x[i, int(k_old[i])] == 1)

        if R[i] > 0:
            model_h.Add(phi[i] == R[i])
            if not np.isnan(k_old[i]):
                model_h.Add(x[i, int(k_old[i])] == 1)
            b[i] = 1

    C_t[i] >= tiempo_actual + 45

    # Def Phi[i] con variable auxiliar
    for i in nodos:
        model_h.Add(phi[i] >= C_t[i] + p[i] + ti0[i] - M * b[i])

    # Restricción de no salir antes del tiempo de llegada estimado
    for i in nodos:
        model_h.Add(C_t[i] - t0i[i] >= 0)

    # Restricciones de plantas
    if plants == 1:
        # Restricción de exclusividad con Big-M
        for (i, j) in arcos:
            if i not in A and j not in A:
                model_h.Add(C_t[i] - t0i[i] >= C_t[j] - t0i[j] + p0[j] - M * (1 - w[i, j]))
                model_h.Add(C_t[i] - t0i[i] <= C_t[j] - t0i[j] - p0[i] + M * w[i, j])

    else:
        # Big M para evitar conflictos (2 plantas default) + Def phi con tiempos de espera
        for (i, j) in arcos:
            if i not in A and j not in A:
                model_h.Add(C_t[i] - t0i[i] <= C_t[j] - t0i[j] - p0[i] + M * (1 - w[i, j]))
                model_h.Add(C_t[j] - t0i[j] <= C_t[i] - t0i[i] + p0[j] + M * w[i, j])

        for i in nodos:
            if i not in A:
                model_h.Add(model_h.Sum(w[i, j] + w[j, i] for j in nodos if j != i and j not in A) >= N - 2)

    #Resolucion del modelo y rescate de resultados

    # Establecer parámetros de tiempo y tolerancia GAP
    model_h.SetTimeLimit(time_limit * 1000)  # OR-Tools usa milisegundos
    model_h.SetSolverSpecificParametersAsString('limits/gap = 0.02')

    # Resolver el modelo
    # Medir tiempo de resolución
    start_time = time.time()
    status = model_h.Solve()
    end_time = time.time()
    solve_time = end_time - start_time

    # Verificar el estado de la solución
    if status == pywraplp.Solver.INFEASIBLE:
        print("El modelo es infactible.")
        return False, model_h, pd.DataFrame(), solve_time

    elif status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        print("Solución Óptima Encontrada")

        # Extraer resultados
        resultados = {
            'Pedido_ID': [],
            'Camion_ID': [],
            'Tiempo_carga': [],
            'Tiempo_llegada_fijo': [],
            'Tiempo_llegada_var': [],
            'ru': [],
            'rl': [],
            'Retorno_depo': []
        }

        # Definir Tiempo_carga en el DataFrame antes de la optimización
        if 'Tiempo_carga' not in df_resultado.columns:
            df_resultado['Tiempo_carga'] = np.nan

        # Extraer las variables de decisión
        for i in nodos:
            for k in vehiculos:
                if x[i, k].solution_value() > 0.9:
                    resultados['Pedido_ID'].append(Viaje[i])
                    resultados['Camion_ID'].append(k)
                    tiempo_carga = C_t[i].solution_value() - t0i[i]
                    resultados['Tiempo_carga'].append(round(tiempo_carga) if not np.isnan(tiempo_carga) else None)
                    resultados['Tiempo_llegada_fijo'].append(C[i])
                    resultados['Tiempo_llegada_var'].append(round(C_t[i].solution_value()))
                    resultados['Retorno_depo'].append(phi[i].solution_value())
                    resultados['ru'].append(round(ru[i].solution_value()))
                    resultados['rl'].append(round(rl[i].solution_value()))

        # Convertir resultados a DataFrame
        df_resultados_camiones = pd.DataFrame(resultados)

        return True, model_h, df_resultados_camiones, solve_time

    else:
        st.info("No se encontró solución óptima.")
        return False, model_h, pd.DataFrame(), solve_time


# #### Dataframe y rescate de resultado del modelo


# In[37]:
def crear_dataframe_optimizado(df_iterado, df_resultados_camiones):
    """
    Procesa los resultados de la optimización para generar un DataFrame detallado.

    Parámetros:
        df_iterado (pd.DataFrame): DataFrame base con los datos iterados.
        df_resultados_camiones (pd.DataFrame): Resultados del modelo optimizado.

    Retorna:
        pd.DataFrame: DataFrame con los resultados procesados.
    """

    if 'Tiempo_carga' not in df_resultados_camiones.columns:
        df_resultados_camiones['Tiempo_carga'] = np.nan  # Evita KeyError en el merge
    
    #Formato Pedido_Id y Viaje_ID
    df_iterado['Viaje_ID'] = df_iterado['Viaje_ID'].astype(str)
    df_resultados_camiones['Pedido_ID'] = df_resultados_camiones['Pedido_ID'].astype(str)
    
    # Realizar el outer join con resultado optimización
    df_merged = pd.merge(
        df_iterado,
        df_resultados_camiones[['Pedido_ID', 'Camion_ID', 'Tiempo_carga', 'Tiempo_llegada_fijo', 
                                'Tiempo_llegada_var', 'ru', 'rl', 'Retorno_depo']].rename(
            columns={'Pedido_ID': 'Viaje_ID'}
        ),
        on='Viaje_ID',
        how='left'
    )

    # Renombrar columnas relevantes
    df_merged = df_merged.rename(columns={
        'Camion_ID_y': 'Camion_ID',
        'Tiempo_llegada_fijo_y': 'Tiempo_llegada_fijo',
        'Tiempo_llegada_var_y': 'Tiempo_llegada_var',
        'Tiempo_carga_y': 'Tiempo_carga',
        'ru_y': 'ru',
        'rl_y': 'rl',
        'Retorno_depo_y': 'Retorno_depo'
    })

    # Dropear columnas antiguas e irrelevantes
    columnas_a_eliminar = [
        'Camion_ID_x', 'Tiempo_llegada_fijo_x', 'Tiempo_llegada_var_x','Tiempo_carga_x', 'ru_x', 'rl_x', 'Retorno_depo_x'
    ]
    df_merged = df_merged.drop(columns=columnas_a_eliminar, errors='ignore')

    # Convertir 'Camion_ID' y tiempos a enteros, ignorando valores nulos
    df_merged['Camion_ID'] = df_merged['Camion_ID'].apply(lambda x: int(x) if not pd.isna(x) else x)
    df_merged['Tiempo_llegada_var'] = df_merged['Tiempo_llegada_var'].apply(lambda x: int(x) if not pd.isna(x) else x)
    df_merged['ru'] = df_merged['ru'].apply(lambda x: int(round(x)) if not pd.isna(x) else x)
    df_merged['rl'] = df_merged['rl'].apply(lambda x: int(round(x)) if not pd.isna(x) else x)

    # Función para convertir minutos totales a formato HH:MM
    
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

    # Crear columnas de horas proyectadas basadas en tiempo de llegada var

    df_merged['Hora_carga'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Tiempo_carga'], row['Turno']), axis=1
    )

    df_merged['Hora_llegada_obra_var'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Tiempo_llegada_var'], row['Turno']), axis=1
    )

    df_merged['Hora_llegada_obra_fijo'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Tiempo_llegada_fijo'], row['Turno']), axis=1
    )

    df_merged['Hora_fin_atencion'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Tiempo_llegada_var'] + row['Atencion_min'], row['Turno']) 
        if not pd.isna(row['Tiempo_llegada_var']) and not pd.isna(row['Atencion_min']) else None, 
        axis=1
    )

    df_merged['Hora_retorno'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Retorno_depo'], row['Turno']), axis=1
    )

    df_merged['Hora_fin_lavado'] = df_merged.apply(
        lambda row: convertir_a_hhmm(row['Retorno_depo'] + row['Lavado_min'], row['Turno']) 
        if not pd.isna(row['Retorno_depo']) and not pd.isna(row['Lavado_min']) else None, 
        axis=1
    )

    # Ordenar por tiempo de carga (prework para mantener la secuencia correcta)
    df_merged = df_merged.sort_values(by='Tiempo_carga', ascending=True)

    # Detectar si se necesita una segunda planta
    necesita_segunda_planta = False
    max_salida_anterior = float('-inf')

    for index, row in df_merged.iterrows():
        if row['Tiempo_carga'] - max_salida_anterior < 15:
            necesita_segunda_planta = True
            break  # Confirmamos y salimos del bucle
        max_salida_anterior = max(max_salida_anterior, row['Tiempo_carga'])

    # Asignar plantas de forma alternada si se necesita la segunda planta
    if necesita_segunda_planta:
        alternador = True  # Alternador entre Planta 1 y Planta 2
        for index in df_merged.index:
            df_merged['Planta_salida'] = df_merged['Planta_salida'].astype(object)
            # Ahora la asignación funcionará sin problemas
            df_merged.at[index, 'Planta_salida'] = 'Planta 1' if alternador else 'Planta 2'

            alternador = not alternador  # Alternar en cada iteración
    else:
        df_merged['Planta_salida'] = 'Planta 1'  # Si no se necesita, todo es Planta 1
        
    # Si el volumen confirmado es 0 o nulo, la planta de retorno debe ser vacía
    df_merged.loc[df_merged['Volumen_confirmado'].isna() | (df_merged['Volumen_confirmado'] == 0), 'Planta_salida'] = ""
    
    # Reordenar columnas en el orden especificado
    columnas_finales = [
        'Fecha', 'Viaje_ID', 'Pedido_ID', 'Empresa', 'Codigo_producto', 'Familia', 'Descripcion', 'Volumen_m3', 'Turno',
        'Camion_ID', 'Volumen_transportado',  'Tiempo_carga','Tiempo_llegada_fijo',
        'Tiempo_llegada_var', 'Retorno_depo', 'Llegada_obra_min', 'Atencion_min', 'Retorno_min', 'Lavado_min', 'Total_vuelta_min', 'Tiempo_pre_descarga',
        'Tiempo_post_descarga', 'Punto_entrega', 'Destino_final', 'Obra', 'Uso', 'Hora_carga', 'Hora_llegada_obra_fijo', 'Hora_llegada_obra_var', 
        'ru', 'rl', 'Hora_fin_atencion', 'Hora_retorno', 'Hora_retorno_proyectado', 'Hora_fin_lavado', 'Prioridad', 'Hora_llamado',
        'Hora_requerida', 'Volumen_confirmado', 'Planta_salida'
    ]

    # Filtrar columnas en el orden deseado
    df_merged = df_merged[columnas_finales]

    return df_merged




# Función para convertir horas a minutos considerando turnos
def convertir_a_minutos_turno(hora, turno):
    if pd.isna(hora) or hora == "":
        return None  # En lugar de NaN, devolver None para evitar valores numéricos no válidos

    try:
        # Si el valor ya es un objeto datetime.time, extraer horas y minutos directamente
        if isinstance(hora, pd.Timestamp):
            horas, minutos = hora.hour, hora.minute
        elif isinstance(hora, str):
            partes = list(map(int, hora.split(":")))
            if len(partes) >= 2:
                horas, minutos = partes[:2]
            else:
                return None
        else:
            return None
        
        # Conversión según turno
        if turno == 'TA':  # Turno de 08:00 a 19:59
            return ((horas - 8) * 60) + minutos if horas >= 8 else ((horas * 60) + minutos - (8 * 60))
        
        elif turno == 'TB':  # Turno de 20:00 a 07:59
            if horas < 20 and horas > 8:
                return (horas * 60) + minutos - (20 * 60)
            if horas >= 20:  # Entre 20:00 y 23:59
                return ((horas - 20) * 60) + minutos
            else:  # Entre 00:00 y 07:59
                return ((horas + 4) * 60) + minutos  
        
        return None
    except (ValueError, IndexError):
        return None

'''
def optimizar_secuenciamiento_camiones_heuristica(df_resultado, tiempo_actual_str, time_limit, plants, K):
    """
    Crea y resuelve un modelo de secuenciamiento de camiones basado en los datos del DataFrame de entrada.

    Parámetros:
        df_resultado (pd.DataFrame): DataFrame con los datos del modelo.
        tiempo_actual_str (str): Tiempo actual en formato "HH:MM".
        Turno (str): Turno de trabajo ("TA" o "TB").
        time_limit (int): Límite de tiempo para la optimización.
        plants (int): Número de plantas.
        K (int): Número de camiones disponibles.

    Retorna:
        model: Modelo de optimización resuelto.
        pd.DataFrame: DataFrame con los resultados obtenidos.
    """

    # Convertir tiempo actual a minutos
    tiempo_actual = convertir_a_minutos_turno(tiempo_actual_str, df_resultado['Turno'].values[0])

    # Crear el modelo
    model_h = Model("Secuenciamiento_camiones_heuristica")

    # Conjuntos    
    N = df_resultado.shape[0] # Nodos a atender en función de la cantidad de pedidos
    A = list(range(N, N + K))  # Índices de nodos de almuerzo (uno por camión)
    nodos = list(range(N + K))  # Se redefine para incluir los nodos de almuerzo asociados a cada camión
    arcos = [(i, j) for i in nodos for j in nodos if i != j]
    vehiculos = list(range(K))
    nodos_almuerzo = {N + i: {'t0i': 0, 'p': 60, 'ti0': 0, 'p0': 0, 'Prioridad': 'BAJA', 'hora_referencia': 300}
                  for i in range(K)} # Definir nodos de almuerzo como un diccionario

    # Parámetros y variables auxiliares
    Li = 0
    Ls = 720
    t0i = np.append(df_resultado['Llegada_obra_min'].values, [nodos_almuerzo[i]['t0i'] for i in A])  # El almuerzo no tiene tiempo de llegada
    p = np.append(df_resultado['Atencion_min'].values, [nodos_almuerzo[i]['p'] for i in A])  # El almuerzo dura 60 minutos
    ti0 = np.append(df_resultado['Retorno_min'].values, [nodos_almuerzo[i]['ti0'] for i in A])  # No hay retorno asociado al almuerzo
    p0 = np.append(df_resultado['Lavado_min'].fillna(0).values, [nodos_almuerzo[i]['p0'] for i in A])  # Asegura que el almuerzo dura 60 min
    Viaje = np.append(df_resultado['Viaje_ID'].values, [f"ALMUERZO_{i}" for i in A])  # Identificación del almuerzo
    Prioridad = np.append(df_resultado['Prioridad'].values, [nodos_almuerzo[i]['Prioridad'] for i in A])  # Prioridad baja para el almuerzo
    C_t_old = np.append(df_resultado['Tiempo_llegada_var'].values, [np.nan for i in A])
    lamb = {i: 1000 if Prioridad[i] == 'ALTA' else 200 if Prioridad[i] == 'MEDIA' else 1 for i in range(len(Prioridad))}
    k_old = np.append(df_resultado['Camion_ID'].values, [np.nan for i in A])
    M = 10000
    C = np.append(df_resultado['Hora_requerido_heuristica_min'].values, 
                  [nodos_almuerzo[i]['hora_referencia'] for i in A])  # Almuerzo centrado en 300 minutos
    col = df_resultado['Hora_retorno_proyectado_min'].fillna(0)
    col = col.infer_objects(copy=False)  # infiere el tipo antes de convertir
    R = np.append(col.astype(int).values, [0 for i in A]) # No tiene retorno proyectado
    b = {i: 0 for i in nodos} # Variable auxiliar para desactivar restricción phi[i] cuando se ingresan llegadas proyectadas  
    
    # Variables de decisión
    x = model_h.addVars(nodos, vehiculos, vtype=GRB.BINARY, name="x")
    z = model_h.addVars(arcos, vehiculos, vtype=GRB.BINARY, name="z")
    C_t = model_h.addVars(nodos, lb=0.0, vtype=GRB.CONTINUOUS, name="C_t")
    phi = model_h.addVars(nodos, lb=0.0, vtype=GRB.CONTINUOUS, name="Phi")
    w = model_h.addVars(arcos, vtype=GRB.BINARY, name="w")
    ru = model_h.addVars(nodos, lb=0.0, vtype=GRB.CONTINUOUS, name="ru")
    rl = model_h.addVars(nodos, lb=0.0, vtype=GRB.CONTINUOUS, name="rl")

    # -------- FUNCIÓN OBJETIVO ----------
    # Minimizar las desviaciones sin afectar el almuerzo
    model_h.setObjective(
        quicksum(lamb[i] * (ru[i] + rl[i]) for i in nodos if i not in A) +
        0.000000001 * quicksum(phi[i] for i in nodos if i not in A),
        GRB.MINIMIZE
    )

    # -------- RESTRICCIONES --------

    # Ventana de almuerzo (de 240 a 360 minutos)
    model_h.addConstrs((C_t[almuerzo_k] >= 240 for almuerzo_k in A), name="Ventana_inferior_almuerzo")    
    model_h.addConstrs((C_t[almuerzo_k] <= 360 for almuerzo_k in A), name="Ventana_superior_almuerzo")

    # Todos los pedidos deben ser atendidos
    model_h.addConstrs((quicksum(x[i, k] for k in vehiculos) == 1 
                        for i in nodos), name="Atencion_demanda")

    model_h.addConstrs((quicksum(x[almuerzo_k, k] for almuerzo_k in A) == 1 
                       for k in vehiculos), name="Todos_almuerzan")

    # Restricción: z[i,j,k] <= x[i,k] // predecisión supeditada a atención
    model_h.addConstrs(z[i, j, k] <= x[i, k] for i, j in arcos if i != j for k in vehiculos)
    model_h.addConstrs(z[i, j, k] <= x[j, k] for i, j in arcos if i != j for k in vehiculos)

    # Restricción: z[i,j,k] + z[j,i,k] >= x[i,k] + x[j,k] - 1
    model_h.addConstrs(z[i, j, k] + z[j, i, k] >= x[i, k] + x[j, k] - 1 for i in nodos for j in nodos for k in vehiculos 
                       if i != j)

    # Restricción: MTZ
    model_h.addConstrs(C_t[j] >= phi[i] + t0i[j] + p0[i] - 
                       M * (1 - quicksum(z[i, j, k] for k in vehiculos)) for i, j in arcos)

    # Restricción: Li[i] <= sum(C_t[i,k]) <= Ls[i]
    model_h.addConstrs(C_t[i] >= Li for i in nodos)
    model_h.addConstrs(C_t[i] <= Ls for i in nodos)

    # Apertura TAP
    model_h.addConstrs(C_t[i] >= 61 + Li for i in nodos) # Atención después de apertura de TAP (9:00 TA y 21:00 TB)
    model_h.addConstrs(C_t[i] <= Ls - 61 for i in nodos) # Atención antes del cierre del TAP (19:00 TA y 07:00 TB) 
    model_h.addConstrs(C_t[i] - t0i[i] >= 45 for i in nodos) # Carga comienza 15 min antes de apertura del TAP
    
    # Vuelta en horario
    model_h.addConstrs(phi[i] <= Ls - 61 for i in nodos) # Llegada a depo 1 hr antes del fin de turno

    # Restricción de número máximo de viajes
    model_h.addConstrs(quicksum(x[i, k] for i in nodos if i not in A) <= 3 for k in vehiculos)

    # Definición de C_t
    model_h.addConstrs(C_t[i] == C[i] + ru[i] - rl[i] for i in nodos)

    # Cotas para ru[i] y rl[i] según prioridad y tiempo fijo
    for i in nodos:
        if Prioridad[i] == 'ALTA':  # Si la prioridad es ALTA, fijar ru[i] y rl[i] a 0
            model_h.addConstr(ru[i] == 0, name=f"Fix_ru_{i}_ALTA")
            model_h.addConstr(rl[i] == 0, name=f"Fix_rl_{i}_ALTA")
        elif Prioridad[i] == 'MEDIA':  # Si la prioridad es MEDIA, limitar ru[i] y rl[i] a 20
            model_h.addConstr(ru[i] <= 20, name=f"Limit_ru_{i}_MEDIA")
            model_h.addConstr(rl[i] <= 20, name=f"Limit_rl_{i}_MEDIA")

        # Si C_t_old no es NaN y C_t_old <= tiempo_actual + 45, entonces C_t[i] = C_t_old[i]
        if not np.isnan(C_t_old[i]) and C_t_old[i] <= tiempo_actual + 45:
            model_h.addConstr(C_t[i] == C_t_old[i], name=f"Fix_C_t_{i}_TimeCondition")  # Congelar hora de llegada
            if not np.isnan(k_old[i]):  # Verifica que no sea NaN antes de indexar
                model_h.addConstr((x[i, int(k_old[i])]) == 1, name=f"Freeze_Camion_{i}")  # Congelar camión asociado

        if R[i] > 0:
            model_h.addConstr(phi[i] == R[i], name=f"Fix_R_{i}_TimeCondition")  # Congelar hora de retorno
            if not np.isnan(k_old[i]):  # Verifica que no sea NaN antes de indexar
                model_h.addConstr((x[i, int(k_old[i])]) == 1, name=f"Freeze_Camion_{i}")  # Congelar camión asociado
            b[i] = 1  # Variable auxiliar para desactivar restricción de phi[i]

    # No explorar nodos pasados
    C_t[i] >= tiempo_actual + 45
    
    # Def Phi[i]
    model_h.addConstrs(phi[i] >= C_t[i] + p[i] + ti0[i] - M * b[i] for i in nodos)  
   
    model_h.addConstrs(C_t[i] - t0i[i] >= 0 for i in nodos)
    if plants == 1:
        # Restricción de exclusividad con Big-M
        model_h.addConstrs(C_t[i] - t0i[i] >= C_t[j] - t0i[j] + p0[j] - M * (1 - w[i,j]) 
                           for (i, j) in arcos if i not in A and j not in A)
        model_h.addConstrs(C_t[i] - t0i[i] <= C_t[j] - t0i[j] - p0[i] + M * w[i,j] 
                           for (i, j) in arcos if i not in A and j not in A)

    else:
        # Big M para evitar conflictos (2 plantas default) + def phi con tiempos de espera
        model_h.addConstrs(C_t[i] - t0i[i] <= C_t[j] - t0i[j] - p0[i] + M * (1 - w[i,j]) 
                           for (i, j) in arcos if i not in A if j not in A)
        model_h.addConstrs(C_t[j] - t0i[j] <= C_t[i] - t0i[i] + p0[j] + M * w[i,j] 
                           for (i, j) in arcos if i not in A if j not in A)
        model_h.addConstrs(quicksum(w[i, j] + w[j, i] for j in nodos if j != i if j not in A) >= N - 2
                           for i in nodos if i not in A)
    
    # Actualizar y optimizar modelo
    model_h.update()
    model_h.setParam(GRB.Param.TimeLimit, time_limit)
    model_h.setParam(GRB.Param.MIPGap, 0.02)
        # Medir tiempo de resolución
    start_time = time.time()
    model_h.optimize()
    end_time = time.time()
    solve_time = end_time - start_time


    # Verificar estado de la solución
    if model_h.status == GRB.INFEASIBLE:
        st.error("El modelo es infactible.")
        return False, model_h, pd.DataFrame(), solve_time
    
    elif model_h.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
        print("Solución Óptima Encontrada")

        # Extraer resultados
        resultados = {
            'Pedido_ID': [],
            'Camion_ID': [],
            'Tiempo_carga': [],
            'Tiempo_llegada_fijo': [],
            'Tiempo_llegada_var': [],
            'ru': [],
            'rl': [],
            'Retorno_depo': []
        }

        for v in model_h.getVars():
            if v.x > 0.9 and v.VarName.startswith('x'):
                var_name = v.VarName.split('[')[1].split(']')[0]
                i, k = map(int, var_name.split(','))

                if i not in A:
                    resultados['Pedido_ID'].append(Viaje[i])
                    resultados['Camion_ID'].append(k)
                    resultados['Tiempo_carga'].append(
                        round(C_t[i].x - t0i[i]) if not np.isnan(C_t[i].x - t0i[i]) else None
                    )
                    resultados['Tiempo_llegada_fijo'].append(C[i])
                    resultados['Tiempo_llegada_var'].append(round(C_t[i].x))
                    resultados['Retorno_depo'].append(phi[i].x)
                    resultados['ru'].append(round(ru[i].x))
                    resultados['rl'].append(round(rl[i].x))

        df_resultados_camiones = pd.DataFrame(resultados)
        return True, model_h, df_resultados_camiones,solve_time

    else:
        st.info("No se encontró solución óptima.")
        return False, model_h, pd.DataFrame(),solve_time 
'''