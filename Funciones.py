# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------------------------------------
#LIBRERIAS A UTILIZAR
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#--------------------------------------------------------------------------------------------------------------------
def calcular_num_camiones(df_std):
    df_std = df_std.copy()
    df_std['Nro_camiones'] = np.ceil(df_std['Volumen Total [m3]'] / 7).astype(int)
    return df_std
#--------------------------------------------------------------------------------------------------------------------
def format_time_columns(df, columns, formatter):
    """
    Convierte las columnas especificadas de un DataFrame a tipo string y aplica un formato de hora.
    
    :param df: DataFrame de pandas.
    :param columns: Lista de nombres de columnas a convertir y formatear.
    :param formatter: Función de formateo que se aplicará a las columnas.
    :return: DataFrame con las columnas formateadas.
    """
    for column in columns:
        df[column] = df[column].astype(str).apply(formatter)
    return df
#--------------------------------------------------------------------------------------------------------------------
#FUNCIÓN QUE GENERA TABLA RESUMEN INICIAL PESTAÑA 1
def generar_tabla_resumen(df):
    # Filtrar las columnas necesarias
    filtered_df = df[['Empresa', 'Fecha', 'Volumen Total [m3]', 'Turno', 'Nro_camiones']].copy()
    
    # Convertir la columna Fecha a formato "día-mes"
    filtered_df['Fecha'] = pd.to_datetime(filtered_df['Fecha']).dt.strftime('%d-%m')
    
    # Calcular métricas por fecha
    daily_totals_m3 = filtered_df.groupby('Fecha')['Volumen Total [m3]'].sum()
    daily_totals_trips = filtered_df.groupby('Fecha')['Nro_camiones'].sum()
    daily_row_count = filtered_df.groupby('Fecha').size()
    
    # Crear un DataFrame consolidado
    summary_df = pd.DataFrame({
        'Volumen Total [m3]': daily_totals_m3,
        'Cantidad de Viajes': daily_totals_trips,
        'Cantidad de Pedidos': daily_row_count
    }).reset_index()
    
    # Transponer la tabla
    transposed_summary = summary_df.T
    transposed_summary.columns = transposed_summary.iloc[0]  # Usar fechas como encabezados
    transposed_summary = transposed_summary.drop(transposed_summary.index[0])  # Eliminar la fila duplicada
    
    # Mostrar la tabla en Streamlit
    st.table(transposed_summary)
#--------------------------------------------------------------------------------------------------------------------
#FUNCIÓN QUE GENERA MAPA DE CALOR PARA SUMA DE M3 POR CLIENTES AGRUPADA POR DIA Y TURNO FINAL DE PESTAÑA 1
def generar_resumen_semanal(dataframe):
    # Filtrar las columnas necesarias
    filtered_df = dataframe[['Empresa', 'Fecha', 'Volumen Total [m3]', 'Turno']].copy()

    # Convertir la columna Fecha a cadenas con el formato "día-mes"
    filtered_df['Fecha'] = pd.to_datetime(filtered_df['Fecha']).dt.strftime('%d-%m')

    # Crear una nueva tabla pivotada con totales de cada turno incluidos
    pivot_table_with_totals = pd.pivot_table(
        filtered_df,
        values='Volumen Total [m3]',
        index=['Turno', 'Empresa'],
        columns='Fecha',
        aggfunc='sum',
        fill_value=0,
        margins=True,
        margins_name='Total.general'
    )

    # Ordenar las filas de manera adecuada para resaltar los subtotales
    pivot_table_with_totals = pivot_table_with_totals.sort_index(level=0)

    # Generar el gráfico de calor
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table_with_totals.iloc[:-1, :-1], annot=True, fmt="g", cmap="YlGnBu", linewidths=.4)
    plt.xlabel('Fecha')
    plt.ylabel('Turno y Empresa')
    plt.tight_layout()

    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)
#--------------------------------------------------------------------------------------------------------------------
#CONVERTIR EN STRING HORA QUE SE SUBE DESDE EXCEL (PESTAÑA 2) 
# Function to format the time string
def format_time(time_str):
  if ":" in time_str:
    parts = time_str.split(":")
    if len(parts) == 3:  # HH:MM:SS format
      return f"{parts[0]}:{parts[1]}"
    elif len(parts) == 2: # HH:MM format
      return f"{parts[0]}:{parts[1]}"
  return time_str # Return as is if not in HH:MM or HH:MM:SS format

#-------------------------------------------------------------------------------------------------------------------
#FUNCIÓN PARA CONVERTIR FECHA EN FORMATO STRING HORA:MINUTO
def formato_h_m(df):
    df1 = df.copy()
    df1["INICIO CARGA"] = df1["INICIO CARGA"].dt.strftime('%H:%M')
    df1["LLEGADA OBRA"] = df1["LLEGADA OBRA"].dt.strftime('%H:%M')
    df1["RETORNO PLANTA"] = df1["RETORNO PLANTA"].dt.strftime('%H:%M')
    df1["HORA REQUERIDA"] = df1["HORA REQUERIDA"].dt.strftime('%H:%M')
    return df1
#-------------------------------------------------------------------------------------------------------------------
# Función para calcular las horas clave (heuristica)
def calcular_horass(df, referencia_fecha):
    """
    Calcula las horas clave (INICIO CARGA, LLEGADA OBRA, RETORNO PLANTA, HORA REQUERIDA)
    usando las columnas de tiempo en formato 'HH:MM' y crea nuevas columnas con formato datetime.datetime.
    Si la hora es menor a las 08:00 AM, se asigna al día siguiente.
    """
    # Convertir la referencia de fecha a datetime
    referencia_base = pd.to_datetime(referencia_fecha, format="%d-%m-%Y", dayfirst=True, errors='coerce')

    def ajustar_fecha(hora):
        hora_dt = pd.to_datetime(hora, format="%H:%M").time()  # Convertir la hora a formato datetime.time
        if hora_dt < pd.to_datetime("08:00", format="%H:%M").time():  # Si es antes de las 08:00 AM
            return pd.Timestamp.combine(referencia_base + pd.Timedelta(days=1), hora_dt)  # Asignar al día siguiente
        else:
            return pd.Timestamp.combine(referencia_base, hora_dt)  # Mantener en el mismo día

    # Aplicar la conversión con ajuste de fecha
    df["INICIO CARGA"] = df["Hora_carga"].apply(ajustar_fecha)
    df["HORA REQUERIDA"] = df["Hora_llegada_obra_fijo"].apply(ajustar_fecha)
    df["LLEGADA OBRA"] = df["Hora_llegada_obra_var"].apply(ajustar_fecha)
    df["RETORNO PLANTA"] = df["Hora_retorno"].apply(ajustar_fecha)

    return df
#--------------------------------------------------------------------------------------------------------------------
#calcular hora para modelo semanal


def calcular_horas_clave(df, referencia_fecha):
    """
    Calcula las columnas de INICIO CARGA, LLEGADA OBRA y RETORNO PLANTA
    a partir de las columnas Hora_carga, Hora_llegada_obra y Hora_retorno.
    Las horas se convierten en objetos datetime, ajustando al día siguiente si la hora es antes de las 08:00 AM.

    Parámetros:
    - df: DataFrame que contiene las columnas de horas en formato 'HH:MM'
    - referencia_fecha: string con la fecha base en formato 'dd-mm-YYYY'

    Retorna:
    - DataFrame con las columnas nuevas añadidas.
    """

    # Asegurarse de trabajar sobre una copia
    df = df.copy()

    # Convertir la fecha base a datetime
    referencia_base = pd.to_datetime(referencia_fecha, format="%d-%m-%Y", dayfirst=True, errors='coerce')

    def ajustar_fecha(hora):
        try:
            hora_dt = pd.to_datetime(hora, format="%H:%M", errors='coerce').time()
            if pd.isnull(hora_dt):
                return pd.NaT
            if hora_dt < pd.to_datetime("08:00", format="%H:%M").time():
                return pd.Timestamp.combine(referencia_base + pd.Timedelta(days=1), hora_dt)
            else:
                return pd.Timestamp.combine(referencia_base, hora_dt)
        except Exception:
            return pd.NaT

    # Aplicar ajustes con .loc para evitar SettingWithCopyWarning
    df.loc[:, "INICIO CARGA"] = df["Hora_carga"].apply(ajustar_fecha)
    df.loc[:, "LLEGADA OBRA"] = df["Hora_llegada_obra"].apply(ajustar_fecha)
    df.loc[:, "RETORNO PLANTA"] = df["Hora_retorno"].apply(ajustar_fecha)

    return df


def calcular_horas_ps(df, referencia_fecha):
    """
    Calcula las horas clave (INICIO CARGA, LLEGADA OBRA, RETORNO PLANTA, HORA REQUERIDA)
    usando las columnas de tiempo en formato 'HH:MM' y crea nuevas columnas con formato datetime.datetime.
    Si la hora es menor a las 08:00 AM, se asigna al día siguiente.
    """
    # Convertir la referencia de fecha a datetime
    referencia_base = pd.to_datetime(referencia_fecha, format="%d-%m-%Y", dayfirst=True, errors='coerce')

    def ajustar_fecha(hora):
        hora_dt = pd.to_datetime(hora, format="%H:%M").time()  # Convertir la hora a formato datetime.time
        if hora_dt < pd.to_datetime("08:00", format="%H:%M").time():  # Si es antes de las 08:00 AM
            return pd.Timestamp.combine(referencia_base + pd.Timedelta(days=1), hora_dt)  # Asignar al día siguiente
        else:
            return pd.Timestamp.combine(referencia_base, hora_dt)  # Mantener en el mismo día

    # Aplicar la conversión con ajuste de fecha
    df["INICIO CARGA"] = df["Hora_carga"].apply(ajustar_fecha)
    df["LLEGADA OBRA"] = df["Hora_llegada_obra"].apply(ajustar_fecha)
    df["RETORNO PLANTA"] = df["Hora_retorno"].apply(ajustar_fecha)

    return df
#--------------------------------------------------------------------------------------------------------
#GRAFICAR CRONOGRAMA RESULTADOS HEURISTICA 

def graficar_cronograma3(data):

    # Asegurarse de trabajar sobre una copia del DataFrame
    data = data.copy()

    # Ordenar los camiones (eje Y) de menor a mayor
    data = data[pd.notnull(data['Camion_ID'])]
    data['Camion_ID'] = data['Camion_ID'].astype(int)
    #data.sort_values('Camion_ID', inplace=False)

    # Asignar un color único a cada cliente
    clientes = data['Empresa'].unique()
    colores_clientes = {cliente: color for cliente, color in zip(clientes, plt.cm.tab10.colors)}

    # Tamaño figura
    fig, ax = plt.subplots(figsize=(22, 10))  # Tamaño ajustado para mejor visibilidad

    # Estilo general del gráfico
    ax.set_facecolor('#f7f7f7')  # Fondo claro y neutro

    # Definir propiedades de la caja de fondo de las horas
    bbox_inicio = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#F5F5DC", alpha=1)
    bbox_llegada = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#f7dc6f", alpha=1)
    bbox_retorno = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#F5F5DC", alpha=1)

    # Graficar los eventos
    for i, camion in enumerate(data['Camion_ID'].unique()):
        camion_data = data[data['Camion_ID'] == camion]
        for _, row in camion_data.iterrows():
            color_cliente = colores_clientes[row['Empresa']]

            # Trayecto completo (Planta -> Obra -> Planta)
            ax.plot([row['INICIO CARGA'], row['LLEGADA OBRA'], row['RETORNO PLANTA']],
                    [i, i, i], color=color_cliente, linewidth=10, solid_capstyle='round')

            # Agregar las horas exactas en los tres eventos con diferentes estilos
            ax.text(row['INICIO CARGA'], i, row['INICIO CARGA'].strftime('%H:%M'),
                    ha='left', va='center', color='black', fontsize=11, fontweight='bold', bbox=bbox_inicio)
            
            ax.text(row['LLEGADA OBRA'], i, row['LLEGADA OBRA'].strftime('%H:%M'),
                    ha='center', va='center', color='black', fontsize=11, fontweight='bold', bbox=bbox_llegada)

            diferencia = (row['LLEGADA OBRA'] - row['HORA REQUERIDA']).total_seconds() / 60
            diferencia_texto = f"{int(diferencia)} min" if diferencia != 0 else "A tiempo"
            ax.text(row['LLEGADA OBRA'], i + 0.19, diferencia_texto,
                   ha='center', va='bottom', color='red' if diferencia > 0 else 'green', fontsize=9, fontweight='bold')

            ax.text(row['RETORNO PLANTA'], i, row['RETORNO PLANTA'].strftime('%H:%M'),
                    ha='right', va='center', color='black', fontsize=11, fontweight='bold', bbox=bbox_retorno)

    # Configurar el eje Y (IDs de camiones)
    ax.set_yticks(range(len(data['Camion_ID'].unique())))
    ax.set_yticklabels(data['Camion_ID'].unique(), fontsize=12, fontweight='bold')
    ax.set_ylabel("Camion_ID", fontsize=14, fontweight='bold', color='black')

    # Configurar el eje X (Hora)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45, fontsize=12, ha='center')
    ax.set_xlabel("Hora", fontsize=14, fontweight='bold', color='black')
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')

    # Título del gráfico
    plt.title("Cronograma de Trayectos de Camiones (Inicio Carga - Llegada Obra - Retorno Planta)", fontsize=18, fontweight='bold', color='black', pad=20)

    # Leyenda
    legend_elements = [Patch(facecolor=color, label=cliente) for cliente, color in colores_clientes.items()]
    ax.legend(
        handles=legend_elements,
        title="Cliente",
        loc='upper left',  # Asegura que el punto de referencia sea la esquina superior izquierda de la leyenda
        bbox_to_anchor=(1.01, 1),  # Desplaza la leyenda hacia la derecha del gráfico
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        edgecolor='black'
    )


    # Mejorar el diseño general
    plt.tight_layout()

    # Mostrar la gráfica
    st.pyplot(fig)
#--------------------------------------------------------------------------------------------------------
#GRAFICAR CRONOGRAMA CON IDENTIFICADORES REALES  
def graficar_cronograma_act(data):

    # Asegurarse de trabajar sobre una copia del DataFrame
    data = data.copy()

    # Ordenar los camiones (eje Y) de menor a mayor
    data = data[pd.notnull(data['MIXER'])]
    #data['MIXER'] = data['MIXER'].astype(int)
    #data.sort_values('MIXER', inplace=True)

    # Asignar un color único a cada cliente
    clientes = data['Empresa'].unique()
    colores_clientes = {cliente: color for cliente, color in zip(clientes, plt.cm.tab10.colors)}

    # Tamaño figura
    fig, ax = plt.subplots(figsize=(22, 10))  # Tamaño ajustado para mejor visibilidad

    # Estilo general del gráfico
    ax.set_facecolor('#f7f7f7')  # Fondo claro y neutro

    # Definir propiedades de la caja de fondo de las horas
    bbox_inicio = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#F5F5DC", alpha=1)
    bbox_llegada = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#f7dc6f", alpha=1)
    bbox_retorno = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#F5F5DC", alpha=1)

    # Graficar los eventos
    for i, camion in enumerate(data['MIXER'].unique()):
        camion_data = data[data['MIXER'] == camion]
        for _, row in camion_data.iterrows():
            color_cliente = colores_clientes[row['Empresa']]

            # Trayecto completo (Planta -> Obra -> Planta)
            ax.plot([row['INICIO CARGA'], row['LLEGADA OBRA'], row['RETORNO PLANTA']],
                    [i, i, i], color=color_cliente, linewidth=10, solid_capstyle='round')

            # Agregar las horas exactas en los tres eventos con diferentes estilos
            ax.text(row['INICIO CARGA'], i, row['INICIO CARGA'].strftime('%H:%M'),
                    ha='left', va='center', color='black', fontsize=11, fontweight='bold', bbox=bbox_inicio)
            
            ax.text(row['LLEGADA OBRA'], i, row['LLEGADA OBRA'].strftime('%H:%M'),
                    ha='center', va='center', color='black', fontsize=11, fontweight='bold', bbox=bbox_llegada)

            diferencia = (row['LLEGADA OBRA'] - row['HORA REQUERIDA']).total_seconds() / 60
            diferencia_texto = f"{int(diferencia)} min" if diferencia != 0 else "A tiempo"
            ax.text(row['LLEGADA OBRA'], i + 0.2, diferencia_texto,
                   ha='center', va='bottom', color='red' if diferencia > 0 else 'green', fontsize=9, fontweight='bold')

            ax.text(row['RETORNO PLANTA'], i, row['RETORNO PLANTA'].strftime('%H:%M'),
                    ha='right', va='center', color='black', fontsize=11, fontweight='bold', bbox=bbox_retorno)

    # Configurar el eje Y (IDs de camiones)
    ax.set_yticks(range(len(data['MIXER'].unique())))
    ax.set_yticklabels(data['MIXER'].unique(), fontsize=12, fontweight='bold')
    ax.set_ylabel("MIXER", fontsize=14, fontweight='bold', color='black')

    # Configurar el eje X (Hora)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45, fontsize=12, ha='center')
    ax.set_xlabel("Hora", fontsize=14, fontweight='bold', color='black')
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')

    # Título del gráfico
    plt.title("Cronograma de Trayectos de Camiones (Inicio Carga - Llegada Obra - Retorno Planta)", fontsize=18, fontweight='bold', color='black', pad=20)

    # Leyenda
    legend_elements = [Patch(facecolor=color, label=cliente) for cliente, color in colores_clientes.items()]
    ax.legend(
        handles=legend_elements,
        title="Cliente",
        loc='upper left',  # Asegura que el punto de referencia sea la esquina superior izquierda de la leyenda
        bbox_to_anchor=(1.01, 1),  # Desplaza la leyenda hacia la derecha del gráfico
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        edgecolor='black'
    )


    # Mejorar el diseño general
    plt.tight_layout()

    # Mostrar la gráfica
    st.pyplot(fig)

def metricas(data):
    data = data.copy()
    data = data[pd.notnull(data['Camion_ID'])]

    # Calcular diferencias en minutos
    data['DIF_MIN'] = (data['LLEGADA OBRA'] - data['HORA REQUERIDA']).dt.total_seconds() / 60

    # Calcular métricas
    atrasados = (data['DIF_MIN'] > 0).sum()
    a_tiempo = ((data['DIF_MIN'] >= 0) & (data['DIF_MIN'] <= 5)).sum()
    adelantados = (data['DIF_MIN'] < 0).sum()

    # Mostrar métricas como tarjetas
    #col1, col2, col3 = st.columns(3)
    st.metric("🟥 Atrasados", atrasados)
    st.metric("🟩 A Tiempo", a_tiempo)
    st.metric("🟦 Adelantados", adelantados)
#--------------------------------------------------------------------------------------------------------
#GRAFICAR CRONOGRAMA RESULTADOS MODELO SEMANAL 

def graficar_cronograma4(data): 

    # Asegurarse de trabajar sobre una copia del DataFrame
    data = data.copy()

    # Ordenar los camiones (eje Y) de menor a mayor
    data = data[pd.notnull(data['Camion_ID']) & ~np.isinf(data['Camion_ID'])]
    data['Camion_ID'] = data['Camion_ID'].astype(int)
    data.sort_values('Camion_ID', inplace=False)

    # Asignar un color único a cada cliente
    clientes = data['Empresa'].unique()
    colores_clientes = {cliente: color for cliente, color in zip(clientes, plt.cm.tab10.colors)}

    # Tamaño figura
    fig, ax = plt.subplots(figsize=(22, 10))  # Tamaño ajustado para mejor visibilidad

    # Estilo general del gráfico
    ax.set_facecolor('#f7f7f7')  # Fondo claro y neutro

    # Definir propiedades de la caja de fondo de las horas
    bbox_inicio = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#F5F5DC", alpha=1)
    bbox_llegada = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#f7dc6f", alpha=1)
    bbox_retorno = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#F5F5DC", alpha=1)

    # Graficar los eventos
    for i, camion in enumerate(data['Camion_ID'].unique()):
        camion_data = data[data['Camion_ID'] == camion]
        for _, row in camion_data.iterrows():
            color_cliente = colores_clientes[row['Empresa']]

            # Trayecto completo (Planta -> Obra -> Planta)
            ax.plot([row['INICIO CARGA'], row['LLEGADA OBRA'], row['RETORNO PLANTA']],
                    [i, i, i], color=color_cliente, linewidth=10, solid_capstyle='round')

            # Agregar las horas exactas en los tres eventos con diferentes estilos
            ax.text(row['INICIO CARGA'], i, row['INICIO CARGA'].strftime('%H:%M'),
                    ha='left', va='center', color='black', fontsize=11, fontweight='bold', bbox=bbox_inicio)
            
            ax.text(row['LLEGADA OBRA'], i, row['LLEGADA OBRA'].strftime('%H:%M'),
                    ha='center', va='center', color='black', fontsize=11, fontweight='bold', bbox=bbox_llegada)

            ax.text(row['RETORNO PLANTA'], i, row['RETORNO PLANTA'].strftime('%H:%M'),
                    ha='right', va='center', color='black', fontsize=11, fontweight='bold', bbox=bbox_retorno)

    # Configurar el eje Y (IDs de camiones)
    ax.set_yticks(range(len(data['Camion_ID'].unique())))
    ax.set_yticklabels(data['Camion_ID'].unique(), fontsize=12, fontweight='bold')
    ax.set_ylabel("Camion ID", fontsize=14, fontweight='bold', color='black')

    # Configurar el eje X (Hora)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45, fontsize=12, ha='center')
    ax.set_xlabel("Hora", fontsize=14, fontweight='bold', color='black')
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')

    # Título del gráfico
    plt.title("Cronograma de Trayectos de Camiones (Inicio Carga - Llegada Obra - Retorno Planta)", fontsize=18, fontweight='bold', color='black', pad=20)

    # Leyenda
    legend_elements = [Patch(facecolor=color, label=cliente) for cliente, color in colores_clientes.items()]
    ax.legend(
        handles=legend_elements,
        title="Cliente",
        loc='upper left',  # Asegura que el punto de referencia sea la esquina superior izquierda de la leyenda
        bbox_to_anchor=(1.01, 1),  # Desplaza la leyenda hacia la derecha del gráfico
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        edgecolor='black'
    )


    # Mejorar el diseño general
    plt.tight_layout()

    # Mostrar la gráfica
    st.pyplot(fig)
#--------------------------------------------------------------------------------------------------------
#GRAFICAR CRONOGRAMA COMPARACION
def graficar_cronograma_comparacion(data):
    """
    Función para graficar el cronograma con MIXER en el eje Y, sin ordenar,
    asegurando que solo se grafican los MIXERs con viajes registrados.
    """
    data = data.copy()
    
    # Convertir las columnas de tiempo a formato datetime
    columnas_tiempo = [
        "Hora_carga", "Hora_llegada_obra_var", "Hora_retorno",
        "Hora_Carga_R", "Hora_Llegada_Planta_R", "Hora_Retorno_R"
    ]
    
    for col in columnas_tiempo:
        data[col] = pd.to_datetime(data[col], format='%H:%M', errors='coerce')

    # Filtrar solo filas donde todos los valores requeridos están completos
    data_filtrada = data.dropna(subset=columnas_tiempo, how='any')

    # Si no hay datos completos, mostrar advertencia y salir
    if data_filtrada.empty:
        st.warning("⚠️ No hay datos ingresados, no se puede realizar la comparación.")
        return

    # Obtener lista de MIXERs con viajes
    mixers_con_viajes = data_filtrada['MIXER'].unique()

    clientes = data_filtrada['Empresa'].unique()
    colores_clientes = {cliente: color for cliente, color in zip(clientes, plt.cm.tab10.colors)}
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(36, 12), sharey=True)
    ax1, ax2 = axes
    fig.patch.set_facecolor('#f0f0f0')
    
    bbox_inicio = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#F5F5DC", alpha=1)
    bbox_llegada = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#f7dc6f", alpha=1)
    bbox_retorno = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#F5F5DC", alpha=1)
    
    # Diferencias de tiempo
    bbox_adelantado = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="#2ECC71", alpha=1)  # Verde
    bbox_atrasado = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="#E74C3C", alpha=1)  # Rojo
    bbox_a_tiempo = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="#F5F5DC", alpha=1)  # Blanco/Crema

    ax1.set_facecolor('#f7f7f7')
    ax1.set_title("Cronograma con Horas Proyectadas", fontsize=20, fontweight='bold', color='black', pad=20)
    
    for i, mixer in enumerate(mixers_con_viajes):
        mixer_data = data_filtrada[data_filtrada['MIXER'] == mixer]
        for _, row in mixer_data.iterrows():
            color_cliente = colores_clientes.get(row['Empresa'], 'gray')
            ax1.plot([row['Hora_carga'], row['Hora_llegada_obra_var'], row['Hora_retorno']],
                     [i, i, i], color=color_cliente, linewidth=12, solid_capstyle='round')
            ax1.text(row['Hora_carga'], i, row['Hora_carga'].strftime('%H:%M'),
                     ha='left', va='center', color='black', fontsize=12, fontweight='bold', bbox=bbox_inicio)
            ax1.text(row['Hora_llegada_obra_var'], i, row['Hora_llegada_obra_var'].strftime('%H:%M'),
                     ha='center', va='center', color='black', fontsize=12, fontweight='bold', bbox=bbox_llegada)
            ax1.text(row['Hora_retorno'], i, row['Hora_retorno'].strftime('%H:%M'),
                     ha='right', va='center', color='black', fontsize=12, fontweight='bold', bbox=bbox_retorno)
    
    ax2.set_facecolor('#f7f7f7')
    ax2.set_title("Cronograma con Horas Reales", fontsize=20, fontweight='bold', color='black', pad=20)
    
    def get_bbox(real, proyectado):
        """Retorna el color del cuadro de texto según la comparación con la hora proyectada."""
        if pd.notnull(real) and pd.notnull(proyectado):
            if real < proyectado:
                return bbox_adelantado  # Verde si llegó antes
            elif real > proyectado:
                return bbox_atrasado  # Rojo si llegó después
            else:
                return bbox_a_tiempo  # Blanco/Crema si llegó exactamente a la hora proyectada
        return bbox_inicio  # Sin diferencia o no se puede calcular
    
    for i, mixer in enumerate(mixers_con_viajes):
        mixer_data = data_filtrada[data_filtrada['MIXER'] == mixer]
        for _, row in mixer_data.iterrows():
            color_cliente = colores_clientes.get(row['Empresa'], 'gray')
            ax2.plot([row['Hora_Carga_R'], row['Hora_Llegada_Planta_R'], row['Hora_Retorno_R']],
                     [i, i, i], color=color_cliente, linewidth=12, solid_capstyle='round')
            
            ax2.text(row['Hora_Carga_R'], i, row['Hora_Carga_R'].strftime('%H:%M') if pd.notnull(row['Hora_Carga_R']) else "N/A",
                     ha='left', va='center', color='black', fontsize=12, fontweight='bold', 
                     bbox=get_bbox(row['Hora_Carga_R'], row['Hora_carga']))
            
            ax2.text(row['Hora_Llegada_Planta_R'], i, row['Hora_Llegada_Planta_R'].strftime('%H:%M') if pd.notnull(row['Hora_Llegada_Planta_R']) else "N/A",
                     ha='center', va='center', color='black', fontsize=12, fontweight='bold', 
                     bbox=get_bbox(row['Hora_Llegada_Planta_R'], row['Hora_llegada_obra_var']))
            
            ax2.text(row['Hora_Retorno_R'], i, row['Hora_Retorno_R'].strftime('%H:%M') if pd.notnull(row['Hora_Retorno_R']) else "N/A",
                     ha='right', va='center', color='black', fontsize=12, fontweight='bold', 
                     bbox=get_bbox(row['Hora_Retorno_R'], row['Hora_retorno']))
    
    for ax in [ax1, ax2]:
        ax.set_yticks(range(len(mixers_con_viajes)))
        ax.set_yticklabels(mixers_con_viajes, fontsize=14, fontweight='bold', color='black')
        ax.set_xlabel("Hora", fontsize=16, fontweight='bold', color='black')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')
    
    ax1.set_ylabel("MIXER", fontsize=16, fontweight='bold', color='black')

    # Agregar leyenda con clientes y diferencias de tiempo
    legend_elements = [Patch(facecolor=color, label=cliente) for cliente, color in colores_clientes.items()]
    legend_elements.append(Patch(facecolor="#2ECC71", edgecolor="black", label="Adelantado"))
    legend_elements.append(Patch(facecolor="#E74C3C", edgecolor="black", label="Retrasado"))
    legend_elements.append(Patch(facecolor="#F5F5DC", edgecolor="black", label="A tiempo"))

    fig.legend(handles=legend_elements, title="Clientes / Diferencias", loc='upper center', bbox_to_anchor=(0.5, 1.02),
               fontsize=12, title_fontsize=14, frameon=True, edgecolor='black', ncol=len(clientes) // 2 + 1)
    plt.tight_layout()
    st.pyplot(fig)

#--------------------------------------------------------------------------------------------------------
#KPIS 
def calcular_kpis(data):
    """
    Función optimizada para calcular KPIs clave del cronograma de transporte y retornar el DataFrame actualizado.
    """
    data = data.copy()
    
    # Convertir columnas a datetime
    columnas_tiempo = [
        "Hora_carga", "Hora_llegada_obra_var", "Hora_retorno",
        "Hora_Carga_R", "Hora_Llegada_Planta_R", "Hora_Retorno_R"
    ]
    for col in columnas_tiempo:
        data[col] = pd.to_datetime(data[col], format='%H:%M', errors='coerce')
    
    # Calcular diferencias de tiempo en minutos si no existen
    if 'Retraso_carga' not in data.columns:
        data['Retraso_carga'] = (data['Hora_Carga_R'] - data['Hora_carga']).dt.total_seconds() / 60
    if 'Retraso_llegada' not in data.columns:
        data['Retraso_llegada'] = (data['Hora_Llegada_Planta_R'] - data['Hora_llegada_obra_var']).dt.total_seconds() / 60
    if 'Retraso_retorno' not in data.columns:
        data['Retraso_retorno'] = (data['Hora_Retorno_R'] - data['Hora_retorno']).dt.total_seconds() / 60
    
    # Reemplazar valores NaN por 0 en los retrasos para evitar errores
    data[['Retraso_carga', 'Retraso_llegada', 'Retraso_retorno']] = data[['Retraso_carga', 'Retraso_llegada', 'Retraso_retorno']].fillna(0)
    
    # KPIs principales
    total_viajes = data.shape[0]
    promedio_retraso_carga = data['Retraso_carga'].mean()
    promedio_retraso_llegada = data['Retraso_llegada'].mean()
    promedio_retraso_retorno = data['Retraso_retorno'].mean()
    viajes_con_retraso = data[(data['Retraso_llegada'] > 5) | (data['Retraso_retorno'] > 5)].shape[0]
    viajes_a_tiempo = data[(data['Retraso_llegada'] <= 5) & (data['Retraso_retorno'] <= 5)].shape[0]
    porcentaje_viajes_a_tiempo = (viajes_a_tiempo / total_viajes) * 100 if total_viajes > 0 else 0
    
    # Identificación de planta, cliente y MIXER con mayor impacto
    planta_mas_retraso = data.groupby('Planta_salida')['Retraso_llegada'].sum().idxmax()
    cliente_top = data['Empresa'].value_counts().idxmax()
    
    return data, {
        "Total de viajes": total_viajes,
        "Promedio retraso carga (min)": round(promedio_retraso_carga, 2),
        "Promedio retraso llegada (min)": round(promedio_retraso_llegada, 2),
        "Promedio retraso retorno (min)": round(promedio_retraso_retorno, 2),
        "Cantidad de viajes con retraso": viajes_con_retraso,
        "% de viajes a tiempo": round(porcentaje_viajes_a_tiempo, 2),
        "Planta con más retrasos": planta_mas_retraso,
        "Cliente con más viajes": cliente_top,
    }
def calcular_kpis_individuales(data):
    """
    Función para calcular KPIs por cliente, incluyendo:
    - Retraso promedio por cliente
    - Adelanto promedio por cliente
    - Cantidad de pedidos a tiempo
    """
    data,kpis = calcular_kpis(data)  # Asegurar que las columnas de retraso están calculadas
    
    # Calcular adelantos (cuando hay valores negativos en retraso significa que llegó antes)
    data['Adelanto_carga'] = data['Retraso_carga'].apply(lambda x: x if x < 0 else 0)
    data['Adelanto_llegada'] = data['Retraso_llegada'].apply(lambda x: x if x < 0 else 0)
    data['Adelanto_retorno'] = data['Retraso_retorno'].apply(lambda x: x if x < 0 else 0)
    
    # Calcular cantidad de pedidos a tiempo (retraso <= 5 min)
    data['Pedidos_a_tiempo'] = ((data['Retraso_carga'] <= 5) & (data['Retraso_llegada'] <= 5) & (data['Retraso_retorno'] <= 5)).astype(int)
    
    # Agrupar por cliente
    kpis_por_cliente = data.groupby('Empresa').agg({
        'Retraso_carga': 'mean',
        'Retraso_llegada': 'mean',
        'Retraso_retorno': 'mean',
        'Adelanto_carga': 'mean',
        'Adelanto_llegada': 'mean',
        'Adelanto_retorno': 'mean',
        'Pedidos_a_tiempo': 'sum',
        'MIXER': 'count'
    }).rename(columns={
        'Retraso_carga': 'Retraso Promedio Carga (min)',
        'Retraso_llegada': 'Retraso Promedio Llegada (min)',
        'Retraso_retorno': 'Retraso Promedio Retorno (min)',
        'Adelanto_carga': 'Adelanto Promedio Carga (min)',
        'Adelanto_llegada': 'Adelanto Promedio Llegada (min)',
        'Adelanto_retorno': 'Adelanto Promedio Retorno (min)',
        'Pedidos_a_tiempo': 'Cantidad Pedidos a Tiempo',
        'MIXER': 'Total de Viajes'
    }).reset_index()
    
    return kpis_por_cliente

def mostrar_kpis(data):
    """
    Función para mostrar un panel de KPIs con diseño profesional en Streamlit.
    """
    data, kpis = calcular_kpis(data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="📦 Total de viajes", value=kpis["Total de viajes"])
        st.metric(label="🔴 Viajes con retraso", value=kpis["Cantidad de viajes con retraso"])
    
    with col2:
        st.metric(label="⏳ Retraso medio en carga (min)", value=kpis["Promedio retraso carga (min)"])
        st.metric(label="⏳ Retraso medio en llegada (min)", value=kpis["Promedio retraso llegada (min)"])
        st.metric(label="⏳ Retraso medio en retorno (min)", value=kpis["Promedio retraso retorno (min)"])
    
    with col3:
        st.metric(label="🟢 % de viajes a tiempo", value=f"{kpis['% de viajes a tiempo']}%")
        st.metric(label="🏭 Planta con más retrasos", value=kpis["Planta con más retrasos"])
        st.metric(label="🏢 Cliente con más viajes", value=kpis["Cliente con más viajes"])

def mostrar_kpis_individuales(data):
    """
    Muestra una tabla con KPIs individuales por cliente en Streamlit.
    """
    kpis_cliente = calcular_kpis_individuales(data)
    
    st.subheader("📊 KPIs por Cliente")
    st.dataframe(kpis_cliente.style.format({
        "Retraso Promedio Carga (min)": "{:.2f}",
        "Retraso Promedio Llegada (min)": "{:.2f}",
        "Retraso Promedio Retorno (min)": "{:.2f}",
        "Adelanto Promedio Carga (min)": "{:.2f}",
        "Adelanto Promedio Llegada (min)": "{:.2f}",
        "Adelanto Promedio Retorno (min)": "{:.2f}",
        "Cantidad Pedidos a Tiempo": "{:.0f}",
        "Total de Viajes": "{:.0f}"
    }))

#--------------------------------------------------------------------------------------------------------
#FUNCION PARA GENERAR TABLA QUE INDICA CANTIDAD DE CAMIONES POR HORA (EN TRANSITO O IDLE)
def generar_tabla_camiones(df):
    """
    Genera una tabla binaria que indica si un camión está en tránsito o en el depósito durante cada rango de horas.
    Trabaja con horas en formato de texto (HH:MM) sin fechas.
    """

    # Convertir las columnas de tiempo de formato de texto (HH:MM) a minutos desde las 00:00
    df['INICIO CARGA'] = pd.to_timedelta(df['INICIO CARGA'] + ':00')
    df['RETORNO PLANTA'] = pd.to_timedelta(df['RETORNO PLANTA'] + ':00')
    df['LLEGADA OBRA'] = pd.to_timedelta(df['LLEGADA OBRA'] + ':00')

    turno = df['Turno'][0]
    if turno == 'TA':
        # Generar los rangos horarios de 08:00 a 20:00 en intervalos de 1 hora
        rangos_horas = [pd.to_timedelta(f"{h}:00:00") for h in range(8, 21)]
    elif turno == 'TB':
        # Generar los rangos horarios de 20:00 a 08:00 en intervalos de 1 hora
        rangos_horas = (
            [pd.to_timedelta(f"{h}:00:00") for h in range(20, 24)] +
            [pd.to_timedelta(f"{h}:00:00") for h in range(0, 9)]
        )

    columnas_horarias = [
        f"{str(r).split()[2][:5]} - {str((r + pd.Timedelta(hours=1))).split()[2][:5]}"
        for r in rangos_horas[:-1]
    ]

    # Crear un DataFrame vacío para la nueva tabla
    tabla_camiones = pd.DataFrame(
        0,
        index=df['Camion_ID'].unique(),
        columns=columnas_horarias
    )

    # Iterar por cada fila y marcar los rangos de tiempo
    for _, row in df.iterrows():
        camion_id = row['Camion_ID']
        hora_salida = row['INICIO CARGA']
        hora_llegada_planta = row['RETORNO PLANTA']
        horallegada_obra = row['LLEGADA OBRA']

        for rango_inicio, columna in zip(rangos_horas[:-1], columnas_horarias):
            rango_fin = rango_inicio + pd.Timedelta(hours=1)

            # Verificar si el camión estuvo en tránsito en el rango de horas
            if (rango_inicio <= hora_salida < rango_fin) or \
               (rango_inicio <= hora_llegada_planta < rango_fin) or \
               (rango_inicio <= horallegada_obra < rango_fin) or \
               (hora_llegada_planta > rango_fin and horallegada_obra < rango_inicio):
                tabla_camiones.loc[camion_id, columna] = 1

    # Ordenar la tabla por Camion_ID (índice) de menor a mayor
    tabla_camiones.sort_index(ascending=True, inplace=True)

    return tabla_camiones

#--------------------------------------------------------------------------------------------------------
def graficar_transito(tabla_camiones):
    """
    Grafica el número de camiones en tránsito y en depósito por rango horario,
    y agrega anotaciones a cada barra. Los valores del eje Y se muestran como enteros.
    """
    def contar_valores(column):
        return pd.Series(column).value_counts()

    cantidad_camiones = tabla_camiones.apply(contar_valores).fillna(0).T
    cantidad_camiones.columns = ['IDLE', 'En Tránsito']
    cantidad_camiones = cantidad_camiones[['En Tránsito', 'IDLE']]

    # Crear gráfico de barras apiladas
    ax = cantidad_camiones.plot(
        kind='bar',
        stacked=True,
        figsize=(8, 5),
        color=['#1f77b4', '#ff7f0e']
    )

    # Configurar etiquetas y título
    plt.xlabel('Rango de Horas')
    plt.ylabel('Cantidad de Camiones')
    plt.title('Estado de Camiones')
    plt.xticks(rotation=45, ha='right')

    # Forzar eje Y con enteros
    max_total = int(cantidad_camiones.sum(axis=1).max())
    ax.set_yticks(range(0, max_total + 1))  # ticks de 1 en 1

    # Ajustar leyenda
    plt.legend(
        ['En Tránsito', 'IDLE'],
        loc='lower right',
        bbox_to_anchor=(1, 1),
        ncol=2
    )

    # Agregar anotaciones
    for bar_group in ax.containers:
        for rect in bar_group:
            height = rect.get_height()
            if height > 0:
                x = rect.get_x() + rect.get_width() / 2
                y = rect.get_y() + height / 2
                ax.annotate(
                    str(int(height)),
                    (x, y),
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black'
                )

    plt.tight_layout()
    st.pyplot(ax.get_figure())



#--------------------------------------------------------------------------------------------------------
#CONVERSION DE HORAS
def convertir_horas_datetime(df,base_date):
    def convertir_timedelta_a_datetime(df, base_date):
        """
        Convierte las columnas de tiempo (SALIDA PLANTA, LLEGADA OBRA, RETORNO PLANTA)
        de formato timedelta a formato datetime con una fecha base.
        """
        # Convertir las columnas de timedelta a datetime
        df['INICIO CARGA'] = pd.to_datetime(base_date) + df['INICIO CARGA']
        df['LLEGADA OBRA'] = pd.to_datetime(base_date) + df['LLEGADA OBRA']
        df['RETORNO PLANTA'] = pd.to_datetime(base_date) + df['RETORNO PLANTA']

        return df

    #Convertir hora a datetime para luego ponerlas en texto para mostrarlas en la tabla
    df = convertir_timedelta_a_datetime(df, base_date)
    df["INICIO CARGA"] = df["INICIO CARGA"].dt.strftime('%H:%M')
    df["LLEGADA OBRA"] = df["LLEGADA OBRA"].dt.strftime('%H:%M')
    df["RETORNO PLANTA"] = df["RETORNO PLANTA"].dt.strftime('%H:%M')

    return df 