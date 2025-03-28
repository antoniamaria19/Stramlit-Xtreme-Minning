# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import io
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import seaborn as sns
from tabulate import tabulate
import sys
import xlsxwriter
#importaciones codigos 
import Funciones as f
#importaciones codigos 
import Funciones as f
import Modelo_Planificacion_Semanal as mps 
import Modelo_Planificacion_Heuristica_Diaria as HD
from datetime import datetime
import os
from datetime import time


# Configuración de la página
st.set_page_config(
    page_title="Planificación de Pedidos - Xtreme Mining",
    page_icon="🚛",
    layout="wide",
)

# Título de la aplicación
st.title("📋 Planificación de Pedidos - Xtreme Mining")

# Inicializar variables globales
planif_semanal = None
data_aux = None
conductores_df = None
tiempos_aux = None
productos_aux = None
prioridades_aux = None
mixer_df = None

def calcular_num_camiones(df_std):
    df_std = df_std.copy()
    df_std['Nro_camiones'] = np.ceil(df_std['Volumen Total [m3]'] / 7).astype(int)
    return df_std


# Crear pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Programa Semanal",
    "📋 Ingreso de Datos - Programa Diario",
    "📆 Panel de Planificación Diaria",
    "🎯 Optimización de Planificación"
])

# VISUALIZACIÓN DATOS PLANIFICACIÓN
with tab1:
    st.header("📊 Programa semanal")

    # Inicializar la lista de uploads si no existe
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []

    # Botón para agregar un nuevo file_uploader dinámicamente
    if st.button("➕ Agregar archivo Excel de Pedidos"):
        st.session_state['uploaded_files'].append(None)

    # Lista para almacenar DataFrames leídos
    dfs_planificacion = []

    # Mostrar todos los file_uploaders dinámicos
    for i in range(len(st.session_state['uploaded_files'])):
        col1, col2 = st.columns([6, 0.2])  # file_uploader a la izquierda, botón eliminar a la derecha

        with col1:
            # Inicializar clave si no existe
            if f"archivo_{i}" not in st.session_state:
                st.session_state[f"archivo_{i}"] = None

            uploaded_file = st.file_uploader(f"Archivo Excel {i+1}", type=["xlsx"], key=f"file_uploader_{i}")

            if uploaded_file is not None:
                st.session_state[f"archivo_{i}"] = uploaded_file

            # Usar archivo desde session_state
            if st.session_state[f"archivo_{i}"] is not None:
                try:
                    df = pd.read_excel(st.session_state[f"archivo_{i}"], sheet_name="BD_Programa")
                    df['Archivo_Origen'] = st.session_state[f"archivo_{i}"].name
                    dfs_planificacion.append(df)
                    #st.success(f"Archivo cargado: {st.session_state[f'archivo_{i}'].name}")
                except Exception as e:
                    st.error(f"Error al procesar el archivo {st.session_state[f'archivo_{i}'].name}: {e}")

        with col2:
            eliminar = st.button("🗑️", key=f"eliminar_{i}")
            if eliminar:
                st.session_state['uploaded_files'].pop(i)
                st.session_state.pop(f"archivo_{i}", None)
                st.rerun()

    # Validar si se cargó al menos un archivo
    if dfs_planificacion:
        try:
            # Concatenar todos los archivos cargados
            planif_semanal = pd.concat(dfs_planificacion, ignore_index=True)

            # Aplicar cálculo de camiones
            planif_semanal = calcular_num_camiones(planif_semanal)

            # Validar columnas esenciales
            if "Fecha" not in planif_semanal.columns or "Turno" not in planif_semanal.columns:
                st.error("El archivo no contiene las columnas requeridas: 'Fecha' o 'Turno'.")
                st.stop()

            # Convertir columna Fecha a datetime
            planif_semanal['Fecha'] = pd.to_datetime(planif_semanal['Fecha'], format="%d-%m-%Y", dayfirst=True, errors='coerce')

            # Guardar en session_state
            st.session_state['planif_semanal'] = planif_semanal

            # Guardar como archivo físico para uso en tab4
            temp_path = os.path.join(".", "temp_planificacion.xlsx")
            planif_semanal.to_excel(temp_path, index=False)
            st.session_state['planif_file'] = temp_path

            # Cargar archivos auxiliares
            script_dir = os.path.dirname(os.path.abspath(__file__))
            ruta_aux = os.path.join(script_dir, "Tablas_aux_datos.xlsx")

            if os.path.exists(ruta_aux):
                df_aux = pd.ExcelFile(ruta_aux)
                data_aux = pd.read_excel(df_aux, sheet_name="Datos")
                conductores_df = pd.read_excel(df_aux, sheet_name="Conductores")
                mixer_df = pd.read_excel(df_aux, sheet_name="Mixers")
                tiempos_aux = pd.read_excel(df_aux, sheet_name="Tiempos")
                productos_aux = pd.read_excel(df_aux, sheet_name="Productos")
                prioridades_aux = pd.read_excel(df_aux, sheet_name="Prioridades")

                # Guardar en session_state para uso en otras pestañas
                st.session_state['MIXERS'] = mixer_df['MIXER'].dropna().unique().tolist()
                st.session_state['CONDUCTORES'] = conductores_df['CONDUCTOR'].dropna().unique().tolist()
            else:
                st.warning("No se encontró el archivo de datos auxiliares.")

            st.success("Archivos cargados y procesados exitosamente.")

            dashboard_data = planif_semanal.copy()

            if dashboard_data is not None:
                # Rango de semana al centro
                fecha_min = dashboard_data['Fecha'].min().strftime('%d-%m-%Y')
                fecha_max = dashboard_data['Fecha'].max().strftime('%d-%m-%Y')
                st.markdown(
                    f"""
                    <div style="text-align: center; font-size: 22px; margin-bottom: 18px;">
                        <strong>Semana: {fecha_min} - {fecha_max}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                
                #st.write(sys.executable) #en que entorno se corre
                
                # Filtros globales arriba
                #st.subheader("Filtros")
                col_filtros1, col_filtros2 = st.columns([1, 1])
  
                # Filtros con estilos
                with col_filtros1:
                    empresas_disponibles = ['Todos'] + list(dashboard_data['Empresa'].unique())
                    empresa_seleccionada = st.selectbox(
                        "📋 **Cliente**",
                        options=empresas_disponibles,
                        index=0,
                        help="Elija el cliente para filtrar los datos."
                    )

                with col_filtros2:
                    turnos_disponibles = ['Todos'] + list(dashboard_data['Turno'].unique())
                    turno_seleccionado = st.selectbox(
                        "⏰ **Turno**",
                        options=turnos_disponibles,
                        index=0,
                        help="Elija el turno para filtrar los datos."
                    )
                
                # Filtrar datos según selección
                data_filtrada = dashboard_data.copy()
                if empresa_seleccionada != 'Todos':
                    data_filtrada = data_filtrada[data_filtrada['Empresa'] == empresa_seleccionada]
                if turno_seleccionado != 'Todos':
                    data_filtrada = data_filtrada[data_filtrada['Turno'] == turno_seleccionado]
                
                st.markdown("<h3 style='font-size:20px;'>Resumen de la Semana</h3>", unsafe_allow_html=True)
 
                f.generar_tabla_resumen(data_filtrada)

                col_tarjetas, col_grafico = st.columns([1, 3])
                
                with col_tarjetas:
                    total_pedidos = len(data_filtrada)
                    total_m3 = data_filtrada['Volumen Total [m3]'].sum()
                    total_viajes = data_filtrada['Nro_camiones'].sum()

                    # Tarjeta: Total de Pedidos
                    st.markdown(
                        f"""
                        <div style='text-align: center; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 5px; padding: 5px 10px; margin: 2px;'>
                            <h4 style='margin: 0; font-size: 14px;'>Cantidad de Pedidos</h4>
                            <h3 style='color: #2c3e50; margin: 0; font-size: 20px;'>{total_pedidos}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Espaciado para separar secciones
                    st.markdown("<hr>", unsafe_allow_html=True)
                    # Tarjeta: Volumen Total
                    st.markdown(
                        f"""
                        <div style='text-align: center; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 5px; padding: 5px 10px; margin: 2px;'>
                            <h4 style='margin: 0; font-size: 14px;'>Demanda Total Estimada (m³)</h4>
                            <h3 style='color: #2c3e50; margin: 0; font-size: 20px;'>{total_m3:.2f}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Espaciado para separar secciones
                    st.markdown("<hr>", unsafe_allow_html=True)
                    # Tarjeta: Total de Viajes
                    st.markdown(
                        f"""
                        <div style='text-align: center; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 5px; padding: 5px 10px; margin: 2px;'>
                            <h4 style='margin: 0; font-size: 14px;'>Cantidad de Viajes</h4>
                            <h3 style='color: #2c3e50; margin: 0; font-size: 20px;'>{total_viajes}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col_grafico:
                    # Selección de tipo de gráfico
                    tipo_grafico = st.radio(
                        "Seleccione el tipo de gráfico:",
                        options=["Demanda de m³ por turno y fecha", "Cantidad de viajes por turno y fecha"],
                        index=0
                    )

                    # Formatear 'Fecha' como DD-MM para agrupar
                    data_filtrada['Fecha'] = data_filtrada['Fecha'].dt.strftime('%d-%m')

                    if tipo_grafico == "Demanda de m³ por turno y fecha":
                        grouped_data = data_filtrada.groupby(['Fecha', 'Turno'])['Volumen Total [m3]'].sum().unstack()
                        ylabel = "Volumen(m³)"
                        title = "Demanda de m³ por Turno y Fecha"
                    else:
                        grouped_data = data_filtrada.groupby(['Fecha', 'Turno'])['Nro_camiones'].sum().unstack()
                        ylabel = "Cantidad de Viajes"
                        title = "Cantidad de Viajes por Turno y Fecha"

                    # Crear el gráfico
                    fig, ax = plt.subplots(figsize=(10, 6))
                    grouped_data.plot(kind='bar', stacked=True, ax=ax)

                    # Agregar etiquetas a cada barra
                    for container in ax.containers:
                        ax.bar_label(container, label_type='center')

                    # Personalizar el gráfico
                    ax.set_title(title)
                    ax.set_xlabel('Fecha')
                    ax.set_ylabel(ylabel)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

                    # Configurar valores del eje Y para "Cantidad de Viajes"
                    if tipo_grafico == "Cantidad de Viajes por Turno y Fecha":
                        # Obtener el máximo valor en el eje Y
                        max_value = grouped_data.sum(axis=1).max()
                        # Crear ticks de 2 en 2
                        yticks = range(0, int(max_value) + 2, 2)
                        ax.set_yticks(yticks)

                    # Leyenda encima del gráfico
                    ax.legend(
                        title='Turno',
                        loc='upper center',
                        bbox_to_anchor=(0.95, 1.38),
                        ncol=2,
                        fontsize=9
                    )

                    plt.tight_layout()

                    # Mostrar gráfico en Streamlit
                    st.pyplot(fig)

            columnas_mostrar = ['Fecha', 'Turno', 'Empresa', 'Codigo Producto', 'Volumen Total [m3]', 'Nro_camiones', 'Punto de Entrega', 'Destino Final' ]
            st.markdown("<h3 style='font-size:20px;'> Detalle de los Pedidos</h3>", unsafe_allow_html=True)
            st.dataframe(data_filtrada[columnas_mostrar], use_container_width=True)


            st.markdown("<h3 style='font-size:20px;'>Resumen Semanal de Pedidos por Cliente (m³)</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                turnos_disponibles2 = ['Todos'] + list(planif_semanal['Turno'].unique())
                turno_seleccionado2 = st.selectbox(
                    "**Turno**",
                    options=turnos_disponibles2,
                    index=0,
                    help="Elija el turno para filtrar los datos."
                )   
            # Filtrar datos según selección
            data_filtrada2 = planif_semanal.copy()
            if turno_seleccionado2 != 'Todos':
                data_filtrada2 = data_filtrada2[data_filtrada2['Turno'] == turno_seleccionado2]
                
            f.generar_resumen_semanal(data_filtrada2)
            #f.generar_resumen_semanal(planif_semanal)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.info("Por favor, suba un archivo Excel para continuar.") 


# Pestaña de planificación diaria
with tab2:
    st.header("Ingreso de datos - Programa diario")
    
    if planif_semanal is not None:
        # Filtros en dos columnas
        col1, col2 = st.columns(2)
        with col1:
            fechas_ordenadas = sorted(
                    planif_semanal['Fecha'].dropna().dt.date.unique()
            )
            fecha_seleccionada = st.selectbox(
                "Seleccione una fecha",
                fechas_ordenadas,
                key="fecha_tab1"
            )
        st.session_state['fecha_seleccionada'] = fecha_seleccionada
        with col2:
            turno_seleccionado = st.selectbox(
                "Seleccione un turno",
                planif_semanal['Turno'].unique(),
                key="turno_tab1"
            )

        if fecha_seleccionada and turno_seleccionado:
            # Filtrar datos por fecha y turno
            data_filtro_fecha_turno = planif_semanal[
                (planif_semanal['Fecha'].dt.date == fecha_seleccionada)
                & (planif_semanal['Turno'] == turno_seleccionado)
            ]
            data_filtro_fecha_turno = data_filtro_fecha_turno.dropna(how="all")  # Eliminar filas completamente vacías

            # Mostrar tarjeta con fecha y turno
            primera_fecha = data_filtro_fecha_turno['Fecha'].dt.strftime('%d-%m').iloc[0]
            primer_turno = data_filtro_fecha_turno['Turno'].iloc[0]
            st.info(f"**Pedidos para el {primera_fecha}, {primer_turno}**")

            st.session_state['data_filtro_fecha_turno'] = data_filtro_fecha_turno

            # Cargar archivo previamente editado (opcional)
            uploaded_file = st.file_uploader("Cargue un archivo editado en formato Excel", type="xlsx", key="upload_excel")
            
            if uploaded_file:
                # Leer archivo Excel y cargar la tabla editada
                tabla_editada = pd.read_excel(uploaded_file)
                tabla_editada = tabla_editada.fillna("")
                # Definir todas las columnas que deben estar en el archivo cargado
                columnas_requeridas = [
                    'Hora_requerida', 'Hora_llamado', 'Fecha', 'Turno', 'Pedido_ID', 'Empresa',
                    'Volumen_confirmado', 'Volumen_transportado', 'Volumen_m3', 
                    'Punto_entrega', 'Destino_final', 'Codigo_producto', 'Familia', 'Prioridad',
                    'Llegada_obra_min', 'Atencion_min', 'Retorno_min', 'Lavado_min',
                    'Descripcion', 'Obra', 'Uso', 'Observaciones', 'Hora_retorno_proyectado'
                ]

                # Verificar qué columnas faltan en el archivo subido
                columnas_faltantes = [col for col in columnas_requeridas if col not in tabla_editada.columns]

                if columnas_faltantes:
                    # Mensaje de error más amigable y con lista de columnas faltantes
                    st.error(f"El archivo cargado no es el correcto falta {columnas_faltantes} ")
                    st.stop()  # Detiene la ejecución si faltan columnas

                tabla_editada['Hora_requerida'] = tabla_editada['Hora_requerida'].astype(str)
                tabla_editada['Hora_llamado'] = tabla_editada['Hora_llamado'].astype(str)

                tabla_editada['Hora_requerida'] = tabla_editada['Hora_requerida'].apply(f.format_time)
                tabla_editada['Hora_llamado'] = tabla_editada['Hora_llamado'].apply(f.format_time)

                planif_diaria = HD.procesar_tabla_editada(tabla_editada)
                # Reset the index to remove duplicates and reset index labels
                planif_diaria = planif_diaria.reset_index(drop=True)
                planif_diaria['ID'] = range(1, len(planif_diaria) + 1)

                st.success("Archivo Excel cargado correctamente. Puede continuar editando.")
            else:
                # Generar planificación desde cero si no se sube archivo
                planif_diaria = HD.procesar_planificacion_inicial(
                    data_filtro_fecha_turno,
                    tiempos_aux,
                    productos_aux, 
                    prioridades_aux,
                    fechas=[fecha_seleccionada],
                    turnos=[turno_seleccionado]
                )
                # Reset the index to remove duplicates and reset index labels
                planif_diaria = planif_diaria.reset_index(drop=True)
                planif_diaria['ID'] = range(1, len(planif_diaria) + 1)

            #st.dataframe(planif_diaria)
            # Seleccionar columnas para la tabla editable
            editable_columns = [
                'ID', 'Pedido_ID','Empresa', 'Hora_llamado', 'Hora_requerida','Volumen_confirmado', 
                'Prioridad', 'Hora_retorno_proyectado', 'Volumen_transportado', 'Punto_entrega', 'Destino_final',
                'Codigo_producto', 'Familia','Llegada_obra_min', 'Atencion_min', 'Retorno_min',
                'Lavado_min','Descripcion','Obra','Uso', 'Volumen_m3',  'Fecha', 'Turno', 'Observaciones'
                ]
            

            # Filtrar solo las columnas relevantes
            editable_df = planif_diaria[editable_columns]
            editable_df = editable_df.dropna(how="all")  # Eliminar filas completamente vacías


            # Verificar si la columna 'Fecha' ya está en el formato '%d-%m-%Y'
            if not all(editable_df['Fecha'].astype(str).str.match(r'\d{2}-\d{2}-\d{4}')):
                # Convertir a datetime y luego formatear
                editable_df['Fecha'] = pd.to_datetime(editable_df['Fecha'], errors='coerce').dt.strftime('%d-%m-%Y')

            
            #listas de info para tabla con campos despegables 
            empresas = list(set(data_aux['Empresas'].unique().tolist() + planif_diaria['Empresa'].unique().tolist()))
            codigos_producto = data_aux['Codigos'].unique().tolist() 
            obras = list(set(data_aux['Obras'].unique().tolist() + planif_diaria['Obra'].unique().tolist()))
            usos = list(set(data_aux['Usos'].unique().tolist() + planif_diaria['Uso'].unique().tolist()))
            fecha = planif_diaria['Fecha'].unique().tolist()
            turno = planif_diaria['Turno'].unique().tolist()

            # Eliminar filas duplicadas basado en todas las columnas relevantes
            editable_df = editable_df.drop_duplicates()
        
        
            # Crear editor interactivo con opciones desplegables
            st.subheader("Confirmación de pedidos")

            edited_df = st.data_editor(
                editable_df,
                column_config={
                    "Volumen_confirmado": st.column_config.SelectboxColumn("Volumen_confirmado", options=[0,1,2,3,4,5,6,7]),
                    "Volumen_transportado": st.column_config.SelectboxColumn("Volumen_transportado", options=[0,1,2,3,4,5,6,7]),
                    "Volumen_m3": st.column_config.SelectboxColumn("Volumen_m3", options=list(range(101))),
                    "Empresa": st.column_config.SelectboxColumn("Empresa", options=empresas),
                    "Turno": st.column_config.SelectboxColumn("Turno", options=turno),
                    "Codigo_producto": st.column_config.SelectboxColumn("Codigo_producto", options=codigos_producto),
                    "Familia": st.column_config.SelectboxColumn("Familia", options=["HORMIGÓN", "SHOTCRETE"]),
                    "Prioridad": st.column_config.SelectboxColumn("Prioridad", options=["ALTA", "MEDIA", "BAJA"]),
                    "Obra": st.column_config.SelectboxColumn("Obra", options=obras),
                    "Uso": st.column_config.SelectboxColumn("Uso", options=usos)
                    
                }, 
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic"
            )

            # Aplicar cambios a la planificación diaria sin duplicar filas
            planif_diaria.update(edited_df)
            planif_diaria = planif_diaria.drop_duplicates()
            planif_diaria['Fecha'] = pd.to_datetime(planif_diaria['Fecha'], errors='coerce', dayfirst=True).dt.strftime('%d-%m-%Y')

            # Obtener el rango de filas desde len(planif_diaria) hasta len(edited_df)
            pedidos_nuevos = edited_df.iloc[len(planif_diaria):len(edited_df)]
            pedidos_nuevos = pedidos_nuevos.dropna(subset=['Hora_requerida'])

            # Mostrar el "pedazo" extraído
            pedidos_nuevos_procesados = HD.procesar_tabla_editada(pedidos_nuevos)

            #st.dataframe(pedidos_nuevos_procesados)

            planif_diaria = pd.concat([planif_diaria, pedidos_nuevos_procesados], ignore_index=True)
            
            planif_diaria['Viaje_ID'] = range(1, len(planif_diaria) + 1)

            try:
                # Convertir a datetime y extraer solo el tiempo
                planif_diaria['Hora_requerida'] = pd.to_datetime(
                    planif_diaria['Hora_requerida'], format='%H:%M'
                ).dt.strftime('%H:%M')  # Convertir de vuelta a texto en formato 'HH:MM'

                planif_diaria['Hora_llamado'] = pd.to_datetime(
                    planif_diaria['Hora_llamado'], format='%H:%M'
                ).dt.strftime('%H:%M')  # Convertir de vuelta a texto en formato 'HH:MM'

                planif_diaria['Hora_retorno_proyectado'] = pd.to_datetime(
                    planif_diaria['Hora_retorno_proyectado'], format='%H:%M'
                ).dt.strftime('%H:%M')  # Convertir de vuelta a texto en formato 'HH:MM'

            except ValueError:
                st.error("Por favor, verifique el formato de las horas ingresadas (HH:MM).")
                st.stop()
  
            # Actualizar tiempos pre y post descarga por si fueron cambiados 
            
            planif_diaria['Tiempo_pre_descarga'] = planif_diaria['Llegada_obra_min']
            planif_diaria['Tiempo_post_descarga'] = (
                planif_diaria['Atencion_min'] + 
                planif_diaria['Retorno_min'] + 
                planif_diaria['Lavado_min']
            )
            planif_diaria['Total_vuelta_min'] = planif_diaria['Tiempo_pre_descarga'] + planif_diaria['Tiempo_post_descarga']

            # Procesar planificación iterada
            df_heur, df_iterado = HD.procesar_planificacion_iterada(planif_diaria)

            # Guardar df_heur en session_state
            st.session_state['df_heur'] = df_heur
            st.session_state['df_iterado'] = df_iterado

            
            # Mostrar los resultados
            st.subheader("Viajes confirmados")
            view_columns = [
                'Viaje_ID', 'Fecha', 'Turno', 'Pedido_ID','Empresa', 'Hora_llamado', 'Hora_requerida', 
                'Volumen_confirmado', 'Volumen_m3','Punto_entrega', 'Destino_final', 'Codigo_producto', 
                'Familia', 'Prioridad','Tiempo_pre_descarga','Tiempo_post_descarga',
                'Descripcion','Obra','Uso', 'Observaciones','Hora_retorno_proyectado'
            ]
            #st.dataframe(df_heur)
                        # Verificar si todas las columnas tienen solo valores nulos
            if df_heur.dropna(how="all").empty:
                st.write("⚠️ No hay viajes confirmados disponibles aún.")
                #st.dataframe(df_heur.fillna(""))  # Muestra la tabla pero con celdas vacías en lugar de None
            else:
                st.dataframe(df_heur[view_columns])


            col1,col2 = st.columns(2)
            with col1: 
                # Botón para descargar la tabla editada en formato Excel
                def to_excel(df):
                    df = df.drop_duplicates()
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Planificacion')
                    return output.getvalue()
                
                excel_data = to_excel(edited_df)
                st.download_button(
                    label="Descargar tabla para editar en Excel",
                    data=excel_data,
                    file_name="tabla_planificacion_editable.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )  
            # Entrada para parámetros de optimización
            with st.container():
                st.markdown("""
                <div style="background-color: #f9f9f9; padding: 15px 20px 10px 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <h4 style="color: #222; text-align: center; margin: 0;"> Parámetros para la optimización</h4>
                """, unsafe_allow_html=True)


                col1, col2 = st.columns(2)
                with col1:
                    time_limit = st.number_input("⏱️ Tiempo límite (seg)", min_value=10, value=120, step=10)
                    P = st.number_input("🏭 Plantas disponibles", min_value=1, value=2, step=1)
                with col2:
                    K = st.number_input("🚚 Camiones a utilizar", min_value=1, value=20, step=1)
                    hora_usuario = st.time_input("🕒 Hora de la ejecución", value=time(0, 0), step=60)

                st.markdown("</div>", unsafe_allow_html=True)  # Cierra el contenedor estilizado


            # Inicializar la variable global de optimización si no existe
            if 'optimizacion_pendiente' not in st.session_state:
                st.session_state['optimizacion_pendiente'] = False


            # Botón de ejecución
            ejecutar_optimizacion = st.button("🚀 Optimizar planificación")

            if ejecutar_optimizacion:
                if 'df_heur' not in st.session_state:
                    st.error("Por favor, confirme los pedidos antes de optimizar.")
                else:
                    # Convertir la hora seleccionada a string en formato "HH:MM"
                    tiempo_actual_str = hora_usuario.strftime("%H:%M")
                    try:
                        # Llamar a la función de optimización con parámetros ingresados
                        st.metric(label = "Hora de la ejecución", value = tiempo_actual_str)
                        status, model_h, df_resultados_camiones, solve_time = HD.optimizar_secuenciamiento_camiones_ortools(
                            st.session_state['df_heur'], tiempo_actual_str= tiempo_actual_str, time_limit=int(time_limit), plants=int(P), K=int(K)
                        )

                        if status == False: 
                            st.info("No fue posible encontrar una solucion factible con los datos ingresados, por favor revise horas y prioridades")
                        
                        if status == True:
                            # Crear el dataframe final filtrando volumen confirmado
                            df_final = HD.crear_dataframe_optimizado(st.session_state['df_iterado'], df_resultados_camiones)
                            df_final = df_final[df_final['Volumen_confirmado'] != 0]

                            # Mostrar el tiempo de resolución y el dataframe
                            st.metric(label="Tiempo de resolución (segundos)", value=round(solve_time, 2))
                            st.dataframe(df_final)

                            # Guardar df_final en session_state
                            st.session_state['df_final'] = df_final

                            # **Activar el flag global de optimización pendiente**
                            st.session_state['optimizacion_pendiente'] = True  

                            # Crear nuevo df_horarios_reales conservando horas reales si existen
                            df_nuevo = df_final.copy()
                            columnas_horas = ['Hora_Carga_R', 'Hora_Llegada_Planta_R', 'Hora_Retorno_R']

                            if 'df_horarios_reales' in st.session_state:
                                # Tomar columnas de horas reales previas
                                df_prev = st.session_state['df_horarios_reales'][['Pedido_ID', 'Viaje_ID'] + columnas_horas]
                                # Hacer merge para conservar horas reales donde coincidan Pedido_ID y Viaje_ID
                                df_nuevo = df_nuevo.merge(df_prev, on=['Pedido_ID', 'Viaje_ID'], how='left')
                            else:
                                # Inicializar las columnas si no existen datos previos
                                for col in columnas_horas:
                                    df_nuevo[col] = ""

                            # Guardar en session_state
                            st.session_state['df_horarios_reales'] = df_nuevo
                            
                            st.success("Optimización completada con éxito.")
                        #else:
                           # st.error("El modelo de optimización no encontró una solución válida.")
                    except Exception as e:
                        st.error(f"Error en la optimización: {str(e)}")

                    def to_excel(df):
                        df = df.drop_duplicates()
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False, sheet_name='Planificacion')
                        return output.getvalue()
                    
                    if 'df_final' in st.session_state: 
                        excel_data = to_excel(st.session_state['df_final'])
                        st.download_button(
                            label="Descargar resultados de la optimización",
                            data=excel_data,
                            file_name="resultados.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )  
    else:
            st.info("No hay datos disponibles. Completa el proceso en la pestaña anterior.")

# Pestaña 3: Usar df_final
with tab3:
    st.header("📆 Planificación Diaria de Pedidos")

    if 'df_final' in st.session_state:
        df_final = st.session_state['df_final']

        # Inicializar session_state si no existe
        if 'filtered_data' not in st.session_state:
            st.session_state['filtered_data'] = df_final.copy()

        filtered_data = st.session_state['filtered_data']

        # Crear subpestañas
        subtab1, subtab2, subtab3 = st.tabs([
            "📅 Planificación", 
            "🕒 Ingreso de Horas Reales", 
            "📊 Comparación - KPI's"
        ])

        with subtab1:
            st.subheader("Planificación de Pedidos")
            # Inicializar la variable global si no existe
            if 'optimizacion_pendiente' not in st.session_state:
                st.session_state['optimizacion_pendiente'] = False

            # Inicializar valores si no existen
            if 'filtered_data' not in st.session_state or 'cronograma_actualizado' not in st.session_state:
                st.session_state['filtered_data'] = df_final.copy()
                st.session_state['cronograma_actualizado'] = False  

            # **Si la optimización está pendiente, actualizar los datos**
            if st.session_state['optimizacion_pendiente']:
                st.session_state['filtered_data'] = df_final.copy()
                st.session_state['cronograma_actualizado'] = False  # Marcar como actualizado
                st.session_state['optimizacion_pendiente'] = False  # **Desactivar el flag**

            filtered_data = st.session_state['filtered_data']

            filtered_data1 = df_final.copy()

            col1, col2 = st.columns(2)

            with col1:
                primera_fecha = df_final['Fecha'].iloc[0]
                #primera_fecha_str = primera_fecha.strftime('%d-%m')
                st.markdown(f"### Planificación del día: {primera_fecha}")
            
            # Tarjetas informativas más pequeñas
            col1, col2, col3 = st.columns(3)

            # Estilos CSS con bordes azul sutil
            st.markdown("""
                <style>
                    .info-card {
                        text-align: center;
                        background-color: #ffffff;
                        border: 2px solid #a3c6f1; /* Borde azul claro */
                        border-radius: 12px;
                        padding: 15px;
                        margin: 5px 0;
                        width: 100%;
                        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.05); /* Sombra sutil */
                        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
                    }
                    .info-card:hover {
                        transform: scale(1.05);
                        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.10); /* Efecto hover */
                    }
                    .info-title {
                        font-size: 18px;
                        font-weight: normal; /* Mantiene la fuente original */
                        color: #34495e;
                        margin-bottom: 5px;
                    }
                    .info-value {
                        font-size: 24px;
                        font-weight: normal; /* Mantiene la fuente original */
                        color: #2c3e50;
                    }
                </style>
            """, unsafe_allow_html=True)

            with col1:
                camiones_utilizados = filtered_data['Camion_ID'].nunique()
                st.markdown(
                    f"""
                    <div class='info-card'>
                        <div class='info-title'>Cantidad de camiones a utilizar</div>
                        <div class='info-value'>{camiones_utilizados}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                cantidad_viajes = filtered_data.shape[0]
                st.markdown(
                    f"""
                    <div class='info-card'>
                        <div class='info-title'>Cantidad de viajes</div>
                        <div class='info-value'>{cantidad_viajes}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                demanda_total = filtered_data['Volumen_confirmado'].sum()
                st.markdown(
                    f"""
                    <div class='info-card'>
                        <div class='info-title'>Demanda de m³</div>
                        <div class='info-value'>{demanda_total}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # Aplicar el cálculo de horas, usando la fecha como referencia
            filtered_data = f.calcular_horass(filtered_data, primera_fecha)
            filtered_data1 = f.calcular_horass(filtered_data1, primera_fecha)

            col1,col2, col3= st.columns([1, 1,1])
            with col1: 
                # Filtro clientes
                clientes_disponibles = df_final['Empresa'].unique()
                cliente_seleccionado = st.selectbox("Filtrar por Empresa", options=['Todas'] + list(clientes_disponibles))
                # Filtrar datos
                if cliente_seleccionado != 'Todas':
                    filtered_data = filtered_data[filtered_data['Empresa'] == cliente_seleccionado]
            with col2: 
                plantas_disponibles = filtered_data['Planta_salida'].unique()
                planta_seleccionada = st.selectbox("Filtrar por Planta", options=['Todas'] + list(plantas_disponibles))
                if planta_seleccionada != 'Todas':
                    filtered_data = filtered_data[filtered_data['Planta_salida'] == planta_seleccionada]

            # Graficar el cronograma correcto según el estado
            st.subheader("Cronograma de Secuenciamiento de Camiones")
            if st.session_state['cronograma_actualizado']:
                col1,col2 = st.columns([6,0.5])
                with col1:
                    f.graficar_cronograma_act(filtered_data)  # Usar la función del cronograma actualizado
                with col2:
                    f.metricas(filtered_data)
            else:
                col1,col2 = st.columns([6,0.5])
                with col1:
                    f.graficar_cronograma3(filtered_data)  # Usar la función del cronograma actualizado
                with col2:
                    f.metricas(filtered_data)


            # Convertir las columnas de horas a formato "HH:MM"
            filtered_data = f.formato_h_m(filtered_data)

            col1, col2 = st.columns([1.5, 2])  
            with col1:
                st.subheader("Asignación de mixers y conductores")
                #Opciones par tabla mixer-conductor
                N = 21
                if mixer_df is not None:
                    MIXERS = mixer_df['MIXER'].unique().tolist()
                if conductores_df is not None:
                    CONDUCTORES = conductores_df['CONDUCTOR'].tolist()
                # Crear DataFrame con el número de camiones seleccionado
                df_mixer = pd.DataFrame({
                    "Camion_ID": list(range(N)),  # ID de camión consecutivo
                    "MIXER": [""] * N,  # Valor por defecto en MIXER
                    "CONDUCTOR": [""] * N  # Valor por defecto en CONDUCTOR
                })

                # Usar st.data_editor para edición interactiva
                edited_df_mixer = st.data_editor(
                    df_mixer,
                    column_config={
                        "MIXER": st.column_config.SelectboxColumn("MIXER", options=MIXERS),
                        "CONDUCTOR": st.column_config.SelectboxColumn("CONDUCTOR", options=CONDUCTORES)
                    },
                    use_container_width=True
                )

                df_mixer.update(edited_df_mixer)

                # Verificar duplicados (ignorando valores vacíos)
                mixers_validos = df_mixer["MIXER"].replace("", None).dropna()
                conductores_validos = df_mixer["CONDUCTOR"].replace("", None).dropna()

                # Identificar los valores duplicados y cuántas veces se repiten
                mixer_duplicados = mixers_validos[mixers_validos.duplicated(keep=False)]
                conductor_duplicados = conductores_validos[conductores_validos.duplicated(keep=False)]

                mixer_repetido = not mixer_duplicados.empty  # Variable de control

                # Mensaje de advertencia solo para mixers repetidos
                if mixer_repetido:
                    mixers_contados = mixer_duplicados.value_counts().to_dict()
                    mixer_mensaje = ", ".join([f"{mixer}: {cantidad} veces" for mixer, cantidad in mixers_contados.items()])
                    st.error(f"🚨 Mixers repetidos: {mixer_mensaje}")

                # Botón de actualización (se deshabilita si hay mixers repetidos)
                actualizar_disabled = mixer_repetido  # Se desactiva si hay mixers repetidos

                if not conductor_duplicados.empty:
                    conductores_contados = conductor_duplicados.value_counts().to_dict()
                    conductor_mensaje = ", ".join([f"{conductor}: {cantidad} veces" for conductor, cantidad in conductores_contados.items()])
                    st.error(f"🚨 Conductores repetidos: {conductor_mensaje}")

                if st.button("Actualizar datos", disabled=actualizar_disabled):
                    if df_mixer["MIXER"].isnull().all():
                        st.warning("⚠️ Debes asignar al menos un mixer antes de actualizar el cronograma.")
                    else:
                        # Actualizar `filtered_data`
                        filtered_data_actualizado = filtered_data.copy()

                        # Verificar si las columnas existen antes de eliminarlas
                        columnas_a_eliminar = ['MIXER', 'CONDUCTOR']
                        filtered_data_actualizado = filtered_data_actualizado.drop(columns=[col for col in columnas_a_eliminar if col in filtered_data_actualizado.columns])

                        if 'Camion_ID' in df_mixer.columns and 'Camion_ID' in filtered_data_actualizado.columns:
                            filtered_data_actualizado = filtered_data_actualizado.merge(df_mixer[['Camion_ID', 'MIXER', 'CONDUCTOR']], 
                                                                                        on='Camion_ID', how='left')

                        # Guardar en session_state
                        st.session_state['filtered_data'] = filtered_data_actualizado
                        st.session_state['cronograma_actualizado'] = True  

                        # También actualizar df_horarios_reales
                        if 'df_horarios_reales' in st.session_state:
                            df_horarios_reales_actualizado = st.session_state['df_horarios_reales'].copy()

                            # Eliminar las columnas antiguas si existen
                            df_horarios_reales_actualizado = df_horarios_reales_actualizado.drop(columns=[col for col in columnas_a_eliminar if col in df_horarios_reales_actualizado.columns])

                            # Hacer el merge con los nuevos datos de MIXER y CONDUCTOR
                            df_horarios_reales_actualizado = df_horarios_reales_actualizado.merge(df_mixer[['Camion_ID', 'MIXER', 'CONDUCTOR']], 
                                                                                                on='Camion_ID', how='left')

                            # Guardar la actualización en session_state
                            st.session_state['df_horarios_reales'] = df_horarios_reales_actualizado

                        st.success("Página actualizada con los nuevos datos ✅")
                        st.rerun()

            with col2: 
                # Calcular la tabla de camiones
                filtered_data1 = f.formato_h_m(filtered_data1)
                if not filtered_data1.empty and 'Turno' in filtered_data1.columns:
                    tabla_camiones = f.generar_tabla_camiones(filtered_data1)
                    st.markdown("""<h3 style='font-size:20px; text-align: center;'>Estado de Camiones por Hora</h3>""", unsafe_allow_html=True)
                    f.graficar_transito(tabla_camiones)
                else:
                    st.warning("No hay datos suficientes para generar este grafico.")
                            #filtered_data = f.convertir_horas_datetime(filtered_data,primera_fecha)

            col1,col2,col3 = st.columns([1, 1,1])
            with col1: 
                # Obtener y ordenar los camiones
                camiones = sorted(filtered_data['Camion_ID'].astype(int).unique())  # Ordenar los valores únicos

                # Crear el filtro ordenado en Streamlit
                camion_seleccionado = st.selectbox("Filtrar por Camion", options=['Todos'] + list(camiones))

                # Filtrar datos si se selecciona un camión específico
                if camion_seleccionado != 'Todos':
                    filtered_data = filtered_data[filtered_data['Camion_ID'] == camion_seleccionado]
                    
            with col2: 
                # Obtener y ordenar los pedidos
                pedidos = sorted(filtered_data['Pedido_ID'].astype(int).unique())  # Ordenar los valores únicos

                # Crear el filtro ordenado en Streamlit
                pedido_seleccionado = st.selectbox("Filtrar por Pedido", options=['Todos'] + list(pedidos))

                # Filtrar datos si se selecciona un pedido específico
                if pedido_seleccionado != 'Todos':
                    filtered_data = filtered_data[filtered_data['Pedido_ID'] == pedido_seleccionado]

            if not st.session_state['cronograma_actualizado']: 
                columnas_mostrar = [
                    "Pedido_ID","Viaje_ID",'Camion_ID', "Empresa", "Codigo_producto",
                    "Familia", "Volumen_confirmado", "Punto_entrega", "Destino_final",
                    "Prioridad",'ru', 'rl', "Planta_salida","HORA REQUERIDA" ,"INICIO CARGA", "LLEGADA OBRA", "RETORNO PLANTA","Hora_fin_lavado"
                ]
            else: 
                columnas_mostrar = [
                    "Pedido_ID","Viaje_ID",'Camion_ID', "MIXER", "CONDUCTOR", "Empresa", "Codigo_producto",
                    "Familia", "Volumen_confirmado", "Punto_entrega", "Destino_final",
                    "Prioridad",'ru', 'rl', "Planta_salida","HORA REQUERIDA" ,"INICIO CARGA", "LLEGADA OBRA", "RETORNO PLANTA","Hora_fin_lavado"
                ]


            st.dataframe(filtered_data[columnas_mostrar])

        with subtab2:
            st.subheader("Ingreso de Horarios Reales")

            # Copiar df_final y agregar columnas nuevas en blanco
            if 'df_horarios_reales' not in st.session_state:
                df_horarios_reales = df_final.copy()
                df_horarios_reales['Hora_Carga_R'] = ""
                df_horarios_reales['Hora_Llegada_Planta_R'] = ""
                df_horarios_reales['Hora_Retorno_R'] = ""
                st.session_state['df_horarios_reales'] = df_horarios_reales
            else:
                df_horarios_reales = st.session_state['df_horarios_reales']

            if st.session_state['cronograma_actualizado'] == True and 'df_horarios_reales' in st.session_state:
                # Mostrar el DataFrame editable
                col_mostrar = [
                        "Pedido_ID","Viaje_ID",'Camion_ID', "MIXER", "CONDUCTOR", "Empresa", "Codigo_producto",
                        "Hora_carga", "Hora_llegada_obra_var", "Hora_retorno", "Hora_Carga_R","Hora_Llegada_Planta_R", "Hora_Retorno_R"
                        ,"Volumen_confirmado", "Punto_entrega", "Destino_final", "Prioridad"
                    ]
                st.info('Deslize la tabla hacia la derecha para completar los campos de horarios reales y confirme con el botón **Guardar cambios**')

                df_editado = st.data_editor(
                    df_horarios_reales[col_mostrar],
                    use_container_width=True)

                # Guardar cambios al actualizar
                if st.button("Guardar cambios"):
                    st.session_state['df_horarios_reales'].update(df_editado)
                    st.success("Horarios reales actualizados ✅")
                    st.info("En la siguiente subpestaña encontrara sus metricas de desempeño📈")

                def to_excel(df):
                    df = df.drop_duplicates()
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Hoja1')
                    return output.getvalue()
                        
                if 'df_horarios_reales' in st.session_state: 
                    excel_data = to_excel(st.session_state['df_horarios_reales'])
                    st.download_button(
                        label="Descargar en excel",
                        data=excel_data,
                        file_name="resultadoss.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ) 
            else: 
                st.info("No hay datos disponibles de los Mixer y Conductores. Complete el proceso en la subpestaña anterior.")


        # Agregar los KPIs en la subpestaña
        with subtab3:
            st.subheader("📊 Panel de Análisis: Planificado VS Ejecutado")
            if st.session_state['cronograma_actualizado'] == True and 'df_horarios_reales' in st.session_state:
                f.mostrar_kpis(st.session_state['df_horarios_reales'])
                f.graficar_cronograma_comparacion(st.session_state['df_horarios_reales'])
                f.mostrar_kpis_individuales(st.session_state['df_horarios_reales'])
            else:
                st.info("No hay datos disponibles. Complete el proceso en la subpestaña anterior.")

    else:
            st.info("No hay datos disponibles. Complete el proceso en la pestaña anterior.")

#EJECUTAR MODELO DE OPTIMIZACION  Y VISUALIZAR DATOS
# Pestaña 4: Optimización de Planificación
with tab4:
    st.header("🎯 Optimización de Planificación")
    if planif_semanal is not None:
        # Filtros en dos columnas
        col1, col2 = st.columns(2)
        with col1:
            fecha_seleccionada = st.selectbox(
                "Seleccione una fecha para la optimización",
                planif_semanal['Fecha'].dt.date.unique(),
                key="fecha_tab4"
            )
        with col2:
            turno_seleccionado = st.selectbox(
                "Seleccione un turno para la optimización",
                planif_semanal['Turno'].unique(),
                key="turno_tab4"
            )

        if fecha_seleccionada and turno_seleccionado:
            # Filtrar los datos según la fecha y turno seleccionados
            data_filtrada = planif_semanal[
                (planif_semanal['Fecha'].dt.date == fecha_seleccionada)
                & (planif_semanal['Turno'] == turno_seleccionado)
            ]
            col1, col2, col3 = st.columns(3)
            with col1:
                PP = st.number_input("Número de plantas disponibles:", min_value=1, value=2, step=1, key="PP")
            with col2:
                T = st.number_input("Límite de tiempo (en segundos):", min_value=60, value=240, step=10, key="T")

            if data_filtrada.empty:
                st.warning("No hay datos disponibles para la fecha y el turno seleccionados.")
            else:
                st.info(f"**Parámetros seleccionados para optimización:** {fecha_seleccionada} - {turno_seleccionado} - {PP} planta(s)")

                # Botón para ejecutar la optimización
                if st.button("Ejecutar Optimización"):
                    try:
                        # Ejecutar la planificación y obtener el resultado
                        # Ejecutar la planificación y obtener el resultado
                        resultado, time2, referencia_fecha = mps.ejecutar_proceso_planificacion(
                            file_path=".",
                            file_name="temp_planificacion.xlsx",
                            df_tiempos= tiempos_aux,
                            df_familia= productos_aux,
                            df_prioridad= prioridades_aux,
                            fechas=[fecha_seleccionada],
                            turnos=[turno_seleccionado],
                            plants=PP,
                            time=T
                        )
                        st.metric(label="Tiempo de resolución (segundos)", value=round(time2, 2))

                        # Guardar resultado en session_state
                        st.session_state['resultado_optimizacion'] = resultado
                        st.session_state['referencia_fecha'] = referencia_fecha

                    except Exception as e:
                        st.error(f"Error durante la optimización {e}")

        # Si hay un resultado en session_state, aplicar filtrado sin reiniciar
        if 'resultado_optimizacion' in st.session_state:
            resultado = st.session_state['resultado_optimizacion']
            referencia_fecha = st.session_state['referencia_fecha']

            # Aplicar filtrado por Planta de Salida
            plantas_disponibles = resultado['Planta_salida'].unique()
            planta_seleccionada = st.selectbox(
                "Filtrar por Planta", 
                options=['Todas'] + list(plantas_disponibles),
                key="planta_tab4"
            )
            if planta_seleccionada != 'Todas':
                resultado = resultado[resultado['Planta_salida'] == planta_seleccionada]

            # Generar cronograma
            resultado = f.calcular_horas_clave(resultado, referencia_fecha)
            f.graficar_cronograma4(resultado)

            resultado = resultado.copy()

            # Formatear resultados
            resultado.loc[:, 'Fecha'] = pd.to_datetime(resultado['Fecha'], dayfirst=True, errors='coerce').dt.strftime('%d-%m-%Y')
            resultado.loc[:, "INICIO CARGA"] = pd.to_datetime(resultado["INICIO CARGA"], errors='coerce').dt.strftime('%H:%M')
            resultado.loc[:, "LLEGADA OBRA"] = pd.to_datetime(resultado["LLEGADA OBRA"], errors='coerce').dt.strftime('%H:%M')
            resultado.loc[:, "RETORNO PLANTA"] = pd.to_datetime(resultado["RETORNO PLANTA"], errors='coerce').dt.strftime('%H:%M')


            st.subheader("Resultados del Modelo de Optimización")
            st.dataframe(resultado)

            # Descargar los resultados como archivo Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                resultado.to_excel(writer, index=False, sheet_name="Resultados")
            st.download_button(
                label="Descargar Resultados",
                data=output.getvalue(),
                file_name="Resultados_Planificación.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Por favor, suba el archivo correspondiente en la pestaña 'Programa Semanal' antes de continuar.")

