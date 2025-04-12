import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
import geopandas as gpd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output




# ======================================
# FUNCIONES AUXILIARES
# ======================================
def limpiar_coordenadas(df):
    """Limpia y convierte coordenadas de manera vectorizada"""
    for col in ['LATITUD', 'LONGITUD']:
        df[col] = (
            df[col].astype(str)
            .str.replace(r'[^\d.-]', '', regex=True)
            .str.replace(',', '.')
            .replace('', np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=["LATITUD", "LONGITUD"])

def crear_grafico_barras(
        df: pd.DataFrame, 
        columna: str, 
        titulo : str, 
        color: str ="#004aad",
        datos_agregados: bool =False,
        columna_categoria: list =None,
        tickangle: int = 0,
        columna_valor: list =None):

    if datos_agregados:
        # Validar parámetros para datos precalculados
        if not columna_categoria or not columna_valor:
            raise ValueError("Se requieren columna_categoria y columna_valor para datos agregados")
            
        # Usar datos directamente sin value_counts()
        conteo = df[[columna_categoria, columna_valor]].copy()
        total = conteo[columna_valor].sum()
        conteo['Porcentaje'] = (conteo[columna_valor] / total * 100).round(1)
        
        categoria = columna_categoria
        valor = columna_valor
        
    else:
        # Procesamiento original para datos crudos
        conteo = df[columna].value_counts().reset_index()
        conteo.columns = [columna, 'Cantidad']
        total = conteo['Cantidad'].sum()
        conteo['Porcentaje'] = (conteo['Cantidad'] / total * 100).round(1)
        
        categoria = columna
        valor = 'Cantidad'

    # Creación del gráfico (compartida para ambos casos)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=conteo[categoria],
        y=conteo[valor],
        marker_color=color,
        text=[f"{c} ({p}%)" for c, p in zip(conteo[valor], conteo['Porcentaje'])],
        textposition="outside",
        hovertemplate=(
            f"<b>{categoria}:</b> %{{x}}<br>"
            f"<b>{valor}:</b> %{{y}}<br>"
            "<b>Porcentaje:</b> %{customdata}%<extra></extra>"
        ),
        customdata=conteo['Porcentaje']
    ))
    
    # Configuración común del layout
    fig.update_layout(
        title={'text': titulo, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title=categoria,
        yaxis_title=valor if datos_agregados else "Cantidad de Siniestros",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        height=600,
        width=None,   # Para que tome el 100% del contenedor
        margin=dict(autoexpand=True)
    )
    
    fig.update_xaxes(tickangle=tickangle, showgrid=False, tickmode = 'linear')
    fig.update_yaxes(showgrid=True, gridcolor="white")
    
    return fig
    
def crear_grafico_apilado(df, grupo, stack, titulo, colores):
    """Crea gráfico de barras apiladas estandarizado"""
    conteo = pd.crosstab(df[grupo], df[stack])
    conteo = conteo.loc[:, conteo.sum().sort_values(ascending=False).index]
    
    fig = go.Figure()
    
    for i, col in enumerate(conteo.columns):
        fig.add_trace(go.Bar(
            x=conteo.index,
            y=conteo[col],
            name=col,
            marker_color=colores.get(col, "#808080"),
            text=conteo[col],
            textposition="inside",
            texttemplate='%{text:.0f}',
            textfont=dict(
                size=14,  # Tamaño base más grande
                color='white',  # Color del texto
                family='Arial',
            ),
            hovertemplate=(
                f"<b>{grupo}:</b> %{{x}}<br>"
                f"<b>{stack}:</b> {col}<br>"
                "<b>Cantidad:</b> %{y}<extra></extra>"
            ),
            marker_line=dict(
                color='white',
                width=1.5
            )
        ))
    
    fig.update_layout(
        title={'text': titulo, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title=grupo,
        yaxis_title="Cantidad",
        barmode="stack",
        legend_title=stack,
        height=600,
        width=None,   # Para que tome el 100% del contenedor
        margin=dict(autoexpand=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        uniformtext=dict(
            mode='hide',  # Oculta texto que no cabe
            minsize=10,   # Tamaño mínimo para mostrar
        ),
        bargap=0.2,
        bargroupgap=0.1
    )
    
    # Ajustes finales de ejes
    fig.update_xaxes(
        tickangle=-45,
        showgrid=False,
        tickfont=dict(size=12)
    )
    
    fig.update_yaxes(
        range=[0, conteo.sum(axis=1).max() * 1.1],
        showgrid=True,
        gridcolor="white"
    )
    
    return fig

def crear_grafico_barras_horizontal(df, col_x, col_y, titulo, color="#004aad", mostrar_porcentaje=False):
    # Ordenar datos
    df = df.sort_values(col_x, ascending=True)
    
    # Calcular porcentajes si es necesario
    total = df[col_x].sum() if mostrar_porcentaje else None
    porcentajes = (df[col_x] / total * 100).round(1) if total else None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df[col_x],
        y=df[col_y],
        orientation='h',
        marker_color=color,
        text=[f"{x}" for x in df[col_x]],
        textposition='outside',
        hovertemplate=(
            f"<b>%{{y}}:</b> %{{x}}" +
            (f"<br><b>Porcentaje:</b> %{{customdata}}%" if mostrar_porcentaje else "") +
            "<extra></extra>"
        ),
        customdata=porcentajes if mostrar_porcentaje else None
    ))
    
    fig.update_layout(
        title={
            'text': titulo,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Cantidad",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        height=600,
        width=None,   # Para que tome el 100% del contenedor
        margin=dict(autoexpand=True),  # Altura dinámica según cantidad de categorías
        xaxis=dict(
            showgrid=True,
            gridcolor="white",
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12)
        ),
        uniformtext=dict(
            minsize=10,
            mode='hide'
        )
    )
    
    return fig

def crear_grafico_comparativo(df: pd.DataFrame,
    col_categoria: str,
    columnas_comparar: list,
    titulo: str,
    colores: list = None,
    nombres_series: list = None,
    orden_categorias: list = None,
    modo_barras: str = "group",
    tickangle: int = 0,
    modo_total: bool = False) -> go.Figure:

     # Validaciones básicas
    if len(columnas_comparar) < 1:
        raise ValueError("Debe especificar al menos una columna para comparar")
        
    if colores and (len(colores) != len(columnas_comparar)):
        raise ValueError("La lista de colores debe coincidir con el número de columnas a comparar")
        
    if nombres_series and (len(nombres_series) != len(columnas_comparar)):
        raise ValueError("Los nombres de serie deben coincidir con el número de columnas")

    if modo_total:
        fig = go.Figure()
        
        # Calcular máximos para ajustar el rango del eje X
        max_valor = max(df[col].sum() for col in columnas_comparar) * 1.2  # +20% de espacio
        
        for idx, (nombre, col) in enumerate(zip(nombres_series, columnas_comparar)):
            fig.add_trace(go.Bar(
                y=[nombre],
                x=[df[col].sum()],
                name=nombre,
                marker_color=colores[idx],
                orientation='h',
                text=[f"{df[col].sum()}"],
                textposition="outside",
                hovertemplate=(
                    f"<b>{nombre}:</b> %{{x}}<br>"
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title={'text': titulo, 'x': 0.05, 'xanchor': 'left'},  # Título alineado a la izquierda
            xaxis_title="Cantidad Total",
            yaxis_title=" ",
            showlegend=False,
            height=600,
            width=None,   # Para que tome el 100% del contenedor
            margin=dict(autoexpand=True),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                range=[0, max_valor],  # Fijar rango desde 0
                showgrid=False
            ),
            yaxis=dict(
                automargin=True,  # Ajuste automático del margen
                tickfont=dict(size=14)
            )
        )
        return fig

    # Lógica original para gráficos comparativos
    if orden_categorias:
        df[col_categoria] = pd.Categorical(df[col_categoria], categories=orden_categorias, ordered=True)
        df = df.sort_values(col_categoria)
    elif col_categoria == "MES":
        meses_orden = ["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
                      "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"]
        df[col_categoria] = pd.Categorical(df[col_categoria], categories=meses_orden, ordered=True)
        df = df.sort_values(col_categoria)

    colores = colores or px.colors.qualitative.Plotly[:len(columnas_comparar)]
    nombres_series = nombres_series or columnas_comparar

    totales = df[columnas_comparar].sum(axis=1).replace(0, 1)
    porcentajes = (df[columnas_comparar].div(totales, axis=0) * 100).round(1)

    fig = go.Figure()
    
    for idx, col in enumerate(columnas_comparar):
        fig.add_trace(go.Bar(
            x=df[col_categoria],
            y=df[col],
            name=nombres_series[idx],
            marker_color=colores[idx],
            text=df[col],
            textposition="outside",
            customdata=porcentajes[col],
            hovertemplate=(
                f"<b>{col_categoria.title()}:</b> %{{x}}<br>"
                f"<b>{nombres_series[idx]}:</b> %{{y}}<br>"
                "<b>Porcentaje del Total:</b> %{customdata}%<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title={'text': titulo, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title=col_categoria.title(),
        yaxis_title="Cantidad",
        barmode=modo_barras,
        legend_title=" ",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        height=600,
        width=None,   # Para que tome el 100% del contenedor
        margin=dict(autoexpand=True),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )

    fig.update_xaxes(tickangle=tickangle, showgrid=False, tickmode='linear')
    fig.update_yaxes(showgrid=True, gridcolor="white", zeroline=False)
    
    return fig

def crear_histograma(
    df: pd.DataFrame, 
    columna_numerica: str, 
    titulo: str, 
    color: str = "#004aad",
    auto_bins: bool = True, 
    num_bins: int = None,
    columna_categoria: str = None,
    subcategoria: str = None,
    tickangle: int = 0
) -> go.Figure:

# Verificar si la columna numérica existe en el DataFrame
    if columna_numerica not in df.columns:
        raise ValueError(f"La columna '{columna_numerica}' no existe en el DataFrame.")
    
    # Normalizar la columna categórica si se especifica
    if columna_categoria is not None:
        if columna_categoria not in df.columns:
            raise ValueError(f"La columna '{columna_categoria}' no existe en el DataFrame.")
        df[columna_categoria] = df[columna_categoria].str.strip().str.lower()
        if subcategoria is not None:
            subcategoria = subcategoria.strip().lower()

    # Filtrar los datos según la categoría y subcategoría (si se especifican)
    if columna_categoria is not None:
        if subcategoria is not None:
            # Filtrar por subcategoría específica
            df_filtrado = df[df[columna_categoria] == subcategoria]
            if df_filtrado.empty:
                raise ValueError(f"No hay datos para la subcategoría '{subcategoria}' en la columna '{columna_categoria}'.")
        else:
            # Usar todos los datos de la categoría
            df_filtrado = df
    else:
        # No se especificó ninguna categoría, usar todo el DataFrame
        df_filtrado = df

    # Crear una copia explícita del DataFrame filtrado para evitar advertencias
    df_filtrado = df_filtrado.copy()

    # Convertir la columna numérica a valores numéricos, forzando valores no numéricos a NaN
    df_filtrado.loc[:, columna_numerica] = pd.to_numeric(df_filtrado[columna_numerica], errors='coerce')

    # Eliminar valores NaN o vacíos en la columna numérica
    df_filtrado = df_filtrado.dropna(subset=[columna_numerica])

    # Validación de datos
    if df_filtrado.empty:
        raise ValueError(f"La columna '{columna_numerica}' no contiene valores numéricos válidos después del filtrado.")

    # Calcular número de bins usando Friedman-Diaconis si auto_bins=True
    if auto_bins:
        q75, q25 = np.percentile(df_filtrado[columna_numerica], [75, 25])
        iqr = q75 - q25
        n = len(df_filtrado[columna_numerica])
        bin_width = 2 * iqr * (n ** (-1/3))
        min_val, max_val = df_filtrado[columna_numerica].min(), df_filtrado[columna_numerica].max()
        if bin_width <= 0:
            bin_width = 1
        num_bins = max(1, int(np.ceil((max_val - min_val) / bin_width)))
    else:
        num_bins = num_bins or 30  # Valor por defecto si no se especifica

    # Calcular bordes de los bins
    min_val = np.floor(df_filtrado[columna_numerica].min())
    max_val = np.ceil(df_filtrado[columna_numerica].max())
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Crear etiquetas de intervalos con valores enteros
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]

    # Crear el histograma
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_filtrado[columna_numerica],
        xbins=dict(
            start=min_val,
            end=max_val,
            size=(max_val - min_val) / num_bins
        ),
        marker_color=color,
        opacity=0.8,
        hovertemplate=(
            f"<b>{columna_numerica}:</b> %{{x}}<br>"
            "<b>Frecuencia:</b> %{y}<br>"
            "<extra></extra>"
        )
    ))

    # Configurar etiquetas de intervalos en el eje x
    fig.update_xaxes(
        tickvals=[(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)],
        ticktext=bin_labels,
        tickangle=-45,
        showgrid=True,
        gridcolor="White"
    )

    # Configuración del layout
    fig.update_layout(
        title={'text': titulo, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title=columna_numerica,
        yaxis_title="Frecuencia",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        height=600,
        width=None,   # Para que tome el 100% del contenedor
        margin=dict(autoexpand=True),
        bargap=0.1
    )

    # Ajustes finales del eje y
    fig.update_xaxes(tickangle=tickangle)
    fig.update_yaxes(showgrid=True, gridcolor="White")

    return fig

def crear_boxplot(df: pd.DataFrame, columna_num: str, columna_cat: str, titulo: str, color: str = None) -> go.Figure:
    
    # Validación de datos
    if columna_num not in df.columns or columna_cat not in df.columns:
        raise ValueError(f"Las columnas '{columna_num}' o '{columna_cat}' no existen en el DataFrame.")
    
    # Verificar si la columna numérica es de tipo numérico
    if not pd.api.types.is_numeric_dtype(df[columna_num]):
        raise ValueError(f"La columna '{columna_num}' no es numérica.")
    
    # Crear el boxplot
    fig = px.box(
        df,
        x=columna_cat,
        y=columna_num,
        color=columna_cat if color is None else None,
        color_discrete_sequence=[color] if color else None,
        title=titulo
    )
    
    # Configuración del layout
    fig.update_layout(
        title={'text': titulo, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title=columna_cat,
        yaxis_title=columna_num,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        height=600,
        width=None,   # Para que tome el 100% del contenedor
        margin=dict(autoexpand=True)
    )
    
    # Ajustes finales de ejes
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")
    
    return fig   

def grafico_comparativo_sencillo(df: pd.DataFrame, col_categoria: str, col_valor: str, titulo: str, colores: list = None, nombres_series: list = None, tickangle: int=0):
    # Validaciones básicas
    if col_categoria not in df.columns or col_valor not in df.columns:
        raise ValueError(f"Las columnas '{col_categoria}' y '{col_valor}' deben estar presentes en el DataFrame.")
    
    # Asignar colores predeterminados si no se especifican
    colores = colores or px.colors.qualitative.Plotly[:len(df)]
    
    # Asignar nombres legibles si no se especifican
    nombres_series = nombres_series or df[col_categoria].tolist()
    
    # Crear la figura
    fig = go.Figure()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        fig.add_trace(go.Bar(
            x=[row[col_categoria]],  # Categoría en el eje x
            y=[row[col_valor]],      # Valor en el eje y
            name=nombres_series[idx],  # Nombre de la serie
            marker_color=colores[idx],  # Color de la barra
            text=[f"{row[col_valor]}"],  # Texto sobre la barra
            textposition="outside",      # Posición del texto
            hovertemplate=(
                f"<b>{col_categoria}:</b> {row[col_categoria]}<br>"
                f"<b>{col_valor}:</b> {row[col_valor]}<br>"
                "<extra></extra>"
            )
        ))
    
    # Configurar el diseño del gráfico
    fig.update_layout(
        title={'text': titulo, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title=col_categoria.title(),
        yaxis_title="Cantidad",
        barmode="group",
        legend_title=" ",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        margin=dict(autoexpand=True),
        height=600,
        width=None,
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    fig.update_xaxes(tickangle=tickangle, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="white", zeroline=False)
    
    return fig

def crear_sunburst_chart(
    df: pd.DataFrame, 
    path: list, 
    values: str, 
    titulo: str = "Sunburst Chart", 
    color: str = None,
    color_continuous_scale: str = "Blues",
) -> px.sunburst:
    # Verificar si las columnas en 'path' existen en el DataFrame
    for col in path:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame.")
    
    # Verificar si la columna 'values' existe
    if values not in df.columns:
        raise ValueError(f"La columna '{values}' no existe en el DataFrame.")
    
    # Convertir la columna 'values' a valores numéricos, forzando valores no numéricos a NaN
    df.loc[:, values] = pd.to_numeric(df[values], errors='coerce')
    
    # Eliminar valores NaN o vacíos en la columna 'values'
    df = df.dropna(subset=[values])
    if df.empty:
        raise ValueError(f"La columna '{values}' no contiene valores numéricos válidos después del filtrado.")

    # Crear el gráfico Sunburst
    fig = px.sunburst(
        df,
        path=path,  # Jerarquía del gráfico
        values=values,  # Valores para el tamaño de los segmentos
        title=titulo,  # Título del gráfico
        color=color or values,  # Colorear según la columna especificada o 'values'
        color_continuous_scale=color_continuous_scale,  # Escala de colores
    )

    # Personalizar diseño
    fig.update_layout(
        height=600,  # Coincide con GRAPH_STYLE
        width=None,   # Para que tome el 100% del contenedor
        margin=dict(autoexpand=True)
    )

    return fig



def cargar_datos():
    # Cargar datos de ambas hojas
    df_vehiculos = pd.read_excel(r"BD_Dashboard.xlsx", sheet_name="Vehiculos_Desagregados")
    df_siniestros = pd.read_excel(r"BD_Dashboard.xlsx", sheet_name="Caracterización del Siniestro")
    df_actores = pd.read_excel(r"BD_Dashboard.xlsx", sheet_name="Caracterización Actores Viales")
    df_poblacion = pd.read_excel(r"BD_Dashboard.xlsx", sheet_name="Demografía Involucrados")
    df_geografia = pd.read_excel(r"BD_Dashboard.xlsx", sheet_name="Datos Geográficos y Temporales")

    # Cargar shapefile una vez
    comunas = gpd.read_file("Comunas/COMUNAS_UNIDAS.shp").to_crs(epsg=4326)
    
    # Limpiar coordenadas en ambos DataFrames (si aplica)
    df_vehiculos = limpiar_coordenadas(df_vehiculos)
    df_siniestros = limpiar_coordenadas(df_siniestros)
    df_actores = limpiar_coordenadas(df_actores)
    df_poblacion = limpiar_coordenadas(df_poblacion)
    df_geografia = limpiar_coordenadas(df_geografia)

    
    # Alinear CRS (usar el DataFrame principal para geometría)
    gdf_puntos = gpd.GeoDataFrame(
        df_vehiculos,  # o df_siniestros según necesidad
        geometry=gpd.points_from_xy(df_vehiculos["LONGITUD"], df_vehiculos["LATITUD"]),
        crs="EPSG:4326"
    )
    
    if comunas.crs != gdf_puntos.crs:
        gdf_puntos = gdf_puntos.to_crs(comunas.crs)
    
    return {
        'vehiculos': df_vehiculos,
        'siniestros': df_siniestros,
        'actores': df_actores,
        'poblacion': df_poblacion,
        'comunas': comunas,
        'gdf_puntos': gdf_puntos,
        'geografia': df_geografia,
    }

# ======================================
# CREACIÓN DE GRÁFICOS
# ======================================
def crear_graficos(data):
    df_vehiculos = data['vehiculos']
    df_siniestros = data['siniestros']
    df_actores = data['actores']
    df_poblacion = data['poblacion']
    df_geografia = data['geografia']

    
    meses = [
        "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
        "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"
    ]

    dias = ["LUNES", "MARTES", "MIERCOLES", "JUEVES", "VIERNES", "SABADO", "DOMINGO"]

    horas = ["00:00","01:00","02:00","03:00","04:00","05:00","06:00","07:00","08:00","09:00","10:00","11:00",
             "12:00","13:00","14:00","15:00","16:00","17:00","18:00","19:00","20:00","21:00","22:00","23:00"]
    
    # Gráficos básicos
    fig_vehiculos = crear_grafico_barras(df_vehiculos, 'CLASE_VEHICULO', 
                                       'Distribución de Siniestros por Tipo de Vehículo')
    
    fig_servicio = crear_grafico_barras(df_vehiculos, 'CLASE_DE_SERVICIO',
                                      'Distribución de Siniestros por Clase de Servicio')
    
    # Gráfico combinado
    colores_servicio = {"PARTICULAR": "#004aad", "PUBLICO": "#6f9ceb", "OTROS": "#808080"}
    fig_combinacion = crear_grafico_apilado(
        df_vehiculos, 'CLASE_VEHICULO', 'CLASE_DE_SERVICIO',
        'Distribución de Tipos de Vehículo por Clase de Servicio',
        colores_servicio
    )
    
    df_vehiculos['MES'] = pd.Categorical(df_vehiculos['MES'], categories=meses, ordered=True)
    
    # Crear diccionario de colores para cada vehículo
    colores_vehiculos = {"CARRETILLA":"#c0d2eb","MOTOCARRO":"#003882","TAXI":"#000A16","VOLQUETA":"#406FAC","BICICLETA":"#001C42"
                         ,"CAMION":"#002557","BUS":"#000000","CAMIONETA":"#80a5d6","AUTOMOVIL":"#004AAD","MOTO":"#00132C"}
    
    fig_mensual = crear_grafico_apilado(
        df_vehiculos.sort_values('MES'), 'MES', 'CLASE_VEHICULO',
        'Distribución de Siniestros por Clase de Vehículo y Mes',
        colores_vehiculos  # Usamos nuestro diccionario personalizado
    )

    # Contar la frecuencia de cada tipo de siniestro
    tipos_siniestros = df_siniestros[['CHOQUE', 'ATROPELLO', 'VOLCAMIENTO', 'CAIDA_O', 'OTRO']].apply(lambda x: x.notna().sum())

    # Convertir a DataFrame para facilitar la visualización
    tipos_siniestros_df = tipos_siniestros.reset_index()
    tipos_siniestros_df.columns = ['Tipo_Siniestro', 'Frecuencia']

    fig_tipos_siniestros = crear_grafico_barras(tipos_siniestros_df, 'Tipo_Siniestro', 'Frecuencia de Tipos de Siniestros', color="#004aad", datos_agregados=True, columna_categoria='Tipo_Siniestro', columna_valor='Frecuencia')

     # Gráfico mensual de heridos/muertos
    df_siniestros['HERIDOS_NUM'] = df_siniestros['HERIDOS'].apply(lambda x: 1 if x == 'X' else 0)
    df_siniestros['MUERTOS_NUM'] = df_siniestros['MUERTOS'].apply(lambda x: 1 if x == 'X' else 0)
    
    resumen_mensual = df_siniestros.groupby('MES')[['HERIDOS_NUM', 'MUERTOS_NUM']].sum().reset_index()
    resumen_mensual['MES'] = pd.Categorical(resumen_mensual['MES'], categories=meses, ordered=True)
    resumen_mensual = resumen_mensual.sort_values('MES')

    # Gráficos básicos
    fig_resumen_mensual = crear_grafico_comparativo(
    df=resumen_mensual,
    col_categoria="MES",
    columnas_comparar=["HERIDOS_NUM", "MUERTOS_NUM"],
    titulo="Heridos vs Muertos por Mes",
    colores=["#004aad", "#d9534f"],
    nombres_series=["Heridos", "Muertos"]
    )

    # Gráfico de actores viales
    actores_viales = df_siniestros[['PEATON', 'ACOMPANANTE', 'PASAJERO', 'CONDUCTOR', 'CICLISTAS']].sum()
    actores_viales_df = actores_viales.reset_index()
    actores_viales_df.columns = ['Actor_Vial', 'Total']
    actores_viales_df = actores_viales_df.sort_values('Total', ascending=True)

    fig_actores_viales = crear_grafico_barras_horizontal(
        actores_viales_df,
        col_x='Total',
        col_y='Actor_Vial',
        titulo="Actores Viales Involucrados en Siniestros",
        color="#004aad",
        mostrar_porcentaje=True
    )

        # 1. Calcular totales
    total_heridos = df_siniestros['TOTAL_HERIDOS'].sum()
    total_muertos = df_siniestros['TOTAL_MUERTOS'].sum()

    # 2. Preparar datos
    df_totales = pd.DataFrame({
        'Metrica': ['Muertos','Heridos'],
        'Total': [total_muertos, total_heridos]
    })
    df_totales = df_totales.pivot(columns='Metrica', values='Total').reset_index(drop=True)
   

    fig_totales= crear_grafico_comparativo(
        df=df_totales,
        col_categoria="index",  # Columna dummy para agrupar
        columnas_comparar=['Muertos','Heridos'],  # Nombres exactos de las nuevas columnas
        titulo="Comparación Total de Heridos vs Muertos",
        colores=["#d9534f","#004aad"],  # 2 colores para 2 métricas
        nombres_series=['Muertos','Heridos'],
        modo_total=True  # Formato horizontal para totales
    )

    # Convertir EDAD a numérico en la hoja "Demografía Involucrados"
    df_actores['EDAD'] = pd.to_numeric(df_actores['EDAD'], errors='coerce')

    # Contar la cantidad de registros por ACTOR_VIAL
    conteo_actores = df_actores["ACTOR_VIAL"].value_counts().reset_index()
    conteo_actores.columns = ["Actor Vial", "Cantidad"]

    # Ordenar los datos por cantidad de mayor a menor
    conteo_actores = conteo_actores.sort_values(by="Cantidad", ascending=False)

    fig_conteo_actores = crear_grafico_barras(
        df=conteo_actores,
        columna="Actor Vial",  # Ignorado en modo_agregado
        titulo="Distribución de Actores Viales",
        color="#004aad",
        datos_agregados=True,
        columna_categoria="Actor Vial",
        columna_valor="Cantidad"
    )

    # Preparar los datos
    conteo_genero = (
        df_actores.groupby(["ACTOR_VIAL", "GENERO"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Asegurarse de que ambos géneros estén presentes (incluso si alguno tiene conteo cero)
    conteo_genero = conteo_genero.reindex(columns=["ACTOR_VIAL", "M", "F"], fill_value=0)

    # Calcular el total por actor vial
    conteo_genero["Total"] = conteo_genero["M"] + conteo_genero["F"]

    # Calcular los porcentajes para hombres y mujeres
    conteo_genero["Porcentaje_M"] = (conteo_genero["M"] / conteo_genero["Total"]) * 100
    conteo_genero["Porcentaje_F"] = (conteo_genero["F"] / conteo_genero["Total"]) * 100

    # Ordenar los datos por el total de mayor a menor
    conteo_genero = conteo_genero.sort_values(by="Total", ascending=False)

    fig_genero_actores = crear_grafico_comparativo(
        df=conteo_genero,
        col_categoria="ACTOR_VIAL",
        columnas_comparar=["M", "F"],
        titulo="Distribución de Género por Actor Vial",
        colores=["#004aad", "#F600E6"],  # Azul para hombres, naranja para mujeres
        nombres_series=["Hombres", "Mujeres"],
        modo_barras="group",  # Usar modo apilado para mostrar porcentajes
        modo_total=False
    )
    fig_boxplot_edad_actores = crear_boxplot(
        df_actores,
        columna_num='EDAD',
        columna_cat='ACTOR_VIAL',
        titulo='Diagrama de Cajas y Bigotes para Edad de Actores Viales',
        color="#004aad"   
    )
    fig_distr_edad_conductores  = crear_histograma(
        df=df_actores,
        columna_numerica="EDAD",
        titulo="Histograma de Edades para Conductores",
        color="#004aad",
        auto_bins=True,
        columna_categoria="ACTOR_VIAL",
        subcategoria="CONDUCTOR"
    )
    fig_distr_edad_pasajeros = crear_histograma(
        df=df_actores,
        columna_numerica="EDAD",
        titulo="Histograma de Edades para Pasajeros",
        color="#004aad",
        auto_bins=True,
        columna_categoria="ACTOR_VIAL",
        subcategoria="PASAJERO"
    )
    fig_distr_edad_peatones  = crear_histograma(
        df=df_actores,
        columna_numerica="EDAD",
        titulo="Histograma de Edades para Ciclistas",
        color="#004aad",
        auto_bins=True,
        columna_categoria="ACTOR_VIAL",
        subcategoria="CICLISTA"
    )
    fig_distr_edad_ciclistas = crear_histograma(
        df=df_actores,
        columna_numerica="EDAD",
        titulo="Histograma de Edades para Peatones",
        color="#004aad",
        auto_bins=True,
        columna_categoria="ACTOR_VIAL",
        subcategoria="PEATON"
    )   
    fig_distribucion_edad = crear_histograma(
        df=df_poblacion,
        columna_numerica="EDAD",
        titulo="Distribución de Edades",
        color="#004aad",
        tickangle=-45,
        auto_bins=False,  # Desactivar Friedman-Diaconis
        num_bins=50      # Número de bins manual
        )
    
    # Eliminar filas con valores nulos en la columna "GENERO"
    df_poblacion = df_poblacion.dropna(subset=["GENERO"])

    # Contar la cantidad de hombres y mujeres
    prop_genero = df_poblacion["GENERO"].value_counts().reset_index()
    prop_genero.columns = ["Género", "Cantidad"]

    fig_prop_genero = grafico_comparativo_sencillo(
        df=prop_genero,
        col_categoria="Género",
        col_valor="Cantidad",
        titulo="Distribución de Género",
        colores=["#004aad", "#F600E6"],  # Azul para hombres, rosa para mujeres
        nombres_series=["Hombres", "Mujeres"]
    )

        # Validar que las columnas necesarias existan
    if "EDAD" not in df_poblacion.columns or "GENERO" not in df_poblacion.columns:
        raise ValueError("Las columnas 'EDAD' y 'GENERO' deben estar presentes en el DataFrame.")

    # Convertir EDAD a numérico y eliminar valores NaN
    df_poblacion['EDAD'] = pd.to_numeric(df_poblacion['EDAD'], errors='coerce')
    df_poblacion = df_poblacion.dropna(subset=["EDAD"])

    # Asegurarse de que haya un rango válido
    edad_min = df_poblacion["EDAD"].min()
    edad_max = df_poblacion["EDAD"].max()

    if edad_min == edad_max:
        raise ValueError("Todos los valores en la columna 'EDAD' son iguales. No se puede crear un histograma.")

    # Crear bordes de los bins
    num_bins = 50
    bins_edges = np.linspace(edad_min, edad_max, num_bins + 1)

    # Crear etiquetas para los rangos de bins
    bin_labels = [f"{int(bins_edges[i])}-{int(bins_edges[i+1])}" for i in range(len(bins_edges) - 1)]

    # Crear una copia explícita para evitar SettingWithCopyWarning
    df_poblacion = df_poblacion.copy()

    # Asignar cada edad a su rango correspondiente
    df_poblacion["Rango Edad"] = pd.cut(df_poblacion["EDAD"], bins=bins_edges, labels=bin_labels, include_lowest=True)

    # Contar la cantidad de hombres y mujeres en cada rango
    conteo_genero = (
        df_poblacion.groupby(["Rango Edad", "GENERO"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Asegurarse de que ambos géneros estén presentes (incluso si alguno tiene conteo cero)
    conteo_genero = conteo_genero.reindex(columns=["Rango Edad", "M", "F"], fill_value=0)

    fig_distr_edad_genero = crear_grafico_comparativo(
        df=conteo_genero,
        col_categoria="Rango Edad",
        columnas_comparar=["M", "F"],
        titulo="Distribución de Género por Rango de Edad",
        colores=["#004aad", "#F600E6"],
        nombres_series=["Hombres", "Mujeres"],
        modo_barras="group",
        tickangle=45,
        modo_total=False,
    )

    # Convertir FECHA a datetime
    df_geografia["FECHA"] = pd.to_datetime(df_geografia["FECHA"])

    # Convertir HORA_O a datetime para facilitar el manejo
    df_geografia['HORA_O'] = pd.to_datetime(df_geografia['HORA_O'], format='%H:%M:%S', errors='coerce')

    # Extraer solo la hora (como entero)
    df_geografia['HORA'] = df_geografia['HORA_O'].dt.hour

    # Clasificar días en "Entre Semana" y "Fines de Semana"
    df_geografia['TIPO_DIA'] = df_geografia['DIA'].apply(lambda x: 'Fin de Semana' if x in ['VIERNES','SABADO', 'DOMINGO'] else 'Entre Semana')


    # Contar siniestros por mes
    siniestros_por_mes = (
        df_geografia["MES"]
        .value_counts()  # Contar ocurrencias de cada mes
        .reindex(meses, fill_value=0)  # Reindexar para asegurar el orden correcto
        .reset_index()  # Convertir a DataFrame 
    )

    
    df_geografia['DIA'] = pd.Categorical(df_geografia['DIA'], categories=dias, ordered=True)

    # Contar siniestros por mes
    siniestros_por_dia = (
        df_geografia["DIA"]
        .value_counts()  # Contar ocurrencias de cada mes
        .reindex(dias, fill_value=0)  # Reindexar para asegurar el orden correcto
        .reset_index()  # Convertir a DataFrame 
    )

    


    siniestros_por_hora = (
        df_geografia["HORA"]
        .value_counts()  # Contar ocurrencias de cada hora
        .reindex(range(24), fill_value=0)  # Reindexar para asegurar todas las horas (0-23)
        .reset_index()  # Convertir a DataFrame  
    )

    siniestros_por_hora['HORA'] = horas

    # Agrupar por COMUNA y contar el número total de accidentes
    accidentes_por_comuna = (
        df_geografia.groupby('COMUNA').size()
        .reset_index(name='TOTAL_ACCIDENTES')
        .sort_values(by='TOTAL_ACCIDENTES', ascending=False)
    )

    # Seleccionar las 3 comunas con más accidentes
    top_3_comunas = accidentes_por_comuna.head(3)['COMUNA']

    # Filtrar el DataFrame para incluir solo las 3 comunas principales
    df_geografia_top_3 = df_geografia[df_geografia['COMUNA'].isin(top_3_comunas)]

    # Agrupar por DIA y COMUNA para preparar los datos para el Sunburst Chart
    accidentes_por_dia_comuna = (
        df_geografia_top_3.groupby(['DIA', 'COMUNA'],observed=True).size()
                .reset_index(name='ACCIDENTES')
    )

    # Ordenar los datos:
    # 1. Primero por DIA (Lunes a Domingo)
    # 2. Luego por ACCIDENTES (de mayor a menor dentro de cada día)
    accidentes_por_dia_comuna = accidentes_por_dia_comuna.sort_values(
        ['DIA', 'ACCIDENTES'], 
        ascending=[True, False]  # Días en orden, comunas de mayor a menor
    )

    fig_distr_siniestros_mes =  crear_grafico_barras(
        df=siniestros_por_mes,
        columna='MES',
        titulo='Distribución de Siniestros por Mes',
        color="#004aad",
        datos_agregados=True,
        columna_categoria='MES',
        columna_valor='count'
    )
    
    fig_distr_siniestros_dia = crear_grafico_barras(
        df=siniestros_por_dia,
        columna='DIA',
        titulo='Distribución de Siniestros por Día',
        color="#004aad",
        datos_agregados=True,
        columna_categoria='DIA',
        columna_valor='count'
    )

    fig_siniestros_hora = crear_grafico_barras(
        df=siniestros_por_hora,
        columna='HORA',
        titulo='Distribución de Siniestros por Hora',
        color="#004aad",
        datos_agregados=True,
        columna_categoria='HORA',
        columna_valor='count'
    )

            # Filtrar datos para Entre Semana y Fin de Semana
    df_entre_semana = df_geografia[df_geografia['TIPO_DIA'] == 'Entre Semana']
    df_fin_semana = df_geografia[df_geografia['TIPO_DIA'] == 'Fin de Semana']

    # Conteo de siniestros por hora
    siniestros_entre_semana = df_entre_semana['HORA'].value_counts().sort_index().reindex(range(24), fill_value=0)
    siniestros_fin_semana = df_fin_semana['HORA'].value_counts().sort_index().reindex(range(24), fill_value=0)

   # Contar siniestros por hora
    siniestros_entre_semana = (
        df_entre_semana["HORA"]
        .value_counts()  # Contar ocurrencias de cada hora
        .reindex(range(24), fill_value=0)  # Reindexar para asegurar todas las horas (0-23)
        .reset_index()  # Convertir a DataFrame  
    )

    siniestros_fin_semana = (
        df_fin_semana["HORA"]
        .value_counts()  # Contar ocurrencias de cada hora
        .reindex(range(24), fill_value=0)  # Reindexar para asegurar todas las horas (0-23)
        .reset_index()  # Convertir a DataFrame  
    )

    # Unir los DataFrames usando la columna "hora"
    df_comparacion= pd.merge(siniestros_entre_semana, siniestros_fin_semana, on="HORA", how="right")

    # Llenar valores NaN con 0 (en caso de que alguna hora no esté presente en uno de los DataFrames)
    df_comparacion = df_comparacion.fillna(0)

    df_comparacion['HORA'] = horas

    fig_comparacion_semana = crear_grafico_comparativo(
        df = df_comparacion,
        col_categoria="HORA",
        columnas_comparar=["count_x", "count_y"],
        titulo="Comparación de Siniestros por Hora entre Semana y Fin de Semana",
        colores=["#004aad", "#ff0000"],
        nombres_series=["Entre Semana", "Fin de Semana"],
        modo_barras="group",
        modo_total=False,
    )
    
    fig_sunburst_comunas = crear_sunburst_chart(
        df=accidentes_por_dia_comuna,
        path=["DIA", "COMUNA"],  # Jerarquía: primero DÍA, luego COMUNA
        values="ACCIDENTES",     # Valores para el tamaño de los segmentos
        titulo="Distribución de Accidentes por Día de la Semana y Comuna (Top 3 Comunas)",
        color="DIA",             # Colorear según el día
        color_continuous_scale="Blues",  # Escala de colores
    )

    # Convertir FECHA a datetime
    df_geografia["FECHA"] = pd.to_datetime(df_geografia["FECHA"])

    # Convertir HORA_O a datetime para facilitar el manejo
    df_geografia['HORA_O'] = pd.to_datetime(df_geografia['HORA_O'], format='%H:%M:%S', errors='coerce')

    # Extraer solo la hora (como entero)
    df_geografia['HORA'] = df_geografia['HORA_O'].dt.hour

    # Clasificar días en "Entre Semana" y "Fines de Semana"
    df_geografia['TIPO_DIA'] = df_geografia['DIA'].apply(lambda x: 'Fin de Semana' if x in ['VIERNES','SABADO', 'DOMINGO'] else 'Entre Semana')


    # Contar siniestros por mes
    siniestros_por_mes = (
        df_geografia["MES"]
        .value_counts()  # Contar ocurrencias de cada mes
        .reindex(meses, fill_value=0)  # Reindexar para asegurar el orden correcto
        .reset_index()  # Convertir a DataFrame 
    )

    
    df_geografia['DIA'] = pd.Categorical(df_geografia['DIA'], categories=dias, ordered=True)

    # Contar siniestros por mes
    siniestros_por_dia = (
        df_geografia["DIA"]
        .value_counts()  # Contar ocurrencias de cada mes
        .reindex(dias, fill_value=0)  # Reindexar para asegurar el orden correcto
        .reset_index()  # Convertir a DataFrame 
    )

    siniestros_por_hora = (
        df_geografia["HORA"]
        .value_counts()  # Contar ocurrencias de cada hora
        .reindex(range(24), fill_value=0)  # Reindexar para asegurar todas las horas (0-23)
        .reset_index()  # Convertir a DataFrame  
    )

    siniestros_por_hora['HORA'] = horas

    # Agrupar por COMUNA y contar el número total de accidentes
    accidentes_por_comuna = (
        df_geografia.groupby('COMUNA').size()
        .reset_index(name='TOTAL_ACCIDENTES')
        .sort_values(by='TOTAL_ACCIDENTES', ascending=False)
    )

    # Seleccionar las 3 comunas con más accidentes
    top_3_comunas = accidentes_por_comuna.head(3)['COMUNA']

    # Filtrar el DataFrame para incluir solo las 3 comunas principales
    df_geografia_top_3 = df_geografia[df_geografia['COMUNA'].isin(top_3_comunas)]

    # Agrupar por DIA y COMUNA para preparar los datos para el Sunburst Chart
    accidentes_por_dia_comuna = (
        df_geografia_top_3.groupby(['DIA', 'COMUNA'],observed=True).size()
                .reset_index(name='ACCIDENTES')
    )

    # Ordenar los datos:
    # 1. Primero por DIA (Lunes a Domingo)
    # 2. Luego por ACCIDENTES (de mayor a menor dentro de cada día)
    accidentes_por_dia_comuna = accidentes_por_dia_comuna.sort_values(
        ['DIA', 'ACCIDENTES'], 
        ascending=[True, False]  # Días en orden, comunas de mayor a menor
    )

    fig_distr_siniestros_mes =  crear_grafico_barras(
        df=siniestros_por_mes,
        columna='MES',
        titulo='Distribución de Siniestros por Mes',
        color="#004aad",
        datos_agregados=True,
        columna_categoria='MES',
        columna_valor='count'
    )
    
    fig_distr_siniestros_dia = crear_grafico_barras(
        df=siniestros_por_dia,
        columna='DIA',
        titulo='Distribución de Siniestros por Día',
        color="#004aad",
        datos_agregados=True,
        columna_categoria='DIA',
        columna_valor='count'
    )

    fig_siniestros_hora = crear_grafico_barras(
        df=siniestros_por_hora,
        columna='HORA',
        titulo='Distribución de Siniestros por Hora',
        color="#004aad",
        datos_agregados=True,
        columna_categoria='HORA',
        columna_valor='count'
    )

            # Filtrar datos para Entre Semana y Fin de Semana
    df_entre_semana = df_geografia[df_geografia['TIPO_DIA'] == 'Entre Semana']
    df_fin_semana = df_geografia[df_geografia['TIPO_DIA'] == 'Fin de Semana']

    # Conteo de siniestros por hora
    siniestros_entre_semana = df_entre_semana['HORA'].value_counts().sort_index().reindex(range(24), fill_value=0)
    siniestros_fin_semana = df_fin_semana['HORA'].value_counts().sort_index().reindex(range(24), fill_value=0)

   # Contar siniestros por hora
    siniestros_entre_semana = (
        df_entre_semana["HORA"]
        .value_counts()  # Contar ocurrencias de cada hora
        .reindex(range(24), fill_value=0)  # Reindexar para asegurar todas las horas (0-23)
        .reset_index()  # Convertir a DataFrame  
    )

    siniestros_fin_semana = (
        df_fin_semana["HORA"]
        .value_counts()  # Contar ocurrencias de cada hora
        .reindex(range(24), fill_value=0)  # Reindexar para asegurar todas las horas (0-23)
        .reset_index()  # Convertir a DataFrame  
    )

    # Unir los DataFrames usando la columna "hora"
    df_comparacion= pd.merge(siniestros_entre_semana, siniestros_fin_semana, on="HORA", how="right")

    # Llenar valores NaN con 0 (en caso de que alguna hora no esté presente en uno de los DataFrames)
    df_comparacion = df_comparacion.fillna(0)

    df_comparacion['HORA'] = horas

    fig_comparacion_semana = crear_grafico_comparativo(
        df = df_comparacion,
        col_categoria="HORA",
        columnas_comparar=["count_x", "count_y"],
        titulo="Comparación de Siniestros por Hora entre Semana y Fin de Semana",
        colores=["#004aad", "#ff0000"],
        nombres_series=["Entre Semana", "Fin de Semana"],
        modo_barras="group",
        modo_total=False,
    )

        # Agrupar por DIA y HORA, y contar los siniestros
    siniestros_por_dia_hora = (
        df_geografia.groupby(['DIA', 'HORA'], observed=True).size()
        .reset_index(name='SINIESTROS')
    )

    # Asegurarse de que la columna HORA sea de tipo int (convertir NaN a 0 si es necesario)
    siniestros_por_dia_hora['HORA'] = pd.to_numeric(siniestros_por_dia_hora['HORA'], errors='coerce').fillna(0).astype(int)

    # Convertir la columna HORA a formato "HH:MM"
    siniestros_por_dia_hora['HORA'] = siniestros_por_dia_hora['HORA'].apply(lambda x: f"{x:02d}:00")

    combinaciones_dia_hora = pd.DataFrame([(dia, hora) for dia in dias for hora in horas], columns=['DIA', 'HORA'])

    # Unir con los datos originales para completar las horas faltantes
    siniestros_por_dia_hora = pd.merge(
        combinaciones_dia_hora,
        siniestros_por_dia_hora,
        on=['DIA', 'HORA'],
        how='left'
    )

    # Rellenar valores NaN con 0 (para las horas sin siniestros)
    siniestros_por_dia_hora['SINIESTROS'] = siniestros_por_dia_hora['SINIESTROS'].fillna(0).astype(int)

        # Filtrar el DataFrame para solo incluir el día "LUNES"
    df_lunes = siniestros_por_dia_hora[siniestros_por_dia_hora['DIA'] == 'LUNES']
    df_martes = siniestros_por_dia_hora[siniestros_por_dia_hora['DIA'] == 'MARTES']
    df_miercoles = siniestros_por_dia_hora[siniestros_por_dia_hora['DIA'] == 'MIERCOLES']
    df_jueves = siniestros_por_dia_hora[siniestros_por_dia_hora['DIA'] == 'JUEVES']
    df_viernes = siniestros_por_dia_hora[siniestros_por_dia_hora['DIA'] == 'VIERNES']
    df_sabado = siniestros_por_dia_hora[siniestros_por_dia_hora['DIA'] == 'SABADO']
    df_domingo = siniestros_por_dia_hora[siniestros_por_dia_hora['DIA'] == 'DOMINGO']


    fig_siniestros_hora_lunes = crear_grafico_barras(
        df=df_lunes,
        columna="HORA",  # Columna categórica (las horas)
        titulo="Siniestros por Hora el Lunes",
        color="#004aad",
        datos_agregados=True,  # Usar datos precalculados
        columna_categoria="HORA",  # Categoría: las horas
        columna_valor="SINIESTROS"  # Valor: número de siniestros
    )

    fig_siniestros_hora_martes = crear_grafico_barras(
        df=df_martes,
        columna="HORA",  # Columna categórica (las horas)
        titulo="Siniestros por Hora el Martes",
        color="#004aad",
        datos_agregados=True,  # Usar datos precalculados
        columna_categoria="HORA",  # Categoría: las horas
        columna_valor="SINIESTROS"  # Valor: número de siniestros
    )

    fig_siniestros_hora_miercoles = crear_grafico_barras(
        df=df_miercoles,
        columna="HORA",  # Columna categórica (las horas)
        titulo="Siniestros por Hora el Miércoles",
        color="#004aad",
        datos_agregados=True,  # Usar datos precalculados
        columna_categoria="HORA",  # Categoría: las horas
        columna_valor="SINIESTROS"  # Valor: número de siniestros
    )

    fig_siniestros_hora_jueves = crear_grafico_barras(
        df=df_jueves,
        columna="HORA",  # Columna categórica (las horas)
        titulo="Siniestros por Hora el Jueves",
        color="#004aad",
        datos_agregados=True,  # Usar datos precalculados
        columna_categoria="HORA",  # Categoría: las horas
        columna_valor="SINIESTROS"  # Valor: número de siniestros
    )

    fig_siniestros_hora_viernes = crear_grafico_barras(
        df=df_viernes,
        columna="HORA",  # Columna categórica (las horas)
        titulo="Siniestros por Hora el Viernes",
        color="#004aad",
        datos_agregados=True,  # Usar datos precalculados
        columna_categoria="HORA",  # Categoría: las horas
        columna_valor="SINIESTROS"  # Valor: número de siniestros
    )

    fig_siniestros_hora_sabado = crear_grafico_barras(
        df=df_sabado,
        columna="HORA",  # Columna categórica (las horas)
        titulo="Siniestros por Hora el Sábado",
        color="#004aad",
        datos_agregados=True,  # Usar datos precalculados
        columna_categoria="HORA",  # Categoría: las horas
        columna_valor="SINIESTROS"  # Valor: número de siniestros
    )

    fig_siniestros_hora_domingo = crear_grafico_barras(
        df=df_domingo,
        columna="HORA",  # Columna categórica (las horas)
        titulo="Siniestros por Hora el Domingo",
        color="#004aad",
        datos_agregados=True,  # Usar datos precalculados
        columna_categoria="HORA",  # Categoría: las horas
        columna_valor="SINIESTROS"  # Valor: número de siniestros
    )
    

    
    return[ 
        fig_vehiculos, 
        fig_servicio, 
        fig_combinacion, 
        fig_mensual,
        fig_tipos_siniestros,
        fig_resumen_mensual,
        fig_actores_viales,
        fig_totales,
        fig_conteo_actores,
        fig_distribucion_edad,
        fig_genero_actores,
        fig_boxplot_edad_actores,
        fig_distr_edad_conductores,
        fig_distr_edad_pasajeros,
        fig_distr_edad_peatones,
        fig_distr_edad_ciclistas,
        fig_prop_genero,
        fig_distr_edad_genero,
        fig_distr_siniestros_mes,
        fig_distr_siniestros_dia,
        fig_siniestros_hora,
        fig_comparacion_semana,
        fig_sunburst_comunas,
        fig_siniestros_hora_lunes,
        fig_siniestros_hora_martes,
        fig_siniestros_hora_miercoles,
        fig_siniestros_hora_jueves,
        fig_siniestros_hora_viernes,
        fig_siniestros_hora_sabado,
        fig_siniestros_hora_domingo,
    ]

# ======================================
# APLICACIÓN DASH
# ======================================
def crear_aplicacion(figs):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Desempaquetar figuras
    ( fig_vehiculos, 
        fig_servicio, 
        fig_combinacion, 
        fig_mensual,
        fig_tipos_siniestros,
        fig_resumen_mensual,
        fig_actores_viales,
        fig_totales,
        fig_conteo_actores,
        fig_distribucion_edad,
        fig_genero_actores,
        fig_boxplot_edad_actores,
        fig_distr_edad_conductores,
        fig_distr_edad_pasajeros,
        fig_distr_edad_peatones,
        fig_distr_edad_ciclistas,
        fig_prop_genero,
        fig_distr_edad_genero,
        fig_distr_siniestros_mes,
        fig_distr_siniestros_dia,
        fig_siniestros_hora,
        fig_comparacion_semana,
        fig_sunburst_comunas,
        fig_siniestros_hora_lunes,
        fig_siniestros_hora_martes,
        fig_siniestros_hora_miercoles,
        fig_siniestros_hora_jueves,
        fig_siniestros_hora_viernes,
        fig_siniestros_hora_sabado,
        fig_siniestros_hora_domingo,) = figs

    # Aplicar autosize a todas las figuras
    for figura in figs:
        if isinstance(figura, go.Figure):
            figura.update_layout(autosize=True)

    nombres_mapas = ["Mapa de Actores Viales", 
                     "Mapa de Caracterización de Siniestros", 
                     "Mapa de Siniestralidad por Mes", 
                     "Mapa de Siniestralidad por día", 
                     "Mapa de Siniestralidad por hora"]
    
    rutas_mapas = [r"\assets\mapa_actores_viales.html",
                   r"\assets\mapa_caracterizacion_siniestros.html",
                   r"\assets\mapa_geografia_siniestros_mes.html",
                   r"\assets\mapa_geografia_siniestros_dia.html",
                   r"\assets\mapa_geografia_siniestros_dia.html"
                   ]


    # Estilos
    ESTILOS = {
        'titulo_principal': {
            'color': 'white',
            'backgroundColor': '#004aad',
            'padding': '20px',
            'borderRadius': '15px',
            'marginBottom': '30px'
        },
        'seccion': {
            'backgroundColor': 'white',
            'borderRadius': '15px',
            'padding': '20px',
            'marginBottom': '20px',
            'boxShadow': '0 4px 6px 0 rgba(0,0,0,0.1)'
        },
        'titulo_seccion': {
            'color': '#004aad',
            'borderBottom': '2px solid #004aad',
            'paddingBottom': '10px',
            'marginBottom': '20px'
        }
    }

    def crear_seccion(titulo, figuras):
        return html.Div([
            html.H3(titulo, style=ESTILOS['titulo_seccion']),
            *[dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=figura,
                        config={'responsive': True},
                        style={'height': '70vh', 'width': '100%'}
                    ),
                    className="mb-4"
                )
            ) for figura in figuras]
        ], style=ESTILOS['seccion'])

    # Layout principal
    app.layout = dbc.Container([
        html.Div(id='dummy-output'),
        
        html.Div([
            html.H1("Análisis de Siniestros Viales", 
                   style=ESTILOS['titulo_principal'],
                   className="text-center")
        ], className="my-4"),
        
        dbc.Tabs(
            id='main-tabs',
            children=[
                # Pestaña Vehículos
                dbc.Tab([
                    crear_seccion("Distribución de Vehículos", [fig_vehiculos]),
                    crear_seccion("Tipo de Servicio", [fig_servicio]),
                    crear_seccion("Combinación de Vehículos", [fig_combinacion]),
                    crear_seccion("Evolución Mensual", [fig_mensual])
                ], label="🚗 Vehículos"),
                
                # Pestaña Siniestros
                dbc.Tab([
                    crear_seccion("Tipos de Siniestros", [fig_tipos_siniestros]),
                    crear_seccion("Resumen Mensual", [fig_resumen_mensual]),
                    crear_seccion("Actores Viales", [fig_actores_viales]),
                    crear_seccion("Totales", [fig_totales])
                ], label="🚨 Siniestros"),
                
                # Pestaña Actores Viales
                dbc.Tab([
                    crear_seccion("Conteo de Actores", [fig_conteo_actores]),
                    crear_seccion("Género de Actores", [fig_genero_actores]),
                    crear_seccion("Distribución de Edad", [fig_boxplot_edad_actores]),
                    crear_seccion("Edad Conductores", [fig_distr_edad_conductores]),
                    crear_seccion("Edad Pasajeros", [fig_distr_edad_pasajeros]),
                    crear_seccion("Edad Peatones", [fig_distr_edad_peatones]),
                    crear_seccion("Edad Ciclistas", [fig_distr_edad_ciclistas]),
                ], label="🚶 Actores"),
                
                # Pestaña Demografía
                dbc.Tab([
                    crear_seccion("Distribución de Edad", [fig_distribucion_edad]),
                    crear_seccion("Proporción de Género", [fig_prop_genero]),
                    crear_seccion("Edad vs Género", [fig_distr_edad_genero])
                ], label="👥 Demografía"),
                
                # Pestaña Temporal
                dbc.Tab([
                    dbc.Tabs(
                        id='inner-tabs',
                        children=[
                            dbc.Tab([
                                crear_seccion("Distribución Mensual", [fig_distr_siniestros_mes]),
                                crear_seccion("Distribución Diaria", [fig_distr_siniestros_dia]),
                                crear_seccion("Distribución Horaria", [fig_siniestros_hora]),
                                crear_seccion("Comparación Semanal", [fig_comparacion_semana]),
                                crear_seccion("Sunburst Comunas", [fig_sunburst_comunas])
                            ], label="📅 Temporalidad"),
                            
                            dbc.Tab([
                                crear_seccion("Lunes", [fig_siniestros_hora_lunes]),
                                crear_seccion("Martes", [fig_siniestros_hora_martes]),
                                crear_seccion("Miércoles", [fig_siniestros_hora_miercoles]),
                                crear_seccion("Jueves", [fig_siniestros_hora_jueves]),
                                crear_seccion("Viernes", [fig_siniestros_hora_viernes]),
                                crear_seccion("Sábado", [fig_siniestros_hora_sabado]),
                                crear_seccion("Domingo", [fig_siniestros_hora_domingo])
                            ], label="🗓️ Por Día")
                        ]
                    )
                ], label="⏱️ Temporal"),
                # Pestaña Mapas (sin subpestañas)
                # Pestaña Mapas (sin subpestañas)
                dbc.Tab([
                    html.Div([
                        html.H3("Mapas", style=ESTILOS['titulo_seccion']),
                        html.P(
                            "Exploración Geográfica de Siniestros Viales. En esta sección, presentamos una serie de mapas interactivos diseñados para analizar y visualizar datos relacionados con siniestros viales desde diferentes perspectivas, para interactuar debes seleccionar un mapa y en la esquina superior izquierda encontraras el boton que despliega las diferentes capas por categoría de cada mapa",
                            style={'textAlign': 'justify', 'marginBottom': '20px'}
                        ),
                        # Contenedor principal con el DropdownMenu centrado
                        html.Div([
                            dbc.DropdownMenu(
                                label="Seleccionar Mapa",
                                children=[
                                    dbc.DropdownMenuItem(
                                        nombres_mapas[i],
                                        id=f"menu-item-{i+1}",
                                        n_clicks=0
                                    )
                                    for i in range(len(nombres_mapas))
                                ],
                                color="primary",
                                direction="down",
                                style={'marginBottom': '20px'}
                            ),
                            html.Iframe(
                                id="iframe-mapas",
                                src=rutas_mapas[0],
                                style={'width': '100%', 'height': '600px', 'border': 'none'}
                            ),
                            # Div para mostrar el texto dinámico
                            html.Div(
                                id="texto-dinamico",
                                style={'textAlign': 'justify', 'marginTop': '20px', 'color': '#333'}
                            )
                        ], style={'textAlign': 'center'})
                    ], style=ESTILOS['seccion'])
                ], label="🗺️ Mapas")
            ]
        )
    ], fluid=True, style={'height': '100vh'})

    # Callbacks para redimensionar
    app.clientside_callback(
        """
        function(active_tab) {
            setTimeout(function() {
                window.dispatchEvent(new Event('resize'));
            }, 200);
            return null;
        }
        """,
        Output('dummy-output', 'children'),
        [Input('main-tabs', 'active_tab'),
         Input('inner-tabs', 'active_tab')]
    )

    
    # Callback para cambiar el mapa en el iframe
    @app.callback(
        Output("iframe-mapas", "src"),
        [Input(f"menu-item-{i+1}", "n_clicks") for i in range(len(nombres_mapas))]
    )
    def actualizar_mapa(*args):
        ctx = dash.callback_context
        if not ctx.triggered:
            return rutas_mapas[0]  # Mostrar el primer mapa por defecto
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        index = int(triggered_id.split("-")[-1]) - 1
        return rutas_mapas[index]

    # Callback para actualizar el texto dinámico
    @app.callback(
        Output("texto-dinamico", "children"),
        [Input(f"menu-item-{i+1}", "n_clicks") for i in range(len(nombres_mapas))]
    )
    def actualizar_texto(*args):
        ctx = dash.callback_context
        if not ctx.triggered:
            return "Seleccione un mapa para ver su descripción."
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        index = int(triggered_id.split("-")[-1]) - 1
        
        # Textos específicos para cada mapa
        textos_mapas = [
            "1. Mapa de Actores Viales: Este mapa muestra la ubicación geográfica de los actores involucrados en siniestros viales.",
            "2. Mapa de Caracterización de Siniestros: Aquí se presenta una visión detallada de los tipos de siniestros ocurridos.",
            "3. Mapa de Siniestralidad por Mes: Este mapa explora la evolución temporal de los siniestros a nivel mensual.",
            "4. Mapa de Siniestralidad por Día: Con este mapa, es posible analizar la distribución diaria de los siniestros.",
            "5. Mapa de Siniestralidad por Hora: Este mapa se centra en la dimensión horaria de los siniestros."
        ]
        
        return textos_mapas[index]
    
    app.layout.children.append(
    dcc.Interval(id="interval-refresh", interval=1000, max_intervals=1)
    )
    return app

# ======================================
# EJECUCIÓN PRINCIPAL
# ======================================
