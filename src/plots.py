import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import plotly.subplots as sp
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy.stats import shapiro, probplot
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.tsa.stattools import adfuller, acf, pacf

def obtener_fecha_dos_meses_antes(fecha_str):
    fecha = datetime.strptime(fecha_str, '%Y-%m-%d')
    fecha_dos_meses_antes = fecha - relativedelta(months=2)
    
    # Ajustar si la fecha cae en sábado o domingo
    if fecha_dos_meses_antes.weekday() == 5:  # Sábado
        fecha_dos_meses_antes -= timedelta(days=1)
    elif fecha_dos_meses_antes.weekday() == 6:  # Domingo
        fecha_dos_meses_antes -= timedelta(days=2)
    
    return fecha_dos_meses_antes.strftime('%Y-%m-%d')

def contar_dias_habiles(fecha_inicio_str, fecha_fin_str):
    fecha_inicio = datetime.strptime(fecha_inicio_str, '%Y-%m-%d')
    fecha_fin = datetime.strptime(fecha_fin_str, '%Y-%m-%d')

    # Asegurarse de que fecha_inicio es menor o igual a fecha_fin
    if fecha_inicio > fecha_fin:
        fecha_inicio, fecha_fin = fecha_fin, fecha_inicio

    total_dias_habiles = 0
    current_date = fecha_inicio

    while current_date <= fecha_fin:
        if current_date.weekday() < 5:  # Lunes a Viernes son 0-4
            total_dias_habiles += 1
        current_date += timedelta(days=1)

    return total_dias_habiles

# Función para calcular la intensidad del proceso de Hawkes
def funcion_intensidad(mu, alpha, beta, eventos, tiempos):
    # Calcula la intensidad actual del proceso de Hawkes
    return mu + alpha * np.sum(eventos * np.exp(-beta * (len(eventos) + 1 - tiempos)))

# Simulación de precios de cierre usando el proceso de Hawkes
def simulated_hawkes_closing_prices(eventos, lambda_0, alpha, beta, initial_price, daily_volatility, start_date, num_days=30):
    number_events = len(eventos)
    
    eventos_historicos = list(eventos)  # Convertir a lista para modificarla

    # Crear un array de tiempos para los eventos históricos
    historical_times = np.arange(1, number_events + 1)

    # Lista para almacenar precios de cierre
    prices = [initial_price]

    current_intensity = lambda_0  # Intensidad actual
    
    simulated_events = np.zeros(1)
    
    for i in range(1, num_days + 1):
        # Actualiza los tiempos históricos añadiendo el nuevo tiempo
        historical_times = np.append(historical_times, number_events + i)

        P = 1 - np.exp(-current_intensity)
        
        price_change = 0  # Inicializar el cambio de precio para el día
        
        if np.random.uniform() < P:
            eventos_historicos.append(1)
            # Calcula el cambio de precio basado en el número de eventos
            event_price_change = np.random.normal(loc=0, scale=daily_volatility)
            price_change += event_price_change
            simulated_events = np.append(simulated_events, 1)
        else:
            simulated_events = np.append(simulated_events, 0)
            eventos_historicos.append(0)

        # Actualiza la intensidad actual del proceso de Hawkes
        current_intensity = funcion_intensidad(lambda_0, alpha, beta, eventos_historicos, historical_times)
        
        price_change += np.random.normal(loc=0, scale=daily_volatility)
        new_price = prices[-1] + price_change
        prices.append(new_price)

    # Generar las fechas desde la fecha de inicio
    dates = pd.bdate_range(start=start_date, periods=num_days)#dates = pd.date_range(start=start_date, periods=num_days + 1)  # +1 para incluir el día inicial

    return dates, prices, simulated_events

def grafico_simulacion_pred(Ticket,data,threshold,ahora,lambda_0,alpha,beta,num_simulations,num_simulations_tests):

    inicio_validacion = obtener_fecha_dos_meses_antes(ahora)

    # Calcular los retornos diarios
    Retornos = data['Close'].pct_change()

    # Establecer un umbral para detectar eventos (e.g., 1% de cambio)
    #threshold = 0.01

    eventos_si_no = Retornos.abs() > threshold

    # Crear una serie de eventos basada en el umbral
    #events = data.index[eventos_si_no]

    initial_price = data['Close'][-1]  # Precio inicial
    if Ticket == 'QQQ':
        daily_volatility = 6#data['Close'].std()/10  # Volatilidad diaria
    else:
        daily_volatility = 1500
    num_days = 140  # Número de días
    start_date = ahora  # Fecha de inicio

    # Inicializar la lista de eventos
    events = eventos_si_no

    # Simulación de múltiples trayectorias
    #num_simulations = 10  # Número de simulaciones que quieres realizar
    all_simulations = []  # Almacena todas las simulaciones


    ####

    # Parámetros del modelo
    lambda_0_tests = lambda_0  # Intensidad base
    alpha_tests = alpha     # Impacto del evento
    beta_tests = beta      # Tasa de decaimiento
    Close_tests = data['Close']
    Close_tests = Close_tests[Close_tests.index <= inicio_validacion]

    initial_price_tests = Close_tests[-1]  # Precio inicial
    daily_volatility_tests = daily_volatility#Close_tests.std()/10  # Volatilidad diaria

    num_days_tests = contar_dias_habiles(inicio_validacion,ahora)  # Número de días
    start_date_tests = inicio_validacion  # Fecha de inicio

    # Inicializar la lista de eventos
    events_tests = eventos_si_no[eventos_si_no.index <= inicio_validacion]

    # Simulación de múltiples trayectorias
    #num_simulations_tests = 3  # Número de simulaciones que quieres realizar
    all_simulations_tests = []  # Almacena todas las simulaciones


    for i in range(max(num_simulations_tests,num_simulations)):
        if i < num_simulations_tests:
            dates_tests, prices_tests, simulated_events_tests = simulated_hawkes_closing_prices(events_tests, lambda_0_tests, alpha_tests, beta_tests, initial_price_tests, daily_volatility_tests, start_date_tests, num_days_tests)
            all_simulations_tests.append((dates_tests, prices_tests, simulated_events_tests))
        if i < num_simulations:
            dates, prices, simulated_events = simulated_hawkes_closing_prices(events, lambda_0, alpha, beta, initial_price, daily_volatility, start_date, num_days)
            all_simulations.append((dates, prices, simulated_events))

    # Crear un gráfico combinado de Plotly
    combined_fig = go.Figure()

    # Añadir las líneas verticales punteadas en las fechas indicadas
    combined_fig.add_shape(
        type="line",
        x0=inicio_validacion,
        y0=0,
        x1=inicio_validacion,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color="black", width=2, dash="dot")
    )
    combined_fig.add_shape(
        type="line",
        x0=ahora,
        y0=0,
        x1=ahora,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color="black", width=2, dash="dot")
    )

    # Añadir el área sombreada entre las dos fechas indicadas
    combined_fig.add_shape(
        type="rect",
        x0=inicio_validacion,
        y0=0,
        x1=ahora,
        y1=1,
        xref='x',
        yref='paper',
        fillcolor="rgba(0, 255, 0, 0.1)",
        line=dict(width=0)
    )

    # Añadir la línea del precio de cierre real
    combined_fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'], 
        mode='lines', 
        name='Close Price',
        line=dict(color='black')
    ))

    # Añadir los eventos detectados como puntos en el gráfico real
    combined_fig.add_trace(go.Scatter(
        x=events[events].index,
        y=data.loc[events, 'Close'], 
        mode='markers', 
        name='Events',
        marker=dict(color='red', size=4)
    ))



    for idx, (dates_tests, prices_tests, simulated_events_tests) in enumerate(all_simulations_tests):
        # Añadir la línea de precios de cierre para cada simulación
        combined_fig.add_trace(go.Scatter(
            x=dates_tests,
            y=prices_tests[1:],  # Saltar el precio inicial para ajustar la longitud de la serie temporal
            mode='lines',
            name=f'Simulación {idx + 1}',
            line=dict(width=3,color = 'rgba(51, 178, 203, 0.7)')  # Cambiar el ancho de línea si lo deseas
        ))

        # Identificar los tiempos de eventos (donde los eventos son 1)
        event_dates_tests = [dates_tests[i-1] for i, e in enumerate(simulated_events_tests) if e == 1 and i < num_days_tests]

        # Añadir el scatter plot para eventos de cada simulación
        combined_fig.add_trace(go.Scatter(
            x=event_dates_tests,
            y=[prices_tests[i] for i, e in enumerate(simulated_events_tests) if e == 1 and i < num_days_tests],
            mode='markers',
            marker=dict(color='dimgray', symbol='0', size=4),
            showlegend=False
        ))



    ####

    for idx, (dates, prices, simulated_events) in enumerate(all_simulations):
        # Añadir la línea de precios de cierre para cada simulación
        combined_fig.add_trace(go.Scatter(
            x=dates,
            y=prices[1:],  # Saltar el precio inicial para ajustar la longitud de la serie temporal
            mode='lines',
            name=f'Simulación {idx + 1}',
            line=dict(width=2,color='goldenrod')  # Cambiar el ancho de línea si lo deseas
        ))

        # Identificar los tiempos de eventos (donde los eventos son 1)
        event_dates = [dates[i-1] for i, e in enumerate(simulated_events) if e == 1 and i < num_days]

        # Añadir el scatter plot para eventos de cada simulación
        combined_fig.add_trace(go.Scatter(
            x=event_dates,
            y=[prices[i] for i, e in enumerate(simulated_events) if e == 1 and i < num_days],
            mode='markers',
            marker=dict(color='gray', symbol='0', size=4),
            showlegend=False
        ))

    # Actualizar el diseño del gráfico combinado
    combined_fig.update_layout(
        title='Comparación del Precio de Cierre Real con Simulaciones usando Proceso de Hawkes',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre (USD)',
        legend_title='Leyenda',
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        )
    )
    return combined_fig, all_simulations_tests


def create_candlestick_volume_chart(data, Ticket):
    # Determine the colors of the volume bars
    colors = ['green' if close > open else 'red' if close < open else 'gray' 
              for open, close in zip(data['Open'], data['Close'])]

    # Create the figure with subplots
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=[0.7, 0.3],  # Adjust the relative height of subplots
        subplot_titles=(f"Candlestick Chart for {Ticket}", "Transaction Volume")
    )

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green', 
        decreasing_line_color='red',
        showlegend=False
    ), row=1, col=1)

    # Add bar chart for volume with conditional colors
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=colors,  # Assign the conditional colors
        showlegend=False
    ), row=2, col=1)

    # Configure the layout of the figure
    fig.update_layout(
        title_text=f'Analysis of {Ticket} - Candlestick and Volume',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,  # Hide the range slider
        template='plotly_white',  # Apply the plotly_white theme
        font=dict(family="Arial", size=12),  # Set the text font
        xaxis=dict(title=dict(text='', standoff=15)),  # Adjust space between X-axis title and labels
        xaxis2=dict(title=dict(text='Date', standoff=15))  # Ensure the X-axis title of the lower subplot is also adjusted
    )

    return fig


def create_event_detection_chart(data, threshold):
    # Calculate daily returns
    data['Return'] = data['Close'].pct_change()

    # Determine events based on the threshold
    eventos_si_no = data['Return'].abs() > threshold

    # Create a series of events based on the threshold
    events = data.index[eventos_si_no]

    # Create a Plotly figure
    fig = go.Figure()

    # Add the closing price line
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'], 
        mode='lines', 
        name='Close Price',
        line=dict(color='blue')
    ))

    # Add the events as points
    fig.add_trace(go.Scatter(
        x=events, 
        y=data.loc[events, 'Close'], 
        mode='markers', 
        name='Events',
        marker=dict(color='red', size=8)
    ))

    # Customize the layout
    fig.update_layout(
        title='Close Price with Detected Events',
        xaxis_title='Date',
        yaxis_title='Close Price (USD)',
        legend_title='Legend',
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey'
        )
    )

    return fig


# Funciones de cálculo de indicadores

# 1. Media Móvil Simple (SMA)
def simple_moving_average(data, period):
    return data.rolling(window=period).mean()

# 2. Media Móvil Exponencial (EMA)
def exponential_moving_average(data, period):
    return data.ewm(span=period, adjust=False).mean()

# 3. Relative Strength Index (RSI)
def relative_strength_index(data, period):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# 4. Moving Average Convergence Divergence (MACD)
def macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data.ewm(span=short_period, adjust=False).mean()
    long_ema = data.ewm(span=long_period, adjust=False).mean()
    
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

# 5. Tasa de Cambio (ROC)
def rate_of_change(data, period):
    return ((data - data.shift(period)) / data.shift(period)) * 100

# 6. Índice de Canal de Commodities (CCI)
def commodity_channel_index(data, period):
    sma = data.rolling(window=period).mean()
    deviation = data - sma
    mean_deviation = deviation.abs().rolling(window=period).mean()
    
    cci = (data - sma) / (0.015 * mean_deviation)
    return cci

# 7. Momentum
def momentum(data, period):
    return data.diff(period)

# 8. Media Móvil Ponderada (WMA)
def weighted_moving_average(data, period):
    weights = np.arange(1, period + 1)
    wma = data.rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    return wma

# 9. Desviación Estándar
def standard_deviation(data, period):
    return data.rolling(window=period).std()

def plot_indicadores(data):
    Close_data = data['Close']
    # Calcular los indicadores
    # Crear gráficos con Plotly
    fig = sp.make_subplots(rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=('','Precio de Cierre y Medias Móviles', 'RSI', 'MACD', 'Momentum, ROC','CCI'))

    # Gráfico de Precios de Cierre y Medias Móviles
    fig.add_trace(go.Scatter(x=Close_data.index, y=Close_data, mode='lines', name='Precio de Cierre', line=dict(color='grey')), row=1, col=2)
    fig.add_trace(go.Scatter(x=Close_data.index, y=data['SMA_14'], mode='lines', name='SMA (14)', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=Close_data.index, y=data['EMA_14'], mode='lines', name='EMA (14)', line=dict(color='green')), row=1, col=2)

    # Gráfico de RSI
    fig.add_trace(go.Scatter(x=Close_data.index, y=data['RSI_14'], mode='lines', name='RSI (14)', line=dict(color='purple')), row=2, col=1)
    fig.add_shape(type='line', x0=Close_data.index[0], x1=Close_data.index[-1], y0=70, y1=70, line=dict(color='red', dash='dash'), row=2, col=1)
    fig.add_shape(type='line', x0=Close_data.index[0], x1=Close_data.index[-1], y0=30, y1=30, line=dict(color='green', dash='dash'), row=2, col=1)

    # Gráfico de MACD
    fig.add_trace(go.Scatter(x=Close_data.index, y=data['MACD_Line'], mode='lines', name='Línea MACD', line=dict(color='blue')), row=2, col=2)
    fig.add_trace(go.Scatter(x=Close_data.index, y=data['Signal_Line'], mode='lines', name='Línea de Señal', line=dict(color='orange')), row=2, col=2)
    fig.add_trace(go.Bar(x=Close_data.index, y=data['MACD_Histogram'], name='Histograma MACD', marker_color='grey'), row=2, col=2)

    # Gráfico de Momentum, ROC, CCI
    fig.add_trace(go.Scatter(x=Close_data.index, y=data['Momentum_14'], mode='lines', name='Momentum (14)', line=dict(color='lightgreen')), row=3, col=1)
    fig.add_trace(go.Scatter(x=Close_data.index, y=data['ROC_14'], mode='lines', name='ROC (14)', line=dict(color='magenta')), row=3, col=1)
    fig.add_trace(go.Scatter(x=Close_data.index, y=data['CCI_14'], mode='lines', name='CCI (14)', line=dict(color='brown')), row=3, col=2)

    # Layout del gráfico
    fig.update_layout(title='Indicadores Técnicos',
                    xaxis_title='Fecha',
                    yaxis_title='Valor',
                    height=800,
                    showlegend=True,
                    legend=dict(x=0, y=1.0))

    fig.update_xaxes(rangeslider_visible=False)
    return fig

def analyze_residuals(prices: pd.Series):
    
    Resultados = {}
    
    # Calcular retornos logarítmicos
    returns = np.log(prices / prices.shift(1)).dropna()

    # Crear un DataFrame con los retornos
    data = pd.DataFrame({'Returns': returns})

    # Ajustar un modelo de regresión lineal simple
    data['Time'] = np.arange(len(data))
    X = sm.add_constant(data['Time'])
    model = sm.OLS(data['Returns'], X).fit()
    data['Fitted'] = model.fittedvalues
    data['Residuals'] = model.resid

    # Mostrar resumen del modelo
    #print("Resumen del Modelo:")
    Resultados['Resumen'] = model.summary()

    # Análisis de residuos
    residuals = data['Residuals']

    # 1. Normalidad de los residuos
    hist_fig = go.Histogram(x=residuals, nbinsx=20, name='Residuals', marker=dict(color='rgba(0, 123, 255, 0.7)'))

    # Prueba de normalidad de Shapiro-Wilk
    shapiro_stat, shapiro_p_value = shapiro(residuals)
    Resultados['Normalidad'] = {}
    Resultados['Normalidad']['Estadistico'] = shapiro_stat
    Resultados['Normalidad']['p-valor'] = shapiro_p_value
    Resultados['Normalidad']['Titulo'] = f"Prueba de Shapiro-Wilk: Estadístico = {shapiro_stat:.4f}, p-valor = {shapiro_p_value:.4f}"
    if shapiro_p_value > 0.05:
        Resultados['Normalidad']['Conclusion'] = "Interpretación: Los residuos siguen una distribución normal (p-valor > 0.05)."
    else:
        Resultados['Normalidad']['Conclusion'] = "Interpretación: Los residuos no siguen una distribución normal (p-valor <= 0.05)."

    # Gráfico Q-Q
    qq = probplot(residuals, dist="norm")
    qq_fig = go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Datos')
    qq_line = go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines', name='Línea de referencia')

    # 2. Homocedasticidad
    scatter_fig = go.Scatter(x=data['Fitted'], y=residuals, mode='markers', name='Residuos')
    scatter_line = go.Scatter(x=[data['Fitted'].min(), data['Fitted'].max()], y=[0, 0], mode='lines', name='Línea 0', line=dict(color='red', dash='dash'))

    # Prueba de heterocedasticidad de Breusch-Pagan
    bp_test = het_breuschpagan(residuals, X)
    
    Resultados['Heterocedasticidad'] = {}
    Resultados['Heterocedasticidad']['Estadistico'] = bp_test[0]
    Resultados['Heterocedasticidad']['p-valor'] = bp_test[1]
    Resultados['Heterocedasticidad']['Titulo'] = f"Prueba de Breusch-Pagan: Estadístico = {bp_test[0]:.4f}, p-valor = {bp_test[1]:.4f}"
    if bp_test[1] > 0.05:
        Resultados['Heterocedasticidad']['Conclusion']="Interpretación: No hay evidencia de heterocedasticidad (p-valor > 0.05)."
    else:
        Resultados['Heterocedasticidad']['Conclusion']="Interpretación: Hay evidencia de heterocedasticidad (p-valor <= 0.05)."

    # 3. Independencia y autocorrelación de residuos
    line_fig = go.Scatter(x=data.index, y=residuals, mode='lines', name='Serie Temporal de Residuos')

    # Correlograma de los residuos
    acf_values = acf(residuals, nlags=20)
    pacf_values = pacf(residuals, nlags=20)

    acf_fig = go.Bar(x=np.arange(len(acf_values)), y=acf_values, name='ACF', marker=dict(color='rgba(255, 99, 71, 0.7)'))
    pacf_fig = go.Bar(x=np.arange(len(pacf_values)), y=pacf_values, name='PACF', marker=dict(color='rgba(71, 99, 255, 0.7)'))

    # Prueba de autocorrelación de Breusch-Godfrey
    bg_test = acorr_breusch_godfrey(model, nlags=5)
    Resultados['Autocorrelacion'] = {}
    Resultados['Autocorrelacion']['Estadistico'] = bg_test[0]
    Resultados['Autocorrelacion']['p-valor'] = bg_test[1]
    Resultados['Autocorrelacion']['Titulo'] = f"Prueba de Breusch-Godfrey: Estadístico = {bg_test[0]:.4f}, p-valor = {bg_test[1]:.4f}"
    if bg_test[1] > 0.05:
        Resultados['Autocorrelacion']['Conclusion'] = "Interpretación: No hay evidencia de autocorrelación (p-valor > 0.05)."
    else:
        Resultados['Autocorrelacion']['Conclusion'] = "Interpretación: Hay evidencia de autocorrelación (p-valor <= 0.05)."

    # ADF Test para verificar si la serie de residuos es estacionaria
    adf_stat, adf_p_value, _, _, _, _ = adfuller(residuals)
    Resultados['Estacionaria'] = {}
    Resultados['Estacionaria']['Estadistico'] = adf_stat
    Resultados['Estacionaria']['p-valor'] = adf_p_value
    Resultados['Estacionaria']['Titulo'] = f"Prueba de ADF (Dickey-Fuller): Estadístico = {adf_stat:.4f}, p-valor = {adf_p_value:.4f}"
    if adf_p_value <= 0.05:
        Resultados['Estacionaria']['Conclusion']="Interpretación: Los residuos son estacionarios (p-valor <= 0.05)."
    else:
        Resultados['Estacionaria']['Conclusion']="Interpretación: Los residuos no son estacionarios (p-valor > 0.05)."

    # Crear subplots
    fig = make_subplots(rows=3, cols=2, 
                        subplot_titles=("Histograma de Residuos", "Gráfico Q-Q de Residuos",
                                        "Residuos vs. Valores Ajustados", "Serie Temporal de Residuos",
                                        "Correlograma ACF", "Correlograma PACF"))

    # Añadir gráficos
    fig.add_trace(hist_fig, row=1, col=1)
    fig.add_trace(qq_fig, row=1, col=2)
    fig.add_trace(qq_line, row=1, col=2)
    fig.add_trace(scatter_fig, row=2, col=1)
    fig.add_trace(scatter_line, row=2, col=1)
    fig.add_trace(line_fig, row=2, col=2)
    fig.add_trace(acf_fig, row=3, col=1)
    fig.add_trace(pacf_fig, row=3, col=2)

    # Actualizar layout
    fig.update_layout(title='Análisis de Residuos', template='plotly_white', showlegend=False, height=900)

    # Mostrar figura
    return fig, Resultados


def volatility_analisys(data):
    # Calcular rendimientos logarítmicos


    # Crear subplots para los gráficos
    fig = sp.make_subplots(rows=3, cols=2, subplot_titles=(
        'Volatilidad Histórica',
        'Bollinger Bands',
        'Average True Range (ATR)',
        'Chaikin Volatility',
        'Keltner Channels',
        'VIX (CBOE Volatility Index)'
    ))

    # Volatilidad Histórica
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Historical Volatility'], mode='lines', name='Volatilidad Histórica', line=dict(color='blue')),
        row=1, col=1
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Precio de Cierre', line=dict(color='black')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Upper Band'], mode='lines', name='Banda Superior', line=dict(color='red', dash='dash')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Middle Band'], mode='lines', name='Banda Media', line=dict(color='blue', dash='dash')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Lower Band'], mode='lines', name='Banda Inferior', line=dict(color='green', dash='dash')),
        row=1, col=2
    )

    # ATR
    fig.add_trace(
        go.Scatter(x=data.index, y=data['ATR'], mode='lines', name='ATR', line=dict(color='purple')),
        row=2, col=1
    )

    # Chaikin Volatility
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Chaikin Volatility'], mode='lines', name='Chaikin Volatility', line=dict(color='orange')),
        row=2, col=2
    )

    # Keltner Channels
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Precio de Cierre', line=dict(color='black')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Upper Keltner'], mode='lines', name='Keltner Superior', line=dict(color='red', dash='dash')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['EMA Close'], mode='lines', name='EMA', line=dict(color='blue', dash='dash')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Lower Keltner'], mode='lines', name='Keltner Inferior', line=dict(color='green', dash='dash')),
        row=3, col=1
    )

    # VIX
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], mode='lines', name='VIX', line=dict(color='magenta')),
        row=3, col=2
    )

    # Layout
    fig.update_layout(
        height=900,
        width=1200,
        title_text='Análisis de Volatilidad y Bandas de Precios para AAPL',
        showlegend=True
    )

    fig.update_xaxes(title_text='Fecha', row=1, col=1)
    fig.update_xaxes(title_text='Fecha', row=1, col=2)
    fig.update_xaxes(title_text='Fecha', row=2, col=1)
    fig.update_xaxes(title_text='Fecha', row=2, col=2)
    fig.update_xaxes(title_text='Fecha', row=3, col=1)
    fig.update_xaxes(title_text='Fecha', row=3, col=2)

    fig.update_yaxes(title_text='Volatilidad', row=1, col=1)
    fig.update_yaxes(title_text='Precio', row=1, col=2)
    fig.update_yaxes(title_text='ATR', row=2, col=1)
    fig.update_yaxes(title_text='Volatilidad', row=2, col=2)
    fig.update_yaxes(title_text='Precio', row=3, col=1)
    fig.update_yaxes(title_text='VIX', row=3, col=2)
    # Actualizar layout
    fig.update_layout(title='Análisis de Volatilidad', template='plotly_white', showlegend=False, height=900)
    return fig


def grafico_ganancia_volumen(df):
    # Crear un subplot con 2 filas
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Capital a lo largo del tiempo', 'Cantidad a lo largo del tiempo'),
                        vertical_spacing=0.1)

    # Añadir el gráfico de líneas para 'capital'
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Capital'], mode='lines', name='Capital'),
                row=1, col=1)

    # Añadir el gráfico de barras para 'cantidad'
    fig.add_trace(go.Bar(x=df['Date'], y=df['Cantidad'], name='Cantidad'),
                row=2, col=1)

    # Actualizar el layout
    fig.update_layout(title_text='Capital and Cantidad Over Time',
                    xaxis_title='', 
                    yaxis_title='Capital',
                    xaxis2_title='Date',
                    yaxis2_title='Cantidad',
                    height=600)
    
    return fig