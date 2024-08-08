# Importing necessary libraries
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
from datetime import datetime
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm

from plotly.subplots import make_subplots
from scipy.stats import shapiro, probplot
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.tsa.stattools import adfuller, acf, pacf

from streamlit_option_menu import option_menu
import numpy as np

# Establece la semilla de aleatoriedad

from src import data_processing as data_processing
from src.plots import simple_moving_average,exponential_moving_average,relative_strength_index,macd,rate_of_change,commodity_channel_index,momentum,weighted_moving_average,standard_deviation
from src import plots as plots

st.set_page_config(
    page_title='Sistema Busqueda',
    page_icon='🔍',
    layout='wide'
)

estrategias_descripcion = {
    'Estrategia_Cruce_Medias': 'Compra cuando la SMA de 14 periodos cruza por encima de la EMA de 14 periodos. Venta cuando la SMA cruza por debajo de la EMA.',
    'Estrategia_RSI': 'Compra cuando el RSI de 14 periodos está por debajo de 30 (sobrevendido). Venta cuando el RSI está por encima de 70 (sobrecomprado).',
    'Estrategia_MACD': 'Compra cuando la línea MACD cruza por encima de la línea de señal. Venta cuando la línea MACD cruza por debajo de la línea de señal.',
    'Estrategia_Bollinger': 'Compra cuando el precio de cierre cruza por debajo de la banda inferior de Bollinger. Venta cuando el precio de cierre cruza por encima de la banda superior de Bollinger.',
    'Estrategia_Mixta_Cruce_Medias_RSI': 'Compra cuando la SMA de 14 periodos cruza por encima de la EMA de 14 periodos y el RSI de 14 periodos está por debajo de 30. Venta cuando la SMA cruza por debajo de la EMA y el RSI está por encima de 70.',
    'Estrategia_Mixta_Cruce_Medias_MACD': 'Compra cuando la SMA de 14 periodos cruza por encima de la EMA de 14 periodos y la línea MACD cruza por encima de la línea de señal. Venta cuando la SMA cruza por debajo de la EMA y la línea MACD cruza por debajo de la línea de señal.',
    'Estrategia_Mixta_Cruce_Medias_Bollinger': 'Compra cuando la SMA de 14 periodos cruza por encima de la EMA de 14 periodos y el precio de cierre cruza por debajo de la banda inferior de Bollinger. Venta cuando la SMA cruza por debajo de la EMA y el precio de cierre cruza por encima de la banda superior de Bollinger.',
    'Estrategia_Mixta_RSI_MACD': 'Compra cuando el RSI de 14 periodos está por debajo de 30 y la línea MACD cruza por encima de la línea de señal. Venta cuando el RSI está por encima de 70 y la línea MACD cruza por debajo de la línea de señal.',
    'Estrategia_Mixta_RSI_Bollinger': 'Compra cuando el RSI de 14 periodos está por debajo de 30 y el precio de cierre cruza por debajo de la banda inferior de Bollinger. Venta cuando el RSI está por encima de 70 y el precio de cierre cruza por encima de la banda superior de Bollinger.',
    'Estrategia_Mixta_MACD_Bollinger': 'Compra cuando la línea MACD cruza por encima de la línea de señal y el precio de cierre cruza por debajo de la banda inferior de Bollinger. Venta cuando la línea MACD cruza por debajo de la línea de señal y el precio de cierre cruza por encima de la banda superior de Bollinger.',
    'Estrategia_Mixta_Cruce_Medias_RSI_MACD': 'Compra cuando la SMA de 14 periodos cruza por encima de la EMA de 14 periodos, el RSI de 14 periodos está por debajo de 30 y la línea MACD cruza por encima de la línea de señal. Venta cuando la SMA cruza por debajo de la EMA, el RSI está por encima de 70 y la línea MACD cruza por debajo de la línea de señal.',
    'Estrategia_Mixta_Cruce_Medias_RSI_Bollinger': 'Compra cuando la SMA de 14 periodos cruza por encima de la EMA de 14 periodos, el RSI de 14 periodos está por debajo de 30 y el precio de cierre cruza por debajo de la banda inferior de Bollinger. Venta cuando la SMA cruza por debajo de la EMA, el RSI está por encima de 70 y el precio de cierre cruza por encima de la banda superior de Bollinger.',
    'Estrategia_Mixta_Cruce_Medias_MACD_Bollinger': 'Compra cuando la SMA de 14 periodos cruza por encima de la EMA de 14 periodos, la línea MACD cruza por encima de la línea de señal y el precio de cierre cruza por debajo de la banda inferior de Bollinger. Venta cuando la SMA cruza por debajo de la EMA, la línea MACD cruza por debajo de la línea de señal y el precio de cierre cruza por encima de la banda superior de Bollinger.',
    'Estrategia_Mixta_RSI_MACD_Bollinger': 'Compra cuando el RSI de 14 periodos está por debajo de 30, la línea MACD cruza por encima de la línea de señal y el precio de cierre cruza por debajo de la banda inferior de Bollinger. Venta cuando el RSI está por encima de 70, la línea MACD cruza por debajo de la línea de señal y el precio de cierre cruza por encima de la banda superior de Bollinger.'
}


estrategias = [
    'Estrategia_Cruce_Medias',
    'Estrategia_RSI',
    'Estrategia_MACD',
    'Estrategia_Bollinger',
    'Estrategia_Mixta_Cruce_Medias_RSI',
    'Estrategia_Mixta_Cruce_Medias_MACD',
    'Estrategia_Mixta_Cruce_Medias_Bollinger',
    'Estrategia_Mixta_RSI_MACD',
    'Estrategia_Mixta_RSI_Bollinger',
    'Estrategia_Mixta_MACD_Bollinger',
    'Estrategia_Mixta_Cruce_Medias_RSI_MACD',
    'Estrategia_Mixta_Cruce_Medias_RSI_Bollinger',
    'Estrategia_Mixta_Cruce_Medias_MACD_Bollinger',
    'Estrategia_Mixta_RSI_MACD_Bollinger'
]


st.title('Aplicacion del proceso de Hawkes en series Financieras')

#Ticket = "QQQ"  # Cambia esto por el símbolo de la acción que desees
opcion1 ='Data Historica'
opcion2 = 'Eventos en la Serie'
opcion3 = 'Simulaciones y Predicciones'
opcion4 = 'Análisis de Residuos'
opcion5 = 'Indicadores Financieros'
opcion6 = 'Volatilidad'

selected = option_menu(
    menu_title=None,
    options = [opcion1,opcion2,opcion3,opcion4,opcion5,opcion6],
    icons= ['bar-chart-line','bar-chart-line','bar-chart-line','bar-chart-line','bar-chart-line','bar-chart-line'],
    orientation='horizontal'
)

st.sidebar.subheader('Fecha inicial y Ticket')

fecha_inicio = st.sidebar.date_input(
    "Fecha Inicial",
    datetime.now() - relativedelta(months=13)
)

Ticket = st.sidebar.selectbox('Ticket: ',['QQQ','BTC-USD'])

st.sidebar.subheader('Parametro Aleatoriedad')
semilla = st.sidebar.number_input('Ingrese Semilla Aleatoriedad:',min_value=1)

st.sidebar.write('---')
st.sidebar.write('---')
st.sidebar.subheader('Autores')
st.sidebar.markdown('Luis Lapo')
st.sidebar.markdown('Mathew Cisneros')
st.sidebar.markdown('Cristian Calle')


if Ticket == 'QQQ':
    np.random.seed(semilla)
else:
    np.random.seed(semilla*2)
data = data_processing.load_data(Ticket,fecha_inicio)
data = data_processing.indicadores_data(data)
estrategia = 'Estrategia_MACD'
capital_inicial = 10000  # Capital inicial en dólares
stop_loss = 0.05  # 5% de stop loss
take_profit = 0.10  # 10% de take profit

resultado = data_processing.estrategia(data, estrategia, capital_inicial, stop_loss, take_profit)
#st.write(resultado)
df = resultado


# Mostrar el gráfico
#st.plotly_chart(fig)
#st.write(data)
threshold = 0.01#= st.select_slider('Umbral',options=[0.01,0.02,0.03,0.04,0.05],value=0.01)

if selected == opcion1:
    if st.checkbox('Mostrar data'):
        st.subheader('Data diaria Obtenida de Yahoo Finance')
        st.write(data.head())

    st.subheader(f'Diagrama de velas japonesas de la data historica del indicador {Ticket} desde {fecha_inicio} hasta el día de hoy')
    fig = plots.create_candlestick_volume_chart(data,Ticket)

    st.plotly_chart(fig)


elif selected == opcion2:

    fig = plots.create_event_detection_chart(data, threshold)

    st.plotly_chart(fig)



elif selected == opcion3:

    st.subheader('Parametros para la función de intensidad')
    col1,col2,col3=st.columns(3)
    # Parametros iniciales

    ahora = datetime.now().strftime('%Y-%m-%d')
    lambda_0 = col1.number_input('Tase de Intensidad base(nu): ',min_value=0.01,value=0.5)
    alpha = col2.number_input('Inpacto del evento: (alpha): ',min_value=0.01,value=0.8)
    beta = col3.number_input('Tasa de decaimiento: (beta): ',min_value=alpha,value=1.5)
    # data
    # threshold
    st.subheader('Cantidad de simulaciones y predicciones')
    c1,c2 = st.columns(2)
    num_simulations = c2.number_input('Numero de predicciones',min_value=1,max_value=10,step=1)
    num_simulations_tests = c1.number_input('Numero de simulaciones',min_value=1,max_value=10,step=1)

    combined_fig,all_simulations_tests = plots.grafico_simulacion_pred(Ticket,data,threshold,ahora,lambda_0,alpha,beta,num_simulations,num_simulations_tests)
    # Mostrar el gráfico combinado
    st.plotly_chart(combined_fig)
    
    contador = 0
    st.title('Análisis de inversiones dadas diferentes estrategias (RESUMEN GENERAL)')
    if Ticket == 'QQQ':
        min_value = 10000
    else:
        min_value = 1000000
    capital_inicial = st.number_input('Capital Inicial',min_value=min_value)
    resultados_generales = {}
    for key,value in estrategias_descripcion.items():
    
        resultados_generales[key] = []
        for idx, (dates_tests, prices_tests, simulated_events_tests) in enumerate(all_simulations_tests):
            # Añadir la línea de precios de cierre para cada simulación
            data1= pd.DataFrame(pd.Series(prices_tests[:-1],index=dates_tests,name='Close'))
            data1 = data_processing.indicadores_data_2(data1)
            columnas = data1.columns.tolist()
            columnas[0] = 'Close_Simulado'
            data1.columns = columnas
            data1 = pd.merge(data1,data[['Close']],left_index=True, right_index=True, how='left').dropna(subset='Close')
            data1.index.name = 'Date'
            data1.reset_index(inplace=True)
            
            data1, resultado = data_processing.estrategia(data1, key, capital_inicial, stop_loss, take_profit)
            resultados_generales[key].append(round(resultado,2))
            contador += 1
            #st.plotly_chart(plots.grafico_ganancia_volumen(data1))
            
    # Definir una función para aplicar estilos
    # Definir una función para aplicar estilos
# Definir una función para aplicar estilos
    def highlight_greater_less(s, threshold):
        return ['background-color: lightgreen' if v > threshold else 'background-color: pink' if v < threshold else '' for v in s]
         
    
    resultados_generales = pd.DataFrame(resultados_generales).transpose()
    styled_df = resultados_generales.round(2).style.apply(highlight_greater_less, subset=[i for i in range(len(all_simulations_tests))],threshold= capital_inicial).format(precision=2)


    # Mostrar el DataFrame estilizado en Streamlit
    #st.dataframe(styled_df)
    st.write(styled_df)
    #st.write()
            #st.metric('Capital Final',round(resultado,2))
        
    st.title('Análisis Individual de estrategias')

    resultados_estrategia = {}
    column1,column2 = st.columns(2)
    numero_simulacion = column1.number_input('Seleccionar simulacion',min_value=1,max_value=len(all_simulations_tests))
    estrategia = column2.selectbox('Seleccionar Estrategia',options=estrategias_descripcion.keys())
    contador = 1
    st.subheader(f'**Estrategia Seleccionada:** {estrategia}')
    st.write(estrategias_descripcion[estrategia])
    for idx, (dates_tests, prices_tests, simulated_events_tests) in enumerate(all_simulations_tests):
        # Añadir la línea de precios de cierre para cada simulación
        if contador == numero_simulacion:
            data1= pd.DataFrame(pd.Series(prices_tests[:-1],index=dates_tests,name='Close'))
            data1 = data_processing.indicadores_data_2(data1)
            columnas = data1.columns.tolist()
            columnas[0] = 'Close_Simulado'
            data1.columns = columnas
            data1 = pd.merge(data1,data[['Close']],left_index=True, right_index=True, how='left').dropna(subset='Close')
            data1.index.name = 'Date'
            data1.reset_index(inplace=True)
            
            data1, resultado = data_processing.estrategia(data1, estrategia, capital_inicial, stop_loss, take_profit)
            resultados_estrategia[f'Simulacion {contador}'] = resultado
            st.plotly_chart(plots.grafico_ganancia_volumen(data1))
            
            st.metric('Capital Final',round(resultado,2))
        contador += 1
            
elif selected == opcion4:

    returns = data['Close'].pct_change()

    st.subheader('Analisis de Residuos')
    grafico, Resultados = plots.analyze_residuals(returns)
    st.plotly_chart(grafico)
    #st.write(Resultados['Resumen'])
    st.subheader('Normalidad')
    st.write(Resultados['Normalidad']['Titulo'])
    st.write(Resultados['Normalidad']['Conclusion'])
    st.subheader('Heterocedasticidad')
    st.write(Resultados['Heterocedasticidad']['Titulo'])
    st.write(Resultados['Heterocedasticidad']['Conclusion'])
    st.subheader('Autocorrelacion')
    st.write(Resultados['Autocorrelacion']['Titulo'])
    st.write(Resultados['Autocorrelacion']['Conclusion'])
    st.subheader('Estacionaria')
    st.write(Resultados['Estacionaria']['Titulo'])
    st.write(Resultados['Estacionaria']['Conclusion'])

elif selected == opcion6:
    st.subheader('Analisis de Volatilidad')
    st.plotly_chart(plots.volatility_analisys(data))
    
elif selected == opcion5:
    st.subheader('Indicadores Financieros')
    st.plotly_chart(plots.plot_indicadores(data))