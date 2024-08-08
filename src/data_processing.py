import yfinance as yf
from datetime import datetime
import numpy as np

from src.plots import simple_moving_average,exponential_moving_average,relative_strength_index,macd,rate_of_change,commodity_channel_index,momentum,weighted_moving_average,standard_deviation

def load_data(Ticket,fecha_inicio):
    # Descargar datos de acciones
    ahora = datetime.now().strftime('%Y-%m-%d')

    data = yf.download(Ticket, start=fecha_inicio, end=ahora,interval='1d')

    return data


def estrategia_cruce_medias(data):
    data['Estrategia_Cruce_Medias'] = 0
    data.loc[(data['SMA_14'] > data['EMA_14']), 'Estrategia_Cruce_Medias'] = 1
    data.loc[(data['SMA_14'] < data['EMA_14']), 'Estrategia_Cruce_Medias'] = -1
    return data


def estrategia_rsi(data):
    data['Estrategia_RSI'] = 0
    data.loc[(data['RSI_14'] < 30), 'Estrategia_RSI'] = 1
    data.loc[(data['RSI_14'] > 70), 'Estrategia_RSI'] = -1
    return data


def estrategia_macd(data):
    data['Estrategia_MACD'] = 0
    data.loc[(data['MACD_Line'] > data['Signal_Line']), 'Estrategia_MACD'] = 1
    data.loc[(data['MACD_Line'] < data['Signal_Line']), 'Estrategia_MACD'] = -1
    return data

def estrategia_bollinger(data):
    data['Estrategia_Bollinger'] = 0
    data.loc[(data['Close'] < data['Lower Band']), 'Estrategia_Bollinger'] = 1
    data.loc[(data['Close'] > data['Upper Band']), 'Estrategia_Bollinger'] = -1
    return data

def estrategia_mixta_cruce_medias_rsi(data):
    data['Estrategia_Mixta_Cruce_Medias_RSI'] = 0
    data.loc[(data['Estrategia_Cruce_Medias'] == 1) & (data['Estrategia_RSI'] == 1), 'Estrategia_Mixta_Cruce_Medias_RSI'] = 1
    data.loc[(data['Estrategia_Cruce_Medias'] == -1) & (data['Estrategia_RSI'] == -1), 'Estrategia_Mixta_Cruce_Medias_RSI'] = -1
    return data

def estrategia_mixta_cruce_medias_macd(data):
    data['Estrategia_Mixta_Cruce_Medias_MACD'] = 0
    data.loc[(data['Estrategia_Cruce_Medias'] == 1) & (data['Estrategia_MACD'] == 1), 'Estrategia_Mixta_Cruce_Medias_MACD'] = 1
    data.loc[(data['Estrategia_Cruce_Medias'] == -1) & (data['Estrategia_MACD'] == -1), 'Estrategia_Mixta_Cruce_Medias_MACD'] = -1
    return data

def estrategia_mixta_rsi_macd(data):
    data['Estrategia_Mixta_RSI_MACD'] = 0
    data.loc[(data['Estrategia_RSI'] == 1) & (data['Estrategia_MACD'] == 1), 'Estrategia_Mixta_RSI_MACD'] = 1
    data.loc[(data['Estrategia_RSI'] == -1) & (data['Estrategia_MACD'] == -1), 'Estrategia_Mixta_RSI_MACD'] = -1
    return data


def estrategia_mixta_cruce_medias_bollinger(data):
    data['Estrategia_Mixta_Cruce_Medias_Bollinger'] = 0
    data.loc[(data['Estrategia_Cruce_Medias'] == 1) & (data['Estrategia_Bollinger'] == 1), 'Estrategia_Mixta_Cruce_Medias_Bollinger'] = 1
    data.loc[(data['Estrategia_Cruce_Medias'] == -1) & (data['Estrategia_Bollinger'] == -1), 'Estrategia_Mixta_Cruce_Medias_Bollinger'] = -1
    return data

def estrategia_mixta_rsi_bollinger(data):
    data['Estrategia_Mixta_RSI_Bollinger'] = 0
    data.loc[(data['Estrategia_RSI'] == 1) & (data['Estrategia_Bollinger'] == 1), 'Estrategia_Mixta_RSI_Bollinger'] = 1
    data.loc[(data['Estrategia_RSI'] == -1) & (data['Estrategia_Bollinger'] == -1), 'Estrategia_Mixta_RSI_Bollinger'] = -1
    return data

def estrategia_mixta_macd_bollinger(data):
    data['Estrategia_Mixta_MACD_Bollinger'] = 0
    data.loc[(data['Estrategia_MACD'] == 1) & (data['Estrategia_Bollinger'] == 1), 'Estrategia_Mixta_MACD_Bollinger'] = 1
    data.loc[(data['Estrategia_MACD'] == -1) & (data['Estrategia_Bollinger'] == -1), 'Estrategia_Mixta_MACD_Bollinger'] = -1
    return data

def estrategia_mixta_cruce_medias_rsi_macd(data):
    data['Estrategia_Mixta_Cruce_Medias_RSI_MACD'] = 0
    data.loc[(data['Estrategia_Cruce_Medias'] == 1) & (data['Estrategia_RSI'] == 1) & (data['Estrategia_MACD'] == 1), 'Estrategia_Mixta_Cruce_Medias_RSI_MACD'] = 1
    data.loc[(data['Estrategia_Cruce_Medias'] == -1) & (data['Estrategia_RSI'] == -1) & (data['Estrategia_MACD'] == -1), 'Estrategia_Mixta_Cruce_Medias_RSI_MACD'] = -1
    return data

def estrategia_mixta_cruce_medias_rsi_bollinger(data):
    data['Estrategia_Mixta_Cruce_Medias_RSI_Bollinger'] = 0
    data.loc[(data['Estrategia_Cruce_Medias'] == 1) & (data['Estrategia_RSI'] == 1) & (data['Estrategia_Bollinger'] == 1), 'Estrategia_Mixta_Cruce_Medias_RSI_Bollinger'] = 1
    data.loc[(data['Estrategia_Cruce_Medias'] == -1) & (data['Estrategia_RSI'] == -1) & (data['Estrategia_Bollinger'] == -1), 'Estrategia_Mixta_Cruce_Medias_RSI_Bollinger'] = -1
    return data


def estrategia_mixta_cruce_medias_macd_bollinger(data):
    data['Estrategia_Mixta_Cruce_Medias_MACD_Bollinger'] = 0
    data.loc[(data['Estrategia_Cruce_Medias'] == 1) & (data['Estrategia_MACD'] == 1) & (data['Estrategia_Bollinger'] == 1), 'Estrategia_Mixta_Cruce_Medias_MACD_Bollinger'] = 1
    data.loc[(data['Estrategia_Cruce_Medias'] == -1) & (data['Estrategia_MACD'] == -1) & (data['Estrategia_Bollinger'] == -1), 'Estrategia_Mixta_Cruce_Medias_MACD_Bollinger'] = -1
    return data

def estrategia_mixta_rsi_macd_bollinger(data):
    data['Estrategia_Mixta_RSI_MACD_Bollinger'] = 0
    data.loc[(data['Estrategia_RSI'] == 1) & (data['Estrategia_MACD'] == 1) & (data['Estrategia_Bollinger'] == 1), 'Estrategia_Mixta_RSI_MACD_Bollinger'] = 1
    data.loc[(data['Estrategia_RSI'] == -1) & (data['Estrategia_MACD'] == -1) & (data['Estrategia_Bollinger'] == -1), 'Estrategia_Mixta_RSI_MACD_Bollinger'] = -1
    return data




def indicadores_data(data):
    data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))

    # Calcular Volatilidad Histórica
    data['Historical Volatility'] = data['Log Returns'].rolling(window=21).std() * np.sqrt(252)

    # Calcular Bollinger Bands
    data['Middle Band'] = data['Close'].rolling(window=20).mean()
    data['Upper Band'] = data['Middle Band'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower Band'] = data['Middle Band'] - 2 * data['Close'].rolling(window=20).std()

    # Calcular ATR (Average True Range)
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = np.abs(data['High'] - data['Close'].shift(1))
    data['Low-Close'] = np.abs(data['Low'] - data['Close'].shift(1))
    data['True Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    data['ATR'] = data['True Range'].rolling(window=14).mean()

    # Calcular Chaikin Volatility
    data['EMA High'] = data['High'].ewm(span=10, adjust=False).mean()
    data['EMA Low'] = data['Low'].ewm(span=10, adjust=False).mean()
    data['Chaikin Volatility'] = ((data['EMA High'] - data['EMA Low']) / data['EMA Low']).rolling(window=10).mean()

    # Calcular Keltner Channels
    data['EMA Close'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Upper Keltner'] = data['EMA Close'] + data['ATR'] * 2
    data['Lower Keltner'] = data['EMA Close'] - data['ATR'] * 2

    Close_data = data['Close']
    # Calcular los indicadores y almacenarlos en el dataframe
    data['SMA_14'] = simple_moving_average(Close_data, 14)
    data['EMA_14'] = exponential_moving_average(Close_data, 14)
    data['RSI_14'] = relative_strength_index(Close_data, 14)
    data['MACD_Line'], data['Signal_Line'], data['MACD_Histogram'] = macd(Close_data)
    data['ROC_14'] = rate_of_change(Close_data, 14)
    data['CCI_14'] = commodity_channel_index(Close_data, 14)
    data['Momentum_14'] = momentum(Close_data, 14)
    data['WMA_14'] = weighted_moving_average(Close_data, 14)
    data['STD_14'] = standard_deviation(Close_data, 14)    
    data = estrategia_cruce_medias(data)
    data = estrategia_rsi(data)
    data = estrategia_macd(data)
    data = estrategia_bollinger(data)
    # Aplicar las estrategias mixtas en pares
    data = estrategia_mixta_cruce_medias_rsi(data)
    data = estrategia_mixta_cruce_medias_macd(data)
    data = estrategia_mixta_cruce_medias_bollinger(data)
    data = estrategia_mixta_rsi_macd(data)
    data = estrategia_mixta_rsi_bollinger(data)
    data = estrategia_mixta_macd_bollinger(data)

    # Aplicar las estrategias mixtas en tripletas
    data = estrategia_mixta_cruce_medias_rsi_macd(data)
    data = estrategia_mixta_cruce_medias_rsi_bollinger(data)
    data = estrategia_mixta_cruce_medias_macd_bollinger(data)
    data = estrategia_mixta_rsi_macd_bollinger(data)
    return data

def indicadores_data_2(data):
    
    data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))

    # Calcular Volatilidad Histórica
    data['Historical Volatility'] = data['Log Returns'].rolling(window=21).std() * np.sqrt(252)

    # Calcular Bollinger Bands
    data['Middle Band'] = data['Close'].rolling(window=20).mean()
    data['Upper Band'] = data['Middle Band'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower Band'] = data['Middle Band'] - 2 * data['Close'].rolling(window=20).std()

    Close_data = data['Close']
    # Calcular los indicadores y almacenarlos en el dataframe
    data['SMA_14'] = simple_moving_average(Close_data, 14)
    data['EMA_14'] = exponential_moving_average(Close_data, 14)
    data['RSI_14'] = relative_strength_index(Close_data, 14)
    data['MACD_Line'], data['Signal_Line'], data['MACD_Histogram'] = macd(Close_data)
    data['ROC_14'] = rate_of_change(Close_data, 14)
    data['CCI_14'] = commodity_channel_index(Close_data, 14)
    data['Momentum_14'] = momentum(Close_data, 14)
    data['WMA_14'] = weighted_moving_average(Close_data, 14)
    data['STD_14'] = standard_deviation(Close_data, 14)    
    data = estrategia_cruce_medias(data)
    data = estrategia_rsi(data)
    data = estrategia_macd(data)
    data = estrategia_bollinger(data)
    # Aplicar las estrategias mixtas en pares
    data = estrategia_mixta_cruce_medias_rsi(data)
    data = estrategia_mixta_cruce_medias_macd(data)
    data = estrategia_mixta_cruce_medias_bollinger(data)
    data = estrategia_mixta_rsi_macd(data)
    data = estrategia_mixta_rsi_bollinger(data)
    data = estrategia_mixta_macd_bollinger(data)

    # Aplicar las estrategias mixtas en tripletas
    data = estrategia_mixta_cruce_medias_rsi_macd(data)
    data = estrategia_mixta_cruce_medias_rsi_bollinger(data)
    data = estrategia_mixta_cruce_medias_macd_bollinger(data)
    data = estrategia_mixta_rsi_macd_bollinger(data)
    return data


def estrategia(data,estrategia,capital_inicial,stop_loss,take_profit):
    copia_data = data.copy().reset_index()
    copia_data = copia_data[['Date','Close',estrategia]]
    copia_data['Capital'] = capital_inicial
    copia_data['Cantidad'] = 0
    copia_data['Transaccion'] = 0
    
    for i in range(1,len(data)):
        # Obtener la señal de la estrategia
        señal = copia_data[estrategia].iloc[i]
        
        if señal==1:
            if copia_data['Capital'].iloc[i-1] > copia_data['Close'].iloc[i]:
                copia_data.at[i, 'Cantidad'] = copia_data['Cantidad'].iloc[i-1] +1
                copia_data.at[i, 'Capital'] = copia_data['Capital'].iloc[i-1] - copia_data['Close'].iloc[i]
                copia_data.at[i, 'Transaccion'] = copia_data['Close'].iloc[i]*1 
        elif señal ==-1:
            if copia_data['Cantidad'].iloc[i-1] > 0:
                copia_data.at[i, 'Cantidad'] = copia_data['Cantidad'].iloc[i-1] -1
                copia_data.at[i, 'Capital'] = copia_data['Capital'].iloc[i-1] + copia_data['Close'].iloc[i]
                copia_data.at[i, 'Transaccion'] = -copia_data['Close'].iloc[i]*1 
            else: 
                copia_data.at[i, 'Capital'] = copia_data['Capital'].iloc[i-1]                
        else:
            copia_data.at[i, 'Cantidad'] = copia_data['Cantidad'].iloc[i-1]
            copia_data.at[i, 'Capital'] = copia_data['Capital'].iloc[i-1]            
    
    total = copia_data['Capital'].iloc[len(data)-1] + copia_data['Cantidad'].iloc[len(data)-1]*copia_data['Close'].iloc[len(data)-1]
    
    return copia_data, total

