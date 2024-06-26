##IMPORTACION DE LIBRERIAS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt

##TRATAMIENTO DE LOS DATOS
# Carga de los datos 
file_path = '../data/Compiled_ABS_2023.csv'  # Reemplaza con la ruta correcta
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
# Normalizar la variable 'Total Absent'
scaler = MinMaxScaler(feature_range=(0, 1))
df_normalized = scaler.fit_transform(df[['Total Absent']])
# Convertir de nuevo a DataFrame
df_normalized = pd.DataFrame(df_normalized, index=df.index, columns=['Total Absent'])
# Crear la secuencias de datos
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
#Secuencia de prediccion
sequence_length = 15
X, y = create_sequences(df_normalized['Total Absent'].values, sequence_length)

##SEGMENTACION DE LOS DATOS
# Dividir en entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
# Reshape para ser compatible con LSTM/GRU/CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

##DEFINICION DE LOS MODELOS
# Definir arquitectura del modelo LSTM
def train_lstm(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=200, batch_size=15, validation_data=(X_test, y_test))
    return model
# Definir arquitectura del modelo GRU
def train_gru(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(GRU(128, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(0.3))
    model.add(GRU(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=200, batch_size=15, validation_data=(X_test, y_test))
    return model
# Definir arquitectura del modelo CNN 1D
def train_cnn(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(sequence_length, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=200, batch_size=15, validation_data=(X_test, y_test))
    return model
# Entrenar y evaluar los modelos
lstm_model = train_lstm(X_train, y_train, X_test, y_test)
gru_model = train_gru(X_train, y_train, X_test, y_test)
cnn_model = train_cnn(X_train, y_train, X_test, y_test)

##PREDICCIONES CON LOS 3 MODELOS
# Hacer predicciones con los modelos
y_pred_lstm = scaler.inverse_transform(lstm_model.predict(X_test))
y_pred_gru = scaler.inverse_transform(gru_model.predict(X_test))
y_pred_cnn = scaler.inverse_transform(cnn_model.predict(X_test))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
# Calcular el MSE para cada modelo
mse_lstm = mean_squared_error(y_test_rescaled, y_pred_lstm)
mse_gru = mean_squared_error(y_test_rescaled, y_pred_gru)
mse_cnn = mean_squared_error(y_test_rescaled, y_pred_cnn)
# Mostrar los resultados
print(f'MSE LSTM: {mse_lstm}')
print(f'MSE GRU: {mse_gru}')
print(f'MSE CNN: {mse_cnn}')

##VISUALIZACIONES
# Visualizar las predicciones vs los valores reales
plt.figure(figsize=(15, 5))
# Modelo LSTM
plt.subplot(1, 3, 1)
plt.plot(y_test_rescaled, label='Real')
plt.plot(y_pred_lstm, label='LSTM')
plt.title('LSTM Predicción vs Real')
plt.legend()
# Modelo GRU
plt.subplot(1, 3, 2)
plt.plot(y_test_rescaled, label='Real')
plt.plot(y_pred_gru, label='GRU')
plt.title('GRU Predicción vs Real')
plt.legend()
# Modelo CNN
plt.subplot(1, 3, 3)
plt.plot(y_test_rescaled, label='Real')
plt.plot(y_pred_cnn, label='CNN')
plt.title('CNN Predicción vs Real')
plt.legend()
# MOstrar todos los graficos
plt.show()