##IMPORTACION DE LIBRERIAS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import holidays
from scipy import stats
from scipy.stats import pearsonr
%matplotlib inline


##EXPLORACION DE LOS DATOS
# Cargar los datos
file_path = "../data/Compiled_ABS_2023_V2_MaskedNames.csv"
df = pd.read_csv(file_path)
# Mostrar la estructura del DataFrame
print(df.head())
print(df.info())
# Mostrar dimension del dataframe
print(f'Número de registros: {df.shape[0]}')
print(f'Número de variables: {df.shape[1]}')
# Mostrar valores faltantes
print(df.isnull().sum())
# Estadística descriptiva de los datos
print(df.describe())


##LIMPIEZA DE LOS DATOS
# Eliminar columnas categóricas que no son útiles para la correlación
excluded_columns = ['ADP', 'LOB', 'Name', 'Coach', 'OM','Abs %']
df_numeric = df.drop(columns=excluded_columns)
# Conversion de formato fecha
df_numeric['Date'] = pd.to_datetime(df_numeric['Date'], errors='coerce')
# Sumar los totales por día para tener el compilado diario
df_daily = df_numeric.groupby('Date').sum().reset_index()
    
    ## Establecer caracteristicas temporales
# Establecer la primera columna "Date" como indice
df_daily.set_index('Date', inplace=True)
# Definir días festivos de Nicaragua usando la librería holidays
nicaragua_holidays = holidays.Nicaragua()
# Agregar características temporales al DataFrame
df_daily['Month'] = df_daily.index.month
df_daily['DayOfMonth'] = df_daily.index.day
df_daily['DayOfWeek'] = df_daily.index.dayofweek
df_daily['IsWeekend'] = df_daily['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df_daily['IsHoliday'] = df_daily.index.to_series().apply(lambda x: 1 if x in nicaragua_holidays else 0)
# Reiniciar los indices
df_daily.reset_index(inplace=True)

    ## Eliminacion de variables no correcionales
# Calcular la matriz de correlación
corr_matrix = df_daily.corr()
# Seleccionar características con alta correlación con respecto a 'Total Absent'
target_variable = 'Total Absent'
correlation_threshold = 0.3  # Umbral para considerar correlaciones significativas
significant_features = corr_matrix[abs(corr_matrix[target_variable]) > correlation_threshold][target_variable]
non_significant_features = corr_matrix[abs(corr_matrix[target_variable]) <= correlation_threshold][target_variable]
# Mostrar características significativamente correlacionadas
print('Características significativamente correlacionadas con Total Absent:')
print(significant_features)
# Mostrar características no significativamente correlacionadas
print('Características no significativamente correlacionadas con Total Absent:')
print(non_significant_features)
# Mostrar el dimension para cada una de las características en cada categoría
print(f'Número de características significativamente correlacionadas: {significant_features.shape[0]}')
print(f'Número de características no significativamente correlacionadas: {non_significant_features.shape[0]}')
# Crear un nuevo DataFrame con las características seleccionadas
selected_features = significant_features.index.tolist()
df_selected = df_daily[ selected_features]

    ## Manejo de valores atipicos
# Aplicar el análisis solo a las columnas correspondientes
for column in df_selected.select_dtypes(include=[np.number]).columns:
    mean = df_selected[column].mean()
    std_dev = df_selected[column].std()
    # Definir los límites inferior y superior usando 3 desviaciones estándar
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    # Filtrar los valores atípicos
    outliers = df_selected[(df_selected[column] < lower_bound) | (df_selected[column] > upper_bound)]
    outliers_column_only = outliers[[column]].reset_index()
# Seleccionar todas las columnas numéricas
numeric_columns = df_selected.select_dtypes(include=[float, int]).columns
# Definir los valores atipicos y su reemplazo con la mediana
def replace_outliers_with_zscore(df, columns):
    """
    Reemplaza los valores atípicos en las columnas especificadas del DataFrame con la mediana de cada columna.
    """
    for column in columns:
        # Calcular el z-score de cada valor en la columna
        z_scores = stats.zscore(df[column].dropna())
        # Calcular la mediana de la columna
        median_value = df[column].median()
        # Reemplazar los valores atípicos con la mediana
        df[column] = df[column].where((z_scores < 3) & (z_scores > -3), median_value)
# Reemplazar los valores atipicos con la mediana
replace_outliers_with_zscore(df_selected, numeric_columns)
# Guardar el DataFrame modificado en un nuevo archivo CSV
output_file_path = "../data/Compiled_ABS_2023.csv"
df_selected.to_csv(output_file_path, index=False)
# Mostrar las primeras filas del DataFrame modificado
print(df_selected.head())


##VISUALIZACIONES
# Visualizar la serie temporal de 'Total Absent'
plt.figure(figsize=(12, 6))
plt.plot(df_daily['Date'], df_daily['Total Absent'], label='Total Absent')
plt.title('Total Absent Over Time')
plt.xlabel('Date')
plt.ylabel('Total Absent')
plt.legend()
plt.grid(True)
plt.show()
# Histogramas para distribuciones de variables numéricas
df_selected.hist(bins=30, figsize=(20, 15))
plt.show()
# Diagramas de dispersión para relaciones entre variables
sns.pairplot(df_selected)
plt.show()
# Matriz de correlación 
correlation_matrix = df_selected.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
# Diagramas de caja y bigote para identificar valores atípicos
df_selected.plot(kind='box', subplots=True, layout=(int(len(df_selected.columns)/3), 3), figsize=(20, 15))
plt.show()