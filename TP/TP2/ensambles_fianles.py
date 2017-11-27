import numpy as np 
import pandas as pd

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# modules
import knn as knnlibrary
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import datetime
import warnings
warnings.filterwarnings('ignore')

############### OBTENER LOS DATOS ###############
print("LEYENDO DATASET...", end='')
properties =knnlibrary.get_dataset()
properties.head()

# transformo el campo fecha
properties_caba = knnlibrary.transform_date(properties)

properties_caba.info()

# filtro por CABA y GBA
# queremos solo las propiedades que tienen precio y eliminamos columnas que sabemos que no son 
#redundantes y que no nos servirian para knn
# eliminamos propiedades con mas de 54 pisos
properties_caba = knnlibrary.clean_dataset(properties)

# las expensas tienen demasiados nulos por lo que voy a eliminar esa columna
properties_caba = properties_caba.drop(['expenses'], axis = 1)

properties_caba.info()

# eliminamos filas con valores nulo
properties_caba = properties_caba.dropna(how='any')
properties_caba.info()

from sklearn.preprocessing import LabelEncoder

# atributos categoricos
encoder = LabelEncoder()
properties_caba = knnlibrary.encoder_attributes(properties_caba, encoder)

properties_caba.tail()

X, y = properties_caba.iloc[:, properties_caba.columns != 'price'].values, properties_caba.iloc[:, properties_caba.columns == 'price'].values

# a cada dato le restamos la media y lo dividimos por su desviacion standard
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)

############### FIN OBTENER LOS DATOS ###############
print("El dataset se encuentra en memoria.")

############### ALGORITMO 1: KNN ###############
print("Entrenando KNN...", end='')
model_knn = KNeighborsRegressor(n_neighbors=22, weights='distance', p=2)
model_knn.fit(X_std, y)
print("Fin entrenamiento KNN")
############### FIN ALGORITMO 1: KNN ###############

############### ALGORITMO 2: RF1 ###############
print("Entrenando Random forest con parametros sin normalizar...", end='')
# Creando modelo con hiper-parametros encontrados
model_rf1 = RandomForestRegressor(n_estimators=3000, max_features='auto', max_depth=100,min_samples_leaf=5,min_samples_split=10 )
model_rf1.fit(X, y)
print("Fin entrenamiento RF1")
############### FIN ALGORITMO 2: RF1 ###############

############### ALGORITMO 3: RF2 ###############
print("Entrenando Random forest con parametros normalizados...", end='')
# Creando modelo con hiper-parametros encontrados
model_rf2 = RandomForestRegressor(n_estimators=3000, max_features='auto', max_depth=100,min_samples_leaf=5,min_samples_split=10 )
model_rf2.fit(X_std, y)
print("Fin entrenamiento RF2")
############### FIN ALGORITMO 3: RF2 ###############

############### ALGORITMO 4: GRADIENT BOOSTING ###############
print("Entrenando Gradient Boosting...", end='')
model_gbr = GradientBoostingRegressor(max_depth=21, learning_rate=0.025, n_estimators=3000, max_features=7, 
                                 min_samples_split=200, subsample=0.95, random_state=10)

model_gbr.fit(X_std,y)
print("Fin entrenamiento Gradient Boostin")
############### FIN ALGORITMO 4: GRADIENT BOOSTING ###############

print("Comenzando predicciones...", end='')
############### PREDICCIONES ###############
# leemos set de test
test_df = pd.read_csv('../data/test/properati_dataset_testing_noprice.csv', low_memory=False)
test_df.head()

# transformamos atributos categoricos
test_df['place_name'] = encoder.fit_transform(test_df[['place_name']])
test_df['state_name'] = encoder.fit_transform(test_df[['state_name']])
test_df['place_with_parent_names'] = encoder.fit_transform(test_df[['place_with_parent_names']])
test_df['property_type'] = encoder.fit_transform(test_df[['property_type']])

# tranformamos fechas
X_test_df = knnlibrary.transform_date(test_df)
X_test_df = X_test_df[['floor', 'lat', 'lon', 'place_name', 'place_with_parent_names',
       'property_type', 'rooms', 'state_name', 'surface_covered_in_m2',
       'surface_total_in_m2', 'created_on_year', 'created_on_month',
       'created_on_day']]

# completamos valores nan
from sklearn.preprocessing import Imputer
imputer_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_test_df['floor'] = X_test_df[['floor']].fillna(1)
X_test_df['rooms'] = X_test_df[['rooms']].fillna(1)

X_test_df["surface_total_in_m2"] = imputer_mean.fit_transform(X_test_df[["surface_total_in_m2"]])
X_test_df["surface_covered_in_m2"] = imputer_mean.fit_transform(X_test_df[["surface_covered_in_m2"]])
X_test_df["lat"] = imputer_mean.fit_transform(X_test_df[["lat"]])
X_test_df["lon"] = imputer_mean.fit_transform(X_test_df[["lon"]])

X_test_std_df = stdsc.transform(X_test_df)

X_test_df.head()

print("Prediciendo modelo...", end='')
y_knn = model_knn.predict(X_test_std_df)
y_rf2 = model_rf2.predict(X_test_std_df)
y_rf1 = model_rf1.predict(X_test_df)
y_gbr = model_gbr.predict(X_test_std_df)
print("OK", end='')

# ensamble
y_final = []
for row1,row2,row3,row4 in zip(y_knn, y_rf1, y_rf2, y_gbr):
    y_final.append((float(row1)+float(row2)+float(row3)+float(row4))/4)

now = datetime.datetime.now()

# escribir al archivo
output = pd.DataFrame( data={"id":test_df["id"], "price_usd":y_final} )
output.to_csv( "../data/result/enmsables_finales_"+str(now)+".csv", index=False, quoting=3 )
print("Achivo","../data/result/enmsables_finales_"+str(now)+".csv","generado")
