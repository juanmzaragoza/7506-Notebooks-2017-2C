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

import datetime
import warnings
warnings.filterwarnings('ignore')

############### OBTENER LOS DATOS ###############
print("LEYENDO DATASET...", end='')
properties = knnlibrary.get_dataset()
properties.head()

# completamos valores nan
imputer_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
properties['floor'] = properties[['floor']].fillna(1)
properties['rooms'] = properties[['rooms']].fillna(1)
print(".", end='')
properties["lat"] = imputer_mean.fit_transform(properties[["lat"]])
properties["lon"] = imputer_mean.fit_transform(properties[["lon"]])
print(".", end='')
properties.head()

# filtro por CABA y GBA
properties_caba = properties[(properties['place_with_parent_names'].str.contains('Capital Federal') \
                             | properties['place_with_parent_names'].str.contains('Bs.As. G.B.A.'))]
print(".", end='')
properties_caba = properties_caba[properties_caba['price'].notnull() & properties_caba['price'] > 0 & properties_caba['place_name'].notnull()]
print(".", end='')
# eliminamos propiedades con mas de 54 pisos
properties_caba = properties_caba[properties_caba['floor'] <= 60]
# eliminamos propiedades con mas de 9 pisos - ver analisis
properties_caba = properties_caba[properties_caba['rooms'] <=  10]
# eliminamos propiedades con mas de 2000 m2 de superficie cubierta - ver analisis
properties_caba = properties_caba[(properties_caba['surface_covered_in_m2'] <= 4000) & (properties_caba['surface_covered_in_m2'] >= 0)]
# eliminamos propiedades con mas de 2000 m2 de superficie cubierta - ver analisis
properties_caba = properties_caba[(properties_caba['surface_total_in_m2']<= 6000)  & (properties_caba['surface_total_in_m2'] >= 0)]
print(".", end='')
properties_caba = knnlibrary.transform_date(properties_caba)
properties_caba.info()

# filtro columnas segun lo que hay en el dataset
attributes = ['created_on_day','created_on_month','created_on_year','property_type','place_name','place_with_parent_names',\
              'country_name','state_name','lat','lon','surface_total_in_m2','surface_covered_in_m2',\
              'floor','rooms', 'price']
properties_caba[attributes].info()
properties_caba_with_price_attributes = properties_caba[attributes]
print(".", end='')


properties_caba_with_price_attributes.fillna('NaN', inplace=True)
# atributos categoricos
encoder = LabelEncoder()
print(".", end='')
properties_caba_with_price_attributes['property_type'] = encoder.fit_transform(properties_caba_with_price_attributes[['property_type']])
properties_caba_with_price_attributes['place_name'] = encoder.fit_transform(properties_caba_with_price_attributes[['place_name']])
properties_caba_with_price_attributes['place_with_parent_names'] = encoder.fit_transform(properties_caba_with_price_attributes[['place_with_parent_names']])
properties_caba_with_price_attributes['country_name'] = encoder.fit_transform(properties_caba_with_price_attributes[['country_name']])
properties_caba_with_price_attributes['state_name'] = encoder.fit_transform(properties_caba_with_price_attributes[['state_name']])
print(".", end='')
properties_caba_with_price_attributes.head()

# separamos el train de traing para validarlo luego usando un 20% de los datos
now = datetime.datetime.now()

columns = properties_caba_with_price_attributes.iloc[:, properties_caba_with_price_attributes.columns != 'price'].columns
print(".", end='')
X, y = properties_caba_with_price_attributes.iloc[:, properties_caba_with_price_attributes.columns != 'price'].values, properties_caba_with_price_attributes.iloc[:, properties_caba_with_price_attributes.columns == 'price'].values
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=50)
print(".", end='')
len(X_test), len(X_train)

# a cada dato le restamos la media y lo dividimos por su desviacion standard
stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print(".", end='')
X_std = stdsc.transform(X)
############### FIN OBTENER LOS DATOS ###############
print("El dataset se encuentra en memoria.")

############### ALGORITMO 1: KNN ###############
print("Entrenando KNN...", end='')
model_knn = KNeighborsRegressor(n_neighbors=22, weights='distance', p=2)
model_knn.fit(X_test_std, y_test)
print("Fin entrenamiento KNN")
############### FIN ALGORITMO 1: KNN ###############

############### ALGORITMO 2: RF1 ###############
print("Entrenando Random forest con parametros sin normalizar...", end='')
# Creando modelo con hiper-parametros encontrados
model_rf1 = RandomForestRegressor(n_estimators=2000, max_features='auto', max_depth=100,min_samples_leaf=5,min_samples_split=10 )
model_rf1.fit(X_train, y_train)
print("Fin entrenamiento RF1")
############### FIN ALGORITMO 2: RF1 ###############

############### ALGORITMO 3: RF2 ###############
print("Entrenando Random forest con parametros sin normalizar...", end='')
# Creando modelo con hiper-parametros encontrados
model_rf2 = RandomForestRegressor(n_estimators=2000, max_features='auto', max_depth=100,min_samples_leaf=5,min_samples_split=10 )
model_rf2.fit(X_train_std, y_train)
print("Fin entrenamiento RF1")
############### FIN ALGORITMO 3: RF2 ###############

print("Comenzando predicciones...", end='')
############### PREDICCIONES ###############
# leemos set de test
test_df = pd.read_csv('../data/test/properati_dataset_testing_noprice.csv', low_memory=False)
test_df.head()

# transformamos atributos categoricos
test_df['country_name'] = encoder.fit_transform(test_df[['country_name']])
test_df['place_name'] = encoder.fit_transform(test_df[['place_name']])
test_df['state_name'] = encoder.fit_transform(test_df[['state_name']])
test_df['place_with_parent_names'] = encoder.fit_transform(test_df[['place_with_parent_names']])
test_df['property_type'] = encoder.fit_transform(test_df[['property_type']])

# tranformamos fechas
X_test_df = knnlibrary.transform_date(test_df)
X_test_df = X_test_df[['created_on_day','created_on_month','created_on_year','property_type','place_name','place_with_parent_names',\
              'country_name','state_name','lat','lon','surface_total_in_m2','surface_covered_in_m2',\
              'floor','rooms']]

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
print("OK", end='')

# ensamble
y_final = []
for row1,row2,row3 in zip(y_knn, y_rf1, y_rf2):
    y_final.append((float(row1)+float(row2)+float(row3))/3)

# escribir al archivo
output = pd.DataFrame( data={"id":test_df["id"], "price_usd":y_final} )
output.to_csv( "../data/result/enmsables_finales_"+str(now)+".csv", index=False, quoting=3 )
print("Achivo","../data/result/enmsables_finales_"+str(now)+".csv","generado")
