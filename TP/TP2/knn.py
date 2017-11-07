import numpy as np 
import pandas as pd

# files
import os
import glob

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# knn libraries
from sklearn.base import clone
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# classes
class SBS():
    def __init__(self, estimator, k_features,
        scoring=accuracy_score,
        test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
        random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
        X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train,
                X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
        
    def _calc_score(self, X_train, y_train,
                        X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

# functions
def get_dataset(limit_files=None):

	extension = 'csv'
	os.chdir("../data")
	results = [i for i in glob.glob('*.{}'.format(extension))]	

	#join de varios datasets
	files = []
	for idx, result in enumerate(results):
	    file = pd.read_csv('../data/'+result, low_memory=False)
	    files.append(file)
	    if (limit_files is not None) and (idx > limit_files): #solo tomo limit_files archivos para no hacerlo tan pesado
	        break

	#la idea es contactenar varios dataframes y despues dropear las filas repetidas
	return pd.concat(files).drop_duplicates()

def transform_date(props):
    props['created_on_year'] = pd.to_datetime(props['created_on']).apply(lambda x: x.year)
    props['created_on_month'] = pd.to_datetime(props['created_on']).apply(lambda x: x.month)
    props['created_on_day'] = pd.to_datetime(props['created_on']).apply(lambda x: x.day)
    return props

def clean_dataset(props):
	# filtro por CABA y GBA
	properties_caba = props[(props['place_with_parent_names'].str.contains('Capital Federal') \
	                             | props['place_with_parent_names'].str.contains('Bs.As. G.B.A.'))]

	# queremos solo las propiedades que tienen precio y eliminamos columnas que sabemos que no son redundantes y que no nos servirian para knn
	properties_caba = properties_caba.drop(['currency','price_usd_per_m2','price_usd_per_m2','price_per_m2','price_aprox_usd','price_aprox_local_currency',\
	                      'id','operation','country_name','properati_url','image_thumbnail','description','title','surface_in_m2',\
	                      'lat-lon','geonames_id','created_on'], axis = 1)

	properties_caba = properties_caba[properties_caba['price'].notnull() & properties_caba['place_name'].notnull()]

	# eliminamos propiedades con mas de 54 pisos
	properties_caba = properties_caba[properties_caba['floor'] <= 54]
	# eliminamos propiedades con mas de 9 pisos - ver analisis
	properties_caba = properties_caba[properties_caba['rooms'] <= 9]
	# eliminamos propiedades con mas de 2000 m2 de superficie cubierta - ver analisis
	properties_caba = properties_caba[(properties_caba['surface_covered_in_m2'] <= 3000) & (properties_caba['surface_covered_in_m2'] >= 0)]
	# eliminamos propiedades con mas de 2000 m2 de superficie cubierta - ver analisis
	properties_caba = properties_caba[(properties_caba['surface_total_in_m2']<= 5000)  & (properties_caba['surface_total_in_m2'] >= 0)]

	return properties_caba

def encoder_attributes(data, encoder):
	#data['currency'] = encoder.fit_transform(data[['currency']])
	data['place_name'] = encoder.fit_transform(data[['place_name']])
	data['state_name'] = encoder.fit_transform(data[['state_name']])
	data['place_with_parent_names'] = encoder.fit_transform(data[['place_with_parent_names']])
	data['property_type'] = encoder.fit_transform(data[['property_type']])
	return data


#main