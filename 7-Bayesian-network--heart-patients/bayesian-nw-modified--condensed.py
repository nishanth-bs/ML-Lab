import pandas as pd
from urllib.request import urlopen
import pgmpy
#get data
"""
from bs4 import BeautifulSoup
nn = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names'
s  = BeautifulSoup(urlopen(nn).read() , 'html.parser')
print(s.prettify())
"""
Cleveland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
heartDisease = pd.read_csv(urlopen(Cleveland_data_URL), names = names) 

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'), 
                       ('exang', 'trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),
                      ('heartdisease','restecg'),('heartdisease','thalach'),('heartdisease','chol')])

# Learing CPDs using Maximum Likelihood Estimators
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
print(model.get_cpds('age'))
print(model.get_cpds('chol'))
print(model.get_cpds('sex'))
model.get_independencies()

# Doing exact inference using Variable Elimination
from pgmpy.inference import VariableElimination
HeartDisease_infer = VariableElimination(model)

# Computing the probability of bronc given smoke.
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 22})
print(q['heartdisease'])
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 128})
print(q['heartdisease'])

