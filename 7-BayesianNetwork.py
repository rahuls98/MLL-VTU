import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

data = pd.read_csv('datasets/7.csv')
data = data.replace("?", np.nan)

model = BayesianModel([
    ('age','trestbps'),
    ('age','fbs'),
    ('sex','trestbps'),
    ('exang','trestbps'),
    ('trestbps','heartdisease'),
    ('fbs','heartdisease'),
    ('heartdisease','restecg'),
    ('heartdisease','thalach'),
    ('heartdisease','chol')
])

model.fit(data, estimator=MaximumLikelihoodEstimator)
infer = VariableElimination(model)
#q = infer.query(variables=['heartdisease'], evidence={'trestbps':150, 'fbs':0})
q = infer.query(variables=['heartdisease'], evidence={'age':60, 'sex':1})

print(q)