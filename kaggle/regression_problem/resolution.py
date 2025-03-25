import pandas as pd
import numpy as np


from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Necesario para IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


import matplotlib.pyplot as plt

# Cargar los conjuntos de datos
train_df = pd.read_csv("train_set.csv")
test_df = pd.read_csv("test_set.csv")
