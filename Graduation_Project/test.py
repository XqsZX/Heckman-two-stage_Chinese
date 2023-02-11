import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from scipy.stats import norm
import Heckman as HK
from statsmodels.discrete.discrete_model import Probit

file_data = r'C:\Users\15245\Desktop\data\test.csv'
plot = pd.read_csv(file_data, encoding='utf-8-sig')
print(plot.iloc[:, 2:5].describe())
print(plot.iloc[:, 5:8].describe())
print(plot.iloc[:, 8:11].describe())
print(plot.iloc[:, 11:14].describe())
print(plot.iloc[:, 14:17].describe())
