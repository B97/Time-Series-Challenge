import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 10)
import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.var_model import VAR



def train(input_data_path):
  
  # data processing (for details and justification check the analysis notebook and readme)
  df = pd.read_csv(input_data_path)
  df.loc[df['sales_quantity'] == 0, 'revenue'] = 0
  df = df.drop_duplicates()
  df1 = df[df['item_number'] == 80317483]
  df1['day'] = pd.to_datetime(df['day'])
  df1.index = df1.day
  df1 = df1.drop(columns = ['day'])
  
  # train on 70% of the data
  train = df1[:int(0.7*(len(df1)))]
  
  model = VAR(endog=train)
  model_fit = model.fit()
  
  return model_fit
  
