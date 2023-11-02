#!/usr/bin/env python
# coding: utf-8

# In[50]:


get_ipython().system('pip install sktime')


# In[2]:


#For data manipulation and analysis
import pandas as pd

#For plotting and data visualization
import matplotlib.pyplot as plt
import plotly.express as px

#For numerical calculations
import numpy as np


# In[29]:


"""
1. Import datasets
2. parse_dates to convert column 0 into datetime
3. set the index to year
"""
sales = pd.read_csv('C:/Users/medinavien/Documents/ING Case/store_sales_per_category.csv', parse_dates=[0], index_col='year')
distance = pd.read_csv('C:/Users/medinavien/Documents/ING Case/store_distances_anonymized.csv')
#Merged csv of total annual sales per store and no. of establishments within 5km
establishments = pd.read_csv('C:/Users/medinavien/Documents/ING Case/total_sales_and_establishments.csv')


# In[4]:


#Data Exploration
sales.head


# In[5]:


#Data Exploration
sales.describe()


# In[33]:


sales_grouped = sales.groupby(['store_id', 'year'])[['Vodka','Tequila','Whiskey','Rum','Gin','Brandy','Other']].sum().reset_index()

# Define the categories for the loop
categories = ['Vodka', 'Tequila', 'Whiskey', 'Rum', 'Gin', 'Brandy', 'Other']

# Iterate through the categories and create figures
for category in categories:
    fig = px.line(sales_grouped, x='year', y=category, color='store_id', height=500, width=800, title=f"Total {category} Sales Annually")
    fig.show()


# In[28]:


#Data Exploration
sales_grouped.describe()


# In[30]:


#Visualize correlation between Total Annual Sales Vs. No. of universities within 5km
locations = ['university', 'supermarket', 'restaurant', 'church', 'gym', 'stadium']
for category in categories:
    for location in locations:
        title = f"Total {category} Sales vs. No. of {location} within 5km"
        fig = px.scatter(establishments, x=location, y=f'Sum of {category}', color='store_id', height=500, width=800, title=title)
        
        fig.show()


# In[8]:


fig15 = px.scatter(distance, x='store_id_1', y='store_id_2', color='distance', title="Distance between Store ID 1 and Store ID 2")


fig15.update_traces(marker=dict(size=5)) 
fig15.show()


# In[53]:


#For time series analysis and forecasting
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series

#Forecasting models from sktime
from sktime.forecasting.fbprophet import Prophet

#Model evaluation from sklearn
from sklearn.metrics import mean_absolute_error, r2_score


# In[47]:


#Separated each liquor category per annual sales and per Store ID
vodka = pd.read_csv('C:/Users/medinavien/Documents/ING Case/Vodka_sales.csv', parse_dates=[0], index_col='Year')
tequila = pd.read_csv('C:/Users/medinavien/Documents/ING Case/Tequila_sales.csv', parse_dates=[0], index_col='Year')
whiskey = pd.read_csv('C:/Users/medinavien/Documents/ING Case/Whiskey_sales.csv', parse_dates=[0], index_col='Year')
other = pd.read_csv('C:/Users/medinavien/Documents/ING Case/Other_sales.csv', parse_dates=[0], index_col='Year')
gin = pd.read_csv('C:/Users/medinavien/Documents/ING Case/Gin_sales.csv', parse_dates=[0], index_col='Year')
brandy = pd.read_csv('C:/Users/medinavien/Documents/ING Case/Brandy_sales.csv', parse_dates=[0], index_col='Year')
rum = pd.read_csv('C:/Users/medinavien/Documents/ING Case/Rum_sales.csv', parse_dates=[0], index_col='Year')


# In[59]:


"""
  Forecast time series data using an sktime forecaster and visualize the results for selected columns.

  Args:
      dataset (pd.DataFrame): Input time series DataFrame with datetime index.
      horizon (int): Forecast horizon.
      forecaster (sktime.forecasting): Configured forecaster.
      validation (bool, optional): Whether to perform validation. Defaults to False.
      confidence (float, optional): Confidence level. Defaults to 0.9.
      frequency (str, optional): Resampling frequency. Defaults to "D".
      exclude_columns (list of str, optional): List of column names to exclude from the visualization. Defaults to None.

  Returns:
      None
  """

def sktime_forecast(dataset, horizon, forecaster, validation=False, confidence=0.9, frequency="D"):

  #Adjust frequency
  forecast_df = dataset.resample(rule=frequency).mean(numeric_only=True)

  #Interpolate missing periods (if any)
  forecast_df = forecast_df.interpolate(method="time")

  columns_to_plot = forecast_df.columns

  for col in columns_to_plot:
      if validation:
          #Use train/test split to validate forecaster
          y = forecast_df[col]

          y_train, y_test = temporal_train_test_split(y, test_size=horizon)

          forecaster.fit(y_train)
          fh = y_test.index
          y_pred = forecaster.predict(fh)
          ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
          mae = mean_absolute_error(y_test, y_pred)
      else:
          #Make predictions beyond the dataset
          y = forecast_df[col].dropna()
          forecaster.fit(y)

          last_date = y.index.max()
          fh = pd.date_range(start=last_date, periods=horizon, freq=frequency)

          y_pred = forecaster.predict(fh)
          ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
          mae = None

      #Visualize results
      plt.plot(forecast_df[col].tail(horizon * 5), label="Actual", color="black")
      plt.plot(fh, y_pred, label="Predicted")
      plt.ylim(bottom=0)
      plt.legend()
      plt.grid(True)
      plt.title(f"{horizon} day forecast for Store ID {col} (MAE: {round(mae, 2) if mae is not None else 'N/A'}, Confidence: {confidence * 100}%)")
      plt.show()
  
      


# In[60]:


#Forecast wach Store ID total sales per year
#Vodka
forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
sktime_forecast(dataset=vodka, horizon=1825, forecaster=forecaster, validation=False)


# In[ ]:


#Forecast wach Store ID total sales per year
#Tequila
forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
sktime_forecast(dataset=tequila, horizon=1825, forecaster=forecaster, validation=False)


# In[ ]:


#Forecast wach Store ID total sales per year
#Whiskey
forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
sktime_forecast(dataset=whiskey, horizon=1825, forecaster=forecaster, validation=False)


# In[ ]:


#Forecast wach Store ID total sales per year
#Other
forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
sktime_forecast(dataset=other, horizon=1825, forecaster=forecaster, validation=False)


# In[ ]:


#Forecast wach Store ID total sales per year
#Gin
forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
sktime_forecast(dataset=gin, horizon=1825, forecaster=forecaster, validation=False)


# In[ ]:


#Forecast wach Store ID total sales per year
#Brandy
forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
sktime_forecast(dataset=brandy, horizon=1825, forecaster=forecaster, validation=False)


# In[ ]:


#Forecast wach Store ID total sales per year
#Rum
forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
sktime_forecast(dataset=rum, horizon=1825, forecaster=forecaster, validation=False)

