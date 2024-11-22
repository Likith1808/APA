#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Import required libraries
import dask.dataframe as dd

# Load NYC taxi data (replace with the actual dataset path)
data = dd.read_csv("yellow_tripdata_2015-01.csv")

# Display basic information
print(data.head())
print(data.info())


# In[4]:


# Assuming columns 'PULocationID' and 'DOLocationID' represent coordinates
data = data[(data['PULocationID'] >= 1) & (data['DOLocationID'] >= 1)]


# In[5]:


data['trip_duration'] = (dd.to_datetime(data['tpep_dropoff_datetime']) - 
                         dd.to_datetime(data['tpep_pickup_datetime'])).dt.total_seconds()

# Remove invalid trip durations (e.g., <1 minute or >6 hours)
data = data[(data['trip_duration'] > 60) & (data['trip_duration'] <= 21600)]


# In[6]:


data['speed'] = data['trip_distance'] / (data['trip_duration'] / 3600)
data = data[(data['speed'] > 0) & (data['speed'] <= 100)]


# In[7]:


# Remove unrealistic trip distances
data = data[(data['trip_distance'] > 0) & (data['trip_distance'] <= 100)]


# In[8]:


# Remove invalid fare values
data = data[(data['fare_amount'] > 0) & (data['fare_amount'] <= 500)]


# In[9]:


# Remove remaining outliers
data = data[(data['trip_distance'] < 50) & (data['speed'] < 100) & (data['fare_amount'] < 500)]


# In[17]:


from sklearn.cluster import MiniBatchKMeans

kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
data_pd['pickup_cluster'] = kmeans.fit_predict(data_pd[['PULocationID', 'DOLocationID']])

# Group by time bin and cluster, and fill missing values
data_pd['pickup_count'] = data_pd.groupby(['pickup_cluster', 'time_bin'])['trip_distance'].transform('count')
data_pd['pickup_count'] = data_pd['pickup_count'].fillna(0)

# Display the result
print(data_pd[['pickup_cluster', 'time_bin', 'pickup_count']].head())


# In[18]:


data_pd['time_bin'] = pd.to_datetime(data_pd['tpep_pickup_datetime']).dt.floor('10min')


# In[13]:


print(data_pd.columns)


# In[31]:


# Group by time bin and cluster, and fill missing values
data_pd['pickup_count'] = data_pd.groupby(['pickup_cluster', 'time_bin'])['trip_distance'].transform('count')
data_pd['pickup_count'] = data_pd['pickup_count'].fillna(0)


# In[32]:


from scipy.fftpack import fft
import numpy as np

# Fourier Transform of pickup count
data_pd['fft_pickup'] = np.abs(fft(data_pd['pickup_count']))


# In[33]:


data_pd['sma'] = data_pd['pickup_count'].rolling(window=5).mean()


# In[34]:


weights = np.arange(1, 6)
data_pd['wma'] = data_pd['pickup_count'].rolling(window=5).apply(
    lambda x: np.dot(x, weights) / weights.sum(), raw=True)


# In[35]:


data_pd['ewma'] = data_pd['pickup_count'].ewm(span=5, adjust=False).mean()


# In[36]:


from sklearn.model_selection import train_test_split

# Select features and target
features = ['pickup_cluster', 'sma', 'wma', 'ewma', 'fft_pickup', 'trip_distance', 'speed']
target = 'pickup_count'

X = data_pd[features]
y = data_pd[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Replace NaNs with mean values
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train and evaluate the model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))


# In[1]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Assuming X and y are your features and target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Random Forest Regressor with reduced estimators and parallelism
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # n_jobs=-1 uses all available cores
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Calculate Mean Squared Error
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))


# In[2]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Assuming X and y are your features and target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# XGBoost Regressor with reduced estimators and parallelism
xg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42, n_jobs=-1)  # n_jobs=-1 uses all cores
xg.fit(X_train, y_train)
y_pred_xg = xg.predict(X_test)

# Calculate Mean Squared Error
print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xg))


# In[ ]:


results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "MSE": [
        mean_squared_error(y_test, y_pred_lr),
        mean_squared_error(y_test, y_pred_rf),
        mean_squared_error(y_test, y_pred_xg)
    ]
})
print(results)


# In[ ]:


def modified_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100  # +1 to avoid division by zero

print("Linear Regression Modified MAPE:", modified_mape(y_test, y_pred_lr))
print("Random Forest Modified MAPE:", modified_mape(y_test, y_pred_rf))
print("XGBoost Modified MAPE:", modified_mape(y_test, y_pred_xg))

