#!/usr/bin/env python
# coding: utf-8

# # Модуль 2
# ---

# Подключение библиотек

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# Загружаем данные

# In[2]:


df = pd.read_csv("c1_result.csv")


# In[3]:


df.head()


# In[4]:


df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df.info()


# ## Конструирование признаков

# Добавим временные признаки

# In[5]:


df["year"] = df["pickup_datetime"].apply(lambda x : x.year)
df["month"] = df["pickup_datetime"].apply(lambda x: x.month)
df["dayofweek"] = df["pickup_datetime"].apply(lambda x: x.dayofweek)
df["hour"] = df["pickup_datetime"].apply(lambda x : x.hour)


# In[6]:


df = df.drop("pickup_datetime", axis = 1)


# In[7]:


df.head()


# ## Визуализация

# In[8]:


plt.figure(figsize = (20, 10))
sns.heatmap(df.corr(), annot = True)


# #### График зависимостей аттрибутов на целевую переменную

# In[9]:


import numpy as np


# In[10]:


plt.figure(figsize = (10, 10))
sns.pointplot(y=np.sort(df["trip_duration"]), x = np.sort(df["passenger_count"]))


# In[11]:


sns.lineplot(y = np.sort(df["trip_duration"]), x = np.sort(df["maximum temperature"]))


# In[12]:


for pr in ["month", "year", "dayofweek", "hour"]:
    sns.scatterplot(y = np.sort(df["trip_duration"]), x = np.sort(df[pr]))
    plt.title(pr)
    plt.show()


# ## Разбиение набора данных

# In[13]:


X = df.drop("trip_duration", axis = 1) 
y = df["trip_duration"].array


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)


# In[15]:


from sklearn.metrics import r2_score, mean_absolute_error
def score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"R^2 : {r2}")
    print("-" * 20)
    print(f"MAE: {mae}")
    print("-" * 20)
    print()


# 1. `RandomForestRegressor`
# 2. `GradientBoostingRegressor`
# 3. `LinearRegression`

# In[16]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# #### RandomForestRegressor

# In[17]:


rfc = RandomForestRegressor(verbose=3, n_jobs=-1)
#Производим обучение
rfc.fit(X_train, y_train)
score(y_test, rfc.predict(X_test))


# In[18]:


rfc.score(X_test, y_test)


# ### GradientBoostingRegressor

# In[19]:


grb = GradientBoostingRegressor(verbose=3)
grb.fit(X_train, y_train)
score(y_test, grb.predict(X_test))


# In[20]:


grb.score(X_test, y_test)


# ### LinearRegression

# In[21]:


lr = LinearRegression()
lr.fit(X_train, y_train)
score(y_test, lr.predict(X_test))


# In[22]:


lr.score(X_test, y_test)


# # Вывод

# 1. Данные были разбиты на тренировочную и тестовую выборки
# 2. Данные провизувлизированы и был совершен пространственный анализ
# 3. Произведено обучение и регрессия на реальных данных. Лучшей моделью себя пока RFR(`RandomForestRegressor()`)
# 4. Был произведен Feature Engineering 
