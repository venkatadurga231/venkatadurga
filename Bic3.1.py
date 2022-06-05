#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pprint import pprint
import glob
import codecs
import json
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


# ## 1.1 Парсинг данных

# In[2]:


#чтение каждого файла
read_files = glob.glob("data/*.geojson")
output_list = []

for f in read_files:
    with open(f, "rb") as infile:
        output_list.append(json.load(infile))

with open("merged_file.json", "w") as outfile:
    json.dump(output_list, outfile)


# In[3]:


#Объединение файлов
with codecs.open('merged_file.json', 'r', 'utf-8') as json_file:  
    data = json.load(json_file)
    
df = pd.json_normalize(data, errors='ignore')
df


# In[4]:


#Нормализация данных
df = pd.json_normalize(data, record_path=['features', 'properties', 'vehicles', 'participants'], meta = [
    ['features', 'properties','id'],
    ['features', 'properties', 'tags'],
    ['features', 'properties', 'light'],
    ['features', 'properties', 'point'], #2 cols
    ['features', 'properties', 'nearby'],
    ['features', 'properties', 'region'],
    ['features', 'properties', 'address'],
    ['features', 'properties', 'weather'],
    ['features', 'properties', 'category'],
    ['features', 'properties', 'datetime'],
    ['features', 'properties', 'severity'],
    ['features', 'properties', 'vehicles', 'year'],
    ['features', 'properties', 'vehicles', 'brand'],
    ['features', 'properties', 'vehicles', 'color'],
    ['features', 'properties', 'vehicles', 'model'],
    ['features', 'properties', 'vehicles', 'category'],
    ['features', 'properties','dead_count'],
    ['features', 'properties','participants'],
    ['features', 'properties','injured_count'],
    ['features', 'properties','parent_region'],
    ['features', 'properties','road_conditions'],
    ['features', 'properties','participants_count'],
    ['features', 'properties','participant_categories'],
], errors='ignore')

df = pd.concat([df.drop('features.properties.point', axis=1), pd.DataFrame(df['features.properties.point'].tolist())], axis=1)
df


# In[5]:


df1 =  (df.set_index('features.properties.id')['features.properties.participants']
       .apply(pd.Series).stack()
         .apply(pd.Series).reset_index().drop('level_1',1))

df = df.merge(df1, how='left', on='features.properties.id')


# In[6]:


df=df.drop('features.properties.participants', axis=1)


# Рассмотрим количество пустых значений

# In[7]:


df.isna().sum()


# ## 1.2 Предобработка данных и выделение значимых атрибутов

# Все данных выгруженные и json представленны. Рассмотрим некоторые статистику и размерность

# In[8]:


#Заполнение недостаюших данных
df=df.fillna(0)


# In[9]:


df.shape


# In[10]:


df.info()


# Рассмотрим количество пустых значений после предобработки

# In[11]:


df.isna().sum()


# ### Предобработка перечеслений в наборе данных

# In[12]:


#Функция explode разворачивает списки в строки
df = df.explode('violations_x')
df = df.explode('features.properties.tags')
df = df.explode('features.properties.nearby')
df = df.explode('features.properties.weather')
df = df.explode('features.properties.road_conditions')
df = df.explode('features.properties.participant_categories')


# In[13]:


df


# In[14]:


#Удаление дубликатов
df=df.drop_duplicates(subset=['features.properties.id'])
df=df.fillna(0)


# In[15]:


df=df[df['features.properties.address']!=0]
df.reset_index(drop=True, inplace=True)


# In[16]:


#Вычисление частоты и количества ДТП
result = pd.merge(df, df.groupby(['features.properties.address']).size().sort_values(ascending=False).to_frame(), on="features.properties.address")
result.rename(columns={0: 'count'},inplace=True)
#result.groupby(['properties.address']).size().sort_values(ascending=False).to_frame()
df = result


# ### Определение наиболее важных атрибутов
# Чтобы найти наиболее значимые атрибуты, построим корреляцию Пирсона на тепловой карте

# In[17]:


# Фомирование корреляции Пирсона
corr=df.drop(['features.properties.id'], axis=1).corr()
plt.figure(figsize=(16, 16))

heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=20)


# Как видим выше наиболее значимыми атрибутами являются: `features.properties.dead_count`, `features.properties.injured_count`, `features.properties.participants_count`

# ## 1.3 Описание структуры набора данных
# 
# * "id": 384094, # идентификатор
# * "tags": ["Дорожно-транспортные происшествия"], # показатели с официального сайта ГИБДД
# * "light": "Светлое время суток", # время суток
# * "point": {"lat": 50.6039, "long": 36.5578}, # координаты
# * "nearby": [ "Нерегулируемый перекрёсток неравнозначных улиц (дорог)", "Жилые дома индивидуальной застройки"], # координаты
# * "region": "Белгород", # город/район
# * "address": "г Белгород, ул Сумская, 30", # адрес
# * "weather": ["Ясно"], # погода
# * "category": "Столкновение", # тип ДТП
# * "datetime": "2017-08-05 13:06:00", # дата и время
# * "severity": "Легкий", # тяжесть ДТП/вред здоровью
# * "vehicles": [ # участники – транспортные средства
# * 
# * "year": 2010, # год производства транспортного средства
# * "brand": "ВАЗ", # марка транспортного средства
# * "color": "Иные цвета", # цвет транспортного средства
# * "model": "Priora", # модель транспортного средства
# * "category": "С-класс (малый средний, компактный) до 4,3 м", # категория транспортного средства
# * "participants": [ # участники внутри транспортных средств
# * 
# * "role": "Водитель", # роль участника
# * "gender": "Женский", # пол участника
# * "violations": [], # нарушения правил участником
# * "health_status": "Раненый, находящийся...", # состояние здоровья участника
# * "years_of_driving_experience": 11 # стаж вождения участника (только у водителей)
# 
# * "dead_count": 0, # кол-во погибших в ДТП
# * "participants": [], # участники без транспортных средств (описание, как у участников внутри транспортных средств)
# * "injured_count": 2, # кол-во раненых в ДТП
# * "parent_region": "Белгородская область", # регион
# * "road_conditions": ["Сухое"], # состояние дорожного покрытия
# * "participants_count": 3, # кол-во участников ДТП
# * "participant_categories": ["Все участники", "Дети"] # категории участников

# In[18]:


df.isna().sum()


# ## 1.4 Формирование дополнительных атрибутов
# Формиование индекса будет производиться на основе количества ДТП, частоты и тяжести.

# In[19]:


#Формирование первичных полей
df['Hazard_level'] = None
count_places_max = df['count'].max()
injured_max = df['features.properties.injured_count'].max()
dead_max = df['features.properties.dead_count'].max()


# In[20]:


#Вычисление индекса
for i in range(len(df)):
    if df['features.properties.dead_count'][i] > 0:
        df['Hazard_level'][i] = (df['features.properties.injured_count'][i]+df['count'][i])/((injured_max+count_places_max)/2)/4
    else:
        df['Hazard_level'][i] = (df['features.properties.dead_count'][i]*100/dead_max)/100/2+0.5


# In[21]:


df.head()


# In[22]:


df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)


# ## Отчёт
# 
# * 1.1 Парсинг данных - Данные загруженны из папки data
# * 1.2 Предобработка данных и выделение значимых атрибутов - Данные предобработаны и выделенны наиболее значимые атрибуты
# * 1.3 Описание структуры набора данных - для каждого атибута представленно описание
# * 1.4 Формирование дополнительных атрибутов - дополнительный индекс на основе данных сформированн
