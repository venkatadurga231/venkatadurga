#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
pd.set_option('display.max_columns', None)


# In[2]:


# Загрузка данных
df=pd.read_csv('result_data.csv')


# In[5]:


df.head()


# ## 2.1 Разбиение набора данных

# Разобъём набор данных таким образом, как это рекомендовано согласно документации `Sklearn`. А именно `30 на 70`. Как представленно в описании, такая выборка является оптимальной, поскольку абсолютное большинство данных должно находится при обучении модели, чтобы получить наиболее оптимизированную модель со стороны её точности
# 
# ### Стратификация
# При разделении стратифицируем данные, чтобы получить одинаковую в процентом соотношении выборку, чтобы не было перевеса на какой-то один класс и такая ситуация не повлияла на некорректное обучение модели

# In[6]:


X=df[['features.properties.dead_count', 'features.properties.injured_count', 'features.properties.participants_count']]
y=df['features.properties.severity']
#Получение выборок
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


# ## 2.3 Классифиткация 
# 
# Рассмотрим три модели классификации
# ### KNeighborsClassifier
# Классификация на основе соседей - это тип обучения на основе экземпляров или необобщающего обучения: он не пытается построить общую внутреннюю модель, а просто сохраняет экземпляры обучающих данных. Классификация вычисляется простым большинством голосов ближайших соседей каждой точки: точке запроса назначается класс данных, который имеет наибольшее количество представителей среди ближайших соседей точки.
# 
# ### RandomForestClassifier
# Случайный лес — это метаоценка, которая соответствует ряду классификаторов дерева решений для различных подвыборок набора данных и использует усреднение для повышения точности прогнозирования и контроля переобучения. Размер подвыборки управляется параметром max_samples, если bootstrap=True (по умолчанию), в противном случае для построения каждого дерева используется весь набор данных
# ### GaussianNB
# Наи́вный ба́йесовский классифика́тор — простой вероятностный классификатор, основанный на применении теоремы Байеса со строгими (наивными) предположениями о независимости. В зависимости от точной природы вероятностной модели, наивные байесовские классификаторы могут обучаться очень эффективно
# 
# ## Матрикики
# Рассмотрим две метрикики для оценивания модели классификации
# 
# ### accuracy f1-score
# Это гармоническое среднее значений точности и полноты. Возьмём её, потому что она дает лучшую оценку неправильно классифицированных случаев
# 
# ### macro avg f1-score
# 
# macro avg f1-score пожалуй, самый простой из многочисленных методов усреднения. Макроусредненная оценка F1 (или макрооценка F1) вычисляется путем взятия среднего арифметического (также известного как невзвешенное среднее) всех оценок F1 для каждого класса. Этот метод будет взят, поскольку он обрабатывает все классы одинаково, независимо от их значений поддержки

# ## 2.4 Обучение

# In[7]:


#Импорт моделей
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[8]:


#Обучение
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
preds=neigh.predict(X_test)
print(classification_report(preds, y_test))


# In[9]:


#Обучение
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_preds=rfc.predict(X_test)
print(classification_report(rfc_preds, y_test))


# In[10]:


#Обучение
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_preds=gnb.predict(X_test)
print(classification_report(gnb_preds, y_test))


# ### Вывод
# Наиболее оптимальной моделью будет `RandomForestClassifier` c accuracy f1-score = `0.68` и macro avg f1-score = `0.71`, поскольку по сравнению с другими он показал наилучший результат.

# ## 3.4 Feature Engineering
# 
# Преобразуем набор данных путём генерации новых данных с целью повышения точности классификатора и использование StandardScaler

# In[13]:


#Генерация данных
result = pd.merge(df, df.groupby(['features.properties.vehicles.brand']).size().sort_values().to_frame(), on='features.properties.vehicles.brand')
result.rename(columns={0: 'brand_count'}, inplace=True)
df = result


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


#Преобразщование с помощью StandardScaler
scaler = StandardScaler()
X=df[['features.properties.dead_count', 'features.properties.injured_count', 'features.properties.participants_count', 'brand_count']]
y=df['features.properties.severity']

#Получение выборок
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

#Обучение
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_preds=rfc.predict(X_test)
print(classification_report(rfc_preds, y_test))


# ## Выводы по Feature Engineering 
# Из резёльтатов выше, преобразование данных для Feature Engineering не привёло к улучшению модели

# ## Отчёт
# * 2.1 Разбиение набора данных - набор данныхз разбит на обучаюшую и тестовую выборки
# * 2.3 Классификация - выбраны 3 алгоритма классификации и метрики для их тестирования
# * 2.4 Обучение - произведена классификация по тяжести ДТП
# * 2.5 Feature Engineering - произведено обечение ещё раз на преобразованных данных
# 

# In[ ]:


# Сохранение данных
df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)

