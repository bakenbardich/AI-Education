#%% md
# ##Abalon - определяем возраст морского ушка.
#%% md
# #Общее исследование данных
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = pd.read_csv('https://github.com/bakenbardich/eda_and_dev_tools/raw/main/abalone.csv')
#%%
df.head()
#%%
df.shape
#%% md
# В датасете 4177 строки и 9 столбцов
#%%
df.info()
#%% md
# Пропущенные значения Diameter, Whole weight и Shell,заменим их на медианные.
#%%
df['Diameter'].fillna(df['Diameter'].median(), inplace=True)
df['Whole weight'].fillna(df['Whole weight'].median(), inplace=True)
df['Shell weight'].fillna(df['Shell weight'].median(), inplace=True)
#%%
df.info()
#%%
df[df.duplicated()]
#%% md
# Повторяющиеся обьекты отсутствуют.
#%% md
# #Исследуем корреляцию
#%%
for i in ['pearson','spearman','kendall']:
    cr =df.corr(numeric_only=True,method=i)
    sns.heatmap(cr,annot=True,cmap='viridis')
    print(f'График корреляции {i}')
    plt.show()
#%% md
# Во всех методаx прослеживается сильная корреляция фичей междуй собой, что может создать проблему мультиколлинеарности при построении моделей ML, особенно линейных. В таких случаях может потребоваться удаление некоторых признаков или применение методов регуляризации.
# Корреляция фичей с таргетом равномерная на уровне 0.5
#%% md
# Рассмотрим как коррелирует единственная категориальная фича с таргетом с помощью ANOVA
#%%
rings = df[['Sex','Rings']]
#%%
rings_group_list = rings.groupby('Sex')['Rings'].apply(list)
#%%
rings_group_list
#%%
from scipy.stats import f_oneway
#%%
AnovaResults = f_oneway(*rings_group_list)

print('P-Value for Anova is: ', AnovaResults[1])

if AnovaResults[1] >= 0.05:
    print('Features are NOT correlated')
else:
    print('Features are correlated')
#%%
Data = []

for c1 in df.columns:
    for c2 in df.columns:
        if df[c1].dtype == 'object' and df[c2].dtype != 'object':
            CategoryGroupLists = df.groupby(c1)[c2].apply(list)
            AnovaResults = f_oneway(*CategoryGroupLists)

            if AnovaResults[1] >= 0.05:
                Data.append({'Category' : c1, 'Numerical' : c2, 'Is correlated' : 'No'})
            else:
                Data.append({'Category' : c1, 'Numerical' : c2, 'Is correlated' : 'Yes'})

AnovaRes = pd.DataFrame.from_dict(Data)
AnovaRes
#%% md
# Взаимосвязь между SEX и таргетом прослеживаеться , так же как и с остальными числовыми фичами
#%% md
# Распределение таргета и фичей 
#%%
sns.countplot(data=df,x = 'Sex',hue=df['Sex']);
#%% md
# В описании три уникальных значения SEX, скорее опечатка, поменяем f на F
#%%
df.loc[df['Sex'] == 'f', 'Sex'] = 'F'
#%%
df.hist(bins=25,figsize=(10,6),color='r');
#%% md
# Целевой возраст приближен к нормальному распределению, хоть и имеется небольшой хвост справа.
# В целом, нормальное распределение признака может облегчить анализ и интерпретацию данных, но необходимо оценивать его применимость к конкретному набору данных и использовать соответствующие методы для проверки и обработки данных, если они не соответствуют нормальному распределению.
# 
#%% md
# Так как у нас таргет-непрерывная переменная, давайте посмотрим на диаграммы рассеивания и оценим, как каждый признак влияет на переменную и есть ли вообще связь.
#%%
sns.pairplot(data=df);
#%% md
# Возраст морских ушек увеличивается с увеличением их длины, диаметра, высоты и веса.
# Общий вес почти линейно изменяется в зависимости от всех других характеристик, кроме возраста.
# Высота имеет наименьшую линейность с остальными функциями.
# Возраст больше всего коррелирует с весом раковины.
# Возраст меньше всего коррелирует со сбросом веса.
#%%

#%% md
# Размножим датасет для проведения тестов с Polars
#%%
df_big = df.copy()

for i in range(400):
    df_big = pd.concat([df_big, df])

df_big.to_csv("big.csv", index=False)
df_big.shape, df.shape
#%%
import polars as pl
#%%

df_pl = pl.read_csv("big.csv")
#%%

df_ = pd.read_csv("big.csv")
#%%

df_.query('Rings > 10')
#%%

df_pl.filter(pl.col('Rings')>10)
#%%

df_.groupby('Sex').agg({'Diameter' : 'mean', 'Height' : 'max'})
#%%

df_pl.group_by('Sex').agg([pl.mean('Diameter'),pl.max('Height')])
#%% md
# Очевидно ,что Polars классная библиотека=)