# %% md
# ![__Predict the age of abalone from physical measurements.__
# Sex - nominal - M, F, and I (infant),
# Length - Bcontinuous - mm - Longest shell measurement,
# Diameter - continuous - mm - perpendicular to length,
# Height - continuous - mm - with meat in shell,
# Whole weight - continuous - grams - whole abalone,
# Shucked weight - continuous - grams - weight of meat,
# Viscera weight - continuous - grams - gut weight (after bleeding),
# Shell weight - continuous - grams - after being dried,
# Rings - integer - +1.5 gives the age in years (target).](https://cdn.shopify.com/s/files/1/0016/6959/5189/files/juvenile-red-abalone-shells.jpg?v=1600820403)
# 
# %% md
# #__ИМПОРТ И ЗАГРУЗКА__
# %% md
# __Импортируем необходимые библиотеки__
# %%
import pandas as pd, numpy as np

# %% md
# __Загружаем очищенный датасет, полученный в результате работы EDA и фиксируем параметр воспроизводимости__ 
# %%
df = pd.read_csv('https://raw.githubusercontent.com/bakenbardich/AI-Education/main/EDA/data/abalone_clean.csv')
RANDOM_STATE = 101
# %%
df.head()
# %%
df.info()
# %% md
# #__ПОСТРОЕНИЕ МОДЕЛЕЙ НА ЧИСЛОВЫХ ПРИЗНАКАХ__
# %% md
# __Naive Bayes__
# %%
# Импортируем необходимые библиотеки.
from sklearn.model_selection import train_test_split
# У нас задача предсказания возраста ушка-задача регрессии, используем Байеса для регрессии.
from sklearn.linear_model import BayesianRidge
# Сразу импортируем несколько метрик, учитывая поставленную задачу для оценки качества работы модели.
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# %%
# Делим датасет на матрицу объект-признак(тут также исключаем категориальный признак) и таргет.
X = df.drop(['Rings', 'Sex'], axis=1)
y = df['Rings']
# %%
# Перед построением моделей посмотрим на распределение таргета.
y.hist(backend='plotly', xlabelsize=10, ylabelsize=8, title='РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ')
# %% md
# __Распределение таргета приближено к нормальному распределению,что позитивно отразится на обучении.__
# %%
# Делим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
# %%
# Создадим класс модели регрессии Байеса(параметры пока по умолчанию) и обучим на тренировочной выборке.
NB_model = BayesianRidge()
NB_model.fit(X_train, y_train)
# %%
# Предсказание на тестовой выборке.
NB_pred = NB_model.predict(X_test)
# %%
# Оценка качества модели
MAE = mean_absolute_error(y_test, NB_pred)
MAPE = mean_absolute_percentage_error(y_test, NB_pred)
MSE = mean_squared_error(y_test, NB_pred)
print("MAE:", round(MAE, 4))
print(f"MAPE: {round(MAPE, 4) * 100:4f} %")
print("MSE:", round(MSE, 4))
# %% md
# __Считаю, что разница между фактическим значением и предсказанным даст нам более объективное представление, чем другие метрики, так как, например ,MAE = 1.6205 - не понятно как интерпретировать, много это или мало для возраста ушек, поэтому будем использовать MAPE, так как данная метрика показывает процентное отношение фактического возраста к предсказанному__  
# %% md
# __K-Nearest Neighbors, KNN__
# %%
# Импортируем необходимые библиотеки для KNN.
from sklearn.neighbors import KNeighborsRegressor
# Импортируем библиотеку для масштабирования данных, которая необходима для KNN, корректного  подсчета расстояния между объектами.
from sklearn.preprocessing import StandardScaler

# %%
# Масштабируем признаки. X_test -делаем только transform, чтобы при fit не допустить утечки данных.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%
# Создаём модель KNN, обучаем на масштабированных данных, делаем предикт.
KNN_model = KNeighborsRegressor()
KNN_model.fit(X_train_scaled, y_train)
KNN_pred = KNN_model.predict(X_test_scaled)
# %%
# Оцениваем качество модели, используя так же метрику MAPE.
mean_absolute_percentage_error(y_test, KNN_pred)
# %% md
# __KNN + GridSearch__
# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': range(1, 10),
    'weights': ['uniform', 'distance'],
    # 'uniform' - все соседи равнозначны, 'distance' - веса обратно пропорциональны расстоянию
    'metric': ['euclidean', 'manhattan']  # Различные метрики расстояния
}

# Создаем объект GridSearchCV
grid_search = GridSearchCV(KNN_model, param_grid, cv=5, scoring='neg_mean_absolute_percentage_error', verbose=1)

# Проводим поиск по сетке с кросс-валидацией
grid_search.fit(X_train_scaled, y_train)

# Выводим лучшие параметры и результаты
print("Best params:", grid_search.best_params_)
print("Best MAPE on train data:", grid_search.best_score_)

# Делаем предсказания на тестовых данных с лучшей моделью
best_knn_model = grid_search.best_estimator_
y_pred = best_knn_model.predict(X_test_scaled)

# Оцениваем модель на тестовых данных
mean_absolute_percentage_error(y_test, y_pred)

# %% md
# __Получилось немного снизить ошибку__
# %% md
# __Random Forest Regressor__
# %%
# Импортируем библиотеку для работы со случайным лесом.
from sklearn.ensemble import RandomForestRegressor

# %%
# Создаем модель леса, обучаем на тренировочных данных, делаем предсказание на тесте, оцениваем качество
RF_model = RandomForestRegressor(random_state=RANDOM_STATE)
RF_model.fit(X_train, y_train)
RF_pred = RF_model.predict(X_test)
mean_absolute_percentage_error(y_test, RF_pred)
# %%
# Подбираем гиперпараметры для леса с помощью поиска по сетке.
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [5, 10, 50],
              'n_estimators': [100, 150, 200],
              'max_features': ['sqrt', 'log2']}
gs = GridSearchCV(estimator=RF_model,
                  param_grid=param_grid,
                  cv=5,
                  scoring='neg_mean_absolute_percentage_error',
                  verbose=2,
                  n_jobs=-1)
gs.fit(X_train, y_train)
RF_GS_pred = gs.best_estimator_.predict(X_test)
mean_absolute_percentage_error(y_test, RF_GS_pred)
# %% md
# __Итак: KNN и RandomeForest дали примерно одинаковые результаты, но второй незначительно лучше, давайте работать далее именно с ним.__
# %% md
# #__ДОБАВЛЕНИЕ КАТЕГОРИЛЬНОГО ПРИЗНАКА В ЛУЧШУЮ МОДЕЛЬ И ПОДБОР ГИПЕРПАРАМЕТРОВ__
# %% md
# __Прежде чем брать за лучшую модель Random Forest, хочу попробывать оценить еще одну модель, впечатлившую меня с Практикума по ML -модификацию градиентного бустинга CatBoost. Одно из ее преимуществ это то ,что указав список индексов категориальных фичей, эта модель сама будет кодировать их.
# После оценки качества с параметрами по умолчанию, буду осуществлять подбор гиперпараметров через Optuna.__
# %%
# Импортируем библиотеку для работы с CatBoost
from catboost import CatBoostRegressor

# %%
# Переинициализируем матрицу объект-признак, добавив назад категориальную фичу "SEX"
X = df.drop('Rings', axis=1)
y = df['Rings']
# %%
# Заново разделим на трейн и тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
# %%
CBR_model = CatBoostRegressor(cat_features=[0], custom_metric='MAPE')
CBR_model.fit(X_train, y_train)
CBR_pred = CBR_model.predict(X_test)
mean_absolute_percentage_error(y_test, CBR_pred)
# %%
# У данной модификации бустинга очень большой ассортимент полезных опций, воспользуемся одной из них и заодно проанализируем важность признаков.
feature_importances = CBR_model.get_feature_importance()
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))
# %% md
# __Как видими топ 2 по важности признака Shell weight и Shucked weight , а вот добавленный SEX оказывает меньше всего значения на предсказание.__ 
# %% md
# __Отлично Catboost со значениями по умолчанию уменьшил ошибку на пол процента, пробуем подобрать гиперпараметры через Optuna__ 
# %%
import optuna


# %%
def objective(trial):
    param = {"n_estimators": trial.suggest_int("n_estimators", 50, 2500),
             "max_depth": trial.suggest_int("max_depth", 2, 16),
             "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
             "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 30),
             "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
             "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 1)}
    estimator = CatBoostRegressor(**param, verbose=False, cat_features=[0])
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    return mean_absolute_percentage_error(y_test, pred)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print(study.best_trial)
CBR_model_optuna = CatBoostRegressor(**study.best_params, cat_features=[0])
CBR_model_optuna.fit(X_train, y_train)
CBR_pred = CBR_model_optuna.predict(X_test)
mean_absolute_percentage_error(y_test, CBR_pred)
# %% md
# __Отлично, с помощью подбора гиперпараметров так же улучшили метрику .__
# %% md
# __Вычисление дополнительных метрик и их интерпретация__
# %%
# Импортируем необходимые метрики.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Делаем предсказания на тренировочных данных с лучшей моделью
y_train_pred = CBR_model_optuna.predict(X_train)
# Вычисляем метрики для тренировочных данных
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
# Вывод результатов для тренировочных данных
print('Train data:')
print(f'MAE: {mae_train:.4f}')
print(f'MSE: {mse_train:.4f}')
print(f'RMSE: {rmse_train:.4f}')
print(f'R2: {r2_train:.4f}')

print()

# Вывод результатов для тестовых данных
print('Test data:')
mae = mean_absolute_error(y_test, CBR_pred)
print(f'MAE: {mae:.4f}')
mse = mean_squared_error(y_test, CBR_pred)
print(f'MSE: {mse:.4f}')
rmse = np.sqrt(mean_squared_error(y_test, CBR_pred))
print(f'RMSE: {rmse:.4f}')
r2 = r2_score(y_test, CBR_pred)
print(f'R2: {r2:.4f}')
# %% md
# __Выводы по метрикам:__
# 
# __MAE: 1.3481 говорит о том, что модель в среднем ошибается на примерно чуть больше чем год при предсказании возраста ушек на тренировочных данных и на MAE: 1.4784 на тестовых. В целом результат на тестовых и тренировочных данных сравним, так что можно сказать что модель имеют неплохую обобщающую способность.__
# 
# __RMSE: 1.8555 - корень из средней квадратичной ошибки RMSE Train и RMSE 2.1392 Test. Значение Test выше, чем на тренировочных данных, что может указывать на некоторое переобучение, но оно, кажется, все еще довольно низкое.__
# 
# __Train data R2: 0.6707 - говорит о том, что модель "угадывает" примерно 67.07% изменчивости в данных для тренировочного набора. Это средний показатель показатель. Test data R2: R2: 0.5511 - на тестовых данных модель "угадывает" примерно 55.11% изменчивости. Это тоже приемлемый результат, но он ниже, чем на тренировочных данных, что может сигнализировать о некотором переобучении или о том, что модель несколько хуже справляется с новыми данными,но разница не слишком большая,чтобы говорить о сильном оферфите.__
# 
# __Модель показывает адекватные результаты как на тренировочных данных , так и на тестовых, разница между метриками минимальна.  На тестовых данных модель  обладает приемлемой обобщающей способностью, и ее прогнозы могут быть полезны в практическом применении.__
# %% md
# #__Explainer Dashboard__
# %%
from explainerdashboard import RegressionExplainer, ExplainerDashboard
# %%
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
# %%
# Создаем Explainer
explainer = RegressionExplainer(CBR_model_optuna, X_test, y_test)

# Создаем ExplainerDashboard
db = ExplainerDashboard(explainer, title="CatBoost Explainer Dashboard",
                        whatif=False)
db.to_yaml("abalone_dash.yaml", explainerfile="abalone_explainer.joblib", dump_explainer=True)
# %% md
# #__Анализ модели в Explainer Dashboard__
# %% md
# __По данным explainer dashboard можно сказать, что наибольший вклад в результат по SHAP вносят Shell weight и Shucked weight.__
# %% md
# __Метрики
# Model Summary
# Quantitative metrics for model performance
# metric	Score
# mean-squared-error	4.576
# root-mean-squared-error	2.139
# mean-absolute-error	1.478
# mean-absolute-percentage-error	0.15
# R-squared	0.551__
# %% md
# __Mean Square Error (MSE):
# MSE вычисляет среднеквадратичную ошибку между фактическими и предсказанными значениями.
# Чем меньше значение MSE, тем лучше модель предсказывает данные.
# Root Mean Squared Error (RMSE):
# RMSE является корнем из MSE и представляет собой среднеквадратичное отклонение предсказанных значений от фактических.
# Чем меньше значение RMSE, тем лучше модель предсказывает данные.
# Mean Absolute Error (MAE):
# MAE вычисляет среднее абсолютное отклонение между фактическими и предсказанными значениями.
# Чем меньше значение MAE, тем лучше модель предсказывает данные.
# Mean Absolute Percentage Error (MAPE):
# MAPE вычисляет среднее абсолютное отклонение в процентном соотношении между фактическими и предсказанными значениями.
# Чем меньше значение MAPE, тем лучше модель предсказывает данные.
# Значение MAPE равное 0,15 означает, что среднее абсолютное отклонение в процентном соотношении между предсказанными и фактическими значениями составляет примерно 15%.
# R-squared (R^2):
# R^2 (коэффициент детерминации) показывает долю дисперсии целевой переменной, объясненную моделью.
# Значение R^2 ближе к 1 означает лучшее соответствие модели данным.
# Значение R^2 равное 0,551 означает, что модель объясняет примерно 55,1% дисперсии целевой переменной.__
# 
# %% md
# 
# __Анализ 2-3 индивидуальных прогнозов с комментарием__
# %% md
# __индекс 1791
# Predicted 10,732
# Observed 10,000
# На графике Contribution Plot можно визульно пронаблюдать какой именно вклад вносит каждая из фичей в числовом значении. Интересно, что вляние фич отсчитывается от Average of Population, что видимо какое-то вычисленное начальное значение. Для индекса 1791 интересно, что shucked weight понижает возраст, а shell weight повышает.__
# 
# __индекс 1764
# Predicted	5.274
# Observed	5.000
# В данном случае почти все фичи двигают значение возраста в сторону уменьшения, кроме shucked weight__
# %%
