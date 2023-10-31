# %% [markdown]
# ## 1. Провести визуализация данных в трехмерном пространстве (3D Plot). Для выборки «Iris» обосновать выбор трех измерений (сделать дополнительные исследования, например карта корреляций). Сравнить трехмерные диаграммы для выборок и сделать предварительные выводы (Исходное количество кластеров считаем неизвестным).  В отчёт включить обоснование выбора измерений (результаты исследований и вывод по ним) и графики
# 
# ### Датасет Ирисы Фишера

# %%
from sklearn.datasets import load_iris
import numpy  as np
import pandas as pd

iris = load_iris()
data_pd = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
data_pd = data_pd.rename(columns={'sepal length (cm)': 'sl', 'sepal width (cm)': 'sw', 'petal length (cm)': 'pl', 'petal width (cm)': 'pw'})

print(data_pd)
print(data_pd.describe())

# %% [markdown]
# Из результатов по исследованиям в предыдущей лабораторной работе, известно, что для параметра sw имеем множество выбросов, поэтому для визуализации используем все параметры за исключением этого параметра и target. 

# %%
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
z = data_pd['sl']
x = data_pd['pl']
y = data_pd['pw']
ax.plot3D(x, y, z, 'b+')
ax.set_xlabel('pl')
ax.set_ylabel('pw')
ax.set_zlabel('sl')
ax.set_title('Представление параметров ирисов.')

plt.show()

# %%
import seaborn as sns
import pingouin

data_x = data_pd.iloc[:,:-1]
corr_matr = data_x.corr()
pcorr_matr = data_x.pcorr()

data_x.boxplot()
plt.show()
sns.heatmap(corr_matr, annot=True, vmin=-1, vmax=1)
plt.title('Парные корреляции')
plt.show()
sns.heatmap(pcorr_matr, annot=True, vmin=-1, vmax=1)
plt.title('Частные корреляции')
plt.show()

# %% [markdown]
# ### Датасет Бейсбол

# %%
baseball_pd = pd.read_csv('Baseball.csv', delimiter=';', decimal=',')

ax = plt.axes(projection='3d')
z = baseball_pd['Height']
x = baseball_pd['Weight']
y = baseball_pd['Age']
ax.plot3D(x, y, z, 'b+')
ax.set_xlabel('Weight')
ax.set_ylabel('Age')
ax.set_zlabel('Height')
ax.set_title('Представление бейсболистов.')
plt.show()

# %% [markdown]
# На трёхмерном представлении параметров ирисов ярко выражены два кластера. А про данные, касающиеся бейсболистов, нельзя точно сказать о наличии отдельных кластеров. Тут данные выглядят хаотично и перемешано.

# %% [markdown]
# ## 2.	Провести иерархическую кластеризацию. Исследовать зависимость результатов иерархической классификации от выбора меры близости (евклидово расстояние, манхэттенское расстояние, расстояние Чебышева, косинусное) и правила объеди-нения кластеров (одиночная связь, полная связь, невзвешенная средняя связь, не-взвешенная центроидная связь, метод Уорда). Проанализировать диаграмму изме-нения расстояний при объединении кластеров. Оценить предположительное число кластеров, на которое разделяется исследуемая совокупность. В отчёт включить наилучшие результаты и обосновать почему они наилучшие (сравнить с пло-хим/средним результатом). Так же включить обоснование предположительного числа кластеров.

# %%
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_dendrogramm(data_x, method, metric):
    cluster_ar = linkage(data_x, method=method, metric=metric)
    link_df = pd.DataFrame(cluster_ar, index=[f'step {i+1}' for i in range(cluster_ar.shape[0])], columns=['cluster1', 'cluster2', 'dist', 'number elements'] )
    fig = plt.figure(figsize=(25,10))
    row_dendr = dendrogram(link_df)
    plt.title(f'Metric: {metric} Method: {method}')
    plt.show()

metrics = ['euclidean', 'cityblock', 'chebyshev', 'cosine']

for metric in metrics:
    plot_dendrogramm(data_x, 'single', metric)


# %%
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from mpl_toolkits.mplot3d import Axes3D

def plot_clusters(data_x, n_clusters, method, metric):
    if method != 'centroid':
        cl = AgglomerativeClustering(n_clusters=n_clusters, linkage=method, metric=metric)
        labels = cl.fit_predict(data_x)
    elif method == 'centroid':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        y_predict = clusterer.fit_predict(data_x)
        clf = NearestCentroid()
        clf.fit(data_x, y_predict)
        labels = clf.predict(data_x)
    else:
        return

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    z = data_pd['sl']
    x = data_pd['pl']
    y = data_pd['pw']
    ax.scatter(x, y, z, c=labels, marker='o', edgecolors=['000']*len(labels))
    ax.set_xlabel('pl')
    ax.set_ylabel('pw')
    ax.set_zlabel('sl')
    ax.set_title(f'Представление параметров ирисов.\nMetric: {metric}, method: {method}')
    plt.show()

for metric in metrics:
    plot_clusters(data_x, 2, 'single', metric)

# %%
methods = ['single', 'complete', 'average', 'centroid', 'ward']

for method in methods:
    plot_dendrogramm(data_x, method, 'euclidean')
    plot_clusters(data_x, 2, method, 'euclidean')

# %% [markdown]
# По трёхмерному представлению всех наблюдений выборки и по всем полученным дендрограммам видно, что данные можно поделить на два явно выделенных кластера. Для дендрограмм все объединения до 2 оставшихся кластеров происходили при небольших значениях расстояний, кроме последнего объединения в 1 кластер.
# 
# При изменении используемых метрик мы получили одинаковые результаты. Различие было в том, что для разных метрик получались различные значения при объединении кластеров. Для косинусной меры близости расстояние между дальними кластерами было гораздо сильнее выделено, в то время как расстояния между остальными кластерами было значительно менее выделено.
# 
# При изменении метод объединения кластеров некоторые результаты получились отличными от большинства. Метод дальнего соседа показал самый худший результат, объединив в один из кластеров множество наблюдений явно не относящихся к нему. Неплохой результат показал метод центроидной связи, выделив лишь пару наблюдений явно не принадлежащих к своему кластеру. Остальные методы показали себя одинаково хорошо😎👍, показав ожидаемый по трёхмерным графикам результат.

# %%
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_x_scaled = scaler.fit_transform(data_x)
pca = decomposition.PCA()
x_pca = pca.fit_transform(data_x_scaled)
data_delta = pca.explained_variance_ratio_
print(data_delta)
print(f'Количество информации: {100*sum(data_delta[:2]):.2f}%')
plt.scatter(x_pca[:,0], x_pca[:,1])
plt.title('Метод главных компонент. Ирисы')
plt.show()
ax = plt.axes(projection='3d')
ax.plot3D(x_pca[:,0], x_pca[:,1], x_pca[:,2], 'b+')
plt.show()


