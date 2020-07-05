import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.decomposition import PCA


dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:19].values
df = pd.DataFrame(X)

corrMatrix = df.corr()
covMatrix = pd.DataFrame.cov(df)

pca = PCA(n_components=5)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

plt.scatter(X_pca[:, 0], X_pca[:, 1],X_pca[:, 2], alpha=0.2)



