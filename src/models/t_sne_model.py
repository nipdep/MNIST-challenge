# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
## t-SNE on MNIST dataset


# %%
from __future__ import  print_function
import time

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

# %%
# access data from the sklearn ftech_openml library
mnist = fetch_openml("mnist_784")
X = mnist.data/255.0
y = mnist.target

print(X.shape, y.shape)

# %%
#convert numpy array into pd.DataFrame
feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))

X,y = None, None 
print(f'Size of tha dataframe: {df.shape}')

# %%
# this skips all the above database operations
mnist = pd.read_csv('../../data/datasets/train.csv')
mnist.rename( columns={'label' : 'y'}, inplace=True)
mnist['label'] =  mnist.iloc[:, 0].apply(lambda i: str(i))

# %%
#shuffel the dataset
np.random.seed(43)
rndperm = np.random.permutation(mnist.shape[0])

feat_cols = mnist.columns[1:-1]
# %%
# plot digits
plt.gray()
fig = plt.figure( figsize=(16,7))
for i in range(15):
    ax = fig.add_subplot(3,5,i+1, title=f'Digit: {mnist.loc[rndperm[i], "label"]}')
    ax.matshow(mnist.loc[rndperm[i], feat_cols].values.reshape(28,28).astype(np.float32))
plt.show()

# %% [markdown]

### Dimensionality reduction using PCA

# %%
# operate PCA upo tree dimesion reduction

pca = PCA(n_components=3)
pca_result = pca.fit_transform(mnist.loc[:, feat_cols].values)

mnist['pca-one'] = pca_result[:, 0]
mnist['pca-two'] = pca_result[:, 1]
mnist['pca-three'] = pca_result[:, 2]

print(f'Explained variation per principal component: {pca.explained_variance_ratio_}')

# %%
# graph variation pca-one with pca-two

plt.figure(figsize=(16,10))
sns.scatterplot(x="pca-one", y="pca-two",hue="y",
            palette=sns.color_palette("hls", 10),
            data=mnist.loc[rndperm, :],
            legend="full",
            alpha=0.3
)

# %%

ax = plt.figure( figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=mnist.loc[rndperm, :]['pca-one'],
    ys=mnist.loc[rndperm, :]['pca-two'],
    zs=mnist.loc[rndperm, :]['pca-three'],
    c=mnist.loc[rndperm, :]['y'],
    cmap='tab10'
)
ax.set_xlabel('pca_one')
ax.set_ylabel('pca_two')
ax.set_zlabel('pca_three')
plt.show()

# %%

N = 10000

df_subset = mnist.loc[rndperm[:N], :].copy()
data_subset = df_subset.loc[:, feat_cols].values

pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)

df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]

print(f'Explained variation per principal component : {pca.explained_variance_ratio_}')

# %%

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_result = tsne.fit_transform(data_subset)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# %%

df_subset['tsne-2d-one'] = tsne_result[:,0]
df_subset['tsne-2d-two'] = tsne_result[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)

# %%
