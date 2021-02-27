# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]

## MNIST challenge - ConvNet model : tensorflow 2.3

# %%
import tensorflow as tf 
import numpy as np 
import pandas as pd 
from tensorflow import keras

from tensorflow.keras.optimizers import RMSprop,SGD,Adam
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten

# %% [markdown]
### load dataset

# %%
mnist_tr_df = pd.read_csv('../../data/datasets/train.csv')
mnist_tr_df.head()
print(mnist_tr_df.shape)


# %%
mnist_ts_df = pd.read_csv('../../data/datasets/test.csv')
mnist_ts_df.head()
print(mnist_ts_df.shape)


# %%
train_df = mnist_tr_df = pd.read_csv('../../data/datasets/train.csv', header=0).sample(frac=1)
y_tol = train_df.iloc[:, 0].values
X_tol = train_df.iloc[:, 1:].values

X_tol = X_tol.astype(np.float32).reshape(-1,28,28,1)/255.0
y_tol = y_tol.astype(np.int32)
y_tol = keras.utils.to_categorical(y_tol, 10)


# %%
X_train, y_train = X_tol[:40000], y_tol[:40000]
X_test, y_test = X_tol[40000:], y_tol[40000:]


# %%
split_point = int(X_train.shape[0]*0.8)
X_tr ,y_tr = X_train[:split_point], y_train[ :split_point]
X_val, y_val = X_train[split_point:], y_train[ split_point:]

# %% [markdown]
### build ConvNet architecture

# %%
model = tf.keras.models.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=[28,28,1]),
    Conv2D(64, (3,3), activation='relu', ),
    MaxPool2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

# %% [markdown]
### defin loss function matrics & optimization algorithm
# %%
model.compile(optimizer=SGD(learning_rate=0.01),
            loss='sparse_categorical_crossentropy',
            metrics = ['accuracy']
)

# %% [markdown]
### train the model
# %%
history = model.fit(
    X_tr, y_tr,
    batch_size=32,
    epochs=20,
    verbose=1,
    validation_data=(X_val, y_val)
)


# %%



