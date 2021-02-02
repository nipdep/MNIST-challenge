# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorboard
import tensorflow as tf 
import numpy as np 
import pandas as pd 
from tensorflow import keras

from tensorflow.keras.optimizers import RMSprop,SGD,Adam
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten
from tensorboard.plugins.hparams import api as hp


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


# %%
X_train, y_train = X_tol[:40000], y_tol[:40000]
X_test, y_test = X_tol[40000:], y_tol[40000:]


# %%
split_point = int(X_train.shape[0]*0.8)
X_tr ,y_tr = X_train[:split_point], y_train[ :split_point]
X_val, y_val = X_train[split_point:], y_train[ split_point:]


# %%
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([64, 128]))
#HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

hparams=[HP_NUM_UNITS, HP_OPTIMIZER]
#%%
model = tf.keras.models.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=[28,28,1]),
    Conv2D(64, (3,3), activation='relu', ),
    MaxPool2D(2,2),
    Flatten(),
    Dense(hparams[HP_NUM_UNITS], activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])


# %%

model.compile(optimizer=hparams[HP_OPTIMIZER],
            loss='sparse_categorical_crossentropy',
            metrics = ['accuracy']
)


# %%
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="../../logs/", update_freq=1, write_graph=True)
history = model.fit(
    X_tr, y_tr,
    batch_size=32,
    epochs=5,
    verbose=1,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback, hp.KerasCallback('../../logs/h_params', hparams),]
)

#%%
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)




# %%
history_frame = pd.DataFrame(history.history)
#history_frame.head()
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()

#%%
model.save('../../data/models/convNet3.pb')





# %%
model.save('../../data/models/convNet3.h5')

# %%
