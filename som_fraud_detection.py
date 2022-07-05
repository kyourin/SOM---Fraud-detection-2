import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

X

from minisom import MiniSom

som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5) # input_len=len(X[0]), sigma=radius

som.random_weights_init(X)

som.train_random(X, 100)

from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)
colorbar()

markers = ["o", "s"]
colors = ["r", "g"]

for i, x in enumerate(X):
  w = som.winner(x)
  plot(w[0]+0.5,
       w[1]+0.5,
       markers[y[i]],
       markeredgecolor = colors[y[i]],
       markerfacecolor = 'None',
       markersize = 10,
       markeredgewidth = 2)
  
show()

mappings = som.win_map(X)

frauds = np.concatenate((mappings[(3,7)], mappings[(7,1)]), axis=0) 
frauds = sc.inverse_transform(frauds)

print("Fraud Customer IDs:")
for i in frauds[:, 0]:
  print(int(i))


from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

dataset=pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

from pylab import bone, pcolor, colorbar, plot, show

som = MiniSom(x=10, y=10, input_len=len(X[0]), sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, 100)

bone()
pcolor(som.distance_map().T)
colorbar()

markers = ["o", "s"]
colors = ["r", "g"]
for i, x in enumerate(X):
  w = som.winner(x) 
  plot(w[0] + 0.5,
       w[1] + 0.5,
       markers[y[i]],
       markeredgecolor = colors[y[i]],
       markerfacecolor = 'None',
       markersize = 10,
       markeredgewidth = 2)

show();

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,8)], mappings[(4,5)], mappings[(3,8)], mappings[(6,1)], mappings[(2,1)]), axis=0)

frauds = sc.inverse_transform(frauds)

frauds.shape

fraud_id = frauds[:, 0]
fraud_id

customers = dataset.iloc[:, 1:].values

is_fraud = np.zeros(shape=(len(customers),))
is_fraud.shape

for i in range(len(dataset)):
  if dataset.iloc[i,0] in frauds:
    is_fraud[i] = 1

dataset.iloc[0,0] 

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
customers = sc.fit_transform(customers)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(2, kernel_initializer="uniform", activation="relu"),
  tf.keras.layers.Dense(1, kernel_initializer="uniform", activation="sigmoid")
])

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.fit(customers,
          is_fraud,
          epochs=2,
          batch_size=1)

model_preds = model.predict(customers)

model_preds = np.concatenate((dataset.iloc[:,0:1], model_preds), axis=1) 
model_preds.shape

model_preds = model_preds[model_preds[:,1].argsort()]

model_preds

