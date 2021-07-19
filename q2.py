import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# load the dataset
url = 'https://raw.githubusercontent.com/jliang33-web/21t2q2_jaxlib_workaround/main/Q2.csv'
data = pd.read_csv(url)
# drop empty rows
data.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)
# get Y data
Ydata = data.iloc[:, -1:]

# remove irrelavent columns for X
dropped = data.drop(
    columns=["transactiondate", "latitude", "longitude", "price"]
)

# normalize the data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dropped)

# split into training and testing data
train_n = int(len(scaled)/2)
test_n = int(len(scaled)-train_n)
split = np.split(scaled, 2)
X_train = split[0]
X_test = split[1]
Y_train = np.ravel(Ydata.iloc[:train_n])
Y_test = np.ravel(Ydata.iloc[-test_n:])

###
# question e
###

import jax.numpy as jnp
from jax import grad

step = 1.0

# build x matrix
X_col1 = np.ones((len(X_train),1))
X_loss_trainer = np.hstack((X_col1,X_train))

def function(xi,yi,w):
    return (((1/4)*(yi-jnp.matmul(w.T,xi))**2+1)**(1/2)-1)

x = X_loss_trainer
y = Y_train
rang = len(x)

def loss(w):
  sum_list = jnp.array([[]])
  for i in range(rang):
    xi = jnp.array([x[i]]).T
    yi = y[i]
    sum_list = jnp.append(sum_list, function(xi,yi,w)[0]*(1/rang))
  return jnp.sum(sum_list)

def test_loss(w):
  x = np.hstack((X_col1,X_test))
  y = Y_test
  sum_list = jnp.array([[]])
  for i in range(rang):
    xi = jnp.array([x[i]]).T
    yi = y[i]
    sum_list = jnp.append(sum_list, function(xi,yi,w)[0]*(1/rang))
  return jnp.sum(sum_list)

w0 = jnp.array([1.0,1.0,1.0,1.0]).T
w = w0

loss_k = loss(w)
loss_delta = loss_k
count = 0
training_loss = np.array([loss_k])
while loss_delta >= 0.0001:
  w = w - step * grad(loss)(w)
  new_loss = loss(w)
  loss_delta = abs(new_loss - loss_k)
  training_loss = np.append(training_loss,new_loss)
  print(loss_delta)
  loss_k = new_loss
  count += 1

print(count)
print(training_loss)
print(w)
print(loss(w))
print(test_loss(w))

