import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np


X = np.array([
                [0., 0.],
                [0., 1.],
                [1., 0.],
                [1., 1.],
        ], dtype = "float32")


Y = np.array([0, 1, 1, 0], dtype = "int32")

model = Sequential()

model.add(Dense(2, input_dim=2, activation= "sigmoid"))

model.add(Dense(1, activation= "sigmoid"))

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.1), metrics = ['acc'])

model.fit(X, Y, epochs = 3000)

pred = model.predict(X)
print(pred)

predict01 = np.where(pred>0.5, 1, 0)
print("=======================================")
print(predict01)
print("=======================================")
predict02 = predict01.flatten()
print(predict02)
print("=======================================")
predict03 = (predict02 == Y)
print(predict03)
print("=======================================")
np.sum(predict03)
print("=======================================")
acc = np.sum(predict03)/len(predict03)
print(acc)
print("=======================================")
arr = np.array([[0,1]], dtype="float32")
pred = model.predict(arr)
print(pred)
print("=======================================")
predict01 = np.where(pred > 0.5, 1, 0)
print(predict01)
print("=======================================")