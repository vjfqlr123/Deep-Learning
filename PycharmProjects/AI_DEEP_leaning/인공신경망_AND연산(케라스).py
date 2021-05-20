import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

# AND연산의  x1, x2입력을 저장하는 배열
X = np.array([
                [0., 0.],
                [0., 1.],
                [1., 0.],
                [1., 1.],
        ], dtype = "float32")

# AND연산의 y출력을 저장하는 배열
Y = np.array([0, 0, 0, 1], dtype = "int32")

model = Sequential()
model.add(Dense(1, input_dim = 2, activation = "sigmoid"))

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.1), metrics = ['acc'])

model.fit(X, Y, epochs = 100)

pred = model.predict(X)
print(pred)

predict01 = np.where(pred>0.5, 1, 0)
print("=======================================")
print(predict01)

predict02 = predict01.flatten()
print(predict02)

predict03 = (predict02 == Y)
print(predict03)

np.sum(predict03)

acc = np.sum(predict03)/4
print(acc)