from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf

data = datasets.load_breast_cancer()
print(data)

X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)

model = Sequential()
model.add(Dense(1, input_dim=30, activation="sigmoid"))

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

model.fit(X_train, Y_train, epochs=1000)

X_test = scaler.transform(X_test)

pred = model.predict(X_test)

predict01 = np.where(pred > 0.5, 1, 0)
print("======================================")
print(predict01)

predict02 = predict01.flatten()
print(predict02)

predict03 = (predict02 == Y_test)
print(predict03)

print(np.sum(predict03))
print(len(predict03))

acc = np.sum(predict03)/len(predict03)
print(acc)