from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.nn import relu
import tensorflow as tf
tf.enable_eager_execution()

data = datasets.load_breast_cancer()
print(data)

X = data.data
print(X)

X = np.array(X, dtype="float32")
print(X)

Y = data.target
print(Y)

Y = np.array(Y, dtype="float32")
print(Y)

Y = Y.reshape(-1, 1)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
print(X_train)

initializer = tf.contrib.layers.xavier_initializer()

w0 = tf.Variable(initializer([30, 30]))
print(w0)

b0 = tf.Variable(initializer([30]))

w1 = tf.Variable(initializer([30, 10]))
print(w1)

b1 = tf.Variable(initializer([10]))

w2 = tf.Variable(initializer([10, 1]))
print(w2)

b2 = tf.Variable(initializer([1]))

tf.matmul(X_train, w0) + b0

hypothesis0 = tf.sigmoid(tf.matmul(X_train, w0) + b0)
print(hypothesis0)

hypothesis1 = tf.sigmoid(tf.matmul(hypothesis0, w1) + b1)
print(hypothesis1)

hypothesis2 = tf.sigmoid(tf.matmul(hypothesis1, w2) + b2)
print(hypothesis2)


optimizer = tf.train.AdamOptimizer(0.01)

for step in range(100):
    with tf.GradientTape() as tape:
        hypothesis0 = tf.sigmoid(tf.matmul(X_train, w0) + b0)
        hypothesis1 = tf.sigmoid(tf.matmul(hypothesis0, w1) + b1)
        hypothesis2 = tf.sigmoid(tf.matmul(hypothesis1, w2) + b2)

        cost = -tf.reduce_mean(Y_train * tf.log(hypothesis2) + (1 - Y_train) * tf.log(1 - hypothesis2))

        grads = tape.gradient(cost, [w0, w1, w2, b0, b1, b2])

        optimizer.apply_gradients(grads_and_vars=zip(grads, [w0, w1, w2, b0, b1, b2]))

        if step % 100 == 0:
            print("="* 50)
            print("step: {}, cost: {}".format(step, cost.numpy()))
            print("=" * 50)


X_test = scaler.transform(X_test)
print(X_test)
print("=" * 50)

hypothesis0 = tf.sigmoid(tf.matmul(X_test, w0) + b0)
hypothesis1 = tf.sigmoid(tf.matmul(hypothesis0, w1) + b1)
predict = tf.sigmoid(tf.matmul(hypothesis1, w2) + b2)
print(predict)

predict01 = tf.cast(predict > 0.5, dtype= tf.float32)
print("============================================")
print(predict01)

ac01 = tf.equal(predict01, Y_test)
print(ac01)

ac02 = tf.cast(ac01, dtype="float32")
print(ac02)

ac03 = tf.reduce_mean(ac02)
print(ac03)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict01, Y_test), dtype= tf.float32))
print("=" * 40)
print(accuracy.numpy())