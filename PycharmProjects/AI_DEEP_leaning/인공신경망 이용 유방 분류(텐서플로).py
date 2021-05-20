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

def sigmoid(h):
    return 1 / (1 + np.exp(-h))

initializer = tf.contrib.layers.xavier_initializer()

w = tf.Variable(initializer([30, 1]))
print(w)

b = tf.Variable(initializer([1]))

tf.matmul(X_train, w) + b

hypothesis = tf.sigmoid(tf.matmul(X_train, w) + b)
print(hypothesis)

optimizer = tf.train.AdamOptimizer(0.001)

for step in range(100):
    with tf.GradientTape() as tape:
        hypothesis = tf.sigmoid(tf.matmul(X_train, w) + b)

        cost = -tf.reduce_mean(Y_train * tf.log(hypothesis) + (1 - Y_train) * tf.log(1 - hypothesis))

        grads = tape.gradient(cost, [w, b])

    optimizer.apply_gradients(grads_and_vars=zip(grads, [w, b]))

    if step % 10 == 0:
        print("="* 50)
        print("step: {}, cost: {}, w: {}, b: {}".format(step, cost.numpy(), w.numpy(), b.numpy()))
        print("=" * 50)

print(w)
print(b)

X_test = scaler.transform(X_test)
print(X_test)

predict = tf.sigmoid(tf.matmul(X_test, w) + b)
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