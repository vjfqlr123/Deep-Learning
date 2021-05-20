import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

X = np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ], dtype = "float32")

Y = np.array([
                [0],
                [1],
                [1],
                [0]
            ], dtype = "float32")

w = np.array([[1], [2]], dtype= "float32")
b = np.array([3], dtype= "float32")

np.dot(X, w) + b

def sigmoid(h):
    return 1 / (1 + np.exp(-h))

hypothesis = np.dot(X, w) + b
hypothesis = sigmoid(np.dot(X, w) + b)
print(hypothesis)
print(hypothesis - Y)

print(X.T)
print(X.T[0])

X_col_2d1 = X.T[0].reshape(1, -1)
print(X_col_2d1)

w1_gred = 1/4 * np.sum(np.dot(X_col_2d1, (hypothesis - Y)))
print(w1_gred)

X_col_2d2 = X.T[1].reshape(1, -1)
print(X_col_2d2)

w2_gred = 1/4 * np.sum(np.dot(X_col_2d2, (hypothesis - Y)))
print(w2_gred)

b_gred = 1/4 * np.sum(hypothesis - Y)
print(b_gred)

learning_rate = 0.1

for i in range(1000):
    for j in range(2):
        hypothesis = sigmoid(np.dot(X, w) + b)
        print(hypothesis)

        cost = -1/4 * (Y * np.log(hypothesis) + (1 - Y) * np.log(1 - hypothesis))
        print(cost)

        X_col_2d = X.T[j].reshape(1, -1)

        w_gred = 1/4 * np.sum(np.dot(X_col_2d, (hypothesis - Y)))
        w[j] = w[j] - learning_rate * w_gred
        print(w)

        b_gred = 1/4 * np.sum(hypothesis - Y)
        b = b - learning_rate * b_gred
        print(b)

print(w)
print(b)
print(np.dot(X, w) + b)

predict = sigmoid(np.dot(X, w) + b)
print(predict)

predict01 = tf.cast(predict > 0.5, dtype= tf.float32)
print("=========================================")
print(predict01)

ac01 = tf.equal(predict01, Y)
print(ac01)

ac02 = tf.cast(ac01, dtype="float32")
print(ac02)

ac03 = tf.reduce_mean(ac02)
print(ac03)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict01, Y), dtype= tf.float32))
print("========================")
print(accuracy.numpy())