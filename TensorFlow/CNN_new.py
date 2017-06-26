#!/usr/bin/python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


K=4
L=8
M=12

W1 = tf.Variable(tf.truncated_normal([5,5,1,K], stddev=0.1))
B1 = tf.Variable(tf.ones([K])/10)

W2 = tf.Variable(tf/truncated_normal([5,5,k,L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)

W3 = tf.Variable(tf.truncated_normal([4,4,L,M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

N = 200

W4 = tf.Variable(tf.truncated_normal([7*7*M, N]), stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)

W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10])/10)

Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') + B1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,2,2,1], padding='SAME') + B2)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,2,2,1], padding='SAME') + B3)

YY = tf.reshape(Y3, shape([-1,7*7*M])

Y4 = tf.nn.relu(tf.matmul(YY,W4)+B4)
Y = tf.nnsoftmax(tf.matmul(Y4,W5)+B5)


