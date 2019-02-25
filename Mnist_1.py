# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("hello")

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

x=tf.placeholder("float",[None,784])
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,w)+b)
y_=tf.placeholder("float",[None,10])

cross_entropy=-tf.reduce_sum((y_*tf.log(y)))
train=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))


sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    batch_x,batch_y=mnist.train.next_batch(100)
    sess.run(train,feed_dict={x:batch_x,y_:batch_y})
    
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
sess.close()




