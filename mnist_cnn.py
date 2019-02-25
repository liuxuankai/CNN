#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:57:24 2019

@author: xuankai
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

def conv2d(x,shape):
    return tf.nn.conv2d(x,shape,strides=[1,1,1,1],padding="SAME")

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bia_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))


x=tf.placeholder("float",shape=[None,784])
y_=tf.placeholder("float",shape=[None,10])

x_image=tf.reshape(x,[-1,28,28,1])


#conv layer
w_conv1=weight_variable([5,5,1,32])
b_conv1=bia_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool(h_conv1)

w_conv2=weight_variable([5,5,32,64])
b_conv2=bia_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool(h_conv2)

#full connected layer
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bia_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

keep_prob=tf.placeholder("float")
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#output layer
w_fc2=weight_variable([1024,10])
b_fc2=bia_variable([10])
y_fc2=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#loss
cross_entropy=-tf.reduce_sum(y_*tf.log(y_fc2))
train=tf.train.GradientDescentOptimizer(0.0002).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#evaluation

correct_prediction=tf.equal(tf.arg_max(y_,1),tf.arg_max(y_fc2,1))
final_acc=tf.reduce_mean(tf.cast(correct_prediction,"float"))


#init and run cnn model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch=mnist.train.next_batch(100)
        if i%100==0:
            train_acc=sess.run(final_acc,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            print("Iter %d ,train accuracy %g"%(i,train_acc))
        #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        sess.run(train,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    print("test accuracy %g"%sess.run(final_acc,{x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
    
            
        
    

















