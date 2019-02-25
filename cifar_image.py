#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:33:16 2019

@author: xuankai
"""

import pickle




f=open('/tmp/cifar10_data/cifar-10-batches-bin/data_batch_2.bin','rb')
d=pickle.load(f)

print(d)
f.close()