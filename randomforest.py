#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
from sklearn.datasets import load_iris
iris=load_iris()

print(iris['target'].shape)

rf=RFC()


rf.fit(iris.data[:150],iris.target[:150])

instance=iris.data[[60,119]]
ins=instance


print('prediction',rf.predict(ins))
print('actual_label',iris.target[60],iris.target[119])