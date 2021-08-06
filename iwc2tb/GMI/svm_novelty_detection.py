#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:27:32 2021

@author: inderpreet
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM
sns.set_style("darkgrid")

from matplotlib import cm
from iwc2tb.GMI.GMI_SatData import GMI_Sat
from iwc2tb.GMI.gmiSatData import gmiSatData
import matplotlib.colors as colors
from iwc2tb.GMI.gmiData import gmiData

from matplotlib import cm
import os
import numpy as np
import xarray
import glob

one_class_svm = OneClassSVM(kernel='rbf', degree=3, gamma='scale')



inputs            = np.array(["ta", "t2m",  "wvp", "stype", "z0"])
#inputs            = np.array(["ta", "t2m",  "wvp", "lat"])
#inputs            = np.array(["ta", "t2m", "wvp", "lat", "stype"])
ninputs           = len(inputs) + 3

outputs           = "iwp"
#latlims           = [45, 65] 
latlims           = [0, 65]
xlog              = True
quantiles         = np.linspace(0.01, 0.99, 128)
print(quantiles)
batchSize         = 256


data = gmiData(os.path.expanduser("~/Dendrite/Projects/IWP/GMI/training_data/TB_GMI_train.nc"), 
               inputs, outputs, latlims = latlims,
               batch_size = batchSize, log = xlog)  


X = data.x
one_class_svm.fit(X)
