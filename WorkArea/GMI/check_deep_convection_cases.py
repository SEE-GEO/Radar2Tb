#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:04:25 2021

@author: inderpreet
"""

import numpy as np
import xarray
import os
from iwc2tb.GMI.gmiSatData import gmiSatData
from iwc2tb.GMI.GMI_SatData import GMI_Sat
import glob
import pickle
import matplotlib.pyplot as plt


month = "01"
year  = "2020"
path = os.path.expanduser("~/Dendrite/UserAreas/Kaur/IWP/with_z0_3")

#--------------------inputs-------------------------------------------
inputs            = np.array(["ta", "t2m",  "wvp", "lat", "stype", "z0"])
#inputs            = np.array(["ta", "t2m",  "wvp", "lat"])
#inputs            = np.array(["ta", "t2m", "wvp", "lat", "stype"])
ninputs           = len(inputs) + 3

outputs           = "iwp"
#latlims           = [45, 65] 
latlims           = [0, 65]
xlog              = True
quantiles         = np.linspace(0.01, 0.99, 128)
batchSize         = 256
#----------------------------------------------------------------------

ncfiles = glob.glob(os.path.join(path, "1B*nc"))
IWP = []
TB  = []

for file in ncfiles:
    print (file)
    dataset = xarray.open_dataset(file)
    
    filename = os.path.basename(file)[:-3] + ".HDF5"
    inpath   = os.path.join("/home/inderpreet/Dendrite/SatData/GMI/L1B/", year, month,  "*")
    
    gmifile = glob.glob(os.path.join(inpath, filename))
    
    gmi_s    = GMI_Sat(gmifile[0])  
    
    validation_data    = gmiSatData(gmi_s, 
                             inputs, outputs,
                             batch_size = batchSize,
                             latlims = latlims,
                             std = [],
                             mean = [],
                             log = xlog)  
    
    tb = validation_data.x[:, :, 0]
    dconv = tb <= 100.0
    
    if np.sum(dconv) > 0:
        
        IWP.append(dataset.iwp_mean.data[dconv])
        TB.append(tb[dconv])
        
    dataset.close()
    
 
IWP = np.concatenate(IWP)
TB  = np.concatenate(TB)

with open("deep_convection_IWP.pickle", "rb") as f:
    IWP = pickle.load(f)
    TB  = pickle.load(f)
    
    f.close()
    
    
fig, ax = plt.subplots(1,1, figsize = [8, 8])
ax.scatter(TB, IWP, alpha = 0.2)
ax.set_xlabel("TB 166V GHz [K]")
ax.set_ylabel(r"IWP [kg m$^{-2}$]")    
fig.savefig("deep_convection_cases.png", bbox_inches = "tight")
    
    
    
    