#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:21:58 2021

@author: inderpreet
"""
from iwc2tb.GMI.gmiData import gmiData

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import pickle
from matplotlib import cm
from iwc2tb.GMI.GMI_SatData import GMI_Sat
from iwc2tb.GMI.gmiSatData import gmiSatData
import matplotlib.colors as colors
from matplotlib import cm
import os
import numpy as np
import xarray
import glob

#%%
def plot_outliers(iwp, data1, data2, ax):    
    
    
    #ax.scatter( tdata1, tdata2, c = "tab:blue", alpha = 0.2)
    
    nanmask = ~np.isnan(iwp)
    mask = iwp[nanmask] > 25
    
    if np.sum(mask) > 0:
        ax.scatter(data1[nanmask][mask],data2[nanmask][mask],  
                        c = "r",)
   
    ax.set_ylabel("PD [166V-166H] [K]")
    ax.set_xlabel("TB 166 V [K]")     
   

def get_data_outliers(x, iwp, stype, lsm, tb, t2m, z0, wvp, lat, iwp_all):
    
    lsm.append(stype)
    tb.append(x[:, :4])
    t2m.append(x[:, 4])
    z0.append(x[:, -1])
    wvp.append(x[:, 5])
    lat.append(x[:, 6])
    iwp_all.append(iwp)

    return lsm, tb, t2m, z0, wvp, lat, iwp_all


def convert_lsm(data):
    
    sindex   = data.get_indices("stype")
    
    he_stype = data.x[:, sindex + 3: sindex + 3 + 11]
    
    stype    = data.to_decode(he_stype)
    
    return stype

def convert_lsm_gmi(data):
    
    sindex   = data.get_indices("stype")
    
    he_stype = data.x[:, :, sindex + 3: sindex + 3 + 11]
    
    stype    = data.to_decode(he_stype)
    
    return stype
#%%    

inputs            = np.array(["ta", "t2m",  "wvp", "lat", "z0", "stype" ])
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


data   = gmiData(os.path.expanduser("~/Dendrite/Projects/IWP/GMI/training_data/TB_GMI_train.nc"), 
               inputs, outputs, latlims = latlims,
               batch_size = batchSize)  

stype  = convert_lsm(data)

tdata1 = data.x[:, 0]
tdata2 = data.x[:, 0] - data.x[:, 1]     

stypes = ["Water", "Land", "Snow", "Sea-ice", "Coast", "Water/Sea-ice"]


with open("high_iwp_files_log.pickle", "rb") as f:
    h_iwp_files = pickle.load(f)    
    f.close()


lsm, tb, t2m, z0, wvp, lat, iwp_all = [], [], [], [], [], [], []

fig, ax = plt.subplots(2, 3, figsize = [25, 15], constrained_layout=True)

axes = ax.ravel()

   

for ix, ilsm in enumerate([0, 1, 2, 3, 4, 6]):
    
    tmask = stype == ilsm
    ax = axes[ix]
    ax.scatter(tdata1[tmask], tdata2[tmask], alpha = 0.2)
    
    
    
for file in h_iwp_files[:]:
    
    a            = xarray.open_dataset(file)
    print (file)
    iwp          = a.iwp_mean.data
    lats         = a.lat.data
    stype_val    = a.stype.data
    
    iwp[np.abs(lats) > 64] = np.nan
 
    filename = os.path.basename(file)[:-3] + ".HDF5"
    inpath   = "/home/inderpreet/Dendrite/SatData/GMI/L1B/2020/01/*"
    
    gmifile = glob.glob(os.path.join(inpath, filename))
    
    #filename = os.path.join(inpath, filename)
    
    gmi_s    = GMI_Sat(gmifile[0])  
    
    validation_data    = gmiSatData(gmi_s, 
                             inputs, outputs,
                             batch_size = batchSize,
                             latlims = latlims,
                             normalize = data.norm,
                             log = xlog)    
    
    data2 = validation_data.x[:, :, 0] - validation_data.x[:, :, 1]
    data1 = validation_data.x[:, :, 0]    
            
    nanmask = ~np.isnan(iwp)
    mask = iwp[nanmask] > 25

    lsm, tb, t2m, z0, wvp, lat, iwp_all = get_data_outliers(validation_data.x[nanmask, :][mask, :], iwp[nanmask][mask],
                                                        stype_val[nanmask][mask], lsm, tb, t2m, z0, 
                                                        wvp, lat, iwp_all)
    
    for ix, ilsm in enumerate([0, 1, 2, 3, 4, 6]):


        smask     = stype_val == ilsm

        
        ax = axes[ix]
        plot_outliers(iwp[smask], data1[smask], data2[smask], ax)

        ax.set_title(stypes[ix])

        ax.set_xlim([50, 320])
        ax.set_ylim([-5, 60])  
fig.savefig("outliers_PD_" +  "_log.png", bbox_inches = "tight")    
 
#%%

lsm = np.concatenate(lsm)
tb  = np.concatenate(tb, axis = 0)
t2m = np.concatenate(t2m)
z0  = np.concatenate(z0)
wvp = np.concatenate(wvp)
lat = np.concatenate(lat)
iwp_all = np.concatenate(iwp_all)

#%%
fig, ax = plt.subplots(1, 1, figsize = [8, 8])
cs = ax.scatter(tb[:, 0], iwp_all, c = lsm, vmin = 0, vmax = 6, cmap = cm.rainbow)
ax.set_xlabel("TB 166 V [K]")
ax.set_ylabel("IWP [kg/m2]")
fig.colorbar(cs)
fig.savefig("high_IWP_log.png", bbox_inches = "tight")






    