#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:35:20 2021

@author: inderpreet
"""


import numpy as np
import pickle
import xarray
from iwc2tb.GMI.grid_field import grid_field  
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.colors as colors
import os
from iwc2tb.GMI.GMI_SatData import GMI_Sat
from iwc2tb.GMI.gmiSatData import gmiSatData
import xarray
from iwc2tb.GMI.remove_oversampling import remove_oversampling
plt.rcParams.update({'font.size': 20})



#%%
def plot_selective_dardar(tlat, tlon, tiwp, latlims, lonlims):

    
    lamask = np.logical_and(tlat > latlims[0], tlat < latlims[1])  
    lomask = np.logical_and(tlon > lonlims[0], tlon < lonlims[1])
    
    tmask  = np.logical_and(lamask, lomask)
    fig, ax = plt.subplots(1, 1, figsize = [20, 10])
    
    m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax)
    m.drawcoastlines()   
    
    cs = m.scatter(tlon[tmask], tlat[tmask],  c =  tiwp[tmask], 
                   norm=colors.LogNorm(vmin=1e-3, vmax= 50),  
                   cmap = cm.rainbow)
    fig.colorbar(cs)
    parallels = np.arange(-80.,80,20.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[True,False,True,False])
    meridians = np.arange(0.,360.,40.)
    m.drawmeridians(meridians,labels=[True,False,False,True])

def zonal_mean(lat, iwp, latbins):
    

    bins     = np.digitize(lat, latbins)
    
    nbins    = np.bincount(bins)
    iwp_mean = np.bincount(bins, iwp)
    
    return iwp_mean, nbins


def histogram(iwp, bins):
    
    hist, _ = np.histogram(iwp, bins)
    
    return hist/np.sum(hist)


def plot_hist(siwp, diwp, giwp, giwp0, tiwp, slat, dlat, glat, tlat, latlims, key = 'all'): 
    
    
    bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,.25,.5,1,2, 5, 10, 20, 50, 100, 200, 1000])
    
    smask = np.logical_and(np.abs(slat) >= latlims[0] , np.abs(slat) <= latlims[1])
    dmask = np.logical_and(np.abs(dlat) >= latlims[0] , np.abs(dlat) <= latlims[1])
    gmask = np.logical_and(np.abs(glat) >= latlims[0] , np.abs(glat) <= latlims[1])   
    tmask = np.logical_and(np.abs(tlat) >= latlims[0] , np.abs(tlat) <= latlims[1])   

    shist = histogram(0.001 * siwp[smask], bins)

    dhist = histogram(diwp[dmask], bins)
    ghist = histogram(giwp[gmask], bins)
    thist = histogram(tiwp[tmask], bins)
    ghist0 = histogram(giwp0[gmask], bins)    
    
    bin_center = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    
    ax.plot(bin_center, shist, 'b-.', markersize = 2, label = "SI" )
    ax.plot(bin_center, dhist, 'b--', markersize = 2, label = "DARDAR" )
    #ax.plot(bin_center, thist, 'o-', markersize = 2,  label = "DARDAR training" )
    ax.plot(bin_center, ghist, 'b-', markersize = 2, label = "GMI QRNN" )
    ax.plot(bin_center, ghist0, 'r:', markersize = 2, label = "GMI GPROF" )

    
    ax.set_xlabel("IWP [kg/m2]")
    ax.set_ylabel("frequency")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(str(latlims[0]) +"-" +str(latlims[1]))
    fig.savefig("Figures/PDF_IWP_" +str(latlims[0]) +"-" +str(latlims[1]) + key + ".png", bbox_inches = "tight")

#%% spare ice data
with open("spareice_jul2009.pickle", "rb") as f:
    slat = pickle.load(f)
    slon = pickle.load(f)
    siwp = pickle.load(f)
f.close()
smask = np.abs(slat) <= 65.0
siwpg, siwpc =  grid_field(slat[smask], slon[smask]%360, siwp[smask],
                                  gsize = 5.0, startlat = 65.0)

with open("gridded_spareice.pickle", "wb") as f:
    pickle.dump(siwpg, f)
    pickle.dump(siwpc, f)
    f.close()
    
#%% dardar data 
with open("dardar_jul2009.pickle", "rb") as f:
    dlat = pickle.load(f)
    dlon = pickle.load(f)
    diwp = pickle.load(f)
f.close()

dmask = np.abs(dlat) <= 65.0 
diwpg, diwpc =  grid_field(dlat[dmask], dlon[dmask]%360, diwp[dmask],
                                  gsize = 5, startlat = 65.0)

with open("gridded_dardar.pickle", "wb") as f:
    pickle.dump(diwpg, f)
    pickle.dump(diwpc, f)
    f.close()

#%% read compiled IWP for one month
# janfile = os.path.expanduser("jan2020_IWP_sgd_old.pickle")
# with open(janfile, "rb") as f:
#       giwp  = pickle.load(f)
#       giwp_mean  = pickle.load(f)
#       giwp0 = pickle.load(f)
#       glon  = pickle.load(f)
#       glat  = pickle.load(f)
#       glsm  = pickle.load(f)
    
#       f.close()

# a = giwp.copy()
# a[:731740, : ] = giwp0
# giwp0 = a.copy()

#giwp_mean = np.concatenate(giwp_mean, axis = 0)

# giwp[giwp < 1e-4] = 0
# giwp_mean[giwp_mean < 1e-4] = 0

# giwp_mean[np.abs(glat)> 65.0] = np.nan
# giwp[np.abs(glat)> 65.0] = np.nan

# # giwp[giwp > giwp_mean]  = np.nan
# # giwp_mean[giwp > giwp_mean]  = np.nan
inputfile = "jul2020_IWP_he_loglinear.nc"
    
dataset = xarray.open_dataset(inputfile)

#giwp = dataset.IWP.data
giwp_mean = dataset.iwp_mean.data
giwp0 = dataset.iwp0.data
glon = dataset.lon.data
glat = dataset.lat.data
glsm = dataset.lsm.data

#giwp[giwp < 1e-4] = 0
giwp_mean[giwp_mean < 1e-4] = 0
#giwp[giwp > giwp_mean]  = np.nan
#giwp_mean[giwp > giwp_mean]  = np.nan
dataset.close()

    
#%% remove oversampling

#glat, glon, glsm, giwp_mean = remove_oversampling(glat.ravel(), glon.ravel(), glsm.ravel(), giwp_mean.ravel())
#%% read in training data
train = xarray.open_dataset("/home/inderpreet/Dendrite/Projects/IWP/GMI/training_data/TB_GMI_train_july.nc") 

tiwp = train.ta.iwp
tlat = train.ta.lat
tlon = train.ta.lon% 360
#%% grid IWP

#giwp[giwp < 0] = 0
nanmask = ~np.isnan(giwp_mean)
gmask = np.abs(glat) <= 65.0

gmask = np.logical_and(gmask, nanmask)
giwpg, giwpc =  grid_field(glat[gmask].ravel(), glon[gmask].ravel()%360, 
                           giwp_mean[gmask].ravel(),
                           gsize = 5.0, startlat = 65.0)   

#%% grid IWP0
nanmask = giwp0 > -9000
gmask = np.abs(glat) <= 65.0

gmask = np.logical_and(gmask, nanmask)
giwp0g, giwp0c =  grid_field(glat[gmask].ravel(), glon[gmask].ravel()%360, 
                           giwp0[gmask].ravel(),
                           gsize = 5.0, startlat = 65.0)    


#%%
with open("gridded_iwp.pickle", "wb") as f:
    
    pickle.dump(giwpg, f)
    pickle.dump(giwpc, f)
    pickle.dump(giwp0g, f)
    pickle.dump(giwp0c, f)
    
    f.close()

#%%
bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,.25,
                 .5,1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 50])
    
bin_center = 0.5 * (bins[1:] + bins[:-1])   

ghist = histogram(giwp_mean.ravel(), bins)
shist = histogram(siwp * 0.001, bins)
dhist = histogram(diwp, bins)


fig, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.plot( bin_center, ghist, label = "adam_cosine")
ax.plot( bin_center, shist, label = "spareice")
ax.plot( bin_center,dhist,  label = "dardar")
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend()

#%% plot histograms, all

giwp = giwp_mean
lsmmask = np.ones(giwp.shape, dtype = "bool")  

#lsmmask = glsm == 0

#tmask = train.ta.stype == 0

latlims = [0, 30]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat,
          dlat, glat[lsmmask], dlat, latlims )

latlims = [30, 45]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat, dlat,
          glat[lsmmask], dlat, latlims )

latlims = [45, 65]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat, dlat,
          glat[lsmmask], dlat, latlims )  
    

#%% plot histograms, water
lsmmask = np.ones(giwp.shape, dtype = "bool")  

lsmmask = glsm == 0
#tmask = train.ta.stype == 0

latlims = [0, 30]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat,
          dlat, glat[lsmmask], dlat, latlims, key = "water" )

latlims = [30, 45]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat, dlat,
          glat[lsmmask], dlat, latlims , key = "water")

latlims = [45, 65]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat, dlat,
          glat[lsmmask], dlat, latlims, key = "water" )  
    

#%% plot histograms, land

lsmmask = np.ones(giwp.shape, dtype = "bool")  

lsmmask = glsm == 1
tmask = train.ta.stype == 1

latlims = [0, 30]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat,
          dlat, glat[lsmmask], dlat, latlims, key = "land" )

latlims = [30, 45]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat, dlat,
          glat[lsmmask], dlat, latlims , key = "land")

latlims = [45, 65]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat, dlat,
          glat[lsmmask], dlat, latlims, key = "land" )   

#%% plot histograms, snow
# lsmmask = np.ones(giwp.shape, dtype = "bool")  

# lsmmask = glsm == 2
# tmask = train.ta.stype == 2

# latlims = [0, 30]
# plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], tiwp[tmask], slat, dlat,
#           glat[lsmmask], tlat[tmask], latlims, key = "snow" )

# latlims = [30, 45]
# plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], tiwp[tmask], slat, dlat,
#           glat[lsmmask], tlat[tmask], latlims,  key = "snow")

# latlims = [45, 65]
# plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], tiwp[tmask], slat, dlat,
#           glat[lsmmask], tlat[tmask], latlims,  key = "snow")  

# #%% plot histograms, sea-ice
# lsmmask = np.ones(giwp.shape, dtype = "bool")  

# lsmmask = glsm == 3
# tmask = train.ta.stype == 3

# latlims = [0, 30]
# plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], tiwp[tmask], slat, dlat,
#           glat[lsmmask], tlat[tmask], latlims, key = "seaice" )

# latlims = [30, 45]
# plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], tiwp[tmask], slat, dlat,
#           glat[lsmmask], tlat[tmask], latlims,  key = "seaice")

# latlims = [45, 65]
# plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], tiwp[tmask], slat, dlat,
#           glat[lsmmask], tlat[tmask], latlims,  key = "seaice")  


#%% plot zonal_means
latbins = np.arange(-65, 66, 1.5)

giwp = giwp_mean
nanmask = giwp0 > -9000
ziwp0, ziwp0c = zonal_mean(glat[nanmask], giwp0[nanmask], latbins)

nanmask = ~np.isnan(giwp)
ziwp, ziwpc = zonal_mean(glat[nanmask], giwp[nanmask], latbins)


ziwp_si, ziwp_sic = zonal_mean(slat, siwp, latbins)
ziwp_d, ziwp_dc = zonal_mean(dlat,  diwp, latbins)


fig, ax = plt.subplots(1, 1, figsize = [15, 15])
ax.plot(ziwp[:-1]/ziwpc[:-1],latbins, 'b-',  label = "QRNN") 
#ax.plot(ziwp/ziwpc,latbins[:-1], 'b-',  label = "QRNN") 


#ax.plot(ziwp/ziwpc,latbins, 'b-',  label = "QRNN") 
#ax.plot(ziwp0[:-1]/ziwp0c[:-1], latbins, 'b.-', label = "GPROF")
#ax.plot(ziwp0/ziwp0c, latbins, 'b.-', label = "GPROF")

ax.plot( (0.001 * ziwp_si/ziwp_sic), latbins, 'b--', label = "SpareIce")
ax.plot(ziwp_d/ziwp_dc,latbins, 'b:', label = "DARDAR") 

ax.set_ylabel("Latitude [deg]")
ax.set_xlabel("IWP [kg/m2]")
ax.legend()
fig.savefig("Figures/zonal_mean_all_jul.png", bbox_inches = "tight")

#%% plot zonal_means from gridded data

lats = np.arange(-65, 65, 1.0)
gziwp_s = np.mean(siwpg/siwpc, axis = 1)
gziwp_d = np.mean(diwpg/diwpc, axis = 1)
gziwp_g = np.mean(giwpg/giwpc, axis = 1)
gziwp_g0 = np.mean(giwp0g/giwp0c, axis = 1)


fig, ax = plt.subplots(1, 1, figsize = [15, 15])

ax.plot(0.001 * gziwp_s, lats, label = "SI")
#ax.plot(gziwp_d, lats, label = "DARDAR")
ax.plot(gziwp_g, lats, label = "QRNN")
ax.plot(gziwp_g0, lats, label = "GPROF")
ax.legend()

ax.set_ylabel("Latitude [deg]")
ax.set_xlabel("IWP [kg/m2]")

fig.savefig("Figures/zonal_mean_gridded.png", bbox_inches = "tight")

  
#%% get avg IWP, weighted by cosine of latitude [g/m2]

lats  = np.arange(-65, 65, 1)
cosines = np.cos(np.deg2rad(lats))


print ("SI mean: ", np.sum(gziwp_s * cosines)/np.sum(cosines))
#print ("DARDAR mean: ", np.sum(gziwp_d * cosines)/np.sum(cosines))
print ("QRNN mean: ", 1000 * np.sum(gziwp_g * cosines)/np.sum(cosines)) # g/m2
print ("GPROF mean: ", np.sum(gziwp_g0 * cosines)/np.sum(cosines))

#%% spatial distribution
lon = np.arange(0, 360, 5)
lat = np.arange(-65, 65, 5)
lon1 = np.arange(0, 360, 5)
lat1 = np.arange(-65, 65, 5)


fig, axes = plt.subplots(4, 1, figsize = [40, 20])
fig.tight_layout()
ax = axes.ravel()
m = Basemap(projection= "cyl", llcrnrlon = 0,  llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax[0])
m.drawcoastlines()  
m.pcolormesh(lon, lat, 0.001 * siwpg/siwpc, norm=colors.LogNorm(vmin=1e-4, vmax= 25),  cmap = cm.rainbow)

parallels = np.arange(-80.,80,20.)
meridians = np.arange(0.,360.,40.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])


m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax[1])
m.pcolormesh(lon1, lat1, diwpg/diwpc, norm=colors.LogNorm(vmin=1e-4, vmax= 25),  cmap = cm.rainbow)
m.drawcoastlines() 
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])

m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax[2])
cs = m.pcolormesh(lon, lat, giwpg/giwpc, norm=colors.LogNorm(vmin=1e-4, vmax= 25),  cmap = cm.rainbow)
m.drawcoastlines() 
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])

m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax[3])
cs = m.pcolormesh(lon, lat, giwp0g/giwp0c, norm=colors.LogNorm(vmin=1e-4, vmax= 25),  cmap = cm.rainbow)
m.drawcoastlines() 
m.drawparallels(parallels,labels=[True,False,True,False])

m.drawmeridians(meridians,labels=[True,False,False,True])
fig.colorbar(cs, label="IWP [kg/m2]", ax = axes)

ax[0].set_title("SpareICE")
ax[1].set_title("DARDAR")
ax[2].set_title("GMI QRNN")
ax[3].set_title("GPROF")
fig.savefig("Figures/IWP_spatial_distribution.png", bbox_inches = "tight")


#%% analyse one file with high values
import xarray
file = "/home/inderpreet/Dendrite/UserAreas/Kaur/IWP/with_z0_2/1B.GPM.GMI.TB2016.20200102-S120411-E133645.033211.V05A.nc"

dataset = xarray.open_dataset(file)

fig, ax = plt.subplots(1, 1, figsize = [20, 8])
m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax)
m.drawcoastlines()   
mask = dataset.iwp_mean > 70.0
cs = m.scatter(dataset.lon.data[mask.data], 
               dataset.lat.data[mask.data],
               c =  dataset.iwp.data[mask.data], vmin = 150, vmax = 1000,  cmap = cm.Pastel1)
fig.colorbar(cs)
parallels = np.arange(-80.,80,20.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,True,False])
meridians = np.arange(0.,360.,40.)
m.drawmeridians(meridians,labels=[True,False,False,True])

basename = os.path.basename(file)
gfile = os.path.join("/home/inderpreet/Dendrite/SatData/GMI/L1B/2020/01/01", basename[:-2] + "HDF5")
gmisat = GMI_Sat(gfile)


inputs             = ["ta", "t2m",  "wvp", "lat", "stype"]
freq               = ['166.5V', '166.5H', '183+-3', '183+-7']
outputs            = "iwp"
batchSize          = 4
latlims            = [0, 65]

validation_data    = gmiSatData(gmisat, 
                             inputs, outputs,
                             batch_size = batchSize,
                             latlims = latlims,
                             std = None,
                             mean = None,
                             log = None)


bins = np.arange(100, 300, 2)
fig, axes = plt.subplots(3, 2, figsize = [20, 20])
ax = axes.ravel()

for i in range(0, 4):

    ax[i].hist(validation_data.x[:, :, i].ravel(), bins, histtype = "step", density = True, label = "GMI")
    ax[i].scatter( validation_data.x[:, :, i][mask], np.repeat(1e-3, 11),)    
    ax[i].hist(train.ta[:, i], bins,  histtype = "step", density = True,  label = "training")
    ax[i].set_yscale("log") 
    ax[i].legend()
    ax[i].set_title(freq[i])


snow = train.ta.stype == 2  
gsnow = validation_data.x[:, :, -1] == 2 
ax[5].hist(train.ta.t2m, bins,  histtype = "step", density = True, label = "training")
ax[5].hist(train.ta.t2m[snow], bins,  histtype = "step", density = True, label = "training only snow")
ax[5].hist(validation_data.x[:, :, 4].ravel(), bins, histtype = "step", density = True, label = "GMI")
ax[5].hist(validation_data.x[:, :, 4][gsnow].ravel(), bins, histtype = "step", density = True, label = "GMI snow")
ax[5].scatter( validation_data.x[:, :, 4][mask], np.repeat(1e-3, 11),)   
ax[5].set_yscale("log") 
ax[5].legend(loc = "upper left")
ax[5].set_title("t2m")

bins = np.arange(0, 100, 2)
ax[4].hist(train.ta.wvp, bins,  histtype = "step", density = True, label = "training")
ax[4].hist(validation_data.x[:, :, 5].ravel(), bins, histtype = "step", density = True, label = "GMI")
ax[4].scatter( validation_data.x[:, :, 5][mask], np.repeat(1e-3, 11),)   
ax[4].set_yscale("log") 
ax[4].legend()
ax[4].set_title("WVP")


fig.savefig("Figures/PDF_training_data.png", bbox_inches = "tight")

#%%plot high IWP values

hmask = (giwp_mean < 5) & (giwp_mean > 3)


fig, ax = plt.subplots(1, 1, figsize = [25, 15])
m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -75, urcrnrlon = 360, urcrnrlat = 65, ax = ax)
m.drawcoastlines()   

cs = m.scatter(glon[hmask], glat[hmask],  c =  giwp_mean[hmask], vmin = 100, vmax = 300,  cmap = cm.rainbow)
fig.colorbar(cs)
parallels = np.arange(-80.,80,20.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,True,False])
meridians = np.arange(0.,360.,40.)
m.drawmeridians(meridians,labels=[True,False,False,True])
fig.savefig("Figures/high_iwp_july.png", bbox_inches = "tight")

#%% check dardar simulated values in the same region as high values
latlims = [40,50]
lonlims = [70, 95]
lonlims = [120, 360 ]

plot_selective_dardar(tlat, tlon, tiwp, latlims, lonlims)

#check dardar values in the same region as high values
plot_selective_dardar(dlat, dlon, diwp, latlims, lonlims)

#%%

lamask = np.logical_and(tlat > latlims[0], tlat < latlims[1])  
lomask = np.logical_and(tlon > lonlims[0], tlon < lonlims[1])

tmask  = np.logical_and(lamask, lomask)
fig, ax = plt.subplots(1, 1, figsize = [20, 10])

m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax)
m.drawcoastlines()   

cs = m.scatter(tlon[tmask], tlat[tmask],  c =  train.ta.z0[tmask], 
               vmin=1e-3, vmax= 6000,  cmap = cm.rainbow)
fig.colorbar(cs)
parallels = np.arange(-80.,80,20.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,True,False])
meridians = np.arange(0.,360.,40.)
m.drawmeridians(meridians,labels=[True,False,False,True])


#%%
latlims = [40,50]
lonlims = [240, 280 ]

bins = np.arange(220, 300, 1)
fig, ax = plt.subplots(1, 1, figsize = [8, 8])
a = train.ta.stype == 2
    
lamask = np.logical_and(tlat > latlims[0], tlat < latlims[1])  
lomask = np.logical_and(tlon > lonlims[0], tlon < lonlims[1])

tmask  = np.logical_and(lamask, lomask)
tmask = np.logical_and(tmask, a)

ax.hist(train.ta.t2m[tmask], bins, density = True, histtype = "step", label = "N.America")

lonlims = [70, 95]
lamask = np.logical_and(tlat > latlims[0], tlat < latlims[1])  
lomask = np.logical_and(tlon > lonlims[0], tlon < lonlims[1])

tmask  = np.logical_and(lamask, lomask)
tmask = np.logical_and(tmask, a)

ax.hist(train.ta.t2m[tmask], bins, density = True, histtype = "step", label = "Asia")

lonlims = [0, 50]
lamask = np.logical_and(tlat > latlims[0], tlat < latlims[1])  
lomask = np.logical_and(tlon > lonlims[0], tlon < lonlims[1])

tmask  = np.logical_and(lamask, lomask)
tmask = np.logical_and(tmask, a)

ax.hist(train.ta.t2m[tmask], bins, density = True, histtype = "step", label = "Europe")

ax.legend()

ax.set_xlabel("t2m [K]")
ax.set_ylabel("PDF")
ax.set_title("t2m distribution over snow regions 40 deg - 50 deg")
fig.savefig("Figures/t2m_comparison.png", bbox_inches = "tight")


#%%

latlims = [40,50]
lonlims = [240, 280 ]

bins = np.arange(220, 300, 1)
fig, ax = plt.subplots(1, 1, figsize = [8, 8])
a = train.ta.stype == 2
    
lamask = np.logical_and(tlat > latlims[0], tlat < latlims[1])  
lomask = np.logical_and(tlon > lonlims[0], tlon < lonlims[1])

tmask  = np.logical_and(lamask, lomask)
tmask = np.logical_and(tmask, a)

ax.hist(train.ta.t2m[tmask], bins, density = True, histtype = "step", label = "N.America")

lonlims = [70, 95]
lamask = np.logical_and(tlat > latlims[0], tlat < latlims[1])  
lomask = np.logical_and(tlon > lonlims[0], tlon < lonlims[1])

tmask  = np.logical_and(lamask, lomask)
tmask = np.logical_and(tmask, a)

ax.hist(train.ta.t2m[tmask], bins, density = True, histtype = "step", label = "Asia")

lonlims = [0, 50]
lamask = np.logical_and(tlat > latlims[0], tlat < latlims[1])  
lomask = np.logical_and(tlon > lonlims[0], tlon < lonlims[1])

tmask  = np.logical_and(lamask, lomask)
tmask = np.logical_and(tmask, a)

ax.hist(train.ta.t2m[tmask], bins, density = True, histtype = "step", label = "Europe")

ax.legend()

ax.set_xlabel("t2m [K]")
ax.set_ylabel("PDF")
ax.set_title("t2m distribution over snow regions 40 deg - 50 deg")
fig.savefig("Figures/t2m_comparison.png", bbox_inches = "tight")


