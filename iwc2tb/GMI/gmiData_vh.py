import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset
from keras.utils import to_categorical
from numpy import argmax
import random
from quantnn.normalizer import Normalizer 
from quantnn.transformations import Log, LogLinear

class gmiData(Dataset):
    """
    Pytorch dataset for the GMI training data for IWP retrievals

    """
    def __init__(self, path, 
                 inputs,
                 outputs,
                 pratio = None,
                 batch_size = None,
                 latlims = None,
                 normalise = None, 
                 transform = None):
        """
        Create instance of the dataset from a given file path.

        Args:
            path: Path to the NetCDF4 containing the data.
            batch_size: If positive, data is provided in batches of this size

        """
        super().__init__()
        
        self.batch_size = batch_size

        self.file = netCDF4.Dataset(path, mode = "r")

        ta = self.file.variables["ta"]
        TB = ta[:]
        
        TB[:, 0] = (ta[:, 0] + ta[:, 1])/2
        TB[:, 1] = ta[:, 2]
        TB[:, 2] = ta[:, 3]

        self.lon   = ta.lon.reshape(-1, 1)
        self.lat   = ta.lat.reshape(-1, 1)
        self.iwp   = ta.iwp.reshape(-1, 1)
        self.rwp   = ta.rwp.reshape(-1, 1)
        self.t0    = ta.t0.reshape(-1, 1)
        self.z0    = ta.z0.reshape(-1, 1)
        self.wvp   = ta.wvp.reshape(-1, 1)
        self.t2m   = ta.t2m.reshape(-1, 1)
        self.p0    = ta.p0.reshape(-1, 1)
        self.pr    = ta.pratio.reshape(-1, 1)
        stype      = ta.stype

        # one hot encoding for categorical variable stype
        self.stype = self.to_encode(stype)
        
        # given all inputs, the chosen variables for
        #training given as "inputs" on init
        all_inputs = [TB[:, :3], 
                      self.t0, 
                      self.lon, 
                      self.lat,
                      self.stype,
                      self.t2m, 
                      self.wvp,
                      self.z0,
                      self.p0] 
        
        inputnames = np.array(["ta",
                               "t0",
                               "lon",
                               "lat",
                               "stype",
                               "t2m",
                               "wvp",
                               "z0", 
                               "p0"])
        

        outputnames = np.array(["iwp", "wvp"])
        
        idy         = np.argwhere(outputnames == outputs)[0][0]
        
        self.inputs = inputs       
        idx = []

        for i in range(len(inputs)):
            idx.append(np.argwhere(inputnames == inputs[i])[0][0]) 
                                                                            

        self.index = idx

        C = []
        for i in idx:
            C.append(all_inputs[i])
            
        x = np.float32(np.concatenate(C, axis = 1))

        
        if latlims is not None:            
            ilat = (np.abs(self.lat) >= latlims[0]) & (np.abs(self.lat) <= latlims[1])
        
        if pratio is not None:
            ipratio = self.pr <= pratio
            
            ilat = np.logical_and(ilat, ipratio)    

            
            x         = x[ilat.ravel(), :]
            self.lat  = self.lat[ilat]
            self.iwp  = self.iwp[ilat]
            self.lon  = self.lon[ilat] 
            self.wvp  = self.wvp[ilat]
            self.rwp  = self.rwp[ilat]
            #self.lst  = self.lst[ilat]
            self.t2m  = self.t2m[ilat]
            self.t0   = self.t0[ilat]
            self.z0   = self.z0[ilat]
            self.p0   = self.p0[ilat]            
            
        all_outputs = [self.iwp, self.wvp]    
            
        tindex = self.get_indices("ta")
        self.chindex = np.arange(tindex, tindex + 3, 1)
        
        
        if normalise is None:
            
            if "stype" in inputs:
                sindex  = self.get_indices("stype") + 3
                exindex = np.arange(sindex, sindex + 11 , 1)
       
            else:
                exindex = None
                
            self.norm = Normalizer(x,
                 exclude_indices= exindex,
                 feature_axis=1)

        else:

            self.norm = normalise      

        
        self.y = np.float32(all_outputs[idy])
        
        self.x = x.data[:]

        # fill in IWP below 1e-4      
        y = np.copy(self.y)
        
        inds = y < 1e-4
        y[inds] = 10 ** np.random.uniform(-6, -4, inds.sum())
        self.y = y[:]
    

        self.file.close()

    def __len__(self):
        """
        The number of entries in the training data. This is part of the
        pytorch interface for datasets.

        Return:
            int: The number of samples in the data set
        """
        if self.batch_size is None:
            return self.x.shape[0]
        else:
            return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i: The index of the sample to return
        """
        if (i == 0):
            indices = np.random.permutation(self.x.shape[0])
            self.x   = self.x[indices, :]
            self.y   = self.y[indices]
            self.lon = self.lon[indices]
            self.lat = self.lat[indices]
            #self.lst = self.lst[indices]

        if self.batch_size is None:
            
            #y = torch.tensor(self.y[[i]])
            #x = torch.tensor(self.x[[i], :])
            
            y = torch.tensor(self.y[i])
            
            x = torch.tensor(self.x[i, :])
            x = self.x[i, :].reshape(1, -1)
            # add new noise to TB in each epoch
            x_noise = np.float32(self.add_noise(x[:, :3], self.chindex))
            
            x[:, :3]      = x_noise
            
            # normalise 
            x_norm       = np.squeeze(np.float32(self.norm(x)))    
            
            x = torch.tensor(x_norm)
            
            #if self.transform is not None:
            #y = self.transform1(y)
            
            return (x, y)
        else:
            i_start = self.batch_size * i
            i_end   = self.batch_size * (i + 1)    
            
            x       = self.x[i_start : i_end, :].copy()
            
            # add new noise to TB in each epoch
            x_noise = np.float32(self.add_noise(x[:, :3], self.chindex))
            
            x[:, :3]      = x_noise
            
            # normalise 
            x_norm       = np.float32(self.norm(x))      
            
            x = torch.tensor(x_norm)
            y = torch.tensor(self.y[i_start : i_end])
            
    
            #y = self.transform1(y)
            
            return (x, y)
     
    def get_indices(self, inputname):
        
        sindex = np.where(self.inputs == inputname)[0][0]
        
        return sindex

        
    def add_noise(self, x, index):        
        """
        Gaussian noise is added to every measurement before used 
        for training again.
        
        Args: 
            the input TB in one batch of size (batch_size x number of channels)
        Returns:
            input TB with noise
            
        """
        
        nedt  = np.array([0.95, # 166 V + H
                          0.47, # 183+-7
                          0.56  # 183+-3                     
                          ])

        
        nedt_subset = nedt[index]
        size_TB = int(x.size/len(nedt_subset))
        x_noise = x.copy()
        if len(index) > 1:
            for ic in range(len(index)):
                noise = np.random.normal(0, nedt_subset[ic], size_TB)
                x_noise[:, ic] += noise
        else:
                noise = np.random.normal(0, nedt_subset, size_TB)
                x_noise[:] += noise
        return x_noise    
 
    def normalise_std(self, x):
        """
        normalise the input data with mean and standard deviation
        Args:
            x
        Returns :
            x_norm
        """          

        x_norm = (x - self.mean)/self.std   
            
        return x_norm 
        
    def normalise_minmax(self, x):

        x_norm = (x - x.min())/(x.max() - x.min())
        return x_norm



    def to_encode(self, stype):
        
        encodedlsm = to_categorical(stype, num_classes=11)
        
        return encodedlsm
    
    def to_decode(self, encodedstype):
        
        lsm = np.argmax(encodedstype, axis = 1)
        
        return lsm
    
    
    def randomize_stype(self, x, nu = 0.02):
        
        sindex = self.get_indices("stype")
      
        enstype  = x[:, sindex:sindex + 10]
        
        print (enstype.shape)
        stype  = self.to_decode(enstype)   
        
        lsm  = stype.copy() 
        nlen = lsm.size
        

        
        n    = max(np.int(nu * nlen), 1)
        
        ix   = random.sample(range(0,  nlen-1), n)
        
        nlsm = np.random.randint(0, 9, n)
        
        lsm[ix] = nlsm
        
        enstype = self.to_encode(lsm)
        
        x[:, sindex:sindex + 10] = enstype
        
        return x
    

    def transform1(self, y):

                
        logy = LogLinear()           
        y    = logy(y)
        
        return y