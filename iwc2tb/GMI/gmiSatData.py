import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset
from iwc2tb.GMI.lsm_gmi2arts import lsm_gmi2arts
from iwc2tb.GMI.swap_gmi_183 import swap_gmi_183
from numpy import argmax
import random
from keras.utils import to_categorical
from quantnn.normalizer import Normalizer 


class gmiSatData(Dataset):
    """
    Pytorch dataset for the GMI training data for IWP retrievals

    """
    def __init__(self, gmi, 
                 inputs,
                 outputs,
                 batch_size = None,
                 latlims = None,
                 normalize = None,                

                 log = False):
        """
        Create instance of the dataset from a given file path.

        Args:
            path: Path to the NetCDF4 containing the data.
            batch_size: If positive, data is provided in batches of this size

        """
        super().__init__()
        
        self.batch_size = batch_size

        self.norm   = normalize

        self.gmi    = gmi

        TB = self.gmi.tb
        
        TB = swap_gmi_183(TB)
        
        self.lsm   = self.gmi.get_lsm()
        self.lon   = self.gmi.lon
        self.lat   = self.gmi.lat
        self.iwp   = self.gmi.iwp
        self.rwp   = self.gmi.rwp
        self.t2m   = self.gmi.t0
        self.wvp   = self.gmi.wvp
        self.lst   = self.gmi.lst
        self.z0    = self.gmi.z0
        stype      = lsm_gmi2arts(self.lsm) 

        # one hot encoding for categorical variable stype
        self.stype = self.to_encode(stype)
        
        # given all inputs, the chosen variables for
        # training given as "inputs" on init
        all_inputs = [TB, 
                      self.lon[:, :, np.newaxis], 
                      self.lat[:, :, np.newaxis],
                      self.stype,
                      self.t2m[:, :, np.newaxis], 
                      self.wvp[:, :, np.newaxis],
                      self.z0[:, :, np.newaxis]] 
        
        inputnames = np.array(["ta",
                               "lon",
                               "lat",
                               "stype",
                               "t2m",
                               "wvp",
                               "z0"])
        

        outputnames = np.array(["iwp", "wvp"])

        
        idy         = np.argwhere(outputnames == outputs)[0][0]
        
        self.inputs = inputs       
        idx = []
        
        for i in range(len(inputs)):
            idx.append(np.argwhere(inputnames == inputs[i])[0][0]) 
                                                                            

        self.index = idx
        self.chindex = [0, 1, 2, 3]
        C = []
        for i in idx:
            C.append(all_inputs[i])
            
        x = np.float32(np.concatenate(C, axis = 2))
        
        ilat = np.logical_and(np.abs(self.lat) >= latlims[0],
                                      np.abs(self.lat) <= latlims[1])
        

        x         = x[ilat[:, 0], :, :]
        self.iwp  = self.iwp[ilat[:, 0], :]
        self.lat  = self.lat[ilat[:, 0], :]
        self.lon  = self.lon[ilat[:, 0], :] 
        self.wvp  = self.wvp[ilat[:, 0], :]
        self.rwp  = self.rwp[ilat[:, 0], :]
        self.lst  = self.lst[ilat[:, 0], :]
        self.t2m  = self.t2m[ilat[:, 0], :]
        self.z0   = self.z0[ilat[:, 0], :]

        self.stype = self.stype[ilat[:, 0], :]
        


        all_outputs = [self.iwp, self.wvp]    

        
        self.y = np.float32(all_outputs[idy])
        
        self.x = x


        nanmask = self.y <= 0
        if log == True:
            self.y[~nanmask] = np.log(self.y[~nanmask])            
            self.y[nanmask]  = np.nan    


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


        if self.batch_size is None:
            return (torch.tensor(self.x[i, :, :]),
                    torch.tensor(self.y[i, :, :]))
        else:
            
            i_start         = self.batch_size * i
            i_end           = self.batch_size * (i + 1)             
            
            x               = self.x[i_start : i_end, :, :]
            
            # normalise 
            x_norm          = x.copy()
            
            i, j, k         = x.shape
            x_norm          = x.reshape(i*j, k)
            
            x               = self.norm(x_norm)
            x               = x.reshape(i, j, k)

            
            return (torch.tensor(x),
                    torch.tensor(self.y[i_start : i_end, :]))
        
  
    def to_encode(self, stype):
        
        # class 11 is defined to handle np.nan, it is "others" class
        stype[np.isnan(stype)] = 10
        
        encodedlsm = to_categorical(stype, num_classes=11)
        
        return encodedlsm
    
    def to_decode(self, encodedstype):
        
        lsm = np.argmax(encodedstype, axis = 2)
        
        return lsm

                
            
            
        
            
