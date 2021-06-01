#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 21:19:33 2021

@author: inderpreet
"""


import numpy as np
       
def expand_lon(lon, A):
    """
    Expands longitude of ERA5 by one point to mimic 
    wrapping of data in interpolation
    
    extra longitudnal point 360.0 is added, the corresponding value is copied
    from longitude 0 deg.
    
    Parameters
    ----------
    lon : np.array containing longitude values from 0 - 359.75
    A : np.array containing the values at grid points defined by lon 
    Returns
    -------
    lon : np.array with extended longitudes 
    A : np.array with extra dimensions 
    """
    
    A_start = A[ :, -1:]
    A_end   = A[ :,  :1]
    A = np.concatenate([A_start, A, A_end], axis = 1)
 
    lon = np.concatenate(([lon.min() - 0.25], lon))    
    lon = np.concatenate((lon, [0.25 + lon.max()]))

    return lon, A    