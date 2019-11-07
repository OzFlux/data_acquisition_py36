#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:10:05 2019

@author: imchugh
"""

import glob
import xarray as xr

def preproc(ds):
    return ds.drop(labels=ds.time[-1].data, dim='time')

base_dir = '/rdsi/market/access_opendap/monthly'

f_list = sorted(glob.glob(base_dir + '/**/AdelaideRiver*'))

ds = xr.open_mfdataset(f_list, concat_dim='time', preprocess=preproc)
