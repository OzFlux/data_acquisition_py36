#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:16:21 2019

@author: ian
"""

import datetime as dt
import numpy as np
import os
import pandas as pd
import xarray as xr
import pdb

def screen_vars(series):
    
    range_lims = range_dict[series.name]
    return series.where((series >= range_lims[0]) & (series <= range_lims[1]))

#------------------------------------------------------------------------------
# CONVERSION ALGORITHMS
#------------------------------------------------------------------------------

def convert_Kelvin_to_celsius(s):
    
    return s - 273.15
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def convert_pressure(df):
    
    return df.ps / 1000.0    
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_Ah(df):
    
    return get_e(df) * 10**6 / (df.Ta * 8.3143) / 18
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_e(df):

    Md = 0.02897   # molecular weight of dry air, kg/mol
    Mv = 0.01802   # molecular weight of water vapour, kg/mol    
    return df.q * (Md / Mv) * (df.ps / 1000)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_energy_components(df):
    
    new_df = pd.DataFrame(index = df.index)
    new_df['Fsu'] = df.Fsd - df.Fn_sw
    new_df['Flu'] = df.Fld - df.Fn_lw
    new_df['Fn'] = (df.Fsd - new_df.Fsu) + (df.Fld - new_df.Flu)
    new_df['Fa'] = df.Fh + df.Fe
    new_df['Fg'] = new_df.Fn - new_df.Fa
    return new_df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_es(df):
    
    return 0.6106 * np.exp(17.27 * (df.Ta - 273.15) / ((df.Ta - 273.15)  + 237.3))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_Rh(df):
    
    return get_e(df) / get_es(df) * 100
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_wind_direction(df):    
    
    s = float(270) - np.arctan2(df.v, df.u) * float(180) / np.pi
    s.loc[s > 360] -= float(360)
    return
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_wind_speed(df):
    
    return np.sqrt(df.u**2 + df.v**2)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def do_conversions(df):
    
    new_df = df[['Habl', 'Fsd', 'Fld', 'Fh', 'Fe', 'Precip']].copy()
    for var in filter(lambda x: 'Ts' in x, df.columns):
        new_df[var] = convert_Kelvin_to_celsius(df[var])
    new_df['Ta'] = convert_Kelvin_to_celsius(df.Ta)
    new_df['Rh'] = get_Rh(df)
    new_df['Ah'] = get_Ah(df)
    new_df = new_df.join(get_energy_components(df))
    return new_df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# GLOBAL DICTS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
vars_dict = {'av_swsfcdown': 'Fsd',
             'av_netswsfc': 'Fn_sw',
             'av_lwsfcdown': 'Fld',
             'av_netlwsfc': 'Fn_lw',
             'temp_scrn': 'Ta',
             'qsair_scrn': 'q',
             'soil_mois': 'Sws',
             'soil_temp': 'Ts',
             'u10': 'u',
             'v10': 'v',
             'sfc_pres': 'ps',
             'inst_prcp': 'Precip',
             'sens_hflx': 'Fh',
             'lat_hflx': 'Fe',
             'abl_ht': 'Habl'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
range_dict = {'av_swsfcdown': [0, 1400],
              'av_netswsfc': [0, 1400],
              'av_lwsfcdown': [200, 600],
              'av_netlwsfc': [-300, 300],
              'temp_scrn': [230, 330],
              'qsair_scrn': [0, 1],
              'soil_mois': [0, 100],
              'soil_temp': [210, 350],
              'u10': [-50, 50],
              'v10': [-50, 50],
              'sfc_pres': [75000, 110000],
              'inst_prcp': [0, 100],
              'sens_hflx': [-200, 1000],
              'lat_hflx': [-200, 1000],
              'abl_ht': [0, 5000]}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
funcs_dict = {'av_swsfcdown': [0, 1400],
              'av_netswsfc': [0, 1400],
              'av_lwsfcdown': [200, 600],
              'av_netlwsfc': [-300, 300],
              'temp_scrn': [230, 330],
              'qsair_scrn': [0, 1],
              'soil_mois': [0, 100],
              'soil_temp': [210, 350],
              'u10': [-50, 50],
              'v10': [-50, 50],
              'sfc_pres': [75000, 110000],
              'accum_prcp': [0, 100],
              'sens_hflx': [-200, 1000],
              'lat_hflx': [-200, 1000],
              'abl_ht': [0, 5000]}
#------------------------------------------------------------------------------

input_path = '/home/ian/Desktop/Adelaide_River.nc'
in_ds = xr.open_dataset(input_path)

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### BEGINNING OF CLASS SECTION ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class access_data_getter(object):
    
    def __init__(self, site_name):
        
        self.site_name = site_name
        

    def open_raw_file(self):
        
        fname = '{}.nc'.format(self.site_name)
        path = os.path.join(input_path, fname)
        return xr.open_dataset(path)
#------------------------------------------------------------------------------


# Rename variables in row (longitude) major, column (latitude) minor format
results = []
for i, this_lat in enumerate(in_ds.lat):
    for j, this_lon in enumerate(in_ds.lon):
        new_df = pd.DataFrame(index = in_ds.time)         
        for var in vars_dict.keys():
            swap_var = vars_dict[var]
            if len(in_ds[var].dims) == 3:
                new_df[swap_var] = screen_vars(in_ds[var].sel(lat=this_lat, 
                                                              lon=this_lon))
            else:
                level = in_ds.soil_lvl[0]
                new_df[swap_var] = screen_vars(in_ds[var].sel(soil_lvl=level, 
                                                              lat=this_lat, 
                                                              lon=this_lon))
        attrs_dict = {'Latitude': this_lat.item(), 'Longitude': this_lon.item()}
        conv_df = do_conversions(new_df)
        conv_df.columns = ['{}_{}'.format(x, str(i) + str(j)) for x in 
                           conv_df.columns]
        conv_ds = conv_df.to_xarray()
        for this_var in conv_df.columns: conv_ds[this_var].attrs = attrs_dict
        results.append(conv_ds)
out_ds = xr.merge(results)
out_ds.attrs = {'nrecs': len(out_ds.time),
                'start_date': dt.datetime.strftime(conv_df.index[0].to_pydatetime(), 
                                                   '%Y-%m-%d %H:%M:%S'),
                'end_date': dt.datetime.strftime(conv_df.index[-1].to_pydatetime(), 
                                                   '%Y-%m-%d %H:%M:%S'),
                'latitude': in_ds.lat[1].item(),
                'longitude': in_ds.lon[1].item()}
in_ds.close()

