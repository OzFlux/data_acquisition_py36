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
import xlrd
import pdb

#------------------------------------------------------------------------------
### BEGINNING OF CLASS SECTION ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class access_data_converter():

    def __init__(self, site_details):

        self.site_name = site_details.name
        self.latitude = round(site_details.Latitude, 4)
        self.longitude = round(site_details.Longitude, 4)
        self.time_step = site_details['Time step']

    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def create_dataset(self):

        fname = '{}.nc'.format(self.site_name.replace(' ',''))
        path = os.path.join(access_file_path, fname)
        in_ds = xr.open_dataset(path)
        results = []
        for i, this_lat in enumerate(in_ds.lat):
            for j, this_lon in enumerate(in_ds.lon):
                df = pd.DataFrame(index=in_ds.time)
                for var in vars_dict.keys():
                    swap_var = vars_dict[var]
                    if len(in_ds[var].dims) == 3:
                        df[swap_var] = _screen_vars(in_ds[var].sel(lat=this_lat,
                                                                   lon=this_lon))
                    else:
                        level = in_ds.soil_lvl[0]
                        df[swap_var] = _screen_vars(in_ds[var].sel(soil_lvl=level,
                                                                   lat=this_lat,
                                                                   lon=this_lon))
                if self.time_step == 30:
                    df = df.resample('30T').interpolate()
                conv_ds = do_conversions(df).to_xarray()
                _set_variable_attributes(conv_ds,
                                         round(this_lat.item(), 4),
                                         round(this_lon.item(), 4))
                _rename_variables(conv_ds, i, j)
                results.append(conv_ds)
        in_ds.close()
        out_ds = xr.merge(results)
        self._set_global_attributes(out_ds)
        return out_ds
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _set_global_attributes(self, ds):

        ds.attrs = {'nrecs': len(ds.time),
                    'start_date': (dt.datetime.strftime
                                   (pd.Timestamp(ds.time[0].item()),
                                   '%Y-%m-%d %H:%M:%S')),
                    'end_date': (dt.datetime.strftime
                                 (pd.Timestamp(ds.time[-1].item()),
                                  '%Y-%m-%d %H:%M:%S')),
                    'latitude': self.latitude,
                    'longitude': self.longitude,
                    'site_name': self.site_name,
                    'time_step': self.time_step,
                    'xl_datemode': 0}
        ds.time.encoding = {'units': 'days since 1800-01-01',
                            '_FillValue': None}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def write_to_netcdf(self, write_path):

        print('Writing netCDF file for site {}'.format(self.site_name))
        dataset = self.create_dataset()
        fname = '{}_ozflux_access.nc'.format(''.join(self.site_name.split(' ')))
        target = os.path.join(write_path, fname)
        dataset.to_netcdf(target, format='NETCDF4')
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### END OF CLASS SECTION ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### BEGINNING OF FUNCTION SECTION ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_ozflux_site_list(master_file_path):

    wb = xlrd.open_workbook(master_file_path)
    sheet = wb.sheet_by_name('Active')
    header_row = 9
    header_list = sheet.row_values(header_row)
    df = pd.DataFrame()
    for var in ['Site', 'Latitude', 'Longitude', 'Time step', 'Start year']:
        index_val = header_list.index(var)
        df[var] = sheet.col_values(index_val, header_row + 1)
    df['Start year'] = df['Start year'].astype(int)
    df.index = df[header_list[0]]
    df.drop(header_list[0], axis=1, inplace=True)
    return df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_var_attrs(var):

    """Make a dictionary of attributes for passed variable"""

    vars_dict = {'Ah': {'long_name': 'Absolute humidity',
                        'units': 'g/m3'},
                 'Fa': {'long_name': 'Calculated available energy',
                        'units': 'W/m2'},
                 'Fe': {'long_name': 'Surface latent heat flux',
                        'units': 'W/m2'},
                 'Fg': {'long_name': 'Calculated ground heat flux',
                        'units': 'W/m2',
                        'standard_name': 'downward_heat_flux_in_soil'},
                 'Fh': {'long_name': 'Surface sensible heat flux',
                        'units': 'W/m2'},
                 'Fld': {'long_name':
                         'Average downward longwave radiation at the surface',
                         'units': 'W/m2'},
                 'Flu': {'long_name':
                         'Average upward longwave radiation at the surface',
                         'standard_name': 'surface_upwelling_longwave_flux_in_air',
                         'units': 'W/m2'},
                 'Fn': {'long_name': 'Calculated net radiation',
                        'standard_name': 'surface_net_allwave_radiation',
                        'units': 'W/m2'},
                 'Fsd': {'long_name': 'average downwards shortwave radiation at the surface',
                         'units': 'W/m2'},
                 'Fsu': {'long_name': 'average upwards shortwave radiation at the surface',
                         'standard_name': 'surface_upwelling_shortwave_flux_in_air',
                         'units': 'W/m2'},
                 'Habl': {'long_name': 'planetary boundary layer height',
                          'units': 'm'},
                 'Precip': {'long_name': 'Precipitation total over time step',
                            'units': 'mm/30minutes'},
                 'ps': {'long_name': 'Air pressure',
                        'units': 'kPa'},
                 'q': {'long_name': 'Specific humidity',
                       'units': 'kg/kg'},
                 'RH': {'long_name': 'Relative humidity',
                        'units': '%'},
                 'Sws': {'long_name': 'soil_moisture_content', 'units': 'frac'},
                 'Ta': {'long_name': 'Air temperature',
                        'units': 'C'},
                 'Ts': {'long_name': 'soil temperature',
                        'units': 'C'},
                 'Wd': {'long_name': 'Wind direction',
                        'units': 'degT'},
                 'Ws': {'long_name': 'Wind speed',
                        'units': 'm/s'}}

    generic_dict = {'instrument': '', 'valid_range': (-1e+35,1e+35),
                    'missing_value': -9999, 'height': '',
                    'standard_name': '', 'group_name': '',
                    'serial_number': ''}

    generic_dict.update(vars_dict[var])
    return generic_dict
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _rename_variables(ds, i, j):

    var_list = [x for x in list(ds.variables) if not x in list(ds.dims)]
    name_swap_dict = {x: '{}_{}'.format(x, str(i) + str(j))
                      for x in var_list}
    ds.rename(name_swap_dict, inplace=True)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _screen_vars(series):

    range_lims = range_dict[series.name]
    return series.where((series >= range_lims[0]) & (series <= range_lims[1]))
#------------------------------------------------------------------------------

#--------------------------------------------------------------------------
def _set_variable_attributes(ds, latitude, longitude):

    for this_var in list(ds.variables):
        if this_var == 'time': continue
        var_attrs = _get_var_attrs(this_var)
        var_attrs.update({'latitude': latitude,
                          'longitude': longitude})
        ds[this_var].attrs = var_attrs
        ds[this_var].encoding = {'_FillValue': -9999}
#--------------------------------------------------------------------------

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

    new_df = pd.DataFrame(index=df.index)
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
    return s
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
    new_df['RH'] = get_Rh(df)
    new_df['Ah'] = get_Ah(df)
    new_df['Ws'] = get_wind_speed(df)
    new_df['Wd'] = get_wind_direction(df)
    new_df = new_df.join(get_energy_components(df))
    return new_df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### END OF FUNCTION SECTION ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### BEGINNING OF GLOBALS ###
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
              'inst_prcp': [0, 100],
              'sens_hflx': [-200, 1000],
              'lat_hflx': [-200, 1000],
              'abl_ht': [0, 5000]}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# User configurations
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
access_file_path = '/home/ian/Desktop'
master_file_path = '/home/ian/Temp/site_master.xls'
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### END OF GLOBALS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### MAIN PROGRAM ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
if __name__ == "__main__":

    sites_df = get_ozflux_site_list(master_file_path)
    for site in sites_df.index[:1]:
        site_details = sites_df.loc[site]
        converter = access_data_converter(site_details)
        converter.write_to_netcdf(access_file_path)
#------------------------------------------------------------------------------