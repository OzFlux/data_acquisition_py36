#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:38:30 2018

@author: ian
"""
# System modules
from collections import namedtuple
from collections import OrderedDict
import datetime as dt
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
from scipy import interpolate, signal
import webbrowser
import xarray as xr
import pdb

# Custom modules
import utils

#------------------------------------------------------------------------------
### Remote configurations ###
#------------------------------------------------------------------------------

api_base_url = 'https://modis.ornl.gov/rst/api/v1/'

#------------------------------------------------------------------------------
### Local configurations ###
#------------------------------------------------------------------------------

master_file_path = '/mnt/OzFlux/Sites/site_master.xls'
#master_file_path = '/home/ian/Temp/site_master.xls'
output_path = '/rdsi/market/CloudStor/Shared/MODIS'
#output_path = '/home/ian/Desktop/MODIS'
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### BEGINNING OF CLASS SECTION ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class modis_data():

    #--------------------------------------------------------------------------
    '''
    Object containing MODIS subset data

    Args:
        * latitude (int or float): decimal latitude of location
        * longitude (int or float): decimal longitude of location
        * product (str): MODIS product for which to retrieve data (note that 
          not all products are available from the web service - use the 
          'get_product_list()' function of this module for a list of the 
          available products)'
        * band (str): MODIS product band for which to retrieve data (use the 
          'get_band_list(<product>)' function for a list of the available 
          bands)
    Kwargs:
        * start_date (python datetime or None): first date for which data is 
          required, or if None, first date available on server
        * end_date (python datetime): last date for which data is required,
          or if None, last date available on server
        * subset_height_km (int): distance in kilometres (centred on location)
          from upper to lower boundary of subset
        * subset_width_km (int): distance in kilometres (centred on location)
          from left to right boundary of subset
        * site (str): name of site to be attached to the global attributes 
          of the xarray dataset
        * qcfiltered (bool): whether to eliminate observations that fail 
          modis qc
    
    Returns:
        * MODIS data class containing the following:
            * band (attribute): MODIS band selected for retrieval
            * cellsize (attribute): actual width of pixel in m
    '''

    def __init__(self, product, band, latitude, longitude, 
                 start_date=None, end_date=None,
                 subset_height_km=0, subset_width_km=0, site=None, 
                 qcfiltered=False):
        
        # Check validity of passed arguments 
        if not product in get_product_list(include_details = False):
            raise KeyError('Product not available from web service! Check '
                           'available products list using get_product_list()')
        try:
            band_attrs = get_band_list(product)[band]
        except KeyError:
            raise KeyError('Band not available for {}! Check available bands '
                           'list using get_band_list(product)'.format(product))
        if start_date is None or end_date is None:
            dates = get_product_dates(product, latitude, longitude)
        if start_date is None: 
            start_date = modis_to_from_pydatetime(dates[0]['modis_date'])
        if end_date is None: 
            end_date = modis_to_from_pydatetime(dates[-1]['modis_date'])

        # Get the data and write additional attributes
        self.data_array = request_subset_by_coords(product, latitude, longitude, 
                                                   band, start_date, end_date, 
                                                   subset_height_km,
                                                   subset_width_km)
        band_attrs.update({'site': site})
        self.data_array.attrs.update(band_attrs)
        
        # QC if requested
        if qcfiltered:
            qc_dict = get_qc_details(product)
            qc_array = request_subset_by_coords(product, latitude, longitude, 
                                                qc_dict['qc_name'], 
                                                start_date, end_date, 
                                                subset_height_km, subset_width_km)
            self._qc_data_array(qc_array, qc_dict)
    #--------------------------------------------------------------------------
        
    #--------------------------------------------------------------------------
    def data_array_by_pixels(self, interpolate_missing = True, 
                             smooth_signal = False, 
                             upsample_to_daily = False):

        d = {}
        var_attrs = {}
        rows = self.data_array.attrs['nrows']
        cols = self.data_array.attrs['ncols']
        for i in range(rows):
            for j in range(cols):
                name_idx = str((i * rows) + j + 1)
                var_name = 'pixel_{}'.format(name_idx)
                d[var_name] = self.data_array.data[i,j,:]
                var_attrs[var_name] = {'x': self.data_array.x[j].item(),
                                       'y': self.data_array.y[i].item(),
                                       'row': i, 'col': j}
        df = pd.DataFrame(d, index = self.data_array.time)
        idx_start, idx_end = df.index[0], df.index[-1]
        if interpolate_missing or smooth_signal:
            df = df.apply(_interp_missing, upsample_to_daily=upsample_to_daily)
            if upsample_to_daily:
                df.index = pd.date_range(idx_start, idx_end, freq='D')
                df.index.name = 'time'
        if smooth_signal:
            df = df.apply(_smooth_signal)
        out_xarr = df.to_xarray()
        out_xarr.attrs = self.data_array.attrs
        for var in var_attrs.keys():
            out_xarr[var].attrs = var_attrs[var]
        return out_xarr
    
    # a=x.data_array.to_dataframe().unstack().T
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_spatial_mean(self, filter_outliers=True, interpolate_missing=True,
                         smooth_signal=False, upsample_to_daily=False):
        
        idx = pd.to_datetime(self.data_array.time.data)
        l = []
        if filter_outliers:
            for i in range(self.data_array.data.shape[2]): 
                l.append(_median_filter(self.data_array.data[:, :, i]))
            arr = np.array(l)
        else:
            arr = self.data_array.mean(['x', 'y']).to_series()
        if interpolate_missing or smooth_signal or upsample_to_daily:
            interp_arr = _interp_missing(pd.Series(arr, index = idx),
                                         upsample_to_daily)
            if upsample_to_daily: idx = pd.date_range(idx[0], idx[-1], freq='D')
            if not smooth_signal: return pd.Series(interp_arr, index = idx)
        if smooth_signal:
            return pd.Series(_smooth_signal(interp_arr), index = idx)
        return pd.Series(arr, index = idx)
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------
    def plot_data(self, pixel = 'centre', plot_to_screen = True):

        state = mpl.is_interactive()
        if plot_to_screen: plt.ion()
        if not plot_to_screen: plt.ioff()
        df = self.data_array_by_pixels().to_dataframe()
        smooth_df = self.data_array_by_pixels(smooth_signal = True).to_dataframe()
        if pixel == 'centre':
            target_pixel = int(len(df.columns) / 2)
        elif isinstance(pixel, int):
            if pixel > len(df.columns) or pixel == 0: 
                raise IndexError('Pixel out of range!')
            pixel = pixel - 1
            target_pixel = pixel
        else:
            raise TypeError('pixel kwarg must be either "mean" or an integer!')
        col_name = df.columns[target_pixel]
        series = df[col_name]
        smooth_series = smooth_df[col_name]
        mean_series = df.mean(axis = 1)
        fig, ax = plt.subplots(1, 1, figsize = (14, 8))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis = 'y', labelsize = 14)
        ax.tick_params(axis = 'x', labelsize = 14)
        ax.set_xlabel('Date', fontsize = 14)
        y_label = '{0} ({1})'.format(self.data_array.attrs['band'],
                                     self.data_array.attrs['units'])
        ax.set_ylabel(y_label, fontsize = 14)
        ax.plot(df.index, df[df.columns[0]], color = 'grey', alpha = 0.1, label = 'All pixels')
        ax.plot(df.index, df[df.columns[1:]], color = 'grey', alpha = 0.1)
        ax.plot(df.index, mean_series, color = 'black', alpha = 0.5, label = 'All pixels (mean)')
        ax.plot(df.index, series, lw = 2, label = col_name)
        ax.plot(df.index, smooth_series, lw = 2, label = '{}_smoothed'.format(col_name))
        ax.legend(frameon = False)
        plt.ion() if state else plt.ioff()
        return fig
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    def _qc_data_array(self, qc_array, qc_dict):
        
        # Apply range limits
        range_limits = self.data_array.attrs['valid_range'].split('to')
        mn, mx = float(range_limits[0]), float(range_limits[-1])
        scale = float(self.data_array.attrs['scale_factor'])
        mn = scale * mn
        mx = scale * mx
        self.data_array = self.data_array.where((self.data_array >= mn) & 
                                                (self.data_array <= mx))
        
        # Apply qc
        max_allowed = qc_dict['reliability_threshold']
        self.data_array = self.data_array.where((qc_array <= max_allowed))
    #--------------------------------------------------------------------------
        
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class modis_data_network(modis_data):
    
    #--------------------------------------------------------------------------
    '''
    Object containing MODIS subset data
    
    Args:
        * product (str): MODIS product for which to retrieve data (note that 
          not all products are available from the web service - use the 
          'get_product_list()' function of this module for a list of the 
          available products)'
        * band (str): MODIS product band for which to retrieve data (use the 
          'get_band_list(<product>)' function for a list of the available 
          bands)
        * network_name (str): network for which to retrieve data (use the 
          'get_network_list()' function for a list of the available networks)
        * site_ID (str): network site for which to retrieve data (use the 
          'get_network_list(<network>)' function for a list of the available 
          sites and corresponding codes within a network)
        * start_date (python datetime or None): first date for which data is 
          required, or if None, first date available on server
        * end_date (python datetime): last date for which data is required,
          or if None, last date available on server
        * qcfiltered (bool): whether or not to impose QC filtering on the data
    
    Returns:
        * MODIS data class containing the following:
            * band (attribute): MODIS band selected for retrieval
            * cellsize (attribute): actual width of pixel in m
    '''
    def __init__(self, product, band, network_name, site_ID,  
                 start_date = None, end_date = None, qcfiltered = False):

        if not product in get_product_list(include_details = False):
            raise KeyError('Product not available from web service! Check '
                           'available products list using get_product_list()')
        try:
            band_attrs = get_band_list(product)[band]
        except KeyError:
            raise KeyError('Band not available for {}! Check available bands '
                           'list using get_band_list(product)'.format(product))
        try:
            site_attrs = get_network_list(network_name)[site_ID]
        except KeyError:
            raise KeyError('Site ID code not found! Check available site ID '
                           'codes using get_network_list(network)')
        if not network_name in get_network_list():
            raise KeyError('Network not available from web service! Check '
                           'available networks list using get_network_list()')
        
        if start_date is None or end_date is None:
            dates = get_product_dates(product, 
                                      site_attrs['latitude'],
                                      site_attrs['longitude'])
        if start_date is None: 
            start_date = modis_to_from_pydatetime(dates[0]['modis_date'])
        if end_date is None: 
            end_date = modis_to_from_pydatetime(dates[-1]['modis_date'])
        
        # Get the data (QC if requested)
        self.data_array = request_subset_by_siteid(product, band, network_name,
                                                   site_ID, start_date, end_date, 
                                                   qcfiltered = qcfiltered)
        band_attrs.update({'site': site_attrs['network_sitename']})
        self.data_array.attrs.update(band_attrs)        
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    def _qc_data_array(self):
        
        print('Not defined for "modis_data_network" class!')
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _error_codes(json_obj):
    
    d = {400: 'Invalid band for product',
         404: 'Product not found'}
    
    status_code = json_obj.status_code
    if status_code == 200: return
    try: 
        error = d[status_code]
    except KeyError:
        error = 'Unknown error code ({})'.format(str(status_code))
    raise RuntimeError('retrieval failed - {}'.format(error))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------    
def get_band_list(product, include_details = True):
    
    """Get available bands for a given product"""
    
    json_obj = requests.get(api_base_url + product + '/bands')
    band_list = json.loads(json_obj.content)['bands']
    d = OrderedDict(list(zip([x.pop('band') for x in band_list], band_list)))
    if include_details: return d
    return list(d.keys())
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_chunks(l, n = 10):

    """yield successive n-sized chunks from list l"""
    
    for i in range(0, len(l), n): yield l[i: i + n]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_product_dates(product, lat, lng):
    
    """Get all available dates for given product and location"""
    
    req_str = "".join([api_base_url, product, "/dates?", "latitude=", str(lat), 
                       "&longitude=", str(lng)])
    json_obj = requests.get(req_str)
    date_list = json.loads(json_obj.content)['dates']
    return date_list
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_product_list(include_details = True):
    
    """Get list of available products"""
    
    json_obj = requests.get(api_base_url + 'products')
    products_list = json.loads(json_obj.content)['products']
    d = OrderedDict(list(zip([x.pop('product') for x in products_list], 
                        products_list)))
    if include_details: return d
    return list(d.keys())
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_product_web_page(product = None):
    
    """Go to web page for product"""
    
    products_list = get_product_list()
    modis_url_dict = {prod: '{}v006'.format(prod.lower()) for prod in 
                      products_list if prod[0] == 'M'}
    viirs_url_dict = {prod: '{}v001'.format(prod.lower()) 
                      for prod in products_list if prod[:3] == 'VNP'}
    modis_url_dict.update(viirs_url_dict)
    base_addr = ('https://lpdaac.usgs.gov/products/{0}')
    if product is None or not product in list(modis_url_dict.keys()):
        print('Product not found... redirecting to data discovery page')
        addr = ('https://lpdaac.usgs.gov')
    else:
        addr = base_addr.format(modis_url_dict[product])
    webbrowser.open(addr)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_network_list(network = None, include_details = True):
    
    """Get list of available networks (if None) or sites within network if 
       network name supplied"""
    
    if network == None: 
        json_obj = requests.get(api_base_url + 'networks')
        return json.loads(json_obj.content)['networks']
    url = api_base_url + '{}/sites'.format(network)
    json_obj = requests.get(url)
    sites_list = json.loads(json_obj.content)
    d = OrderedDict(list(zip([x.pop('network_siteid') for x in sites_list['sites']], 
                    sites_list['sites'])))
    if include_details: return d
    return list(d.keys())
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_pixel_subset(x_arr, pixels_per_side = 3):
    
    """Create a spatial subset of a larger dataset (relative to centre pixel)"""
    
    try:
        assert x_arr.nrows == x_arr.ncols
    except AssertionError: 
        raise RuntimeError('Malformed data array!')
    if not x_arr.nrows % 2 != 0:
        raise TypeError('pixels_per_side must be an odd integer!')
    if not pixels_per_side < x_arr.nrows:
        print('Pixels requested exceeds pixels available!')
        return x_arr
    centre_pixel = int(x_arr.nrows / 2)
    pixel_min = centre_pixel - int(pixels_per_side / 2)
    pixel_max = pixel_min + pixels_per_side
    new_data = []
    for i in range(x_arr.data.shape[2]):
        new_data.append(x_arr.data[pixel_min: pixel_max, pixel_min: pixel_max, i])
    new_x = x_arr.x[pixel_min: pixel_max]
    new_y = x_arr.y[pixel_min: pixel_max]
    attrs_dict = x_arr.attrs.copy()
    attrs_dict['nrows'] = pixels_per_side
    attrs_dict['ncols'] = pixels_per_side
    attrs_dict['xllcorner'] = new_x[0].item()
    attrs_dict['yllcorner'] = new_y[0].item()
    return xr.DataArray(name = x_arr.band,
                        data = np.dstack(new_data),
                        coords = [new_y, new_x, x_arr.time],
                        dims = [ "y", "x", "time" ],
                        attrs = attrs_dict)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_qc_details(product = None):

    """Get the qc variable details for the product"""
    
    d = {'MOD13Q1': {'qc_name': '250m_16_days_pixel_reliability',
                     'reliability_threshold': 1,
                     'bitmap': {'0': 'Good data', '1': 'Marginal data',
                                '2': 'Snow/Ice', '3': 'Cloudy'}}}
    if not product: return d
    return d[product]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _interp_missing(series, upsample_to_daily=False):
    
    """Interpolate (Akima) signal"""
    
    days = np.array((series.index - series.index[0]).days)
    data = np.array(series)
    valid_idx = np.where(~np.isnan(data))
    f = interpolate.Akima1DInterpolator(days[valid_idx], data[valid_idx])
    if upsample_to_daily: days = np.arange(days[0], days[-1] + 1, 1)
    return f(days)
#------------------------------------------------------------------------------

#--------------------------------------------------------------------------
def _median_filter(arr, mult = 1.5):
    
    """Filter outliers from the 2d spatial array"""
  
    n_valid = sum(~np.isnan(arr)).sum()
    if n_valid == 0: return np.nan
    pct75 = np.nanpercentile(arr, 75)
    pct25 = np.nanpercentile(arr, 25)
    iqr = pct75 - pct25
    range_min = pct25 - mult * iqr
    range_max = pct75 + mult * iqr
    filt_arr = np.where((arr < range_min) | (arr > range_max), np.nan, arr)
    return np.nanmean(filt_arr)
#--------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def modis_to_from_pydatetime(date):
    
    """Convert between MODIS date strings and pydate format"""
    
    if isinstance(date, str): 
        return dt.datetime.strptime(date[1:], '%Y%j').date()
    return dt.datetime.strftime(date, 'A%Y%j')
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def _process_data(data, prod, band):
    
    """Process the raw data into a more human-intelligible format (xarray)"""

    meta = {key:value for key,value in list(data[0].items()) 
            if key != "subset" }
    meta['product'] = prod
    meta['band'] = band
    if 'scale' in meta:
        try: scale = float(meta['scale'])
        except ValueError: scale = None
    if not 'scale' in meta: scale = None
    data_dict = {'dates': [], 'arrays': [], 'metadata': meta}
    for i in data:
        for j in i['subset']:
            if j['band'] == meta['band']:
                data_dict['dates'].append(j['calendar_date'])
                data_list = []
                for obs in j['data']:
                    try: data_list.append(float(obs))
                    except ValueError: data_list.append(np.nan)
                new_array = np.array(data_list).reshape(meta['nrows'], 
                                                        meta['ncols'])
                data_dict['arrays'].append(new_array)
    stacked_array = np.dstack(data_dict['arrays'])
    if scale: stacked_array = stacked_array * scale
    dtdates = [dt.datetime.strptime(d,"%Y-%m-%d") for d in data_dict['dates']]
    xcoordinates = ([float(meta['xllcorner'])] + 
                    [i * meta['cellsize'] + float(meta['xllcorner']) 
                     for i in range(1, meta['ncols'])])
    ycoordinates = ([float(meta['yllcorner'])] + 
                     [i * meta['cellsize'] + float(meta['yllcorner'])
                      for i in range(1, meta['nrows'])])
    ycoordinates = list(reversed(ycoordinates))
    return xr.DataArray(name = meta['band'], data = stacked_array,
                        coords = [np.array(ycoordinates), 
                                  np.array(xcoordinates), dtdates],
                        dims = [ "y", "x", "time" ], attrs = meta)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def request_subset_by_coords(prod, lat, lng, band, start_date, end_date, 
                             ab = 0, lr = 0, qcfiltered = False):
    
    """Get the data from ORNL DAAC by coordinates - are we double-handling dates here?"""
    
    def getSubsetURL(this_start_date, this_end_date):
        return( "".join([api_base_url, prod, "/subset?",
                     "latitude=", str(lat),
                     "&longitude=", str(lng),
                     "&band=", str(band),
                     "&startDate=", this_start_date,
                     "&endDate=", this_end_date,
                     "&kmAboveBelow=", str(ab),
                     "&kmLeftRight=", str(lr)]))    
            
    if not (isinstance(ab, int) and isinstance(lr, int)): 
        raise TypeError('km_above_below (ab) and km_left_right (lr) must be '
                        'integers!')    
    dates = [x['modis_date'] for x in get_product_dates(prod, lat, lng)]
    pydates_arr = np.array([modis_to_from_pydatetime(x) for x in dates])
    if not start_date: start_date = pydates_arr[0]
    if not end_date: end_date = pydates_arr[-1]
    start_idx = abs(pydates_arr - start_date).argmin()
    end_idx = abs(pydates_arr - end_date).argmin()
    date_chunks = list(_get_chunks(dates[start_idx: end_idx]))
    subsets = []
    print('Retrieving data for product {0}, band {1}:'.format(prod, band))
    for i, chunk in enumerate(date_chunks):
        print('[{0} / {1}] {2} - {3}'.format(str(i + 1), str(len(date_chunks)),
              chunk[0], chunk[-1]))
        url = getSubsetURL(chunk[0], chunk[-1])
        subset = request_subset_by_URLstring(url)
        subsets.append(subset)
    return _process_data(subsets, prod, band)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def request_subset_by_siteid(prod, band, network, siteid, start_date, end_date, 
                             qcfiltered = False):
    
    """Get the data from ORNL DAAC by network and site id"""
    
    modis_start = modis_to_from_pydatetime(start_date)
    modis_end = modis_to_from_pydatetime(end_date)
    print('Retrieving data for product {0}, band {1}'.format(prod, band))
    subset_str = '/subsetFiltered?' if qcfiltered else '/subset?'
    url = (''.join([api_base_url, prod, '/', network, '/', siteid, 
                    subset_str, band, '&startDate=', modis_start,
                    '&endDate=', modis_end]))
    subset = request_subset_by_URLstring(url)
    return _process_data([subset], prod, band)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def request_subset_by_URLstring(URLstr):

    """Submit request to ORNL DAAC server"""
    
    header = {'Accept': 'application/json'}
    response = requests.get(URLstr, headers = header)
    _error_codes(response)
    return json.loads(response.text)    
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _smooth_signal(series, n_points = 11, poly_order = 3):
    
    """Smooth (Savitzky-Golay) signal"""
    
    return signal.savgol_filter(series, n_points, poly_order, mode = "mirror")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def modis_object(by_coords=True):
    if by_coords:
        return namedtuple('modis_by_coords', 
                          ['product', 'band', 'latitude', 'longitude', 
                           'start_date', 'end_date', 'subset_height_km', 
                           'subset_width_km'])
        return namedtuple('modis_by_network', 
                          ['product', 'band', 'network_name', 'site_ID', 
                           'start_date', 'end_date'])
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
### MAIN PROGRAM
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
if __name__ == "__main__":

    # Get sites info for processing
    sites = sites=utils.get_ozflux_site_list(master_file_path)
    
    # Get list of ozflux sites that are in the MODIS collection (note Wombat 
    # has designated site name 'Wombat', so change in dict)
    ozflux_modis_collection_sites = get_network_list('OZFLUX')
    coll_dict = {ozflux_modis_collection_sites[x]['network_sitename']: 
                 x for x in ozflux_modis_collection_sites.keys()}
    coll_dict['Wombat State Forest'] = coll_dict.pop('Wombat')

    # Get site data and write to netcdf
    for site in sites.index:
        
        print('Retrieving data for site {}:'.format(site))
        
        nc_file_name = '{}_EVI.nc'.format(site.replace(' ', '_'))
        plot_file_name = '{}_EVI.png'.format(site.replace(' ', '_'))
        full_nc_path = os.path.join(output_path, nc_file_name)
        full_plot_path = os.path.join(output_path, plot_file_name)
        try: first_date = dt.date(int(sites.loc[site, 'Start year']) - 1, 7, 1)
        except (TypeError, ValueError): first_date = None
        try: last_date = dt.date(int(sites.loc[site, 'End year']) + 1, 6, 1)
        except (TypeError, ValueError): last_date = None
        
        # Get sites in the collection
        if site in coll_dict.keys():
            site_code = coll_dict[site]
            x = modis_data_network('MOD13Q1', '250m_16_days_EVI', 'OZFLUX', 
                                   site_code, first_date, last_date,
                                   qcfiltered=True)
            x.data_array = _get_pixel_subset(x.data_array, pixels_per_side = 5)
        
        # Get sites not in the collection
        else:
            x = modis_data('MOD13Q1', '250m_16_days_EVI', 
                           sites.loc[site, 'Latitude'], 
                           sites.loc[site, 'Longitude'], first_date, last_date,
                           1, 1, site, qcfiltered=True)
            x.data_array = _get_pixel_subset(x.data_array, pixels_per_side = 5)
        
        # Get outputs and write to file
        thisfig = x.plot_data(plot_to_screen=False)
        thisfig.savefig(full_plot_path)
        plt.close(thisfig)
        ozflux_data_array = x.data_array_by_pixels(smooth_signal = True)
        ozflux_data_array['EVI'] = (['time'], x.get_spatial_mean())
        ozflux_data_array['EVI_smoothed'] = (['time'], 
                                             x.get_spatial_mean(smooth_signal=True))
        new_xarray = ozflux_data_array.resample({'time': '30T'}).interpolate()
        new_xarray.time.encoding = {'units': 'days since 1800-01-01',
                                    '_FillValue': None}
        new_xarray[['EVI', 'EVI_smoothed']].to_netcdf(full_nc_path, format='NETCDF4')