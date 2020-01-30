#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:38:30 2018

@author: ian
"""
# System modules
from collections import namedtuple
from collections import OrderedDict
import copy as cp
import datetime as dt
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
from requests.exceptions import ConnectionError
from scipy import interpolate, signal
from time import sleep
from types import SimpleNamespace
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

#master_file_path = '/mnt/OzFlux/Sites/site_master.xls'
master_file_path = '/home/unimelb.edu.au/imchugh/Temp/site_master.xls'
#output_path = '/rdsi/market/CloudStor/Shared/MODIS'
output_path = '/home/unimelb.edu.au/imchugh/Desktop/MODIS'
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

        # Apply range limits
        self.data_array: self._apply_range_limits()

        # QC if requested
        if qcfiltered: self._qc_data_array()
        #     qc_dict = get_qc_details(product)
        #     if not qc_dict: print('No QC variable defined!'); return
        #     qc_array = request_subset_by_coords(product, latitude, longitude,
        #                                         qc_dict['qc_name'],
        #                                         start_date, end_date,
        #                                         subset_height_km, subset_width_km)
        #     self._qc_data_array(qc_array, qc_dict)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _apply_range_limits(self):

        range_limits = self.data_array.attrs['valid_range'].split('to')
        mn, mx = float(range_limits[0]), float(range_limits[-1])
        if 'scale_factor' in self.data_array.attrs:
            scale = float(self.data_array.attrs['scale_factor'])
            mn = scale * mn
            mx = scale * mx
        self.data_array = self.data_array.where((self.data_array >= mn) &
                                                (self.data_array <= mx))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def data_array_by_pixels(self, interpolate_missing=True,
                             smooth_signal=False):

        d = {}
        var_attrs = {}
        rows = self.data_array.attrs['nrows']
        cols = self.data_array.attrs['ncols']
        for i in range(rows):
            for j in range(cols):
                name_idx = str((i * rows) + j + 1)
                var_name = 'pixel_{}'.format(name_idx)
                d[var_name] = self.data_array.data[i, j, :]
                var_attrs[var_name] = {'x': self.data_array.x[j].item(),
                                       'y': self.data_array.y[i].item(),
                                       'row': i, 'col': j}
        df = pd.DataFrame(d, index=self.data_array.time.data)
        if interpolate_missing or smooth_signal: df = df.apply(_interp_missing)
        if smooth_signal: df = df.apply(_smooth_signal)
        out_xarr = df.to_xarray()
        out_xarr.attrs = self.data_array.attrs
        for var in var_attrs.keys():
            out_xarr[var].attrs = var_attrs[var]
        return out_xarr

    # a=x.data_array.to_dataframe().unstack().T
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_spatial_mean(self, filter_outliers=True, interpolate_missing=True,
                         smooth_signal=False):

        """Must be a better way to index the time steps than as numpy arrays"""

        da = cp.deepcopy(self.data_array)
        idx = pd.to_datetime(da.time.data)
        idx.name = 'time'
        if filter_outliers:
            for i in range(da.data.shape[2]):
                da.data[:, :, i] = _median_filter(da.data[:, :, i])
        s = da.mean(['x', 'y']).to_series()
        if interpolate_missing or smooth_signal: s = _interp_missing(s)
        if smooth_signal: s = pd.Series(_smooth_signal(s), index=s.index)
        s.name = 'Mean'
        return s
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def plot_data(self, pixel='centre', plot_to_screen=True):

        state = mpl.is_interactive()
        if plot_to_screen: plt.ion()
        if not plot_to_screen: plt.ioff()
        df = self.data_array_by_pixels().to_dataframe()
        smooth_df = self.data_array_by_pixels(smooth_signal=True).to_dataframe()
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
        series_label = 'centre_pixel' if pixel == 'centre' else col_name
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
        ax.plot(df.index, series, lw = 2, label = series_label)
        ax.plot(df.index, smooth_series, lw = 2,
                label = '{}_smoothed'.format(series_label))
        ax.legend(frameon = False)
        plt.ion() if state else plt.ioff()
        return fig
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _qc_data_array(self, qc_array):

        # Apply qc
        qc_dict = get_qc_details(self.data_array.product)
        if not qc_dict: print('No QC variable defined!'); return
        if qc_dict['is_binary']:
            f = lambda x: int(bin(int(x)).split('b')[1].zfill(8)[:-5],2)
            vector_f = np.vectorize(f)
            qc_array.data = vector_f(qc_array.data)

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

        # Get the data and write additional attributes
        self.data_array = request_subset_by_siteid(product, band, network_name,
                                                   site_ID, start_date, end_date,
                                                   qcfiltered = qcfiltered)
        band_attrs.update({'site': site_attrs['network_sitename']})
        self.data_array.attrs.update(band_attrs)

        # Apply range limits
        self._apply_range_limits()
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _apply_range_limits(self):

        range_limits = self.data_array.attrs['valid_range'].split('to')
        mn, mx = float(range_limits[0]), float(range_limits[-1])
        if 'scale_factor' in self.data_array.attrs:
            scale = float(self.data_array.attrs['scale_factor'])
            mn = scale * mn
            mx = scale * mx
        self.data_array = self.data_array.where((self.data_array >= mn) &
                                                (self.data_array <= mx))
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### END OF CLASS SECTION ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### BEGINNING OF FUNCTION SECTION ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# def bin_converter(x): return int(bin(int(x)).split('b')[1].zfill(8)[:-5],2)

    # data_list = []
    # for x in series:
    #     this_bin = bin(int(x)).split('b')[-1].zfill(8)
    #     data_list.append(int(this_bin[:-5], 2))
    # return data_list
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
                     'is_binary': False, 'reliability_threshold': 1,
                     'bitmap': {'0': 'Good data', '1': 'Marginal data',
                                '2': 'Snow/Ice', '3': 'Cloudy'}},
         'MOD17A2H': {'qc_name': 'Psn_QC_500m',
                      'is_binary': True, 'reliability_threshold': 1,
                      'bitmap': {'0': 'Good data', '1': 'Marginal data',
                                 '2': 'Snow/Ice', '3': 'Cloudy'}}}
    if not product in d: return None
    return d[product]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _interp_missing(series):

    """Interpolate (Akima) signal"""

    days = np.array((series.index - series.index[0]).days)
    data = np.array(series)
    valid_idx = np.where(~np.isnan(data))
    f = interpolate.Akima1DInterpolator(days[valid_idx], data[valid_idx])
    return pd.Series(f(days), index=series.index)
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
    for this_try in range(5):
        try:
            response = requests.get(URLstr, headers = header)
            break
        except ConnectionError:
            response = None
            sleep(5)
    if response is None: raise RuntimeError('Connection error - '
                                            'server not responsing')
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
def get_config_file_by_coords(product, band, latitude, longitude,
                              start_date=None, end_date=None,
                              above_below_km=0, left_right_km=0):

    try:
        assert -180 <= longitude <= 180
        assert -90 <= latitude <= 90
    except AssertionError:
        print ('Latitude or longitude out of bounds'); raise RuntimeError
    try:
        assert isinstance(above_below_km, int)
        assert isinstance(left_right_km, int)
    except AssertionError:
        print ('"above_below_km" and "left_right_km" kwargs must be integers')
    unique_dict = {'latitude': latitude, 'longitude': longitude,
                   'above_below_km': above_below_km, 'left_right_km': left_right_km}
    common_dict = _do_common_checks(product, band, latitude, longitude,
                                    start_date, end_date)
    common_dict.update(unique_dict)
    return SimpleNamespace(**common_dict)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _do_common_checks(*args):

    product, band = args[0], args[1]
    latitude, longitude = args[2], args[3]
    start_date, end_date = args[4], args[5]

    # Check product and band are legit
    try:
        assert product in get_product_list(include_details = False)
    except AssertionError:
        print('Product not available from web service! Check available '
              'products list using get_product_list()'); raise KeyError

    try:
        get_band_list(product)[band]
    except KeyError:
        print('Band not available for {}! Check available bands '
              'list using get_band_list(product)'.format(product)); raise

    # Check and set MODIS dates
    avail_dates = get_product_dates(product, latitude, longitude)
    py_avail_dates = np.array([dt.datetime.strptime(x['modis_date'], 'A%Y%j').date()
                               for x in avail_dates])
    if start_date:
        py_start_dt = dt.datetime.strptime(start_date, '%Y%m%d').date()
    else:
        py_start_dt = py_avail_dates[0]
    if end_date:
        py_end_dt = dt.datetime.strptime(end_date, '%Y%m%d').date()
    else:
        py_end_dt = py_avail_dates[-1]
    start_idx = abs(py_avail_dates - py_start_dt).argmin()
    end_idx = abs(py_avail_dates - py_end_dt).argmin()

    return {'product': product, 'band': band,
            'start_date': avail_dates[start_idx]['modis_date'],
            'end_date': avail_dates[end_idx]['modis_date']}


#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### END OF FUNCTION SECTION ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### MAIN PROGRAM
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _product_band_to_retrieve():

    return {'MOD13Q1': ['250m_16_days_EVI'],
            'MOD17A2H': ['Gpp_500m']}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _band_short_name(band):

    d = {'250m_16_days_EVI': 'EVI', 'Gpp_500m': 'GPP'}
    return d[band]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
if __name__ == "__main__":

    # Get sites info for processing
    sites = sites=utils.get_ozflux_site_list(master_file_path)

    products_dict = _product_band_to_retrieve()

    # Get list of ozflux sites that are in the MODIS collection (note Wombat
    # has designated site name 'Wombat', so change in dict)
    ozflux_modis_collection_sites = get_network_list('OZFLUX')
    coll_dict = {ozflux_modis_collection_sites[x]['network_sitename']:
                 x for x in ozflux_modis_collection_sites.keys()}
    coll_dict['Wombat State Forest'] = coll_dict.pop('Wombat')



    # Iterate on product (create dirs where required)
    for product in products_dict:
        this_path = os.path.join(output_path, product)
        if not os.path.exists(this_path): os.makedirs(this_path)

        # Iterate on band
        for band in products_dict[product]:

            short_name = _band_short_name(band)

            # Get site data and write to netcdf
            for site in sites.index[:1]:

                print('Retrieving data for site {}:'.format(site))

                target = os.path.join(this_path,
                                      '{0}_{1}'.format(site.replace(' ', '_'),
                                                       short_name))
                full_nc_path = target + '.nc'
                full_plot_path = target + '.png'
                try: first_date = dt.date(int(sites.loc[site, 'Start year']) - 1, 7, 1)
                except (TypeError, ValueError): first_date = None
                try: last_date = dt.date(int(sites.loc[site, 'End year']) + 1, 6, 1)
                except (TypeError, ValueError): last_date = None

                # Get sites in the collection
                if site in coll_dict.keys():
                    site_code = coll_dict[site]
                    x = modis_data_network(product, band, 'OZFLUX',
                                           site_code, first_date, last_date,
                                           qcfiltered=True)

                # Get sites not in the collection
                else:
                    x = modis_data(product, band,
                                   sites.loc[site, 'Latitude'],
                                   sites.loc[site, 'Longitude'], first_date, last_date,
                                   1, 1, site, qcfiltered=True)

                # Reduce the number of pixels to 5 x 5
                x.data_array = _get_pixel_subset(x.data_array, pixels_per_side = 5)

                # Get outputs and write to file (plots then nc)
                thisfig = x.plot_data(plot_to_screen=False)
                thisfig.savefig(full_plot_path)
                plt.close(thisfig)
                da = (pd.DataFrame({short_name: x.get_spatial_mean(),
                                    short_name + '_smoothed': x.get_spatial_mean(smooth_signal=True)})
                      .to_xarray())
                da.attrs = x.data_array.attrs
                resampled_da = da.resample({'time': '30T'}).interpolate()
                resampled_da.time.encoding = {'units': 'days since 1800-01-01',
                                              '_FillValue': None}
                resampled_da.to_netcdf(full_nc_path, format='NETCDF4')