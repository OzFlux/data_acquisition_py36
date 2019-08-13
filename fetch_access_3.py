#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:19:53 2019

@author: ian
"""

from bs4 import BeautifulSoup
from datetime import datetime
import netCDF4
import numpy as np
import os
import pandas as pd
import requests
from subprocess import call as spc
import xlrd
import pdb

#------------------------------------------------------------------------------
def check_seen_files(opendap_url, base_dir, site_list):
    
    """Check local files to see which of the available files on the opendap 
       server have been seen and processed"""
    
    opendap_files = _list_opendap_dirs(opendap_url)
    month_dirs = np.unique([x[:6] for x in opendap_files]).astype(str)
    check_paths = [os.path.join(base_dir, 'Monthly_files', x) for x in month_dirs]
    data = (np.tile(False, len(opendap_files) * len(site_list))
            .reshape(len(opendap_files), len(site_list)))
    seen_df = pd.DataFrame(data, index = opendap_files, columns = site_list)
    for site in seen_df.columns:
        seen_dirs = []
        for target_path in check_paths:
            target = os.path.join(target_path, '{}.nc'.format(site))
            try:
                nc = netCDF4.Dataset(target)
            except IOError:
                continue
            dts = sorted(netCDF4.num2date(nc.variables['time'][:], 
                         units = nc.variables['time'].units))
            seen_dates = [datetime.strftime(x, '%Y%m%d') for x in dts]
            seen_hours = [str(x.hour - x.hour % 6).zfill(2) for x in dts]
            seen_dirs += list(set([x[0] + x[1] for x in zip(seen_dates, 
                                                            seen_hours)]))
        seen_df[site] = list(map(lambda x: x in seen_dirs, opendap_files))
    seen_df = seen_df.T
    seen_dict = {}
    for site in seen_df.columns:
        l = list(seen_df[seen_df[site]==False].index)
        if l:
            seen_dict[site] = l    
    return seen_dict
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def check_set_subdirs(base_dir):
    
    """Check if required directories reside in base directory 
       and create if not"""
    
    missing_dirs = []
    for sub_dir in ['Continental_files', 'Monthly_files', 
                    'Precip_forecast_files', 'Working_files']:
        expected_path = os.path.join(base_dir, sub_dir)
        if os.path.exists(expected_path): continue
        os.makedirs(expected_path)
        missing_dirs.append(sub_dir)
    if not missing_dirs:
        print ('All required subdirectories present')
    else:
        missing_str = ', '.join(missing_dirs)
        print ('The following directories were missing and have been created {}'
               .format(missing_str))
    return
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_files_from_datestring(datestring):
    
    """Turn standard datestring format for ACCESS directories into list of 
       file IDs (0-5)"""
    
    return map(lambda x: '{}_{}'.format(datestring, str(x).zfill(3)), range(7))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_ozflux_site_list(master_file_path):
    
    """Create a dataframe containing site names (index) and lat, long and 
       measurement interval"""
    
    wb = xlrd.open_workbook(master_file_path)
    sheet = wb.sheet_by_name('Active')
    header_row = 9
    header_list = sheet.row_values(header_row)
    df = pd.DataFrame()
    for var in ['Site', 'Latitude', 'Longitude']:
        index_val = header_list.index(var)
        df[var] = sheet.col_values(index_val, header_row + 1)   
    df.index = map(lambda x: '_'.join(x.split(' ')), df.Site)
    df.drop(header_list[0], axis = 1, inplace = True)
    return df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _list_opendap_dirs(url):
    
    """Scrape list of directories from opendap surface url"""
    
    full_url = url.format('dodsC')
    page = requests.get(full_url).text
    soup = BeautifulSoup(page, 'html.parser')    
    dir_list = [url + '/' + node.get('href') for node in soup.find_all('a') 
                if node.get('href').endswith('html')]
    new_list = []
    for path in dir_list:
        path_list = path.replace('//', '/').split('/')[1:]
        try:
            path_list.remove('catalog.html')
            datetime.strptime(path_list[-1], '%Y%m%d%H')
            new_list.append(path_list[-1])
        except: 
            continue
    return new_list
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def nco_exec(base_directory, site_name, date_directory, latitude, longitude):

    """Call the shell script that cuts out the site coordinates and 
       concatenates with existing data (using NCO)"""
    
    exec_string = ('./nco_shell.sh "{0}" "{1}" "{2}" "{3}" "{4}"'
                   .format(base_directory, site_name, date_directory, 
                           latitude, longitude))    
    if spc(exec_string, shell = True):
        raise RuntimeError('Error in command: {}'.format(exec_string))
    return
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def purge_dir(directory, file_ext = '.tmp'):
    
    """Dump any files not required"""
    
    f_list = filter(lambda x: os.path.splitext(x)[1] == '.tmp', 
                    os.listdir(directory))
    for f in [os.path.join(directory, x) for x in f_list]:
        os.remove(f)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def wget_exec(read_path, write_path, server_file_ID):
    
    """Build the complete wget string and retrieve temp file"""
    
    print ('Retrieving forecast files for {}, hours 0-6'.format(server_file_ID))
    tmp_fname = os.path.join(write_path, '{}_access.tmp'.format(server_file_ID))
    wget_prefix = '/usr/bin/wget -nv -a Download.log -O'
    server_dir = server_file_ID.split('_')[0]
    full_read_path = read_path.format('fileServer') + server_dir
    server_fname = os.path.join(full_read_path,
                                'ACCESS-R_{}_surface.nc'.format(server_file_ID))
    cmd = '{0} {1} {2}'.format(wget_prefix, tmp_fname, server_fname)
    if spc(cmd, shell=True):
        raise RuntimeError('Error in command: {}'.format(cmd))
    return tmp_fname
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# USER PATH CONFIGURATIONS
#------------------------------------------------------------------------------

retrieval_path = 'http://opendap.bom.gov.au:8080/thredds/{}/bmrc/access-r-fc/ops/surface/'
base_dir = '/home/ian/Desktop/access_test'
#base_dir = '/rdsi/market/access_test'
master_file_path = '/home/ian/Temp/site_master.xls'
#master_file_path = '/mnt/OzFlux/Sites/site_master.xls'

#------------------------------------------------------------------------------
# MAIN PROGRAM
#------------------------------------------------------------------------------

# Check the base directory contains the required subdirs, create if not
check_set_subdirs(base_dir)

# Get site details
site_df = get_ozflux_site_list(master_file_path)

# Set the path for continental file retrieval
continental_file_path = os.path.join(base_dir, 'Continental_files')

# Cross check available files on the opendap server against content of existing
# files 
files_dict = check_seen_files(retrieval_path, base_dir, site_df.index)

# Pre-purge the continental file path for all temp files
purge_dir(continental_file_path)

# For each six-hour directory...
for this_dir in sorted(files_dict.keys())[:2]:

    # Get a list of the files we want to extract (UTC + 0-7)
    file_list = get_files_from_datestring(this_dir)

    # Get a list of the sites we need to collect data for
    sites_list = sorted(files_dict[this_dir])

    # Grab the continent-wide file        
    for f in file_list:    
        wget_exec(retrieval_path, continental_file_path, f)
    
    # Cut out site from continent-wide file and append 
    # (see shell script nco_shell.sh)    
    for site in site_df.index:
        
        try:
            site_details = site_df.loc[site]
            nco_exec(base_dir, site_details.name, this_dir, 
                     site_details.Latitude, site_details.Longitude)            
        except RuntimeError as e:
            print (e)
            continue