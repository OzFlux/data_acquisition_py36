#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:08:18 2018

@author: ian
"""

import datetime as dt
import ftplib
import math
import numpy as np
import os
import pandas as pd
from pytz import timezone
#from string import maketrans
from io import StringIO, BytesIO
from timezonefinder import TimezoneFinder as tzf
import zipfile
import pdb

ftp_server = 'ftp.bom.gov.au'
ftp_dir = 'anon2/home/ncc/srds/Scheduled_Jobs/DS010_OzFlux/'

##------------------------------------------------------------------------------
#def compile_station_details():
#    
#    columns_list = ['record_id', 'station_id', 'rainfall_district_id', 
#                    'station_name', 'month_year_opened', 'month_year_closed',
#                    'lat', 'lon', 'coords_derivation', 'State', 
#                    'height_stn_asl', 'height_barom_asl', 'wmo_id', 
#                    'first_file_year', 'last_file_year', 'pct_complete',
#                    'pct_vals_Y', 'pct_vals_N', 'pct_vals_W', 'pct_vals_S',
#                    'pct_vals_I', 'eor']
#    station_files_dir = os.path.join(local_dir, 'Station_lists')
#    file_list = [os.path.join(station_files_dir, x) for x in 
#                 os.listdir(station_files_dir)]
#    df_list = []
#    for this_file in file_list:
#        df = pd.read_csv(this_file, header = None)
#        df.columns = columns_list
#        df_list.append(df)
#    master_df = pd.concat(df_list)
#    master_df = master_df[columns_list[1:13]]
#    master_df.index = map(lambda x: x.zfill(6), 
#                          master_df.station_id.astype('str'))
#    master_df.drop('station_id', axis = 1, inplace = True)
#    master_df.sort_index(inplace = True)
#    master_df = master_df[~master_df.index.duplicated(keep = 'first')]
#    tz = []
#    for ID in master_df.index:
#        lat = master_df.loc[ID, 'lat']
#        lon = master_df.loc[ID, 'lon']
#        tz.append(tzf().timezone_at(lng = lon, lat = lat))
#    master_df['timezone'] = tz
#    return master_df
##------------------------------------------------------------------------------

##------------------------------------------------------------------------------
#def get_combined_station_list():
#    
#    """Like get_station_list, but instead of using only stations.txt, uses all 
#       available files in the ftp directory to construct a dataframe"""
#    
#    file_list = get_station_list_files()
#    df_list = []
#    for f in file_list:
#        print f
#        df_list.append(get_station_list(f, as_dataframe=True))
#    df = pd.concat(df_list, sort=True)
#    df.drop_duplicates('Site', inplace = True)
#    df.index = df.Site
#    df.drop('Site', axis = 1, inplace = True)
#    df.sort_index()
#    return df
##------------------------------------------------------------------------------
#
##------------------------------------------------------------------------------
#def get_station_list_files():
#    
#    """Gets a list of all available sitelist files on the BOM permanent ftp 
#       server"""
#    
#    ftp_server = 'ftp.bom.gov.au'
#    ftp_dir = 'anon2/home/ncc/metadata/sitelists/'
#    ftp = ftplib.FTP(ftp_server)
#    ftp.login('anonymous', 'guest')
#    f_list = [f for f in ftp.nlst(ftp_dir) if '.txt' in f]
#    return [f.split('/')[-1] for f in f_list if not ' ' in f]
##------------------------------------------------------------------------------

##------------------------------------------------------------------------------
#def subset_station_list(files_list, target_ID_list):
#    
#    unq_files_list = sorted(list(set([f for f in files_list if 'Data' in f])))
#    avail_list = [f.split('_')[2] for f in unq_files_list]
#    common_list = list(set(avail_list).intersection(target_ID_list))
#    file_name_list = []
#    for f in common_list:
#        idx = avail_list.index(f)
#        file_name_list.append(unq_files_list[idx])
#    return file_name_list
##------------------------------------------------------------------------------

#------------------------------------------------------------------------------
################### BOM permanent server-specific section #####################
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def _get_station_list_dataframe(station_list):
    
    """Returns a dataframe from passed list, using simplified names"""
    
    ref_dict = {'AWS_end': 'End', 'AWS_start': 'Start', 'Bar. Ht': 'Bar_ht', 
                'Co-ord Source': 'Source', 'Site name': 'Site_name',
                'STA': 'Sta', 'Long': 'Lon', 'AeroHt': 'Aero_ht'}
    
    length_list = [len(x) for x in station_list[1].rstrip('\r\n').split()]
    pos = np.cumsum([x + 1 for x in length_list])
    end_pos = pos - 1
    start_pos = np.concatenate([np.array([0]), pos[:-1]])
    tuples = zip(start_pos, end_pos)
    labels = [station_list[0][i[0]:i[1]].strip() for i in tuples]
    new_labels = [ref_dict[label] if label in ref_dict else label for label in labels]
    d = {}
    for i, label in enumerate(new_labels):
        start, end = tuples[i][0], tuples[i][1]
        d[label] = [stn[start:end].strip() for stn in station_list[4:-6]]
    return pd.DataFrame(d)   
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_all_stations_list(file_name = 'stations.txt', as_dataframe = False):
    
    """Gets contents of stations.txt file on BOM permanent ftp server"""
    
    ftp_server = 'ftp.bom.gov.au'
    ftp_dir = 'anon2/home/ncc/metadata/sitelists/'
    ftp = ftplib.FTP(ftp_server)
    ftp.login('anonymous', 'guest')
    full_file_name = os.path.join(ftp_dir, file_name)
    if not full_file_name in ftp.nlst(ftp_dir): 
        print ('File not found!')
        return
    f_str = 'RETR {0}'.format(full_file_name)
    sio = StringIO.StringIO()
    ftp.retrbinary(f_str, sio.write)
    ftp.close()
    sio.seek(0)
    data_list = sio.readlines()
    header_list = []
    for i, line in enumerate(data_list[:10]):
        try: l = line.split()[0]
        except IndexError: continue
        if l == 'Site':
            header_list.append(line)
            break
    new_list = header_list + data_list[i + 1:]
    if not as_dataframe: return new_list
    return _get_station_list_dataframe(new_list)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
####################### OZFLUX server-specific section ########################   
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_station_file_list():

    """Gets a list of all of the available files on the OzFlux-specific ftp
       server"""
    
    ftp = ftplib.FTP(ftp_server)
    ftp.login()    
    zf_list = ftp.nlst(ftp_dir)
    name_list = []
    for this_file in zf_list:
        if not 'AWS' in this_file: continue
        f_str = 'RETR {0}'.format(this_file)
        sio = StringIO.StringIO()
        ftp.retrbinary(f_str, sio.write)
        sio.seek(0)
        zf = zipfile.ZipFile(sio)
        name_list = name_list + zf.namelist()
    ftp.close()
    return name_list
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_station_details_formatting(truncate_description = True):

    """Gets the file format for the station details file on the ozflux-specific 
       ftp server"""
    
    simple_list = ['record_id', 'station_id', 'rainfall_district_id', 
                    'station_name', 'month_year_opened', 'month_year_closed',
                    'lat', 'lon', 'coords_derivation', 'State', 
                    'height_stn_asl', 'height_barom_asl', 'wmo_id', 
                    'first_file_year', 'last_file_year', 'pct_complete',
                    'pct_vals_Y', 'pct_vals_N', 'pct_vals_W', 'pct_vals_S',
                    'pct_vals_I', 'eor']
    start_line = 207
    end_line = 229    
    zf = _get_ftp_data(search_list = ['Notes'])
    with zf.open(zf.namelist()[0]) as file_obj:
        notes_list = file_obj.readlines()
    byte_start = ['Byte_start']
    byte_length = ['Byte_length']
    desc = ['Description']
    if truncate_description: desc += simple_list
    for line in notes_list[start_line:end_line]:
        try:
            line_list = [x.strip() for x in line.decode().split(',')]
        except TypeError:
            pdb.set_trace()
        byte_start.append(int(line_list[0].split('-')[0]))
        byte_length.append(int(line_list[1]))
        if not truncate_description: desc.append(line_list[2].rstrip())
    return pd.DataFrame({byte_start[0]: byte_start[1:],
                         byte_length[0]: byte_length[1:],
                         desc[0]: desc[1:]})
#------------------------------------------------------------------------------    

#------------------------------------------------------------------------------
def get_data_file_formatting():

    """Gets the file format for the data files on the OzFlux-specific ftp 
       server"""
    
    zf = _get_ftp_data(search_list = ['Notes'])
    with zf.open(zf.namelist()[0]) as file_obj:
        notes_list = file_obj.readlines()
    bad_chars = '*_-.'
    start_line = 19
    end_line = 44    
    byte_start = ['Byte_start']
    byte_length = ['Byte_length']
    desc = ['Description']
    for i, line in enumerate(notes_list[start_line:end_line]):
        line_list = [x.strip() for x in line.decode().split(',')]
        byte_start.append(int(line_list[0].split('-')[0]))
        byte_length.append(int(line_list[1]))
        if i > 1: line_list[2] = line_list[2].translate({ord(c): None for c in 
                                                         bad_chars})
        if len(line_list) > 2: line_list[2] = ','.join(line_list[2:])
        desc.append(line_list[2].rstrip('.').lstrip())
    return pd.DataFrame({byte_start[0]: byte_start[1:],
                         byte_length[0]: byte_length[1:],
                         desc[0]: desc[1:]})
#------------------------------------------------------------------------------        

#------------------------------------------------------------------------------
def get_aws_station_details():
    
    """Retrieves a dataframe containing details of all AWS stations on the 
       OzFlux ftp server site"""
       
    zf = _get_ftp_data(['StnDet'], get_first = False)
    data_list = []
    for f in zf.namelist():
        with zf.open(f) as file_obj:
            for line in file_obj:
                data_list += [line.decode().split(',')]
    notes_df = get_station_details_formatting()
    df = pd.DataFrame(data_list, columns = notes_df.Description)
    for col in [x for x in df.columns if not 'station_id' in x]:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue
    df.index = df['station_id']
    return df.sort_index()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = (math.sin(dlat / 2)**2 + math.cos(lat1) * 
         math.cos(lat2) * math.sin(dlon / 2)**2)
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_closest_bom_station(lat, lon, current = True, nearest_n = 5):
    
    stations = get_aws_station_details()
    stations['dist (km)'] = map(lambda x: haversine(lat, lon, 
                                                    stations.loc[x, 'lat'], 
                                                    stations.loc[x, 'lon']),
                           stations.index)
    if current:
        year = str(dt.datetime.now().year)
        stations = stations.loc[stations.last_file_year == year]
    df = stations.sort_values(['dist (km)']).head(nearest_n)
    return df[['station_name', 'lat', 'lon', 'dist (km)']]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_ftp_data(search_list = None, get_first = True):
    
    """Function to retrieve zipfile containing data files for AWS stations 
       on OzFlux ftp server site
       Args: 
           * search_list (list or None, default None): specifies substrings 
             used to search for matching file names (if None returns all data 
             available)
           * get_first (boolean, default True): if true, returns only the first
             file with matching substring"""
    
    if search_list:
        if not isinstance(search_list, list):
            raise TypeError('argument search_list must be of type list!')
    
    # Login to ftp server      
    ftp = ftplib.FTP(ftp_server)
    ftp.login()   
    
    # Open the separate zip files and combine in a single zip file 
    # held in dynamic memory - ignore the solar data
    master_bio = BytesIO() 
    master_zf = zipfile.ZipFile(master_bio, 'w')
    zip_file_list = [os.path.split(f)[1] for f in ftp.nlst(ftp_dir)]   
    for this_file in zip_file_list:
        if 'globalsolar' in this_file: continue
        in_file = os.path.join(ftp_dir, this_file)
        f_str = 'RETR {0}'.format(in_file)
#        sio = StringIO()
        bio = BytesIO()
        ftp.retrbinary(f_str, bio.write)
        bio.seek(0)
        zf = zipfile.ZipFile(bio)
        if not search_list is None:
            file_list = []
            for this_str in search_list:
                if not isinstance(this_str, str):
                    raise TypeError('search_list elements must be of type str!')
                for f in zf.namelist(): 
                    if this_str in f:
                        file_list.append(f)
                        if get_first: search_list.remove(this_str)
        else:
            file_list = zf.namelist()
        for f in file_list:
            if not f in master_zf.namelist():
                master_zf.writestr(f, zf.read(f))
        zf.close()
        if not search_list is None and len(search_list) == 0: break
    ftp.close()

    return master_zf
#------------------------------------------------------------------------------