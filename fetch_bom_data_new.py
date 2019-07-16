#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:48:26 2019

@author: ian
"""

# Standard modules
import datetime as dt
import importlib
import json
import os
import pandas as pd
from pytz import timezone
from pytz.exceptions import UnknownTimeZoneError
import requests
from time import sleep
from timezonefinder import TimezoneFinder as tzf

import pdb

# Custom modules
import BOM_ftp_functions as bomftp

# Reloads
importlib.reload(bomftp)

class bom_data(object):
    
    def __init__(self):
        
        self.stations = get_station_dataframe()
        self.file_format = bomftp.get_data_file_formatting()
        self._zfile = bomftp._get_ftp_data()

    #--------------------------------------------------------------------------
    def file_id_dict(self):
        
        """Get a dictionary referencing the file name by key of BOM site ID"""
        
        data_list = [x for x in self._zfile.namelist() if 'Data' in x]
        return dict(zip([x.split('_')[2] for x in data_list], data_list)) 
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def generate_dummy_line(self, station_id, datetime):
    
        """Write out a line of dummy data with correct spacing and site 
           details, but no met data"""
        
        station_details = self.stations.loc()
        start_list = ['dd', station_details.station_id, 
                      station_details.station_name.zfill(40)]
        try:
            local_datetime = get_local_datetime(datetime, 
                                                station_details.timezone)
            local_datetime_str = dt.datetime.strftime(local_datetime, 
                                                      '%Y,%m,%d,%H,%M')
        except UnknownTimeZoneError:
            local_datetime_str = '    ,  ,  ,  ,  '
        start_list.append(local_datetime_str)
        start_list.append(dt.datetime.strftime(datetime, '%Y,%m,%d,%H,%M'))
        blank_list = [' ' * x for x in self.file_format.Byte_length]
        new_str = ','.join((start_list + blank_list[5: -1] + ['#\r\n']))
        return set_line_order(new_str)                        
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def header_list(self):
    
        """Get formatted header list for the output file"""
        
        headers = self.file_format.Description.tolist()
        new_headers = headers[:2] + headers[3:9] + headers[11:]
        new_headers[0] = new_headers[0].split('-')[1].lstrip()
        new_headers[1] = 'Station Number'
        new_headers[12] = new_headers[12].replace('km/h', 'm/s')
        new_headers[14] += ' true'
        new_headers[16] = new_headers[16].replace('km/h', 'm/s')
        new_headers[-1] = new_headers[-1].replace(',', ' ')
        new_header_str = ','.join(new_headers) + '\n'
        return new_header_str
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _make_tz_request(lat, lon):
        
        """Get timezone data from timezonedb API where python timezone fails"""
        
        api_key = 'UT66CKBKU8MM'
        base_url_str = 'http://api.timezonedb.com/v2.1/get-time-zone'
        end_str = ('?key={0}&format=json&by=position&lat={1}&lng={2}'
                   .format(api_key, lat, lon))
        sleep(1)
        json_obj = requests.get(base_url_str + end_str)
        if json_obj.status_code == 200:
            return json.loads(json_obj.content)
        else: 
            print ('Timezone request returned status code {}'
                   .format(json_obj.status_code))
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------
def check_line_integrity(line):
    
    """Check line length and number of elements are as expected, and that final
       character is #"""
    
    # Set values for validity checks
    line_len = 161
    element_n = 33   

    # Do checks
    line_list = line.decode().split(',')
    try:
        assert len(line) == line_len # line length consistent?
        assert len(line_list) == element_n # number elements consistent?
        assert '#' in line_list[-1] # hash last character (ex carriage return)?
    except AssertionError:
        return False
    return True
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#def get_header_list():
#        
#    l = ['hm',
#         'Station Number',
#         'Year Month Day Hour Minutes in YYYY,MM,DD,HH24,MI format in Local time',
#         'Year Month Day Hour Minutes in YYYY,MM,DD,HH24,MI format in Local standard time',
#         'Precipitation since 9am local time in mm',
#         'Quality of precipitation since 9am local time',
#         'Air Temperature in degrees C',
#         'Quality of air temperature',
#         'Dew point temperature in degrees C',
#         'Quality of dew point temperature',
#         'Relative humidity in percentage %',
#         'Quality of relative humidity',
#         'Wind speed in m/s',
#         'Wind speed quality',
#         'Wind direction in degrees true',
#         'Wind direction quality',
#         'Speed of maximum windgust in last 10 minutes in m/s',
#         'Quality of speed of maximum windgust in last 10 minutes',
#         'Station level pressure in hPa',
#         'Quality of station level pressure',
#         'AWS Flag',
#         '#\r\n']
#         
#    return [','.join(l)]

##------------------------------------------------------------------------------
#def get_header_list():
#
#    """Get formatted header list for the output file"""
#    
#    headers = format_df.Description.tolist()
#    new_headers = headers[:2] + headers[3:9] + headers[11:]
#    new_headers[0] = new_headers[0].split('-')[1].lstrip()
#    new_headers[1] = 'Station Number'
#    new_headers[12] = new_headers[12].replace('km/h', 'm/s')
#    new_headers[14] += ' true'
#    new_headers[16] = new_headers[16].replace('km/h', 'm/s')
#    new_headers[-1] = new_headers[-1].replace(',', ' ')
#    new_header_str = ','.join(new_headers) + '\n'
#    return new_header_str
##------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_local_datetime(dt_obj, tz_name):
    
    """Convert standard local datetime to local datetime using timezone"""
    
    tz_obj = timezone(tz_name)
    try:
        dst_offset = tz_obj.dst(dt_obj)
    except:
        dst_offset = tz_obj.dst(dt_obj + dt.timedelta(seconds = 3600))
    return dt_obj + dst_offset 
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
def set_line_order(line):

    """Drop wet bulb temperature and convert wind speed"""
    
    def convert_kmh_2_ms(kmh):
        try:
            return str(round(float(kmh) / 3.6, 1)).rjust(5)
        except:
            return kmh.rjust(5)

    line_list = line.decode().split(',')
    line_list[23] = convert_kmh_2_ms(line_list[23])
    line_list[27] = convert_kmh_2_ms(line_list[27])
    return ','.join(line_list[:2] + line_list[3:17] + line_list[19:])
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_write_list(content, ftp_df):

    """Compare the existing file content with the content of the dataframe,
       and keep all lines that are missing from the local (add dummy lines)
       if there is a gap"""
    
    if len(content) == 0:
        write_list = [get_header_list()] + ftp_df.Data.tolist()
    else:
        last_date_str = ','.join(content[-1].split(',')[7:12])
        last_date = dt.datetime.strptime(last_date_str, '%Y,%m,%d,%H,%M')
        try:
            int_loc = ftp_df.index.get_loc(last_date)
            write_list = ftp_df.iloc[int_loc + 1:]['Data'].tolist()
        except KeyError:
            date_range = (pd.date_range(last_date, ftp_df.index[0], 
                                        freq = '30T')
                          .to_pydatetime())[1:-1]
            write_list = [generate_dummy_line(station_df, x) 
                          for x in date_range] + ftp_df.Data.tolist()
    return write_list
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def zipfile_to_dataframe(file_obj, station_df):

    """Convert zipfile to dataframe with dummy spaces"""
    
    header_line = 0
    content = file_obj.readlines()
    valid_data = []
    for line in content[header_line + 1:]:
        if check_line_integrity(line):
            new_line = set_line_order(line)
            valid_data.append(new_line)        
    if len(valid_data) == 0: return None
    dtstr_list = [','.join(x.split(',')[7: 12]) for x in valid_data]
    dt_list = [dt.datetime.strptime(x, '%Y,%m,%d,%H,%M') for x in dtstr_list]
    valid_df = pd.DataFrame(valid_data, index = dt_list, columns = ['Data'])
    new_index = pd.date_range(valid_df.index[0], valid_df.index[-1], freq = '30T')
    missing_dates = [x.to_pydatetime() for x in new_index 
                     if not x in valid_df.index]
    dummy_data = [generate_dummy_line(station_df, x) for x in missing_dates]
    dummy_df = pd.DataFrame(dummy_data, index = missing_dates, columns = ['Data'])
    df = pd.concat([valid_df, dummy_df])
    df.sort_index(inplace = True)
    return df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_station_dataframe():
    
    """Get the station details from the BOM ftp server and add timezone info"""
    
    # Get basic station data
    stations_df = bomftp.get_aws_station_details()

    # Create a timezone variable using lookup from python package
    stations_df['timezone'] = [tzf().timezone_at(lng = stations_df.loc[x, 'lon'], 
                                                 lat = stations_df.loc[x, 'lat']) 
                               for x in stations_df.index]

    # Try to find missing timezones from web API, then give up (sites with 
    # unsuccessful tz lookup will have only empty data appended to the local 
    # date and time slots of the text file)
    for id_code in stations_df[pd.isnull(stations_df.timezone)].index:
        rq = make_tz_request(stations_df.loc[id_code, 'lat'], 
                             stations_df.loc[id_code, 'lon'])
        stations_df.loc[id_code, 'timezone'] = rq['zoneName']

    return stations_df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

# Set path to read existing and write new data
write_path = '/home/ian/Desktop/BOM_data' #'/rdsi/market/AWS_BOM_all'

x = bom_data()

## Get info about stations
#stations_df = get_station_dataframe()
#
## Get formatting of data
#format_df = bomftp.get_data_file_formatting()
#
## Get a zipfile containing most recent ftp data
#z_file = bomftp._get_ftp_data()
#
## Make a lookup dict for IDs and server file names
#data_list = [x for x in z_file.namelist() if 'Data' in x]
#id_dict = dict(zip([x.split('_')[2] for x in data_list],
#                   data_list)) 
#
#missing_list = []
#
## Process each file
#for station_id in stations_df.index:
#    
#    # Get site info
#    station_df = stations_df.loc[station_id]    
#    
#    # Print details to screen
#    print ('Processing site {0} ({1})'.format(station_id, 
#                                              station_df['station_name'].rstrip()))
#    
#    # Get all the data from the ftp file and make a dataframe
#    readf_name = id_dict[station_id]
#    with z_file.open(readf_name) as zf:
#        ftp_df = zipfile_to_dataframe(zf, station_df)
#    if not ftp_df: 
#        missing_list.append(station_id)
#        continue # Skip to next of no new data
#
#    # Write any new data to the existing file - open existing in append+, 
#    # then make a local dataframe from the existing data (if any)
#    writef_name = 'HM01X_Data_{}.txt'.format(station_id)    
#    with open(os.path.join(write_path, writef_name), 'a+') as f:
#        content = f.readlines()
#        write_list = get_write_list(content, ftp_df)
#        f.writelines(write_list)
#    
#    # List missing data files    
#    print ('The ftp files for the following stations had no data:'
#           .format(station_id, station_df['station_name'].rstrip()))