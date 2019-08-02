#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:55:30 2019

@author: ian
"""

import datetime as dt
import os
import pandas as pd
import pdb

def get_dummy_lines(df):
    
    check_dates = pd.date_range(df.index[0], df.index[-1], freq = '30T')
    if len(df) == len(check_dates): return
    missing_dates = [x.to_pydatetime() for x in check_dates 
                     if not x in df.index]
    ref_list = df.data.iloc[0].split(',')
    new_list_start = ['dd'] + [ref_list[1]]
    new_list_end = [' ' * len(x) for x in ref_list[12:-1]] + ['#\n']
    date_str_list = [dt.datetime.strftime(x, '%Y,%m,%d,%H,%M') 
                     for x in missing_dates]
    dummy_data = []
    for date in date_str_list:
        line = ','.join(new_list_start + [date] + [date] + new_list_end)
        dummy_data.append(line)
    dummy_df = pd.DataFrame(dummy_data, index = missing_dates, columns = ['data'])
    new_df = pd.concat([df, dummy_df])
    new_df.sort_index(inplace = True)
    return new_df
    
def combine_dataframes(df1, df2):
    
    combined_df = pd.concat([df1, df2])
    combined_df.sort_index(inplace = True)
    combined_df.drop_duplicates(inplace = True)
    combined_df = combined_df[~combined_df.index.duplicated()]
    check_dates = pd.date_range(combined_df.index[0], combined_df.index[-1],
                                freq = '30T')
    if not len(combined_df) == len(check_dates):
        combined_df = get_dummy_lines(combined_df)
    return combined_df
    
def get_dates(data):
    
    dates = []
    for line in data:
        dates.append(dt.datetime.strptime(''.join(line.split(',')[7:12]), 
                                          '%Y%m%d%H%M'))
    return dates

def get_dataframe(read_path, header_lines):
    
    with open(read_path) as f:
        data = f.readlines()
        header = data[:header_lines]
        dates = get_dates(data[header_lines:])
        df = pd.DataFrame(data[header_lines:], 
                          index = dates, columns = ['data'])
        return df, header

old_header_lines = 1
new_header_lines = 2
old_path = '/home/ian/Desktop/AWS_BOM_all'
new_path = '/home/ian/Desktop/BOM_data/'
output_path = '/home/ian/Desktop/BOM_new'

old_file_list = sorted(os.listdir(old_path))
new_file_list = sorted(os.listdir(new_path))

for fname in old_file_list:
    
    print ('Processing file {}'.format(fname))
    
    old_read_path = os.path.join(old_path, fname)
    new_read_path = os.path.join(new_path, fname)

    old_df, old_header = get_dataframe(old_read_path, old_header_lines)

    try:
        new_df, new_header = get_dataframe(new_read_path, new_header_lines)
        no_new_file = False
    except FileNotFoundError:
        no_new_file = True
        
    write_path = os.path.join(output_path, fname)
    if not no_new_file:
        combined_df = combine_dataframes(old_df, new_df)
    else:
        combined_df = old_df
    
    with open(write_path, 'w') as f:
        
        f.write(new_header[1])
        data = combined_df.data.tolist()
        f.writelines(data)