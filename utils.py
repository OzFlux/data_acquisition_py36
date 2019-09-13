#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:41:50 2019

@author: ian
"""

import pandas as pd
import xlrd

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
    df.drop(header_list[0], axis = 1, inplace = True)
    return df
#------------------------------------------------------------------------------