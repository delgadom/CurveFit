#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:54:15 2020

@author: hiroakimachida
"""

import pandas as pd
import datetime

base = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
confirmed = 'time_series_covid19_confirmed_'
death = 'time_series_covid19_deaths_'
recovered = 'time_series_covid19_recovered_'

def data_country(selected_country, dataset='confirmed'):
    """ return dataset timeseries for a selected country """
    
    #select the right database
    if dataset == 'confirmed':
        url = base+confirmed
    elif dataset == 'death':
        url = base+death
    elif dataset == 'recovered':
        url = base+recovered
    
    if selected_country != 'US':
        df = pd.read_csv(url+'global.csv').groupby(['Country/Region']).sum()
        df.drop(['Lat', 'Long'], axis=1, inplace=True)
        df = df.loc[selected_country]
    else:
        df = pd.read_csv(url+'US.csv').groupby('Country_Region').sum()
        df.drop(['UID', 'code3', 'FIPS', 'Lat', 'Long_'], axis=1, inplace=True)
        if dataset == 'death':
            df.drop(['Population'], axis=1, inplace=True)
        df = df.sum()
    return df.index, df.values

date_confirmed, confirmed = data_country('Japan', 'confirmed') # or Canada, etc. for confirmed cases

date_death, death = data_country('Japan', 'death') # or Canada, etc. for confirmed cases



for i, e in enumerate(date_confirmed):
    date = datetime.datetime.strptime(date_confirmed[i], '%m/%d/%y').strftime('%Y/%m/%d')
    print(date, death[i], confirmed[i])