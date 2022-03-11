#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme()


def main():
    zips = pd.read_csv(os.path.join(os.path.dirname(__file__), "Data/zip_code_market_tracker.csv"), sep='\t')

    data = zips[['period_begin', 'period_end', 'region', 'state', 'state_code', 'property_type', 'sold_above_list',
                 'off_market_in_two_weeks']]

    data['period_begin'] = pd.to_datetime(data['period_begin'], format="%Y-%m-%d")
    data['period_end'] = pd.to_datetime(data['period_end'], format="%Y-%m-%d")
    data = data.dropna(subset=['sold_above_list'])
    data.off_market_in_two_weeks.fillna(0, inplace=True)
    data['estimated_investor_purchase'] = data.sold_above_list * data.off_market_in_two_weeks
    temp = data.loc[(data.period_begin.dt.year == 2013) & (data.region == 'Zip Code: 78501')].sort_values(
        by='period_begin')
    temp = temp.groupby(by=['period_begin', 'region', 'state', 'state_code']).median().reset_index()
    temp.groupby(by=[temp.period_begin.dt.year, 'region', 'state', 'state_code']).median()

    investor = data.groupby(by=['period_begin', 'region', 'state', 'state_code']).median().reset_index()
    investor = investor.groupby(
        by=[investor.period_begin.dt.year, 'region', 'state', 'state_code']).median().reset_index()
    investor['zipcode'] = investor.region.str.extract(r'(\d+)')
    investor.drop(columns=['region'], inplace=True)

    from uszipcode import SearchEngine
    sr = SearchEngine()
    county = []
    for zip_code in investor.zipcode:
        z = sr.by_zipcode(zip_code)
        county.append(z.county)

    investor.to_csv(os.path.join(os.path.dirname(__file__), "Data/investor.csv"))
