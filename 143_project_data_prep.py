import quandl
import json
import pandas as pd
import numpy as np

def price_by_date(indicator_ids=['ZSFH'], region_ids=['99999'], redo=False):
    """Grabs data from Zillow and pivots it so that the columns include dates and prices at those dates.

    :bool redo: If true, loads and reattaches all of the tables and saves them to a csv. If False, loads csv from file.
    """
    if redo:
        data = quandl.get_table('ZILLOW/DATA', indicator_id=indicator_ids, region_id=region_ids, paginate=True)
        data = data.pivot(index=['region_id', 'indicator_id'],columns='date', values='value').reset_index().rename_axis(None, axis=1)
        data.to_csv('prices.csv', index=False)
        return data
    else:
        return pd.read_csv('prices.csv')

def regions(redo=False):
    """Pulls region data from Zillow and attaches fips code.

    :bool redo: If true, loads and reattaches all of the tables and saves them to a csv. If False, loads csv from file.
    """
    if redo:
        with open('zip2fips.json') as json_file:
            zip2fips = json.load(json_file)
        data  = quandl.get_table('ZILLOW/REGIONS', region_type='zip', paginate=True)
        data[['zip', 'state', 'city', 'county', 'subcity']] = data['region'].str.split("; ", expand=True)
        data[['FIPS']] = data[['zip']].replace(zip2fips)
        data[['FIPS']] = data[['FIPS']].astype('Int64')
        data[['region_id', 'FIPS', 'zip']].to_csv('region_data.csv', index=False)
        return data[['region_id', 'FIPS', 'zip']]

    else:
        return pd.read_csv('region_data.csv')

def attach_tables(redo=False):
    """Attach multiple tables together using fips.

    :bool redo: If true, loads and reattaches all of the tables and saves them to a csv. If False, loads csv from file.
    """
    if redo:
        UIC = pd.read_csv('UIC_codes.csv')
        UIC = UIC.drop(columns=['UIC_2013', 'Description'])

        education = pd.read_csv('education.csv')
        education['FIPS'] = education['FIPS Code']
        education=education.drop(columns=['FIPS Code', 'State', 'Area name', '2003 Rural-urban Continuum Code', '2003 Urban Influence Code', '2013 Rural-urban Continuum Code', '2013 Urban Influence Code', 'City/Suburb/Town/Rural 2013'])

        unemployment = pd.read_csv('unemployment.csv')
        unemployment['FIPS'] = unemployment['FIPS_Code']
        unemployment=unemployment.drop(columns=['FIPS_Code', 'State', 'Area_name', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013', 'City/Suburb/Town/Rural'])
        
        data = pd.merge(pd.merge(UIC, education, on='FIPS', how='inner'), unemployment, on='FIPS', how='inner')
        data.to_csv('employment_education.csv', index=False)
        return data

    else:
        return pd.read_csv('employment_education.csv')


quandl.ApiConfig.api_key = 'wW1JzdsRZwhQ78d-dkq2'

region_data = regions()
# region_ids =['99999', '99998']
# print()
# ee_data = attach_tables()
# ee_region_data = region_data.merge(ee_data, on='FIPS', how='inner')
# ee_region_data.to_csv('ee_region_data.csv', index=False)

# ee_region_data = pd.read_csv('ee_region_data.csv')
# print(ee_region_data)
#price_data = price_by_date(region_ids=ee_region_data['region_id'].tolist()[0:100], redo=True)
#print(price_data.head())
#total_data = ee_region_data.merge(price_data, on='region_id', how='inner')
#total_data.to_csv('total_data.csv', index=False)
#print(total_data)