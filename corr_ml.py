import pandas as pd
import dataprep
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Created data\\employment_education_region_data.csv')
price_data = price_by_date(indicator_ids="ZSFH").drop(columns=["indicator_id"]).set_index('region_id')
# The indicator id selects which file to draw data from.
# First time run price_by_date with redo=True, this pivots the data to use dates as columns.

data = price_data[price_data.columns[2:]].astype('float').interpolate(axis=1).dropna(axis=0).merge(data, on='region_id')
# Select all price columns (columns 0 and 1 are region id and index((I dont remember if its index, but it doesnt need interpolating))
# interpolate(axis=1) interpolates across columns. There are different methods of doing this, default is linear. Doesn't really matter imo, affects 3 columns
#   thought of this in write up, worth checking how many rows are affected.
#       Easily done by checking amount of rows in price_data[price_data.columns[2:]].astype('float').interpolate(axis=1).dropna(axis=0) vs price_data[price_data.columns[2:]].astype('float').dropna(axis=0)
# dropna(axis=0) drops rows with NaN values. It drops about half of the ZSFH. This is due to leading NaN that interpolate cannot fill in
# merge with data drops a bit more data. employment_education_region_data.csv was done through zip codes. Anything without a zip code is dropped.
#   This can be changed in line 28 of dataprep.py in the regions function by changing the region_type
#       data  = quandl.get_table('ZILLOW/REGIONS', region_type='zip', paginate=True)
#   Some other changes might be necessary to make that work as I'm using zip2fips
#   Key thing to note, combining 'regions' with 'employment_education' is done through FIPS, though maybe it's possible through county.

data = data.dropna(axis=0)
# Drop rows again with NaN values. 
# I wouldn't interpolate this. columns next to each other are often unrelated 
#   unemployment, high school diplomas, etc. -> would lead to funky and false numbers

data = data.corr()
#print(data)
#data.to_csv('Created Data\\correlation_ZSFH.csv', index=True)
# Full correlation matrix

#data.loc['2021-07-31'].to_csv('Created Data\\correlation_ZSFH_2021_7_31.csv', index=True)
# Latest housing price correlation.


### ML attempt, experimental, not finished, not even sure if useful
data_y = data['2021-07-31']
data_x = data.drop(columns= ['2021-07-31'])
model = LinearRegression().fit(data_x, data_y)
print(model.score(data_x, data_y))
print(model.coef_)