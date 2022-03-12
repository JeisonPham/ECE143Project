#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
sns.set_theme()

def main():
    prices = pd.read_csv(os.path.join(os.path.dirname(__file__), "Data/metro.csv"))
    zips = pd.read_csv(os.path.join(os.path.dirname(__file__), "Data/investor.csv"))

    counties = {
        'Boston': ('Suffolk County', 'MA'),
        'Denver': ('Denver County', 'CO'),
        'Los Angeles': ('Los Angeles County', 'CA'),
        'McAllen': ('Hildago County', "TX"),
        'Merced': ("Merced County", 'CA'),
        'New York': ("New York County", "NY"),
        'San Diego': ('San Diego County', 'CA'),
        'San Francisco' : ('San Francisco County', 'CA'),
        'San Jose' : ('Santa Clara County', 'CA'),
        'St. Louis' : ('St. Louis County', 'MO'),
        'Ventura' : ('Ventura County', 'CA'),
        'Wichita' : ('Wichita County', 'KS')
    }


    investment = []
    for region, county in counties.items():
        data = zips[(zips.county == county[0]) & (zips.state_code == county[1])]
        data = data.groupby(by=['period_begin', 'county']).median().reset_index()
        data['Region'] = region


        county = prices[(prices.RegionName.str.contains(region)) & (prices.RegionName.str.contains(county[1]))]
        county = county.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName'])
        county = pd.melt(county, id_vars='RegionName', value_name='value')
        county.columns = ['RegionName', 'date', 'value']
        county['date'] = pd.to_datetime(county['date'], format='%Y-%m-%d')
        county = county.groupby(by=[county.date.dt.year]).mean().reset_index()


        combined = data.merge(county, left_on=['period_begin'], right_on=['date'], how='left')
        combined['investor change'] = combined['estimated_investor_purchase'].diff(1)
        combined['value change'] = combined['value'].diff(1)
        investment.append(combined)

    combined = pd.concat(investment).reset_index().dropna()


    combined = combined[['date', 'estimated_investor_purchase', 'Region', 'value', 'investor change', 'value change']]



    from matplotlib.offsetbox import AnchoredText
    colors = sns.color_palette("tab10")
    for index, date in enumerate(combined.date.unique()[3:]):
        data = combined[combined.date == date]
        vals = ['estimated_investor_purchase', 'value']
        labels = ["Estimated Investor Purchase %", "Median House Value"]
        width = 0.5
        _X = np.arange(len(data.Region.tolist()))
        fig, ax = plt.subplots(figsize=(15, 10))
        anc = AnchoredText(date.astype(str), loc="upper right", frameon=False)
        ax.add_artist(anc)
        ax.set_ylim([0, 100])
        ax.bar(_X, data[vals[0]] * 100, width=width/2, align="center", color=colors[0])
        ax.set_ylabel('Investor Purchase Percentage', color = colors[0])
        ax.tick_params(axis ='y', labelcolor = colors[0])

        ax2 = ax.twinx()
        ax2.set_ylim([0, 2e6])
        ax2.bar(_X + 0.125, data[vals[1]], width = width / 2, align='edge', color=colors[1])
        ax2.set_ylabel('Median Home Value', color = colors[1])
        ax2.tick_params(axis ='y', labelcolor = colors[1])

        plt.xticks(_X, data.Region.tolist())
        plt.title(f"Percentage Investor Purchases and Home Price since 2015")
        plt.savefig(os.path.join(os.path.dirname(__file__), f"Investor Graphs/{index:03d}.png"), bbox_inches='tight')
        plt.show()

    temp = pd.pivot_table(combined, index='date', columns=['Region'], values=['estimated_investor_purchase'])
    temp.columns = temp.columns.get_level_values(1)
    temp['Group 1'] = temp[['New York', 'Boston']].mean(axis=1)
    temp['Group 2'] = temp[['Ventura', 'Los Angeles', 'San Francisco', 'Merced']].mean(axis=1)
    temp['Group 3'] = temp[['St. Louis', 'Denver', 'San Jose']].mean(axis=1)
    temp = temp[['San Diego', 'Group 1', 'Group 2', 'Group 3']].reset_index()
    temp = pd.melt(temp, id_vars=['date'], value_vars=['San Diego', 'Group 1', 'Group 2', 'Group 3'])
    temp1 = temp



    temp = pd.pivot_table(combined, index='date', columns=['Region'], values=['value'])
    temp.columns = temp.columns.get_level_values(1)
    temp['Group 1'] = temp[['New York', 'Boston']].mean(axis=1)
    temp['Group 2'] = temp[['Ventura', 'Los Angeles', 'San Francisco', 'Merced']].mean(axis=1)
    temp['Group 3'] = temp[['St. Louis', 'Denver', 'San Jose']].mean(axis=1)
    temp = temp[['San Diego', 'Group 1', 'Group 2', 'Group 3']].reset_index()
    temp = pd.melt(temp, id_vars=['date'], value_vars=['San Diego', 'Group 1', 'Group 2', 'Group 3'])



    fig, ax = plt.subplots(figsize=(20, 10))
    for region in temp.Region.unique():
        data = temp[temp.Region == region]
        data2 = temp1[temp1.Region == region]
        ax.plot(data.date, data.value / (data2.value * 100), label=region)
    plt.legend()
    plt.title("Home Price to Estimated Investor Purchase % over time")
    plt.ylabel("Ratio")
    plt.xlabel("Years")
    plt.show()


    inv = pd.pivot(combined, index='date', columns='Region', values=['estimated_investor_purchase'])
    inv.columns = inv.columns.get_level_values(1)
    value = pd.pivot(combined, index='date', columns='Region', values=['value'])
    value.columns = value.columns.get_level_values(1)
    corr = value.corrwith(inv)
    corr = pd.DataFrame(corr)
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.heatmap(corr, annot=True)
    plt.xlabel("Home Price")
    plt.title("Investor Percentage Corr")
    plt.show()


    import geopandas as gpd
    import geoplot as gplt

    # Load the json file with county coordinates
    geoData = gpd.read_file('https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ca_california_zip_codes_geo.min.json')
    geoData.ZCTA5CE10 = geoData.ZCTA5CE10.astype(str).astype(int)


    investment = []
    for region, county in counties.items():
        if region == 'San Diego':
            data = zips.loc[(zips.county == county[0]) & (zips.state_code == county[1])]
            data = data.groupby(by=['zipcode']).mean().reset_index()[['zipcode', 'estimated_investor_purchase']]
            data['Region'] = region

            investment.append(data)
    map_data = pd.concat(investment).reset_index()


    temp = geoData.merge(map_data, left_on=geoData.ZCTA5CE10, right_on=map_data.zipcode, how='inner')


    gplt.polyplot(temp, figsize=(20,20))

    import mapclassify as mc
    scheme = mc.Quantiles(temp['estimated_investor_purchase'] , k=5)




    fig, ax = plt.subplots(figsize=(20, 20))
    gplt.choropleth(temp,
        hue="estimated_investor_purchase",
        scheme=scheme,linewidth=1, cmap='inferno_r',
        legend=True,
        edgecolor='black',
        ax=ax
    );




