import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from datetime import datetime
import os
sns.set_theme()

def main():

    counties = ['Boston', 'Denver', 'Los Angeles', 'McAllen', 'Merced', 'New York', 'San Diego', 'San Francisco', 'San Jose', 'St. Louis', 'Ventura']

    files = glob(os.path.join(os.path.dirname(__file__), "Data/inventory data/*"))


    prices = pd.read_csv(os.path.join(os.path.dirname(__file__), "Data/metro.csv"))


    inventory = []
    for file, county in zip(files, counties):
        data = pd.read_csv(file)
        data.columns = ['date', 'Inventory']
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        data['Region'] = county
        data = data.reset_index()

        county = prices[prices['RegionName'].str.contains(county)]
        county = county.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName'])
        county = pd.melt(county, id_vars='RegionName', value_name='value')
        county.columns = ['RegionName', 'date', 'value']
        county['date'] = pd.to_datetime(county['date'], format='%Y-%m-%d')
        county['value'] /= 1000

        data = data.merge(county, left_on=data['date'].apply(lambda x: (x.year, x.month)),
                                  right_on=county['date'].apply(lambda x: (x.year, x.month)), how='inner')
        data['date'] = data['date_x']

        inventory.append(data)

    pop = pd.concat(inventory).reset_index().dropna()


    color = plt.cm.rainbow(np.linspace(0, 1, 15))
    fig, ax = plt.subplots(figsize=(10, 10))
    for index, region in enumerate(pop.Region.unique()):
        data = pop[pop.Region == region]
        data['date'] = pd.to_datetime(data['date'])
        ax.plot(data['date'], data['value'] / data['Inventory'], label=region, color=color[index])
        ax.annotate(region, (data['date'].iloc[-1], data['value'].iloc[-1] / data['Inventory'].iloc[-1]))
    plt.legend()
    plt.axvspan(datetime(2020, 1, 1), datetime(2022, 1, 1), facecolor='black', alpha=0.2)
    plt.show()


    group1 = ['New York', 'McAllen', 'Los Angeles', 'St. Louis']
    group2 = ['San Jose', 'Denver', 'San Francisco', 'Merced', 'Ventura']
    temp = pop
    temp['ratio'] = temp['value'] / temp['Inventory']
    temp =  pd.pivot_table(temp, index='date', columns='Region', values='ratio')
    temp['Group 1'] = temp[group1].mean(axis=1)
    temp['Group 2'] = temp[group2].mean(axis=1)
    temp = temp[['San Diego', 'Boston', 'Group 1', 'Group 2']].reset_index()
    temp = pd.melt(temp, id_vars=['date'], value_vars=['San Diego', 'Boston', 'Group 1', 'Group 2'])
    temp['ratio'] = temp['value']

    fig, ax = plt.subplots(figsize=(15, 10))
    for index, region in enumerate(temp.Region.unique()):
        data = temp[temp.Region == region]
        data['date'] = pd.to_datetime(data['date'])
        ax.plot(data['date'], data['ratio'], label=region)
    plt.legend()
    plt.title("Home Price to Inventory Ratio")
    plt.ylabel("Ratio")
    plt.xlabel("Year")
    plt.show()

    for index, date in enumerate(sorted(pop.date.unique())):
        print(date)
        fig, ax = plt.subplots(figsize=(10, 10))
        data = pop[pop.date == date]
        data = data[["Inventory", "value"]]
        corr = data.corr()
        sns.heatmap(corr, annot=True, cbar=False)
        date = pd.to_datetime(date, format="%Y")
        plt.title(f'Correlation of Inventory and Home Value {date.strftime("%Y-%b")}')
        plt.savefig(os.path.join(os.path.dirname(__file__), f"Inventory/corr_{index:03d}.png"), bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    inv = pd.pivot(pop, index='date', columns='Region', values=['Inventory'])
    inv.columns = inv.columns.get_level_values(1)
    value = pd.pivot(pop, index='date', columns='Region', values=['value'])
    value.columns = value.columns.get_level_values(1)


    corr = value.corrwith(inv)
    corr = pd.DataFrame(corr)
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.heatmap(corr, annot=True)
    plt.xlabel("Home Price")
    plt.title("Inventory Correlation")
    plt.show()





