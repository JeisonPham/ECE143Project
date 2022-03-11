#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from glob import glob
import os


sns.set_theme()

def main():

    # In[2]:


    files = glob(os.path.join(os.path.dirname(__file__), "Data/migration data/*.csv"))


    # In[3]:


    files = list(filter(lambda x: "pop" in x, files))


    # In[4]:


    prices = pd.read_csv(os.path.join(os.path.dirname(__file__), "Data/metro.csv"))


    # In[20]:


    populations = []
    counties = ['Boston', 'Denver', 'Los Angeles', 'McAllen', 'Merced', 'New York', 'San Diego', 'San Francisco', 'San Jose', 'St. Louis', 'Ventura', 'Wichita']
    for file, county in zip(files, counties):
        data = pd.read_csv(file)
        data.columns = ['date', 'pop']
        data['net'] = data['pop'].diff(1)
        data['net normal'] = data['net'] / data['net'].abs().max()
        data['percent pop'] = (data['pop'] / data['pop'].iat[0] - 1) * 100
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d').dt.year
        data['Region'] = county
        data = data.reset_index()

        county = prices[prices['RegionName'].str.contains(county)]
        county = county.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName'])
        county = pd.melt(county, id_vars='RegionName', value_name='value')
        county.columns = ['RegionName', 'date', 'value']
        county['date'] = pd.to_datetime(county['date'], format='%Y-%m-%d')
        county = county.groupby([county['date'].dt.year, 'RegionName']).mean().reset_index()
        county = county[county.date > 2000]
        county['price difference'] = county['value'].diff(1)
        county['price difference'] /= county['price difference'].abs().max()
        county['percent value'] = (county['value'] / county['value'].iat[0] - 1) * 100



        data = data.merge(county, left_on=['date'], right_on=['date'], how='inner')

        populations.append(data)

    pop = pd.concat(populations).reset_index().dropna()


    # In[21]:


    pop.dropna(inplace=True)
    pop


    # In[7]:


    counter = 0
    color = plt.cm.rainbow(np.linspace(0, 1, 15))
    print(color)
    for date in sorted(pop.date.unique()):
        data = pop[pop.date == date]
        fig, ax = plt.subplots(figsize=(10, 10))
        bar = ax.barh(data.Region, data.net, align='center')
        for i in range(len(bar)):
            bar[i].set_color(color[i])
        plt.xlim([-100, 100])
        plt.xlabel("Net Population in Thousands")
        date = pd.to_datetime(str(date))
        plt.title(date.strftime("%Y"))
        plt.savefig(os.path.join(os.path.dirname(__file__), f"Migration Graphs/bar_{counter:05d}.png"), bbox_inches='tight')
        plt.show()
        counter += 1


    # In[8]:


    fig, ax = plt.subplots(figsize=(20, 10))
    for region in pop.RegionName.unique():
        regions = pop[pop.RegionName == region]
        ax.plot(regions.index, regions['value'] / regions['net'].abs(), label=region, alpha=0.5)
    plt.legend()
    plt.xlabel("Ratio")
    plt.ylabel("Years")
    plt.show()


    # In[9]:



    for index, date in enumerate(sorted(pop.date.unique())):
        fig, ax = plt.subplots(figsize=(10, 10))
        data = pop[pop.date == date]
        data = data[["pop", "net", "value"]]
        corr = data.corr()
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, annot=True, cbar=False)
        date = pd.to_datetime(date, format="%Y")
        plt.title(f'Correlation of Population, Net, and Home Value {date.strftime("%Y")}')
        plt.savefig(os.path.join(os.path.dirname(__file__), f"Migration Graphs/corr_{index:03d}.png"), bbox_inches='tight')
        plt.tight_layout()
        plt.show()


    # In[10]:


    fig, ax = plt.subplots(figsize=(10, 10))
    data = pop[['pop', 'net', 'value']]
    corr = data.corr()
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, cbar=False)
    plt.title(f'Correlation of Population, Net, and Home Value from 2003 - 2020')
    plt.savefig(os.path.join(os.path.dirname(__file__), f"Migration Graphs/corr.png"), bbox_inches='tight')
    plt.tight_layout()
    plt.show()


    # In[11]:



    # for date in pop.date.unique():
    #     p = pop[pop.date == date]
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     for index, region in enumerate(p.Region.unique()):
    #         data = p[p.Region == region]
    #         ax.scatter(data['net normal'], data['price difference'], label=region, color=color[index], alpha=1)
    #         ax.annotate(region, (data['net normal'], data['price difference']))
    #     plt.title(f"Net Population Change vs Home Price Difference {date}")
    #     plt.xlim([-1.1, 1.1])
    #     plt.ylim([-1.1, 1.1])
    #     plt.axvspan(0, 2, ymin=0, ymax=0.5, facecolor='g', alpha=0.3)
    #     plt.axvspan(0, 2, ymin=0.5, ymax=1, facecolor='y', alpha=0.3)
    #     plt.axvspan(-2, 0, ymin=0, ymax=0.5, facecolor='y', alpha=0.3)
    #     plt.axvspan(-2, 0, ymin=0.5, ymax=1, facecolor='r', alpha=0.3)
    #     plt.xlabel("Net Population")
    #     plt.ylabel("Home Price")
    #     plt.savefig(f"Migration Graphs/xy_{date}.png", bbox_inches='tight')
    #     plt.show()
    for index, region in enumerate(pop.Region.unique()):
        fig, ax = plt.subplots(figsize=(10, 10))
        data = pop[pop.Region == region]
        data = data.iloc[-5:]
        ax.plot(data['net normal'], data['price difference'], color=color[index])
        ax.scatter(data['net normal'], data['price difference'], color=color[index])
        ax.annotate(data.date.iloc[-1], (data['net normal'].iloc[-1], data['price difference'].iloc[-1]))
        ax.annotate(data.date.iloc[0], (data['net normal'].iloc[0], data['price difference'].iloc[0]))
        ax.annotate(region, (-1, 0.9), fontsize=24, color='white')
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.axvspan(0, 2, ymin=0, ymax=0.5, facecolor='g', alpha=0.3)
        plt.axvspan(0, 2, ymin=0.5, ymax=1, facecolor='y', alpha=0.3)
        plt.axvspan(-2, 0, ymin=0, ymax=0.5, facecolor='y', alpha=0.3)
        plt.axvspan(-2, 0, ymin=0.5, ymax=1, facecolor='r', alpha=0.3)
        plt.xlabel("Net Population")
        plt.ylabel("Home Price")
        plt.title(f"Net Population vs Home Price")
        plt.savefig(os.path.join(os.path.dirname(__file__), f"Migration Graphs/xy_{region}.png"), bbox_inches='tight')
        plt.show()
        plt.show()


    # In[15]:


    pop


    # In[41]:


    from matplotlib.offsetbox import AnchoredText
    pop['value'] /= 100
    def plot_bar_graph(df):
        counter = 0
        for date in df.date.unique():
            data = df[df.date == date]
            regions = list(data.Region.unique())
            vals = ['percent pop', 'percent value']
            labels = ["Population", "Home Price"]
            n = len(vals)
            width = 0.25
            _X = np.arange(len(regions))
            fig, ax = plt.subplots(figsize=(15, 10))
            anc = AnchoredText(date.astype(str), loc="upper right", frameon=False)
            ax.add_artist(anc)
            for i in range(n):
                bar = ax.bar(_X - width/2. + i/float(n)*width, data[vals[i]],
                        width=width/float(n), align="edge", label=labels[i])
            plt.xticks(_X, regions)
            plt.ylim([-10, 200])
            plt.ylabel("Percentage")
            plt.xlabel("County")
            plt.title(f"Percentage Growth of Income and Home Price since 2001")
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(os.path.dirname(__file__), f"Migration Graphs/bar_{date}.png"), bbox_inches='tight')
            plt.show()
            counter += 1


    # In[42]:


    plot_bar_graph(pop)


    # In[34]:


    long = pd.pivot_table(pop[['date','pop', 'Region']], index='date', columns='Region', values=['pop'])
    long.columns = long.columns.get_level_values(1)
    long


    # In[36]:


    price = pd.pivot_table(pop[['date','value', 'Region']], index='date', columns='Region', values=['value'])
    price.columns = price.columns.get_level_values(1)
    price


    # In[37]:


    corr = pd.DataFrame(price.corrwith(long))
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.heatmap(corr, annot=True)
    plt.xlabel("Home Price")
    plt.title("Population Correlation")
    plt.show()




