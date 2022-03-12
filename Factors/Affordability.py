import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from matplotlib.offsetbox import AnchoredText
sns.set_theme()
import os

def main():

    def combine_data(Regions:list):
        """
        Combines region price data with median income
        """
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/metro.csv'))
        counties = []
        for region in Regions:
            county = df[df['RegionName'].str.contains(region)]
            county = county.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName'])
            county = pd.melt(county, id_vars='RegionName', value_name='value')
            county.columns = ['RegionName', 'date', 'value']
            county['date'] = pd.to_datetime(county['date'], format='%Y-%m-%d')
            county = county.groupby([county['date'].dt.year, 'RegionName']).mean().reset_index()
            county = county[county.date > 2000]



            median_income = pd.read_csv(os.path.join(os.path.dirname(__file__), f"Data/{region} median income.csv"))
            median_income.columns = ['date', 'Median Income']

            median_income['date'] = pd.to_datetime(median_income.date).astype(str)
            median_income['date'] = pd.to_datetime(median_income['date']).dt.year
            median_income = median_income[median_income['date'] > 2000]
            median_income['Median Income'] = median_income['Median Income'].astype(float)


            ratio = county.merge(median_income, on='date', how='left')
            ratio['Region'] = region

            fill = ratio['Median Income'].shift(1) * 1.01
            ratio['Median Income'].fillna(fill, inplace=True)

            ratio.dropna(inplace=True)

            ratio['median_growth'] = (ratio['Median Income'] / ratio['Median Income'].iat[0] - 1) * 100
            ratio['value_growth'] = (ratio['value'] / ratio['value'].iat[0] - 1) * 100

            counties.append(ratio)

        counties = pd.concat(counties).reset_index().drop(columns=['index'])
        return counties

    combined = combine_data(['San Diego', 'Los Angeles', 'San Francisco', 'St. Louis', 'New York', 'Boston', "Youngstown", 'San Jose', 'Ventura', 'Denver'])


    def plot_bar_graph(df):
        counter = 0
        for date in df.date.unique():
            data = df[df.date == date]
            regions = list(data.Region.unique())
            vals = ['median_growth', 'value_growth']
            labels = ["Median Income Growth %", "Median House Value Growth %"]
            n = len(vals)
            width = 0.25
            _X = np.arange(len(regions))
            fig, ax = plt.subplots(figsize=(10, 10))
            anc = AnchoredText(date.astype(str), loc="upper right", frameon=False)
            ax.add_artist(anc)
            for i in range(n):
                bar = ax.bar(_X - width/2. + i/float(n)*width, data[vals[i]],
                        width=width/float(n), align="edge", label=labels[i])
            plt.xticks(_X, regions)
            plt.ylim([0, 175])
            plt.ylabel("Percentage")
            plt.xlabel("County")
            plt.title(f"Percentage Growth of Income and Home Price since 2001")
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(os.path.dirname(__file__), f"Median Income Graphs/{counter:03d}.png"), bbox_inches='tight')
            plt.show()
            counter += 1
            
    plot_bar_graph(combined)
    combined['ratio'] = combined['value'] / combined['Median Income']

    def group_values(df, values):
        temp =  pd.pivot_table(df, index='date', columns='Region', values=values)
        temp['Group 1'] = temp[['Youngstown', 'St. Louis']].mean(axis=1)
        temp['Group 2'] = temp[['Ventura', 'Boston', 'New York']].mean(axis=1)
        temp['Group 3'] = temp[['Los Angeles', 'San Francisco', 'San Jose']].mean(axis=1)
        temp = temp[['San Diego', 'Group 1', 'Group 2', 'Group 3']].reset_index()
        temp = pd.melt(temp, id_vars=['date'], value_vars=['San Diego', 'Group 1', 'Group 2', 'Group 3'])
        temp[values] = temp['value']
        return temp
    
    temp1 = group_values(combined, 'value_growth')
    temp = group_values(combined, 'median_growth')
    plot_bar_graph(temp.merge(temp1, on=['date', 'Region']))
    temp = group_values(combined, 'value').merge(group_values(combined, 'Median Income'), on=['date', 'Region'])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    temp['ratio'] = temp['value_x'] / temp['Median Income']
    for region in temp.Region.unique():
        regions = temp[temp.Region == region]
        dates = pd.to_datetime(regions.date.astype(str))
        ax.plot(dates.dt.year, regions.ratio, label=region)
    plt.legend()
    plt.locator_params(axis='x', nbins=4)
    plt.title("Home Price / Median Income vs Time")
    plt.xlabel("Dates")
    plt.ylabel("Ratio")
    plt.axvspan(2005, 2009, facecolor='g', alpha=0.2, label="Housing Market Crash")
    plt.axvspan(2019, 2021, facecolor='r', alpha=0.2, label='Covid')
    plt.show()

    fig, ax = plt.subplots(figsize=(20, 10))
    for region in combined.RegionName.unique():
        data = combined[combined.RegionName == region]
        ax.scatter(x = data.date, y = data['Median Income'], s = data.value / 500, alpha = 0.5)

    plt.show()



