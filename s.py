import quandl

# data = quandl.get_table('ZILLOW/DATA', indicator_id=indicator_ids, region_id=region_ids, paginate=True)
# quandl.ApiConfig.api_key = '9FJNsK3v2iL5JCZZHpNQ'
import matplotlib.pyplot as plt 
from urllib.request import urlopen
import json
import plotly.express as px
import pandas as pd
#################################Data Process Method :
####No Need to ran this part of code. since I post the result for all of the functions.
#Zillow_Data is a large dataset. My laptop can not go through all of them.
#Zillow-Data is first 10000 rows of complite list. 
#partial/full result is the combined list of price, time, region_id,Fips, zips. partial result only conatins region range 99999-99951
#yr-17-19 and yrr-17-19_parcial data is data selected in year2017-2019.
# quandl.ApiConfig.api_key = '9FJNsK3v2iL5JCZZHpNQ'
#data=pd.read_csv(r'C:\Users\kicph\python_code\ECE143\project\ECE143Project\Combine\region_data_full.csv')
# ## data set from Myro
# print(data)

#data1=pd.read_csv(r'C:\Users\kicph\python_code\ECE143\project\ECE143Project\Combine\ZILLOW-DATA.csv')
# # print(data1)

# #data2 = quandl.get_table('ZILLOW/DATA', paginate=True)
# data2 = pd.read_csv(r'C:\Users\kicph\python_code\ECE143\project\ECE143Project\Combine\ZILLOW_DATA.csv')

#data

# data3=pd.merge(data1,data,how='left',on='region_id')
# data3.to_csv('part_result.csv')
# data3 = pd.read_csv(r'C:\Users\kicph\python_code\ECE143\project\ECE143Project\part_result.csv')
# data_yr1 = pd.read_csv(r'C:\Users\kicph\python_code\ECE143\project\ECE143Project\Combine\yr17-19_data.csv')
# data_yr1.drop(['state','county','city'],axis=1)
# data3 = pd.read_csv(r'C:\Users\kicph\python_code\ECE143\project\ECE143Project\full_result.csv')
# data3.dropna(subset=["FIPS"],inplace=True)
# data3.dropna(subset=["region_id"],inplace=True)
# dic={}# dict
# for i in range(0,data3.shape[0]):
#     dic[data3.at[i,"region_id"]]=[]
# for i in range(0,data3.shape[0]):
#     dic[data3.at[i,"region_id"]].append([int(data3.at[i,"value"]),data3.at[i,"date"]])
# print(dic)


def select_time(data3):
    """a function to select the housing price of some timerange
    Args:
        data3 (_type_): _description_
    """
# query :
# search all price in selected time range:     通过构造 字符串类型语句 的筛选条件去筛选
    print(data3.dtypes)   
# print(data3) #
    start="2017-12"    
    data3['date'].astype(str)  #String type
    end='2019-12'  
# query of time start and time end
    data3['new_date']=data3['date'].apply(lambda x: x[0:7])
#create a query to show time range 
    query_item="'{}'<=new_date<='{}' ".format(start,end)
    print(query_item)
    data_yr1=data3.query(query_item)  
    print(data_yr1)
    data_yr1.to_csv('yr17-19_data_partial.csv')

    
def indicator_cut(data_yr1,id):
    col = [9,10,11]
    data_yr1.drop(data_yr1.columns[col], axis=1)
    ZSFH_data = data_yr1.loc[data_yr1['indicator_id'] == id]
    print(ZSFH_data)
    ZSFH_data.to_csv('indicator=' + id +'.csv')


    
    


with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    ##link given from Jeison

df = pd.read_csv(r"C:\Users\kicph\python_code\ECE143\project\ECE143Project\yr17-19_data.csv",
                   dtype={"fips": str})

#df = pd.read_csv(r"C:\Users\kicph\python_code\ECE143\project\ECE143Project\full_result.csv",
#                    dtype={"fips": str})


# print(min(df.value),max(df.value))
#print(type(df._data))

fig = px.choropleth(df, geojson=counties, locations='FIPS', color='value', color_continuous_scale="Viridis",
                           range_color=(min(df.value),max(df.value)),
                           scope="usa", labels={'value':'Housing_price_data'} )

#px.choropleth_mapbox
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

plt.savefig('yr_plot.jpg', dpi=50)

import pandas as pd
data1=pd.read_csv(r"C:\Users\kicph\python_code\ECE143\project\ECE143Project\select\indicator=ZALL.csv")
df = data1.iloc[: , :-4]
print(df)
df.to_csv('dropyrindicator=ZALL.csv')