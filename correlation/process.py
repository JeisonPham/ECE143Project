import pandas as pd
import os

data1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'full_result.csv')
# data3 = pd.read_csv(r'C:\Users\kicph\python_code\ECE143\project\ECE143Project\part_result.csv')
# data_yr1 = pd.read_csv(r'C:\Users\kicph\python_code\ECE143\project\ECE143Project\Combine\yr17-19_data.csv')



def select_time(data3,st,ed):
    """a function to select the housing price of some timerange
    Args:
        data3 (_type_): _description_
    """
# query :
# search all price in selected time range:     通过构造 字符串类型语句 的筛选条件去筛选
    #print(data3.dtypes)   
# print(data3) #
    #Example: # start="2017-12"   # end='2019-12'  
    data3['date'].astype(str)  #String type
     
# query of time start and time end
    data3['new_date']=data3['date'].apply(lambda x: x[0:7])
#create a query to show time range 
    query_item="'{}'<=new_date<='{}' ".format(st,ed)
    print(query_item)
    data_yr1=data3.query(query_item)  
    print(data_yr1)
    data_yr1.to_csv(os.path.join(os.path.dirname(__file__), 'yr' + st +'-' +ed + '_data_partial.csv'))
    
    
# select_time(data1,"2017-12",'2018-12')    
    
    
def indicator_cut(data_yr1,id):
    # col = [9,10,11]
    # data_yr1.drop(data_yr1.columns[col], axis=1)
    ZSFH_data = data_yr1.loc[data_yr1['indicator_id'] == id]
    print(ZSFH_data)
    ZSFH_data.to_csv('indicator=' + id +'.csv')
 
 
# ind = data1['indicator_id'].tolist()  
# print(ind)
ind = ['ZSFH','ZALL']
# indicator_cut(data_yr1,'ZSFH')
# indicator_cut(data1,'ZSFH')
# indicator_cut(data1,'ZSFH')

# indicator_cut(data1,'ZALL')
# indicator_cut(data1,'ZSFH')
indicator_cut(data1,'ZATT')
# indicator_cut(data1,'ZSFH')
    
