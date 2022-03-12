import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import plotly
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import seed
from numpy.random import normal
from numpy import savetxt
from sklearn import metrics
import plotly.graph_objects as go
import os

def main():
    data2 = pd.read_csv(os.path.join(os.path.dirname(__file__), "features_to _train.csv"))
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "list_of_train.csv"))
    def Trainset_selected(list_of_train):
        X = list_of_train.iloc[:,0:-1]
        y = list_of_train.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        # print("X is :", X_train)
        return X_train, X_test, y_train, y_test
    
    def find_the_score(X_train,X_test, y_train,y_test):
        linreg =GradientBoostingRegressor()
        linreg.fit(X_train, y_train)
        # print (linreg.intercept_)
        # print (linreg.coef_)
        y_pred = linreg.predict(X_test)
    #     print("r2_score:",metrics.r2_score(y_test, y_pred)) 
    # # MSE
    #     print("MSE:",metrics.mean_squared_error(y_test, y_pred)) 
    # # RMSE
    #     print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
        return metrics.r2_score(y_test, y_pred),metrics.mean_absolute_error(y_test, y_pred),metrics.mean_squared_error(y_test, y_pred),np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    corrmat = data.corr()
    abs_c = np.absolute(corrmat)


    def map_score(t, model='LinearReg'):
        features_l = []
        temp = abs_c.iloc[-1,:]
        for i in range(93):
            if temp[i] > t:    
               features_l.append(i)
        after_threshold = data.iloc[:,features_l]
        # print(after_threshold)
        x1,x2,x3,x4 = Trainset_selected(after_threshold)  
        # print("Train set is :" , x1)
        # print("Y is",x3)
        if model == 'LinearReg':
            return find_the_score(x1,x2,x3,x4)
        # elif model == 'SVD':
        #     return svd_score(x1,x2,x3,x4)
        # elif model == "RFR":
        #     return rfr_score(x1,x2,x3,x4)
        #  elif model == 'AVG':
        #     return avg_score(x1,x2,x3,x4)

    k_v,r2,mae,mse,msre = [],[],[],[],[]

    for k in np.arange(0.5,0.95,0.05):
    # for k in [0.1,0.9]:
        a1,a4,a2,a3 = map_score(k)
        r2.append(a1)
        mse.append(a2)
        msre.append(a3)
        mae.append(a4)
        k_v.append(k)
    plt.plot(k_v,mse)
    plt.title('MSE score change with step = 0.05')
    plt.xlabel('threshold value')
    plt.ylabel('MSE value')
    plt.show()


    # In[56]:


    plt.plot(k_v,mae)
    plt.title('MSE score change with step = 0.05')
    plt.xlabel('threshold value')
    plt.ylabel('MSE value')
    plt.show()


    # In[57]:


    plt.plot(k_v,r2)
    plt.title('MSE score change with step = 0.05')
    plt.xlabel('threshold value')
    plt.ylabel('MSE value')
    plt.show()


    # In[58]:


    plt.plot(k_v,msre)
    plt.title('MSE score change with step = 0.05')
    plt.xlabel('threshold value')
    plt.ylabel('MSE value')
    plt.show()

    X = data2.iloc[:,0:-1]

    y =data2.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    model = GradientBoostingRegressor()                   
    model.fit(X_train, y_train)
    x_range = np.linspace(0, 6200, 100)
    y_range = model.predict(X_test)
    print(y_range)
    x1 = X_test.squeeze().copy()
    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x1, y=y_range, name='prediction by GradientBoostingRegressor',marker_color='rgba(200, 100, 1, 100.8)',opacity=0.4)
    ])
    fig.show()


    # In[47]:



    

    X = data2.iloc[:,0:-1]

    y =data2.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state = 1)
    model = GradientBoostingRegressor()                   
    model.fit(X_train, y_train)
    x_range = np.linspace(0, 6200, 100)
    y_range = model.predict(X_test)
    print(y_range)
    x1 = X_test.squeeze().copy()
    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x1, y=y_range, name='prediction by GradientBoostingRegressor',marker_color='rgba(200, 100, 1, 100.8)',opacity=0.4)
    ])
    fig.show()


    X = data2.iloc[:,0:-1]

    y =data2.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
    model = GradientBoostingRegressor()                   
    model.fit(X_train, y_train)
    x_range = np.linspace(0, 6200, 100)
    y_range = model.predict(X_test)
    print(y_range)
    x1 = X_test.squeeze().copy()
    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x1, y=y_range, name='prediction by GradientBoostingRegressor',marker_color='rgba(200, 100, 1, 100.8)',opacity=0.4)
    ])
    fig.show()


    X = data2.iloc[:,0:-1]

    y =data2.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
    model = RandomForestRegressor()                   
    model.fit(X_train, y_train)
    x_range = np.linspace(0, 6200, 100)
    y_range = model.predict(X_test)
    x1 = X_test.squeeze().copy()
    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x1, y=y_range, name='prediction by RandomForestRegressor',marker_color='rgba(200, 100, 1, 100.8)',opacity=0.4)
    ])
    fig.show()


    X = data2.iloc[:,0:-1]
    y =data2.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state = 1)

    model = LinearRegression()
    model.fit(X_train, y_train)
    x_range = np.linspace(0, 6200, 100)
    y_range = model.predict(X_test)
    x1 = X_test.squeeze().copy()
    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x1, y=y_range, name='prediction by LinearRegression',marker_color='rgba(200, 100, 1, 100.8)',opacity=0.2)
    ])
    fig.show()


    # In[52]:


    X = data2.iloc[:,0:-1]
    e = []
    y =data2.iloc[:,-1]
    
    model1 = GradientBoostingRegressor()
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

    for i in range(100):
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
        model1.fit(X_train, y_train)
        res = model1.predict(X_test)
        a = np.subtract(res,y_test)
        e = e + list(a)


    # In[53]:


    model1 = RandomForestRegressor()
    d = []
    for i in range(100):
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
        model1.fit(X_train, y_train)
        res = model1.predict(X_test)
        a = np.subtract(res,y_test)
        d = d + list(a)


    # In[54]:


    X = data2.iloc[:,0:-1]
    c = []
    y =data2.iloc[:,-1]
    model1 = LinearRegression()
    for i in range(100):
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
        model1.fit(X_train, y_train)
        res = model1.predict(X_test)
        a = np.subtract(res,y_test)
        c = c + list(a)


    # In[55]:




    fig = go.Figure()
    fig.add_trace(go.Box(
        y=c,
        name="Linear Resression",
        boxpoints='outliers', # only outliers
        marker_color='rgb(7,40,89)',
        line_color='rgb(7,40,89)'
    ))

    fig.add_trace(go.Box(
        y=d,
        name="RandomForestResression",
        boxpoints='outliers', # only outliers
        marker_color='rgb(8,81,156)',
        line_color='rgb(8,81,156)'
    ))


    fig.add_trace(go.Box(
        y=e,
        name="GradientBoostingRegressor",
        boxpoints='outliers', # only outliers
        marker_color='rgb(107,174,214)',
        line_color='rgb(107,174,214)'
    ))


    fig.update_layout(title_text="Box Plot Styling Outliers")
    fig.show()

