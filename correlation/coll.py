#!/usr/bin/env python
# coding: utf-8

# In[2]:




def main():
    import pandas as pd
    import os
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "employment_education_with Value_for 2010 (1).xlsx"),sheet_name='CA')
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "2010_data.csv"))
    df.dropna(subset = ["value_2010_avg"], inplace=True)


    # In[3]:


    # df_s = df.loc[:, ['County_Name','value_2010_avg','Population_2010','Unemployment_rate_2010',"Civilian_labor_force_2010"]]
    # df_s.to_csv(r'C:\Users\kicph\python_code\ECE143\project\ECE143Project\select\2010_data.csv')


    # In[4]:


    import seaborn as sns
    sns.pairplot(data) 


    # In[45]:


    import matplotlib.pyplot as plt
    import numpy as np
    corrmat = data.corr()
    fig, ax = plt.subplots(figsize = (18, 10))
    sns.heatmap(corrmat, annot = True, annot_kws={'size': 12})


    # In[2]:


    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "2019_data_drp.csv"),header = [0,1])
    # data = df.values # note that the data is array type
    # index1 = list(df.keys()) # title list
    # data = list(map(list, zip(*data))) # map()convert to list
    # data = pd.DataFrame(data, index=index1) # swap row and column
    # data.to_csv(r"C:\Users\kicph\python_code\ECE143\project\ECE143Project\select\2019_data.csv")

    df.columns.tolist()


    # In[4]:


    idx = pd.IndexSlice
    data = df.loc[:,[idx['Value','Median (dollars)'],idx['Sex and Age','rate over 65'],idx['Sex and Age','Median age (years)'],idx["Disability Status of the Civilian Noninstitutionalized Population","With a disability"],idx["Residence 1 Year Ago",'Different house in the U.S.'],idx['Employment Status','Unemployment Rate'],idx['Commuting to Work','Public transportation (excluding taxicab)'],idx['Commuting to Work','Walked'],idx['Commuting to Work','Worked from home'],idx['Commuting to Work','Mean travel time to work (minutes)'],idx['Year Householder Moved into Unit','Moved in 2000 to 2009'],idx['Housing Tenure','Average household size of owner-occupied unit']]]


    # In[9]:


    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    corrmat = data.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.loc[:,'Value'], annot = True)

    #sns.heatmap(corrmat, annot = True, annot_kws={'size': 12})
    #sns.heatmap(np.array(corrmat.iloc[0,:]).reshape((1,12)), annot = True, annot_kws={'size': 12})


    # In[13]:





    # In[43]:


    idx = pd.IndexSlice
    data_age = df.loc[:, [idx["Sex and Age","Under 5 years"],idx["Sex and Age","10 to 14 years"],idx["Sex and Age","15 to 19 years"],idx["Sex and Age","20 to 24 years"],idx["Sex and Age","25 to 34 years"],idx["Sex and Age","35 to 44 years"],idx["Sex and Age","45 to 54 years"],idx["Sex and Age","55 to 59 years"],idx["Sex and Age","60 to 64 years"],idx["Sex and Age","65 to 74 years"],idx["Sex and Age","75 to 84 years"],idx["Sex and Age","85 years and over"],idx["Sex and Age","Total population"]]]
    for column in data_age:
        # print(column)
        data_age[column] = data_age[column]/data_age['Sex and Age', 'Total population']
    data_age['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_age['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    corrmat = data_age.corr()

    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.droplevel(0).loc[:,'Value'], annot = True)
    #sns.heatmap(corrmat, annot = True, annot_kws={'size': 12})


    # In[44]:


    corrmat = data_age.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.loc[:,'Value'], annot = True)
    #sns.heatmap(corrmat, annot = True, annot_kws={'size': 12})
    data_age = data_age.iloc[:,0:-1]

    #data_age.rename(columns={idx['Value','Median (dollars)']: "value_key"})


    # In[46]:


    idx = pd.IndexSlice
    data_BirthPlace = df.loc[:, [idx["Place of Birth","Native"],idx["Place of Birth","Born in United States"],idx["Place of Birth","State of residence"],idx["Place of Birth","Different state"],idx["Place of Birth","Born in Puerto Rico, U.S. Island areas, or born abroad to American parent(s)"],idx['Place of Birth','Foreign born'],idx["Sex and Age","Total population"]]]
    for column in data_BirthPlace:
        # print(column)
        data_BirthPlace[column] = data_BirthPlace[column]/data_BirthPlace['Sex and Age', 'Total population']
    data_BirthPlace['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_BirthPlace['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    corrmat = data_BirthPlace.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.droplevel(0).loc[:,'Value'], annot = True)
    #sns.heatmap(corrmat, annot = True, annot_kws={'size': 12})
    data_BirthPlace = data_BirthPlace.iloc[:,0:-2]


    # In[47]:


    idx = pd.IndexSlice
    data_R = df.loc[:, [idx["Residence 1 Year Ago","Same house"],idx["Residence 1 Year Ago","Different house in the U.S."],idx["Residence 1 Year Ago","Same county"],idx["Residence 1 Year Ago","Different county"],idx["Residence 1 Year Ago","Same state"],idx['Residence 1 Year Ago','Different state'],idx["Residence 1 Year Ago","Abroad"],idx[['Sex and Age', 'Total population']]]]
    for column in data_R:
        # print(column)
        data_R[column] = data_R[column]/data_R['Sex and Age', 'Total population']
    data_R['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_R['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    corrmat = data_R.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.droplevel(0).loc[:,'Value'], annot = True, annot_kws={'size': 12})
    data_R = data_R.iloc[:,0:-2]


    # In[48]:


    idx = pd.IndexSlice
    data_Commuting = df.loc[:, [idx['Commuting to Work','Car, truck, or van -- drove alone'],idx['Commuting to Work','Car, truck, or van -- carpooled'],idx['Commuting to Work','Public transportation (excluding taxicab)'],idx['Commuting to Work','Walked'],idx["Commuting to Work","Worked from home"],idx['Sex and Age', 'Total population']]]
    for column in data_Commuting:
        # print(column)
        data_Commuting[column] = data_Commuting[column]/data_Commuting['Sex and Age', 'Total population']

    data_Commuting['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_Commuting['Commuting to Work','Mean travel time to work (minutes)'] = df.loc[:,idx['Commuting to Work','Mean travel time to work (minutes)']]
    data_Commuting['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    # data_Commuting.drop(index = 0)
    corrmat = data_Commuting.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.droplevel(0).loc[:,'Value'], annot = True, annot_kws={'size': 12})


    data_Commuting = data_Commuting.iloc[:,0:-1]
    data_Commuting = data_Commuting.drop(data_Commuting.columns[-2] ,axis = 1)


    # In[50]:


    idx = pd.IndexSlice
    data_Occupation = df.loc[:, [idx['Occupation','Civilian employed population 16 years and over'],idx['Occupation','Management, business, science, and arts occupations'],idx['Occupation','Service occupations'],idx['Occupation','Sales and office occupations'],idx['Occupation','Natural resources, construction, and maintenance occupations'],idx["Occupation","Production, transportation, and material moving occupations"],idx['Sex and Age', 'Total population']]]
    for column in data_Occupation:
        # print(column)
        data_Occupation[column] = data_Occupation[column]/data_Occupation['Sex and Age', 'Total population']
    data_Occupation['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    #data_Occupation['Commuting to Work','Mean travel time to work (minutes)'] = df.loc[:,idx['Commuting to Work','Mean travel time to work (minutes)']]
    data_Occupation['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    # data_Occupation.drop(index = 0)
    corrmat = data_Occupation.corr()
    fig, ax = plt.subplots(figsize = (1, 5))

    sns.heatmap(corrmat.droplevel(0).loc[:,'Value'], annot = True)
    #sns.heatmap(corrmat, annot = True, annot_kws={'size': 12})
    data_Occupation = data_Occupation.iloc[:,0:-2]


    # In[51]:


    idx = pd.IndexSlice
    data_Industry = df.loc[:, 'Industry']
    data_Industry['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    for column in data_Industry:
        # print(column)
        data_Industry[column] = data_Industry[column]/data_Industry['Sex and Age', 'Total population']
    data_Industry['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    #data_Industry['Commuting to Work','Mean travel time to work (minutes)'] = df.loc[:,idx['Commuting to Work','Mean travel time to work (minutes)']]
    data_Industry['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    corrmat = data_Industry.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.loc[:,[('Value', 'Median (dollars)')]], annot = True)
    #sns.heatmap(corrmat, annot = True, annot_kws={'size': 12})
    data_Industry = data_Industry.iloc[:,0:-2]


    # In[130]:


    idx = pd.IndexSlice
    data_Worker = df.loc[:, 'Class of Worker']
    data_Worker['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    for column in data_Worker:
        # print(column)
        data_Worker[column] = data_Worker[column]/data_Worker['Sex and Age', 'Total population']
    data_Worker['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_Worker['Commuting to Work','Mean travel time to work (minutes)'] = df.loc[:,idx['Commuting to Work','Mean travel time to work (minutes)']]
    data_Worker['Employment Status','Unemployment Rate'] = df.loc[:,idx['Employment Status','Unemployment Rate']]
    data_Worker['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    corrmat = data_Worker.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.loc[:,[('Value', 'Median (dollars)')]], annot = True, annot_kws={'size': 12})
    data_Worker = data_Worker.iloc[:,0:-1]
    data_Worker = data_Worker.drop(data_Worker.columns[-3],axis = 1)


    # In[58]:


    idx = pd.IndexSlice
    data_Income = df.loc[:, 'Income and Benefits (In 2019 inflation-adjusted dollars)']
    data_Income['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    for column in data_Income:
        # print(column)
        data_Income[column] = data_Income[column]/data_Income['Sex and Age', 'Total population']
    data_Income['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    #data_Income['Commuting to Work','Mean travel time to work (minutes)'] = df.loc[:,idx['Commuting to Work','Mean travel time to work (minutes)']]
    data_Income['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]

    corrmat = data_Income.corr()
    fig, ax = plt.subplots(figsize = (1, 10))
    sns.heatmap(corrmat.loc[:,[('Value', 'Median (dollars)')]], annot = True, annot_kws={'size': 12})
    data_Worker = data_Worker.iloc[:,0:-2]


    # In[59]:


    idx = pd.IndexSlice
    data_Health = df.loc[:, 'Health Insurance Coverage']
    data_Health['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    for column in data_Health:
        # print(column)
        data_Health[column] = data_Health[column]/data_Health['Sex and Age', 'Total population']
    data_Health['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_Health['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    corrmat = data_Health.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.loc[:,[('Value', 'Median (dollars)')]], annot = True, annot_kws={'size': 12})
    data_Worker = data_Worker.iloc[:,0:-2]


    # In[55]:


    idx = pd.IndexSlice
    data_Education = df.loc[:, 'Educational Attainment']
    data_Education['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    for column in data_Education.columns[0:-3]:
        # print(column)
        data_Education[column] = data_Education[column]/data_Education['Sex and Age', 'Total population']
    data_Education['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    #data_Education['Commuting to Work','Mean travel time to work (minutes)'] = df.loc[:,idx['Commuting to Work','Mean travel time to work (minutes)']]
    data_Education['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    corrmat = data_Education.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.loc[:,[('Value', 'Median (dollars)')]], annot = True, annot_kws={'size': 12})
    # print(data_Education)
    # data_Education.rename(columns={idx['Value','Median (dollars)']: "value_key"})

    data_Education = data_Education.iloc[:,0:-2]


    # In[129]:


    idx = pd.IndexSlice
    data_Race = df.loc[:, 'Race']
    for column in data_Race:
        # print(column)
        data_Race[column] = data_Race[column]/data_Race['Total population']
    #data_Race['Commuting to Work','Mean travel time to work (minutes)'] = df.loc[:,idx['Commuting to Work','Mean travel time to work (minutes)']]
    data_Race['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_Race = data_Race.iloc[:,1:]
    corrmat = data_Race.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.loc[:,[('Value', 'Median (dollars)')]], annot = True, annot_kws={'size': 12})


    # In[61]:


    import sklearn
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    threshold_list = np.arange(0.1,1,0.1)
    list_of_train =pd.concat([data_age,data_BirthPlace,data_R,data_Commuting,data_Occupation,data_Industry,data_Worker,data_Income,data_Health,data_Education,data_Race], axis=1)
    def Trainset_selected(list_of_train):
        X = list_of_train.iloc[:,0:-1]
        y = list_of_train.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        return X_train, X_test, y_train, y_test


    # In[62]:


    from sklearn import metrics
    def find_the_score(X_train,X_test, y_train,y_test):
        linreg = LinearRegression()
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


    # In[127]:


    from sklearn.svm import SVR
    #https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
    def svd_score(X_train,X_test, y_train,y_test):
        svr = SVR()
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        return metrics.r2_score(y_test, y_pred),metrics.mean_absolute_error(y_test, y_pred),metrics.mean_squared_error(y_test, y_pred),np.sqrt(metrics.mean_squared_error(y_test, y_pred))


    # In[128]:


    from sklearn.ensemble import RandomForestRegressor
    #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    def rfr_score(X_train,X_test, y_train,y_test):
        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_test)
        return metrics.r2_score(y_test, y_pred),metrics.mean_absolute_error(y_test, y_pred),metrics.mean_squared_error(y_test, y_pred),np.sqrt(metrics.mean_squared_error(y_test, y_pred))


    # In[134]:


    def avg_score(X_train,X_test, y_train,y_test):
        svr = SVR()
        svr.fit(X_train, y_train)
        y_pred_1 = svr.predict(X_test)

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)
        y_pred_2 = rfr.predict(X_test)

        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        y_pred_3 = linreg.predict(X_test)

        y_pred = (y_pred_1+y_pred_2+y_pred_3)/3
        return metrics.r2_score(y_test, y_pred),metrics.mean_absolute_error(y_test, y_pred),metrics.mean_squared_error(y_test, y_pred),np.sqrt(metrics.mean_squared_error(y_test, y_pred))    


    # In[82]:


    list_of_train =pd.concat([data_age,data_BirthPlace,data_R,data_Commuting,data_Occupation,data_Industry,data_Worker,data_Income,data_Health,data_Education,data_Race], axis=1)
    list_of_train = list_of_train*100
    list_of_train = list_of_train.iloc[:,0:-1]
    list_of_train['value_price'] = df.loc[:,idx['Value','Median (dollars)']]
    corrmat = list_of_train.corr()
    abs_c = np.absolute(corrmat)
    from numpy.random import seed
    from numpy.random import normal
    from numpy import savetxt

    print (list_of_train)
    def map_score(t, model='LinearReg'):
        features_l = []
        temp = abs_c.iloc[-1,:]
        for i in range(97):
            if temp[i] > t:    
               features_l.append(i)
        after_threshold = list_of_train.iloc[:,features_l]

        x1,x2,x3,x4 = Trainset_selected(after_threshold)
        if model == 'LinearReg':
            return find_the_score(x1,x2,x3,x4)
        elif model == 'SVD':
            return svd_score(x1,x2,x3,x4)
        elif model == "RFR":
            return rfr_score(x1,x2,x3,x4)
        elif model == 'AVG':
            return avg_score(x1,x2,x3,x4)

    import matplotlib.pyplot as plt
    k_v,r2,mae,mse,msre = [],[],[],[],[]

    for k in np.arange(0,0.95,0.05):
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


    # for i in np.arange(0,1,0.1):


    # In[83]:


    plt.plot(k_v,msre)
    plt.title('MSRE score change with step = 0.05')
    plt.xlabel('threshold value')
    plt.ylabel('MSRE value')
    plt.show()


    # In[84]:


    ###DONT RUN THIS PART CODE. Just SEE the result
    plt.plot(k_v,r2)
    plt.title('R-2 score change with step = 0.05')
    plt.xlabel('threshold value')
    plt.ylabel('r_2 value')
    plt.show()


    # In[85]:



    plt.plot(k_v,mae)
    plt.title('MAE score change with step = 0.05')
    plt.xlabel('threshold value')
    plt.ylabel('MAE value')
    plt.show()


    # In[76]:


    list_of_train =pd.concat([data_age,data_BirthPlace,data_R,data_Commuting,data_Occupation,data_Industry,data_Worker,data_Income,data_Health,data_Education,data_Race], axis=1)
    list_of_train = list_of_train*100
    list_of_train = list_of_train.iloc[:,0:-1]
    list_of_train['value_price'] = df.loc[:,idx['Value','Median (dollars)']]
    corrmat = list_of_train.corr()
    abs_c = np.absolute(corrmat)
    from numpy.random import seed
    from numpy.random import normal
    from numpy import savetxt

    def map_score(t):
        features_l = []
        temp = abs_c.iloc[-1,:]
        for i in range(97):
            if temp[i] > t:    
               features_l.append(i)
        after_threshold = list_of_train.iloc[:,features_l]
        return after_threshold

    data = map_score(0.8)
    data = data.loc[:,~data.columns.duplicated()]
    #print(data.loc[:,[('Value', 'Median (dollars)')]])
    corrmat = data.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    sns.heatmap(corrmat.loc[:,[('Value', 'Median (dollars)')]], annot = True, annot_kws={'size': 12})


    # In[126]:


    list_of_train =pd.concat([data_age,data_BirthPlace,data_R,data_Commuting,data_Occupation,data_Industry,data_Worker,data_Income,data_Health,data_Education,data_Race], axis=1)
    list_of_train = list_of_train*100
    list_of_train = list_of_train.iloc[:,0:-1]
    list_of_train.drop(columns=['Bachelor\'s degree', 'Graduate or professional degree', 'Mean household income (dollars)', 'Private wage and salary workers', 'Civilian employed population 16 years and over', 'Professional, scientific, and management, and administrative and waste management services'], inplace=True)
    list_of_train['value_price'] = df.loc[:,idx['Value','Median (dollars)']]
    list_of_train.drop(columns=[('Place of Birth', 'State of residence')], inplace=True)

    corrmat = list_of_train.corr()
    abs_c = np.absolute(corrmat)
    from numpy.random import seed
    from numpy.random import normal
    from numpy import savetxt

    def map_score(t):
        features_l = []
        temp = abs_c.iloc[-1,:]
        for i in range(abs_c.shape[1]):
            if temp[i] > t:    
               features_l.append(i)
        after_threshold = list_of_train.iloc[:,features_l]
        return after_threshold

    data = map_score(0.67)
    data = data.loc[:,~data.columns.duplicated()]
    #print(data.loc[:,[('Value', 'Median (dollars)')]])
    data['Commuting to Work, Walked from home'] = list_of_train.loc[:,[('Commuting to Work', 'Walked')]]
    data['Population Age 45-54'] = list_of_train.loc[:,[('Sex and Age','45 to 54 years')]]
    corrmat = data.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    corrmat.drop(index=[('Value', 'Median (dollars)')], inplace=True)
    corrmat.drop(index=['value_price'], inplace=True)
    corrmat.rename(index={'Information':"('Occupation', 'Information')", 'With public coverage':"Public health coverage", "$150,000 to $199,000":"Household income, $150,000-$199,000", "$200,000 or more":"Household income, $200,000 or more"}, inplace=True)
    sns.heatmap(corrmat.loc[:,[('Value', 'Median (dollars)')]], annot = True, annot_kws={'size': 12})




