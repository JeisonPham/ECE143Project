#!/usr/bin/env python
# coding: utf-8


def main():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from numpy.random import seed
    from numpy.random import normal
    from numpy import savetxt


    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "2019_data_drp.csv"),header = [0,1])
    df.columns.tolist()
    idx = pd.IndexSlice

    #data = df.loc[:,[idx['Value','Median (dollars)'],idx['Sex and Age','rate over 65'],idx['Sex and Age','Median age (years)'],idx["Disability Status of the Civilian Noninstitutionalized Population","With a disability"],idx["Residence 1 Year Ago",'Different house in the U.S.'],idx['Employment Status','Unemployment Rate'],idx['Commuting to Work','Public transportation (excluding taxicab)'],idx['Commuting to Work','Walked'],idx['Commuting to Work','Worked from home'],idx['Commuting to Work','Mean travel time to work (minutes)'],idx['Year Householder Moved into Unit','Moved in 2000 to 2009'],idx['Housing Tenure','Average household size of owner-occupied unit']]]
    data_age = df.loc[:, [idx["Sex and Age","Under 5 years"],idx["Sex and Age","10 to 14 years"],idx["Sex and Age","15 to 19 years"],idx["Sex and Age","20 to 24 years"],idx["Sex and Age","25 to 34 years"],idx["Sex and Age","35 to 44 years"],idx["Sex and Age","45 to 54 years"],idx["Sex and Age","55 to 59 years"],idx["Sex and Age","60 to 64 years"],idx["Sex and Age","65 to 74 years"],idx["Sex and Age","75 to 84 years"],idx["Sex and Age","85 years and over"],idx["Sex and Age","Total population"]]]
    for column in data_age:
        data_age[column] = data_age[column]/data_age['Sex and Age', 'Total population']
    data_age['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_age['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_age = data_age.iloc[:,0:-1]

    data_BirthPlace = df.loc[:, [idx["Place of Birth","Native"],idx["Place of Birth","Born in United States"],idx["Place of Birth","State of residence"],idx["Place of Birth","Different state"],idx["Place of Birth","Born in Puerto Rico, U.S. Island areas, or born abroad to American parent(s)"],idx['Place of Birth','Foreign born'],idx["Sex and Age","Total population"]]]
    for column in data_BirthPlace:
        data_BirthPlace[column] = data_BirthPlace[column]/data_BirthPlace['Sex and Age', 'Total population']
    data_BirthPlace['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_BirthPlace['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_BirthPlace = data_BirthPlace.iloc[:,0:-2]

    data_R = df.loc[:, [idx["Residence 1 Year Ago","Same house"],idx["Residence 1 Year Ago","Different house in the U.S."],idx["Residence 1 Year Ago","Same county"],idx["Residence 1 Year Ago","Different county"],idx["Residence 1 Year Ago","Same state"],idx['Residence 1 Year Ago','Different state'],idx["Residence 1 Year Ago","Abroad"],idx[['Sex and Age', 'Total population']]]]
    for column in data_R:
        data_R[column] = data_R[column]/data_R['Sex and Age', 'Total population']
    data_R['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_R['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_R = data_R.iloc[:,0:-2]

    data_Commuting = df.loc[:, [idx['Commuting to Work','Car, truck, or van -- drove alone'],idx['Commuting to Work','Car, truck, or van -- carpooled'],idx['Commuting to Work','Public transportation (excluding taxicab)'],idx['Commuting to Work','Walked'],idx["Commuting to Work","Worked from home"],idx['Sex and Age', 'Total population']]]
    for column in data_Commuting:
        data_Commuting[column] = data_Commuting[column]/data_Commuting['Sex and Age', 'Total population']
    data_Commuting['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_Commuting['Commuting to Work','Mean travel time to work (minutes)'] = df.loc[:,idx['Commuting to Work','Mean travel time to work (minutes)']]
    data_Commuting['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_Commuting = data_Commuting.iloc[:,0:-1]
    data_Commuting = data_Commuting.drop(data_Commuting.columns[-2] ,axis = 1)

    data_Occupation = df.loc[:, [idx['Occupation','Civilian employed population 16 years and over'],idx['Occupation','Management, business, science, and arts occupations'],idx['Occupation','Service occupations'],idx['Occupation','Sales and office occupations'],idx['Occupation','Natural resources, construction, and maintenance occupations'],idx["Occupation","Production, transportation, and material moving occupations"],idx['Sex and Age', 'Total population']]]
    for column in data_Occupation:
        data_Occupation[column] = data_Occupation[column]/data_Occupation['Sex and Age', 'Total population']
    data_Occupation['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_Occupation['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_Occupation = data_Occupation.iloc[:,0:-2]

    data_Industry = df.loc[:, 'Industry']
    data_Industry['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    for column in data_Industry:
        data_Industry[column] = data_Industry[column]/data_Industry['Sex and Age', 'Total population']
    data_Industry['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_Industry['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_Industry = data_Industry.iloc[:,0:-2]

    data_Worker = df.loc[:, 'Class of Worker']
    data_Worker['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    for column in data_Worker:
        data_Worker[column] = data_Worker[column]/data_Worker['Sex and Age', 'Total population']
    data_Worker['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_Worker['Commuting to Work','Mean travel time to work (minutes)'] = df.loc[:,idx['Commuting to Work','Mean travel time to work (minutes)']]
    data_Worker['Employment Status','Unemployment Rate'] = df.loc[:,idx['Employment Status','Unemployment Rate']]
    data_Worker['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_Worker = data_Worker.iloc[:,0:-1]
    data_Worker = data_Worker.drop(data_Worker.columns[-3],axis = 1)

    data_Income = df.loc[:, 'Income and Benefits (In 2019 inflation-adjusted dollars)']
    data_Income['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    for column in data_Income:
        data_Income[column] = data_Income[column]/data_Income['Sex and Age', 'Total population']
    data_Income['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_Income['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_Worker = data_Worker.iloc[:,0:-2]

    data_Health = df.loc[:, 'Health Insurance Coverage']
    data_Health['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    for column in data_Health:
        data_Health[column] = data_Health[column]/data_Health['Sex and Age', 'Total population']
    data_Health['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_Health['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_Worker = data_Worker.iloc[:,0:-2]

    data_Education = df.loc[:, 'Educational Attainment']
    data_Education['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    for column in data_Education.columns[0:-3]:
        data_Education[column] = data_Education[column]/data_Education['Sex and Age', 'Total population']
    data_Education['Sex and Age', 'Total population'] = df.loc[:,idx['Sex and Age', 'Total population']]
    data_Education['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_Education = data_Education.iloc[:,0:-2]

    data_Race = df.loc[:, 'Race']
    for column in data_Race:
        data_Race[column] = data_Race[column]/data_Race['Total population']
    data_Race['Value','Median (dollars)'] = df.loc[:,idx['Value','Median (dollars)']]
    data_Race = data_Race.iloc[:,1:]

    list_of_train =pd.concat([data_age,data_BirthPlace,data_R,data_Commuting,data_Occupation,data_Industry,data_Worker,data_Income,data_Health,data_Education,data_Race], axis=1)
    list_of_train = list_of_train*100
    list_of_train = list_of_train.iloc[:,0:-1]
    list_of_train.drop(columns=['Bachelor\'s degree', 'Graduate or professional degree', 'Mean household income (dollars)', 'Private wage and salary workers', 'Civilian employed population 16 years and over', 'Professional, scientific, and management, and administrative and waste management services'], inplace=True)
    list_of_train['value_price'] = df.loc[:,idx['Value','Median (dollars)']]
    list_of_train.drop(columns=[('Place of Birth', 'State of residence')], inplace=True)

    corrmat = list_of_train.corr()
    abs_c = np.absolute(corrmat)

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
    data['Commuting to Work, Walked from home'] = list_of_train.loc[:,[('Commuting to Work', 'Walked')]]
    data['Population Age 45-54'] = list_of_train.loc[:,[('Sex and Age','45 to 54 years')]]
    corrmat = data.corr()
    fig, ax = plt.subplots(figsize = (1, 5))
    corrmat.drop(index=[('Value', 'Median (dollars)')], inplace=True)
    corrmat.drop(index=['value_price'], inplace=True)
    corrmat.rename(index={'Information':"('Occupation', 'Information')", 'With public coverage':"Public health coverage", "$150,000 to $199,000":"Household income, $150,000-$199,000", "$200,000 or more":"Household income, $200,000 or more"}, inplace=True)
    sns.heatmap(corrmat.loc[:,[('Value', 'Median (dollars)')]], annot = True, annot_kws={'size': 12})
