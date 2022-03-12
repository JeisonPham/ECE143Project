import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def county_data(rid,fip):



	df1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'rentals.csv'))

	df2= pd.read_csv(os.path.join(os.path.dirname(__file__), 'housing.csv'))

	df_new_1 = df1[df1.RegionID==rid]

	df_new_2 = df2[df2.RegionID==rid]


	y1=df_new_1.iloc[:,3:-1]


	y2=df_new_2.iloc[:,173:-1]



	x1= list(y1)

	#flat_list = [item for sublist in x for item in sublist]

	z1= y1.values.tolist()

	flat_list_1= [item for sublist in z1 for item in sublist]



	x2= list(y2)

	#flat_list = [item for sublist in x for item in sublist]

	z2= y2.values.tolist()

	flat_list_2 = [item for sublist in z2 for item in sublist]

	rent = pd.DataFrame({'date':x1, 'rent':flat_list_1})

	house=pd.DataFrame({'date':x2, 'house_price':flat_list_2})


	rent.date = pd.to_datetime(rent.date)

	rent=rent.groupby(rent['date'].dt.year)['rent'].agg(['mean','sum'])


	rent=rent.reset_index()

	rent=rent.rename(columns={'mean':'mean_rent'})

	rent['percent_change_rent']= 100*(rent['mean_rent'].pct_change())

	#print(rent)



	house.date = pd.to_datetime(house.date)

	house=house.groupby(house['date'].dt.year)['house_price'].agg(['mean'])


	house=house.reset_index()

	house=house.rename(columns={'mean':'mean_house_price'})
	house['percent_change_price']= 100*house['mean_house_price'].pct_change()
	#print(house)


	house['price_to_rent_ratio']=house['mean_house_price'].div(rent['sum'])
	house['percent_change']= 100*house['price_to_rent_ratio'].pct_change()



	df_3 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'new_unemployment.csv'))


	df_new_3 = df_3[df_3.FIPS_Code==fip]




	x3= list(df_new_3.iloc[:,3:])

	#flat_list = [item for sublist in x for item in sublist]

	z3= df_new_3.iloc[:,3:].values.tolist()

	flat_list_3= [item for sublist in z3 for item in sublist]

	unemployment = pd.DataFrame({'date':x3, 'unemployment_rate':flat_list_3})

	#print(unemployment)



	#print("house",house)

	return(rent, house, unemployment)




def main():

    county1= "San Diego, CA"

    county2= "San Francisco, CA"

    county3= "Denver, CO"

    county4= "Boston, MA"

    county5= "Wichita, KS"


    rent1, house1, unemployment1= county_data(395056,6073)
    rent2, house2, unemployment2= county_data(395057,6075)
    rent3, house3, unemployment3= county_data(394530,8031)
    rent4, house4, unemployment4= county_data(394404,25025)
    rent5, house5, unemployment5= county_data(395224,20203)



    sns.set_theme() 


    fig1, axes1= plt.subplots(2,1)
    fig1.suptitle("Rent and House price change over years")

    z1=pd.concat({
        county1: rent1.set_index('date').percent_change_rent, county2: rent2.set_index('date').percent_change_rent, county3: rent3.set_index('date').percent_change_rent,
        county4: rent4.set_index('date').percent_change_rent, county5: rent5.set_index('date').percent_change_rent 
    }, axis=1).plot.bar(ax=axes1[0])

    z1.set_xlabel("Year")
    z1.set_ylabel("Percent change of Rent")

    z2=pd.concat({
        county1: house1.set_index('date').percent_change_price, county2: house2.set_index('date').percent_change_price,county3: house3.set_index('date').percent_change_price,
        county4: house4.set_index('date').percent_change_price, county5: house5.set_index('date').percent_change_price
    }, axis=1).plot.bar(ax=axes1[1])

    z2.set_xlabel("Year")
    z2.set_ylabel("Percent change of House Price")


    z1.set_xticklabels(z1.get_xticklabels(), rotation=0, ha='right')
    z2.set_xticklabels(z2.get_xticklabels(), rotation=0, ha='right')


    fig2, axes2= plt.subplots(2,1)
    fig2.suptitle("Rent vs House Price")


    z1=pd.concat({
        county1: rent1.set_index('date').mean_rent, county2: rent2.set_index('date').mean_rent, county3: rent3.set_index('date').mean_rent, 
        county4: rent4.set_index('date').mean_rent, county5: rent5.set_index('date').mean_rent
    }, axis=1).plot.bar(ax=axes2[0])


    z1.set_xlabel("Year")
    z1.set_ylabel("Mean Rent")

    z2=pd.concat({
        county1: house1.set_index('date').mean_house_price, county2: house2.set_index('date').mean_house_price, county3: house3.set_index('date').mean_house_price,
        county4: house4.set_index('date').mean_house_price,county5: house5.set_index('date').mean_house_price
    }, axis=1).plot.bar(ax=axes2[1])

    z2.set_xlabel("Year")
    z2.set_ylabel("Mean House Price")



    z1.set_xticklabels(z1.get_xticklabels(), rotation=0, ha='right')
    z2.set_xticklabels(z2.get_xticklabels(), rotation=0, ha='right')


    fig3, axes3= plt.subplots()
    fig3.suptitle("Price to Annual Rent ratio ")
    z1=pd.concat({
        county1: house1.set_index('date').price_to_rent_ratio, county2: house2.set_index('date').price_to_rent_ratio, county3: house3.set_index('date').price_to_rent_ratio,
        county4: house4.set_index('date').price_to_rent_ratio, county5: house5.set_index('date').price_to_rent_ratio
    }, axis=1).plot.bar(ax=axes3)

    z1.set_ylabel("Price to rent ratio")




    z1.set_xticklabels(z1.get_xticklabels(), rotation=0, ha='right')

    z1=pd.concat({
        county1: unemployment1.set_index('date').unemployment_rate, county2: unemployment2.set_index('date').unemployment_rate,
        county3: unemployment3.set_index('date').unemployment_rate, county4: unemployment4.set_index('date').unemployment_rate,
        county5: unemployment5.set_index('date').unemployment_rate
    }, axis=1).plot.bar()

    z1.set_ylabel("Unemployment rate")
    z1.set_xticklabels(z1.get_xticklabels(), rotation=0, ha='right')



    plt.show()
