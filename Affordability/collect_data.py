import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def county_data(rid,fip):



	df1 = pd.read_csv('rentals.csv')

	df2= pd.read_csv('housing.csv')

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



	df_3 = pd.read_csv('new_unemployment.csv')


	df_new_3 = df_3[df_3.FIPS_Code==fip]




	x3= list(df_new_3.iloc[:,3:])

	#flat_list = [item for sublist in x for item in sublist]

	z3= df_new_3.iloc[:,3:].values.tolist()

	flat_list_3= [item for sublist in z3 for item in sublist]

	unemployment = pd.DataFrame({'date':x3, 'unemployment_rate':flat_list_3})

	#print(unemployment)



	#print("house",house)

	return(rent, house, unemployment)





county1= "Los Angeles, CA"

county2= "Ventura, CA"

county3= "St. Louis, MO"

county4= "New York, NY"



rent1, house1, unemployment1= county_data(753899,6037)
rent2, house2, unemployment2= county_data(394952,6111)
rent3, house3, unemployment3= county_data(395121,27137)
rent4, house4, unemployment4= county_data(394913,36061)


print(county1,"\n",rent1,"\n", house1,"\n", unemployment1 )
print(county2,"\n",rent2,"\n", house2,"\n", unemployment2 )
print(county3,"\n",rent3,"\n", house3,"\n", unemployment3 )
print(county4,"\n",rent4,"\n", house4,"\n", unemployment4 )

