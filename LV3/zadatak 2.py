import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

# a)

plt.figure()
data['CO2 Emissions (g/km)'].plot(kind="hist", bins=40, title = "Histogram")
plt.show()

# b)

data['Fuel Type'] = data['Fuel Type'].astype('category')
dotColor = {'Z': 'red', 'X': 'blue', 'E': 'green', 'D': 'black'}
data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c=data["Fuel Type"].map(dotColor), s=30)
plt.title('Dijagram raspršenja')
plt.show()

# c) 

data.boxplot(column='CO2 Emissions (g/km)', by='Fuel Type')
plt.title('Kutijasti dijagram za broj vozila po tipu goriva')
plt.show()

# d)

fuel = data.groupby('Fuel Type').size()
fuel.plot(kind ='bar', xlabel='Fuel Type', ylabel='Number of vehicles', title='Stupčati dijagram - broj vozila po tipu goriva')
plt.show()

# e)

cylinder = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
cylinder.plot(kind='bar', x=cylinder.index, y=cylinder.values, xlabel='Cylinders', ylabel='CO2 emissions (g/km)', title='Stupčasti dijagram - prosječna CO2 emisija vozila s obzirom na broj cilindara')
plt.show()
