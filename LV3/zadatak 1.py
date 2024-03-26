import pandas as pd
import numpy as np

# a)

data=pd.read_csv('data_C02_emission.csv')

print(f"Data frame sadrži {len(data)} mjerenja")

for i in data.columns:
    print(f"{i} je tipa {data[i].dtype}")

print(f"Broj izostalih vrijednosti: {data.isnull().sum().sum()}")
print(f"Broj dupliciranih vrijednosti: {data.duplicated().sum()}")

data.dropna(axis = 0)
data.dropna(axis = 1)
data.drop_duplicates()
data = data.reset_index ( drop = True )

print (data)
print (data.info())
print (data.describe())

# b)

most = data.nlargest(3, 'Fuel Consumption City (L/100km)')
least = data.nsmallest(3, 'Fuel Consumption City (L/100km)')


print("Najveća potrošnja: ")
print(most[['Make', 'Model', 'Fuel Consumption City (L/100km)' ]])
print("Najmanja potrošnja: ")
print(least[['Make', 'Model', 'Fuel Consumption City (L/100km)' ]])

# c)

engineCount = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
count = len(engineCount['Make'])
print(f"Vozila koja imaju veličinu motora između 2.5 i 3.5 L: {count}")
print(f"Njihova prosječna CO2 emisija plinova je {round(engineCount['CO2 Emissions (g/km)'].mean(),2)} g/km")

# d)

audiCound = data[data['Make'] == 'Audi']
print(f'{len(audiCound)} mjerenja se odnosi na vozila proizvđača Audi.')

gasEmission = audiCound[audiCound['Cylinders'] == 4]
print(f'Prosječna emisija CO2 plinova automobila proizvođača Audi koji imaju 4 cilindra iznosi {round(gasEmission['CO2 Emissions (g/km)'].mean(),2)} g/km')

# e)

cylinderCound = data['Cylinders'].value_counts().sort_index()
print(cylinderCound)

cylinderCound = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().round(2)
print("Cylinder emissions: ")
print(cylinderCound)

# f)

diesel = data[data['Fuel Type'] == 'D']
petrol = data[data['Fuel Type'] == 'Z']

print(f"Dizel - Prosječna: {round(diesel['Fuel Consumption City (L/100km)'].mean(),2)} - Medijalna vrijednost: {diesel['Fuel Consumption City (L/100km)'].median()}")
print(f"Benzin - Prosječna: {round(petrol['Fuel Consumption City (L/100km)'].mean(),2)} - Medijalna vrijednost: {petrol['Fuel Consumption City (L/100km)'].median()}")

# g)

fourCylinders = diesel[diesel['Cylinders'] == 4]
print(f"Dizel s 4 cilindra koji ima najveću gradsku potrošnju:\n {fourCylinders.nlargest(1, 'Fuel Consumption City (L/100km)')}")

# h)

manual = data[(data['Transmission'].str[0] == 'M')]
print(f"Postoji {len(manual['Make'])} vozila s rucnim mjenjacem")

# i)

print(data.corr(numeric_only=True))

# Komentar za zadnji zadatak
# Dobivamo visoke pozitivne i negativne korelacije
# Primjer za Engine Size i Cylinders : veći motori imaju više cilindara
# Primjer za CO2 Emissions i Fuel Consumption Comb (mpg) : što vozilo ima manji broj u ovom slučaju milja po galonu to mu je emisija CO2 veća