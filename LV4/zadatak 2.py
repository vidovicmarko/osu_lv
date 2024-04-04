import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import max_error
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

input = ['Engine Size (L)','Fuel Type','Cylinders','Fuel Consumption City (L/100km)']
output = 'CO2 Emissions (g/km)'

X = data[input]
y = data[output]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1)

encoder = OneHotEncoder()
X_encoded_train = encoder.fit_transform(X_train[['Fuel Type']]).toarray()
X_encoded_test = encoder.fit_transform(X_test[['Fuel Type']]).toarray()

linModel = lm.LinearRegression()
linModel.fit(X_encoded_train, y_train)
y_test_p = linModel.predict(X_encoded_test)

ME = max_error(y_test, y_test_p)
print(f"Max Error: {ME}")

error = np.abs(y_test_p, y_test)
print(round(np.max(error),2))

max_error_index = np.argmax(error)
max_error_model = data.iloc[max_error_index, 1]
print(f"Model s najvećom pogrškom: {max_error_model}")

#Maksimalna pogrška iznosi 292.84, radi se o modelu vozila S5 Cabriolet