import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


# a)
data = pd.read_csv('data_C02_emission.csv')

input = ['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)']
output = 'CO2 Emissions (g/km)'

X = data[input]
y = data[output]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1)

# b)

plt.scatter(X_train['Cylinders'], y_train, c='Red')
plt.scatter(X_test['Cylinders'], y_test, c='Blue')

plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('CO2 emisije u usporedbi sa brojem cilindara')
plt.show()

# c)

plt.hist(X_train['Engine Size (L)'])
plt.show()

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
plt.hist(X_train_n[:, 0])
plt.title("Dijagram nakon skaliranja")
plt.show()

X_test_n = sc.transform(X_test)

# d)

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)

# e)

y_test_p = linearModel.predict(X_test_n)
plt.scatter(y_test, y_test_p)
plt.title("Odnos između stvarnih vrijednosti izlazne veličine i procjene dobivene modelom")
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Procjenjene vrijednosti")
plt.show()

# f)

MAE = mean_absolute_error(y_test , y_test_p)
MSE = mean_squared_error(y_test , y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
RMSE = mean_squared_error(y_test, y_test_p, squared=False)
R_TWO = r2_score(y_test, y_test_p)

print(f"MAE: {MAE}, MSE: {MSE}, MAPE: {MAPE}, RMSE: {RMSE}, R2: {R_TWO}")

