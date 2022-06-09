
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("FuelConsumption.csv")


#using required features
cdf = df[['AGE','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


#Training Data and Predictor Variable

x = cdf.iloc[:, :4]
y = cdf.iloc[:, -1]


regressor = LinearRegression()


regressor.fit(x, y)


pickle.dump(regressor, open('model.pkl','wb'))

