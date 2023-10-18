import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


class EM_Price_point_forecast():

# Generate some example data

 df = pd.read_csv('Electricity_Market_Price.csv')
 y = df['EM_Price'].to_numpy()
 y = y.reshape(-1, 1)

 X = np.arange(0.0, 1440.0, 1.0)
 X = X.reshape(-1, 1)

 regressor = RandomForestRegressor(n_estimators=100, random_state=42)

 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


 print(y_train)



 print(y_test)
 regressor.fit(X_train, y_train)
 y_pred = regressor.predict(X_test)
 y_pred = y_pred.reshape(-1, 1)
 print(y_pred - y_test)



 mse = mean_squared_error(y_test, y_pred)
 mae = mean_absolute_error(y_test, y_pred)

 print("Mean Squared Error:", mse)
 print("Mean Absolute Error", mae)

# Import necessary libraries


# Generate synthetic data for binary classification
# Replace this with your own data


#
 #print(y_probabilities)

# Now, y_pred is the mean prediction, and sigma is the standard deviation, providing a measure of uncertainty.

# You can plot the results to visualize the probabilistic forecast.