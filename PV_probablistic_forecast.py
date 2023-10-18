import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


class PV_probablistic_forecasting():

# Generate some example data

 X = pd.read_excel('Solar Data.xlsx', sheet_name= 'SolarPV_X')
 print(X)
 y = pd.read_excel('Solar Data.xlsx', sheet_name= 'SolarPV_y')
 print(y)


# Import necessary libraries


# Generate synthetic data for binary classification
# Replace this with your own data


# Train a Gaussian Naive Bayes classifier
 model = GaussianNB()

 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 model.fit(X_train, y_train)

# Predict class probabilities on the test set
 y_probabilities = model.predict_proba(X_test)
 y_pred = model.predict(X_test)
# Evaluate the model's performance
 print(classification_report(y_test, y_pred))

 print(model.predict(X_test))
 print(y_test)

 #print(y_probabilities)

# Now, y_pred is the mean prediction, and sigma is the standard deviation, providing a measure of uncertainty.

# You can plot the results to visualize the probabilistic forecast.