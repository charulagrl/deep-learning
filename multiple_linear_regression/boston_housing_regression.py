'''
	Program to predict Boston housing prices
	Using multiple linear regression to train the model
	The model will not be a straight line. It will be a curve 
	y = m1x1 + m2x2 + m3x3 + ... + mnxn + b

'''

from sklearn import linear_model
# The dataset consists of 13 features of 506 houses
from sklearn.datasets import load_boston

# read data
boston_data = load_boston()
x = boston_data['data']
y = boston_data['target']

# train the model 
linear_reg = linear_model.LinearRegression()
linear_reg.fit(x, y)

# sample data for prediction
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]

# predict the outcome for sample data
prediction = linear_reg.predict(sample_house)
print prediction

