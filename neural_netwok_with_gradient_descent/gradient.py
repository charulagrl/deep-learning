import numpy as np
import pandas as pd

# Calculate sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Calculate derivative of sigmoid function
def sigmoid_prime(x):
	return x * (1 - x)

def prepare_data():

	admissions = pd.read_csv('data.csv')

	# Make dummy variable for rank
	data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
	data = data.drop('rank', axis=1)

	# Standardize data
	for field in ['gre', 'gpa']:
		mean, std = data[field].mean(), data[field].std()
		data.loc[:, field] = (data[field] - mean)/std

	# Split off random 10% of the data for testing
	np.random.seed(42)
	sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
	data, test_data = data.ix[sample], data.drop(sample)

	# Split data into features and target data
	features, targets = data.drop('admit', axis=1), data['admit']
	features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

	return features, targets, features_test, targets_test

# Function that calculates the gradient descent
def gardient_descent():

	# fetch the data from prepare_data function
	features, targets, features_test, targets_test = prepare_data()

	n_records, n_features = features.shape

	# Use the same seed to make the debugging easier
	np.random.seed(42)
	# Initialize the weights, epochs and learning rate
	weights = np.random.normal(scale=1 / n_features**0.5, size=n_features)
	epochs = 1000
	learnrate = 0.2


	for e in range(epochs):
		del_w = np.zeros(weights.shape)
		# Loop through all the records with x as input and y as target variable
		for x, y in zip(features.values, targets):
			# Calculate the output
			output = sigmoid(np.dot(weights, x))

			# calculate the error
			error = y - output

			# Calculate change in weights i.e. partial derivative of error with respect to w
			del_w  += error * sigmoid_prime(np.dot(weights, x)) * x

		# Update weights
		weights += learnrate * del_w / n_records

		last_loss = None
		# Printing the mean squared error on the tarining set
		if e % (epochs / 10) == 0:
			output = sigmoid(np.dot(features, weights))
			loss = np.mean((output - targets) ** 2)

			if last_loss and last_loss < loss:
				print("Train loss: ", loss, "  WARNING - Loss Increasing")
			else:
				print("Train loss: ", loss)
			last_loss = loss

	tes_out = sigmoid(np.dot(features_test, weights))
	predictions = tes_out > 0.5
	accuracy = np.mean(predictions == targets_test)
	print("Prediction accuracy: {:.3f}".format(accuracy))

if __name__ == "__main__":
	prepare_data()
	gardient_descent()


