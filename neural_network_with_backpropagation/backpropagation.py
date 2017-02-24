import pandas as pd
import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
	return x * (1 - x)

def prepare_data():

	# Read csv file
	admissions = pd.read_csv('data.csv')

	# Create dummy columns for rank
	data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
	data = data.drop('rank', axis=1)

	# Standardize the data
	for field in ['gre', 'gpa']:
		mean, std = data[field].mean(), data[field].std()
		data.loc[:, field] = (data[field] - mean)/std

	# Split off 10% of the data for testing
	np.random.seed(42)
	sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
	data, test_data = data.ix[sample], data.drop(sample)

	# Split data into features and target
	features, targets = data.drop('admit', axis=1), data['admit']
	features_test, target_test = test_data.drop('admit', axis=1), test_data['admit']

	return features, targets, features_test, target_test


def backpropagation():
	features, targets, features_test, target_test = prepare_data()

	# Setting the hyper parameters
	n_hidden = 3
	epochs = 500
	learnrate = 0.1
	last_loss = 0

	n_records, n_features = features.shape
	
	# Setting the weights
	weights_input_to_hidden = np.random.normal(scale = 1 / n_features ** 0.5, size=(n_features, n_hidden))
	weights_hidden_to_output = np.random.normal(scale = 1 / n_features ** 0.5, size=n_hidden)

	for e in range(epochs):
		del_w_input_hidden = np.zeros(weights_input_to_hidden.shape)
		del_w_hidden_output = np.zeros(weights_hidden_to_output.shape)

		for x, y in zip(features.values, targets):
			# Feed forward neural network
			hidden_input = np.dot(x, weights_input_to_hidden)
			hidden_activations = sigmoid(hidden_input)

			input_to_output_layer = np.dot(hidden_activations, weights_hidden_to_output)
			final_output = sigmoid(input_to_output_layer)

			# Calculate errors
			error_output_layer = (y - final_output) * final_output * (1 - final_output)

			error_hidden_layer = np.dot(error_output_layer, weights_hidden_to_output) * hidden_activations * (1 - hidden_activations)

			# Gradient_descent_step, update the change in weights
			del_w_hidden_output += error_output_layer * hidden_activations
			del_w_input_hidden += error_hidden_layer * x[:, None]

		# Update the weights
		weights_input_to_hidden += learnrate * del_w_input_hidden / n_records
		weights_hidden_to_output += learnrate * del_w_hidden_output / n_records

		if (e % 10 == 0):
			hidden_activations = sigmoid(np.dot(x, weights_input_to_hidden))
			output = sigmoid(np.dot(hidden_activations, weights_hidden_to_output))

			loss = np.mean((output - targets) ** 2)

			if last_loss and last_loss < loss:
				print("Train loss: ", loss, "  WARNING - Loss Increasing")
			else:
				print ("Train loss:", loss)
			last_loss = loss

	# Test and find the accuracy on testing data
	hidden_activations = sigmoid(np.dot(features_test, weights_input_to_hidden))
	output = sigmoid(np.dot(hidden_activations, weights_hidden_to_output))
	predictions = output > 0.5
	accuracy = np.mean(predictions == target_test)
	print("Prediction accuracy: {:.3f}".format(accuracy))


backpropagation()
