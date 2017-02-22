'''
	Program to improve the error in a linear regression model by gradient descent.
	Using gradient descent we can slowly and iteratively we can minimize errors.
'''

from numpy import *

# Calculating mean squared error
def compute_error(b, m, points):
	total_error = 0

	# Error will be sum of square of error for each point
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]


		total_error += (y - (m * x + b)) ** 2

	return total_error / float(len(points))

# Function that calculates the gradient at each step
def step_gradient(b, m, learning_rate, points):
	b_gradient = 0
	m_gradient = 0

	N = float(len(points))
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]

		# Calculate the gradient with respect to b amd m
		b_gradient += - (2 / N) * (y - (m * x + b))
		m_gradient += - (2 / N) * x * (y - (m * x + b))

	new_b = b - (learning_rate * b_gradient)
	new_m = m - (learning_rate * m_gradient)

	return [new_b, new_m]


# Run gradient descent for n number of iterations
def gradient_descent_runner(num_iterations, starting_m, starting_b, learning_rate, points):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		b, m = step_gradient(b, m, learning_rate, points)

	return [b, m]



def run():
	# read data
	points = genfromtxt("data.csv", delimiter=",")

	# Initialize the values for hyper parameters
	learning_rate = 0.0001
	initial_b = 0
	initial_m = 0
	num_iterations = 1000

	print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, \
		compute_error(initial_b, initial_m, points))
	
	print "Running...."

	[b, m] = gradient_descent_runner(num_iterations, initial_m, initial_b, learning_rate, points)
	
	print "After {0}  iterations b = {1}, m = {2}, error ={3}".format(num_iterations, b, m, compute_error(b, m, points))

# Run the main function
if __name__ == "__main__":
	run()