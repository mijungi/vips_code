import numpy as np
import numpy.random as rn

mvnrnd = rn.multivariate_normal


# define sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def generate_data(d, N, alpha):
	# generate x and y
	mean = np.zeros(d)
	cov = np.eye(d)
	X = mvnrnd(mean, cov, N)
	normaliser = np.max(np.sqrt(np.sum(np.abs(X) ** 2, axis=-1)))
	X = X / normaliser

	# generate theta
	theta = 2 * mvnrnd(np.zeros(d), 1 / alpha * np.eye(d))
	Odds = sigmoid(np.dot(X, theta))

	y = 1 * (Odds > 0.5)

	return y, X, theta


def computeOdds(X, theta):
	return sigmoid(np.dot(X, theta))


def generate_testdata(Ntst, theta):
	d = np.shape(theta)
	d = d[0]

	# generate x and y
	mean = np.zeros(d)
	cov = np.eye(d)
	Xtst = mvnrnd(mean, cov, Ntst)
	normaliser = np.max(np.sqrt(np.sum(np.abs(Xtst) ** 2, axis=-1)))
	Xtst = Xtst / normaliser

	Odds = sigmoid(np.dot(Xtst, theta))

	ytst = 1 * (Odds > 0.5)

	return ytst, Xtst


def mean_squared_error(ytst, ypred, Ntst):
	return np.sqrt(np.sum((ytst - ypred) ** 2) / np.float(Ntst))


def generate_data_linear_regression(d, N, alpha):
	# generate x and y
	mean = np.zeros(d)
	cov = np.eye(d)
	X = mvnrnd(mean, cov, N)
	normaliser = np.max(np.sqrt(np.sum(np.abs(X) ** 2, axis=-1)))
	X = X / normaliser

	# generate theta
	theta = 2 * mvnrnd(np.zeros(d), 1 / alpha * np.eye(d))
	Xtheta = np.dot(X, theta)

	y = Xtheta + mvnrnd(0, 1, Xtheta.shape())

	return y, X, theta
