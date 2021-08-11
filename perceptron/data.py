import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('perceptron_data.txt', delimiter='\t')
X, y = data[:, :2], data[:, 2]
y = y.astype(int)


# Shuffling & train/test split
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y = X[shuffle_idx], y[shuffle_idx]

X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]


# Normalize (mean zero, unit variance)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma


# Plot Training set
#plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
#plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')
#plt.title('Training set')
#plt.xlabel('feature 1')
#plt.ylabel('feature 2')
#plt.xlim([-3, 3])
#plt.ylim([-3, 3])
#plt.legend()
#plt.show()


# Plot Test set
#plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='class 0', marker='o')
#plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='class 1', marker='s')
#plt.title('Test set')
#plt.xlabel('feature 1')
#plt.ylabel('feature 2')
#plt.xlim([-3, 3])
#plt.ylim([-3, 3])
#plt.legend()
#plt.show()
