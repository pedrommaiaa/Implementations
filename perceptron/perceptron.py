import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.activation_func = self.step_func
        self.weights = None
        self.bias = None

    def step_func(self, x):
        return np.where(x>=0,1,0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iter):

            for index, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[index] - y_predicted)

                self.weights += update * x_i
                self.bias += update



if __name__ == "__main__":
   
    from sklearn.model_selection import train_test_split
    from sklearn import datasets


    def accuracy(y_true, y_pred):
        accuracy = (np.sum(y_true == y_pred) / len(y_true))*100
        return accuracy


    X,y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)

    p = Perceptron(lr=0.01, n_iter=1000)

    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print(f"Perceptron classification accuracy {accuracy(y_test, predictions)}%")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
 
    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()
