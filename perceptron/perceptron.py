import numpy as np

class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros((num_features, 1), dtype=np.float64)
        self.bias = np.zeros(1, dtype=np.float64)

    def activation_function(self, linear):
        return np.where(linear > 0., 1, 0)

    def forward(self, x):
        linear = np.dot(x, self.weights) + self.bias
        predictions = self.activation_function(linear)
        return predictions

    def backward(self, x, y):
        predictions = self.forward(x)
        errors = y - predictions
        return errors

    def train(self, x, y, epochs):
        
        for _ in range(epochs):
            for i in range(y.shape[0]):
                errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
                self.weights += (errors*x[i]).reshape(self.num_features, 1)
                self.bias += errors

    def evaluate(self, x, y):
        predictions = self.forward(x).reshape(-1)
        accuracy = np.sum(predictions == y) / y.shape[0]
        return accuracy



if __name__ == "__main__":
    from data import X_train, y_train, X_test, y_test 

    perceptron = Perceptron(num_features=2)
    perceptron.train(X_train, y_train, epochs=5)

    train_acc = perceptron.evaluate(X_train, y_train)
    print(f"Train set accuracy: {(train_acc*100):.2f}%")

    test_acc = perceptron.evaluate(X_test, y_test)
    print(f"Test set accuracy: {(test_acc*100):.2f}%")
