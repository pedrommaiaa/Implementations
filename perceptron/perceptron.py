import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def random_weights(X, random_state: int):
    '''create vector of random weights
    Parameters
    ----------
    X: 2-dimensional array, shape = [n_samples, n_features]
    Returns
    -------
    w: array, shape = [w_bias + n_features]'''
    rand = np.random.RandomState(random_state)
    w = rand.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
    return w

def net_input(X, w):
    '''Compute net input as dot product'''
    return np.dot(X, w[1:]) + w[0]


def predict(X, w):
    '''Return class label after unit step'''
    return np.where(net_input(X, w) >= 0.0, 1, -1)


def fit(X, y, eta=0.001, n_iter=100):
    '''loop over exemplars and update weights'''
    errors = []
    w = random_weights(X, random_state=1)
    for exemplar in range(n_iter):
        error = 0
        for xi, target in zip(X, y):
            delta = eta * (target - predict(xi, w))
            w[1:] += delta * xi
            w[0] += delta
            error += int(delta != 0.0)
        errors.append(error)
    return w, errors

def species_generator(mu1, sigma1, mu2, sigma2, n_samples, target, seed):
    '''creates [n_samples, 2] array

    Parameters
    ----------
    mu1, sigma1: int, shape = [n_samples, 2]
        mean feature-1, standar-dev feature-1
    mu2, sigma2: int, shape = [n_samples, 2]
        mean feature-2, standar-dev feature-2
    n_samples: int, shape= [n_samples, 1]
        number of sample cases
    target: int, shape = [1]
        target value
    seed: int
        random seed for reproducibility

    Return
    ------
    X: ndim-array, shape = [n_samples, 2]
        matrix of feature vectors
    y: 1d-vector, shape = [n_samples, 1]
        target vector
    ------
    X'''
    rand = np.random.RandomState(seed)
    f1 = rand.normal(mu1, sigma1, n_samples)
    f2 = rand.normal(mu2, sigma2, n_samples)
    X = np.array([f1, f2])
    X = X.transpose()
    y = np.full((n_samples), target)
    return X, y


if __name__ == "__main__":

    albatross_weight_mean = 9000 # in grams
    albatross_weight_variance =  800 # in grams
    albatross_wingspan_mean = 300 # in cm
    albatross_wingspan_variance = 20 # in cm
    albatross_target = 1
    
    owl_weight_mean = 1000 # in grams
    owl_weight_variance =  200 # in grams
    owl_wingspan_mean = 100 # in cm
    owl_wingspan_variance = 15 # in cm
    owl_target = -1
    
    n_samples = 100
    seed = 100

    # aX: feature matrix (weight, wingspan)
    # ay: target value (1)
    aX, ay = species_generator(albatross_weight_mean, albatross_weight_variance,
                               albatross_wingspan_mean, albatross_wingspan_variance,
                               n_samples,albatross_target,seed )


    albatross_dic = {'weight-(gm)': aX[:,0],
                     'wingspan-(cm)': aX[:,1],
                     'species': ay
                    }

    # put values in a relational table (pandas dataframe)
    albatross_df = pd.DataFrame(albatross_dic)


    # oX: feature matrix (weight, wingspan)
    # oy: target value (1)
    oX, oy = species_generator(owl_weight_mean, owl_weight_variance,
                               owl_wingspan_mean, owl_wingspan_variance,
                               n_samples,owl_target,seed )

    owl_dic = {'weight-(gm)': oX[:,0],
                 'wingspan-(cm)': oX[:,1],
                 'species': oy
              }

    # put values in a relational table (pandas dataframe)
    owl_df = pd.DataFrame(owl_dic)

    df = albatross_df.append(owl_df, ignore_index=True)

    df_shuffle = df.sample(frac=1, random_state=1).reset_index(drop=True)
    X = df_shuffle[['weight-(gm)','wingspan-(cm)']].to_numpy()
    y = df_shuffle['species'].to_numpy()

    w, errors = fit(X, y, eta=0.01, n_iter=200)

    y_pred = predict(X, w)
    num_correct_predictions = (y_pred == y).sum()
    accuracy = (num_correct_predictions / y.shape[0]) * 100
    print(f"Perceptron accuracy: {accuracy:.2f}%")


    error_df = pd.DataFrame({'error':errors, 'time-step': np.arange(0, len(errors))})

    plt.plot(error_df['error'])
    plt.title("error-rate")
    plt.xlabel("time-step")
    plt.ylabel("error")
    plt.show()
