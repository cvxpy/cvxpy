import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp


def loss_fn(X, Y, beta):
    return cp.pnorm(cp.matmul(X, beta) - Y, p=2)**2


def regularizer(beta):
    return cp.pnorm(beta, p=2)**2


def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


def generate_data(m: int = 1000, n: int = 30, sigma: int = 40):
    """Generates data for regression.

    To experiment with your own data, just replace the contents of this
    function with code that loads your dataset.

    Args
    ----
    m : int
        The number of examples.
    n : int
        The number of features per example.
    sigma : positive float
        The standard deviation of the additive noise.

    Returns
    -------
    X : np.array
        An array of featurized examples, shape (m, n), m the number of
        examples and n the number of features per example.

    Y : np.array
        An array of shape (m,) containing the observed labels for the
        examples.

    beta_star : np.array
        The true parameter. This is the quantity we are trying to
        estimate.
    """
    beta_star = np.random.randn(n)
    # Generate an ill-conditioned data matrix
    X = np.random.randn(m, n)
    U, _, V = np.linalg.svd(X)
    s = np.linspace(30, 1, min(m, n))
    S = np.zeros((m, n))
    S[:min(m, n), :min(m, n)] = np.diag(s)
    X = np.dot(U, np.dot(S, V))
    # Corrupt the observations with additive Gaussian noise
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y, beta_star


def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (mSE)")
    plt.show()


def plot_regularization_path(lambd_values, beta_values):
    num_coeffs = len(beta_values[0])
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()


if __name__ == "__main__":
    m = 1000
    n = 30
    sigma = 4

    X, Y, beta_star = generate_data(m, n, sigma)
    X_train = X[:800, :]
    Y_train = Y[:800]
    X_test = X[800:, :]
    Y_test = Y[800:]

    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(
                    cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

    lambd_values = np.logspace(-2, 2, 50)
    train_errors = []
    test_errors = []
    beta_values = []
    for v in lambd_values:
        lambd.value = v
        problem.solve()
        train_errors.append(mse(X_train, Y_train, beta))
        test_errors.append(mse(X_test, Y_test, beta))
        beta_values.append(beta.value)

    plot_train_test_errors(train_errors, test_errors, lambd_values)
    plot_regularization_path(lambd_values, beta_values)
