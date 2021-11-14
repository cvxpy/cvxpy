import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp


def loss_fn(X, Y, beta):
    return cp.norm2(cp.matmul(X, beta) - Y)**2


def regularizer(beta):
    return cp.norm1(beta)


def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


def generate_data(m: int = 100, n: int = 20, sigma: int = 5, density: float = 0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(m,n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y, beta_star


def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
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
    m = 100
    n = 20
    sigma = 5
    density = 0.2

    X, Y, _ = generate_data(m, n, sigma)
    X_train = X[:50, :]
    Y_train = Y[:50]
    X_test = X[50:, :]
    Y_test = Y[50:]


    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(
                    cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

    lambd_values = np.logspace(-2, 3, 50)
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
