import numpy as np
import cvxpy as cp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(42)

# ---------------------------------------------------------
# 1. Load the dataset and select 4000 of digits 0 and 1 only
# ---------------------------------------------------------
data = np.load("data/mnist.npz")
X, y = data["X"], data["y"]
mask = (y == 0) | (y == 1)
X, y = X[mask], y[mask]
total_samples = 4000
indices = np.random.choice(len(y), total_samples, replace=False)
X, y = X[indices], y[indices]
y = 2*y - 1 # change binary labels from 0/1 to -1/+1
print("Number of -1s and 1s:", np.sum(y == -1), np.sum(y == 1))

# ---------------------------------------------------------
# 2. Standardize features and split data
# ---------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, shuffle=True)
m, n = X_train.shape
percent_adversarial = 0.01
num_adversarial = int(percent_adversarial * m)
y_test = (y_test + 1) // 2  # convert back to 0/1 labels

d = n + 1
X_train_augmented = np.hstack((np.ones((m, 1)), X_train))

# ---------------------------------------------------------
#       Standard logistic regression on original data
# ---------------------------------------------------------
theta = cp.Variable(d)
log_likelihood = cp.sum(cp.logistic(-cp.multiply(y_train, X_train_augmented @ theta)))
prob = cp.Problem(cp.Minimize(log_likelihood))
print("Solving standard logistic regression on original data...")
prob.solve(verbose=False, solver=cp.MOSEK)
logits_test = X_test @ theta.value[1:] + theta.value[0]
y_pred = (logits_test >= 0).astype(int)
acc = accuracy_score(y_test, y_pred)
print("\nTest accuracy standard logistic regression:", acc)

# ---------------------------------------------------------
#           create adversarial examples
# --------------------------------------------------------- 
logits_train = X_train_augmented @ theta.value
probs = 1 / (1 + np.exp(-logits_train))
top_indices = np.argsort(probs)[-int(num_adversarial):][::-1]
y_train_adv = y_train.copy()
for idx in top_indices:
    y_train_adv[idx] = -y_train_adv[idx] 


# ---------------------------------------------------------
#    Standard logistic regression on adversarial examples
# ---------------------------------------------------------
log_likelihood_adv = cp.sum(cp.logistic(-cp.multiply(y_train_adv, X_train_augmented @ theta)))
prob_adv = cp.Problem(cp.Minimize(log_likelihood_adv))
print("Solving standard logistic regression on adversarially flipped data...")
prob_adv.solve(verbose=False, solver=cp.MOSEK)    
logits = X_test @ theta.value[1:] + theta.value[0]
y_pred = (logits >= 0).astype(int)
acc_adv = accuracy_score(y_test, y_pred)
print("\nTest accuracy after adversarial label flipping:", acc_adv)

# ----------------------------------------------------------
# Trimmed logistic regression on adversarial examples
# ----------------------------------------------------------
theta.value = None
weights = cp.Variable(m, bounds=[0, 1])
obj = cp.sum(cp.multiply(weights, cp.logistic(-cp.multiply(y_train_adv, X_train_augmented @ theta))))
constraints = [cp.sum(weights) == 0.95 * m]  # keep 95% of data
objective = cp.Minimize(obj)
prob = cp.Problem(objective, constraints)

print("Solving trimmed logistic regression on adversarially flipped data...")
# knitro takes two minutes, ipopt needs many more iterations and takes about an hour
prob.solve(nlp=True, solver=cp.KNITRO, verbose=True, algorithm=0)
#prob.solve(nlp=True, solver=cp.IPOPT, verbose=True, least_square_init_duals='no')    
# predictions on original test set
logits = X_test @ theta.value[1:] + theta.value[0]
y_pred = (logits >= 0).astype(int)
acc_trimmed = accuracy_score(y_test, y_pred)
print("\nTest accuracy with trimming after adversarial label flipping:", acc_trimmed)

print("weights of trimmed data points:", weights[top_indices].value)