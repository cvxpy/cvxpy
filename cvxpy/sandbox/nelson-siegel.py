import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def nelson_siegel(tau, beta0, beta1, beta2, lambda_param):
    """
    Calculate Nelson-Siegel yield curve values.
    
    Parameters:
    tau: maturity times
    beta0, beta1, beta2: model parameters
    lambda_param: decay parameter
    """
    exp_term = cp.exp(-tau / lambda_param)
    factor1 = (1 - exp_term) / (tau / lambda_param)
    factor2 = factor1 - exp_term
    
    return beta0 + beta1 * factor1 + beta2 * factor2

def fit_nelson_siegel_cvxpy(maturities, yields, lambda_init=1.0, 
                           beta_bounds=(-10, 10), lambda_bounds=(0.1, 10)):
    """
    Fit Nelson-Siegel model using CVXPY with NLP solver.
    
    Parameters:
    maturities: array of maturity times
    yields: observed yields
    lambda_init: initial value for lambda parameter
    beta_bounds: bounds for beta parameters
    lambda_bounds: bounds for lambda parameter
    """
    n = len(maturities)
    
    # Define variables
    beta0 = cp.Variable()
    beta1 = cp.Variable()
    beta2 = cp.Variable()
    lambda_param = cp.Variable(pos=True)
    
    # Calculate model predictions using Nelson-Siegel formula
    # Note: We need to handle the division by zero case when tau approaches 0
    predictions = []
    for tau in maturities:
        if tau < 1e-6:  # Handle near-zero maturity
            pred = beta0 + beta1
        else:
            exp_term = cp.exp(-tau / lambda_param)
            factor1 = (1 - exp_term) / (tau / lambda_param)
            factor2 = factor1 - exp_term
            pred = beta0 + beta1 * factor1 + beta2 * factor2
        predictions.append(pred)
    
    predictions = cp.vstack(predictions)
    
    # Define objective: minimize sum of squared errors
    objective = cp.Minimize(cp.sum_squares(predictions - yields.reshape(-1, 1)))
    
    # Define constraints
    constraints = [
        beta0 >= beta_bounds[0], beta0 <= beta_bounds[1],
        beta1 >= beta_bounds[0], beta1 <= beta_bounds[1],
        beta2 >= beta_bounds[0], beta2 <= beta_bounds[1],
        lambda_param >= lambda_bounds[0], 
        lambda_param <= lambda_bounds[1]
    ]
    
    # Set initial values (important for NLP solvers)
    beta0.value = np.mean(yields)
    lambda_param.value = lambda_init
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    
    # Solve using NLP solver (e.g., IPOPT through CVXPY)
    # Note: You need to have an NLP solver installed
    problem.solve(solver=cp.IPOPT, verbose=True, nlp=True)
    
    if problem.status not in ["infeasible", "unbounded"]:
        return {
            'beta0': beta0.value,
            'beta1': beta1.value,
            'beta2': beta2.value,
            'lambda': lambda_param.value,
            'objective': problem.value,
            'status': problem.status
        }
    else:
        raise ValueError(f"Optimization failed with status: {problem.status}")

def plot_results(maturities, yields, fitted_params, title="Nelson-Siegel Fit"):
    """Plot observed vs fitted yield curve."""
    
    # Generate smooth curve for plotting
    tau_smooth = np.linspace(min(maturities), max(maturities), 100)
    
    # Calculate fitted values
    beta0 = fitted_params['beta0']
    beta1 = fitted_params['beta1']
    beta2 = fitted_params['beta2']
    lambda_val = fitted_params['lambda']
    
    # Nelson-Siegel formula for smooth curve
    exp_term = np.exp(-tau_smooth / lambda_val)
    factor1 = np.where(tau_smooth < 1e-6, 1, (1 - exp_term) / (tau_smooth / lambda_val))
    factor2 = factor1 - exp_term
    y_fitted_smooth = beta0 + beta1 * factor1 + beta2 * factor2
    
    # Calculate fitted values at observed points
    exp_term_obs = np.exp(-maturities / lambda_val)
    factor1_obs = np.where(maturities < 1e-6, 1, 
                           (1 - exp_term_obs) / (maturities / lambda_val))
    factor2_obs = factor1_obs - exp_term_obs
    y_fitted_obs = beta0 + beta1 * factor1_obs + beta2 * factor2_obs
    
    plt.figure(figsize=(10, 6))
    plt.scatter(maturities, yields, color='blue', label='Observed', s=50)
    plt.plot(tau_smooth, y_fitted_smooth, 'r-', label='Fitted NS Curve', linewidth=2)
    plt.scatter(maturities, y_fitted_obs, color='red', alpha=0.5, s=30)
    
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    param_text = f'β₀={beta0:.4f}, β₁={beta1:.4f}, β₂={beta2:.4f}, λ={lambda_val:.4f}'
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample yield curve data
    np.random.seed(42)
    
    # Maturities in years
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    
    # True parameters for generating synthetic data
    true_beta0 = 5.0
    true_beta1 = -2.0
    true_beta2 = 1.5
    true_lambda = 2.0
    
    # Generate synthetic yields with some noise
    exp_term = np.exp(-maturities / true_lambda)
    factor1 = (1 - exp_term) / (maturities / true_lambda)
    factor2 = factor1 - exp_term
    true_yields = true_beta0 + true_beta1 * factor1 + true_beta2 * factor2
    
    # Add noise
    noise = np.random.normal(0, 0.1, len(maturities))
    observed_yields = true_yields + noise
    
    print("Fitting Nelson-Siegel model using CVXPY...")
    print("-" * 50)
    
    # Fit the model
    fitted_params = fit_nelson_siegel_cvxpy(
        maturities, 
        observed_yields,
        lambda_init=2.0,
        beta_bounds=(-10, 10),
        lambda_bounds=(0.5, 5.0)
    )
    
    print(f"Optimization Status: {fitted_params['status']}")
    print(f"Objective Value (SSE): {fitted_params['objective']:.6f}")
    print("\nFitted Parameters:")
    print(f"  β₀ (level):     {fitted_params['beta0']:.4f} (true: {true_beta0:.4f})")
    print(f"  β₁ (slope):     {fitted_params['beta1']:.4f} (true: {true_beta1:.4f})")
    print(f"  β₂ (curvature): {fitted_params['beta2']:.4f} (true: {true_beta2:.4f})")
    print(f"  λ (decay):      {fitted_params['lambda']:.4f} (true: {true_lambda:.4f})")
    
    # Calculate R-squared
    exp_term_fit = np.exp(-maturities / fitted_params['lambda'])
    factor1_fit = (1 - exp_term_fit) / (maturities / fitted_params['lambda'])
    factor2_fit = factor1_fit - exp_term_fit
    y_fitted = (fitted_params['beta0'] + 
                fitted_params['beta1'] * factor1_fit + 
                fitted_params['beta2'] * factor2_fit)
    
    ss_res = np.sum((observed_yields - y_fitted) ** 2)
    ss_tot = np.sum((observed_yields - np.mean(observed_yields)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"\nR-squared: {r_squared:.4f}")
    
    # Plot results
    plot_results(maturities, observed_yields, fitted_params, 
                "Nelson-Siegel Model Fit with CVXPY")
    
    # Alternative: Using different lambda initialization
    print("\n" + "=" * 50)
    print("Testing sensitivity to lambda initialization...")
    
    for lambda_init in [0.5, 1.0, 3.0, 4.5]:
        try:
            params = fit_nelson_siegel_cvxpy(
                maturities, observed_yields, 
                lambda_init=lambda_init
            )
            print(f"λ_init={lambda_init:.1f}: λ_final={params['lambda']:.4f}, "
                  f"SSE={params['objective']:.4f}")
        except Exception as e:
            print(f"λ_init={lambda_init:.1f}: Failed - {e}")
