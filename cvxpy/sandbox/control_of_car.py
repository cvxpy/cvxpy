import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp


def solve_car_control(x_final, L=0.1, N=5, h=0.1, gamma=10):
    """
    Solve the nonlinear optimal control problem for car trajectory planning.
    
    Parameters:
    - x_final: tuple (p1, p2, theta) for final position and orientation
    - L: wheelbase length
    - N: number of time steps
    - h: time step size
    - gamma: weight for control smoothness term
    
    Returns:
    - x_opt: optimal states (N+1 x 3)
    - u_opt: optimal controls (N x 2)
    """
    # Add random seed for reproducibility
    np.random.seed(78)

    x = [cp.Variable(3) for _ in range(N+1)]
    # Controls: u[k] = [s(k), phi(k)]
    u = [cp.Variable(2) for _ in range(N)]
    # initial guess for controls between 0 and 1
    for k in range(N):
        u[k].value = np.random.uniform(0, 1, 2)
    # Initial state (starting at origin with zero orientation)
    x_init = np.array([0, 0, 0])
    
    # Objective function
    objective = 0
    
    # Sum of squared control inputs
    for k in range(N):
        objective += cp.sum_squares(u[k][:])
    
    # Control smoothness term
    for k in range(N-1):
        objective += gamma * cp.sum_squares(u[k+1][:] - u[k][:])

    # Constraints
    constraints = []
    constraints.append(x[0][:] == x_init)

    for k in range(N):
        # x[k+1] = f(x[k], u[k])
        # where f(x, u) = x + h * [u[0]*cos(x[2]), u[0]*sin(x[2]), u[0]*tan(u[1])/L]
        
        # Position dynamics
        constraints.append(x[k+1][0] == x[k][0] + h * u[k][0] * cp.cos(x[k][2]))
        constraints.append(x[k+1][1] == x[k][1] + h * u[k][0] * cp.sin(x[k][2]))

        # Orientation dynamics
        constraints.append(x[k+1][2] == x[k][2] + h * (u[k][0] / L) * cp.tan(u[k][1]))

    # Final state constraint
    constraints.append(x[N][:] == x_final)

    # Steering angle limits (optional but realistic)
    # Assuming max steering angle of 45 degrees
    max_steering = np.pi / 4
    constraints.append(u[:][1] >= -max_steering)
    constraints.append(u[:][1] <= max_steering)
    
    # Create and solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

    return x, u


def plot_trajectory(x_opt, u_opt, L, h, title="Car Trajectory"):
    """
    Plot the car trajectory with orientation indicators.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Plot trajectory
    ax.plot(x_opt[:, 0], x_opt[:, 1], 'b-', linewidth=2, label='Trajectory')
    
    # Plot car position and orientation at several time steps
    car_length = L
    
    # Select time steps to show car outline (every 5th step)
    steps_to_show = range(0, len(x_opt), 5)
    
    for k in steps_to_show:
        p1, p2, theta = x_opt[k]
        
        # Car outline (simplified rectangle)
        # Front of car
        front_x = p1 + (car_length/2) * np.cos(theta)
        front_y = p2 + (car_length/2) * np.sin(theta)
        
        # Rear of car
        rear_x = p1 - (car_length/2) * np.cos(theta)
        rear_y = p2 - (car_length/2) * np.sin(theta)
        
        # Draw car as a line with orientation
        ax.plot([rear_x, front_x], [rear_y, front_y], 'k-', linewidth=3, alpha=0.5)
        
        # Draw steering angle indicator if not at final position
        if k < len(u_opt):
            phi = u_opt[k, 1]
            # Steering direction from front of car
            steer_x = front_x + (car_length/3) * np.cos(theta + phi)
            steer_y = front_y + (car_length/3) * np.sin(theta + phi)
            ax.plot([front_x, steer_x], [front_y, steer_y], 'r-', linewidth=2, alpha=0.7)
    
    # Mark start and end points
    ax.plot(x_opt[0, 0], x_opt[0, 1], 'go', markersize=10, label='Start')
    ax.plot(x_opt[-1, 0], x_opt[-1, 1], 'ro', markersize=10, label='Goal')
    
    ax.set_xlabel('p1')
    ax.set_ylabel('p2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    return fig, ax


# Example usage
if __name__ == "__main__":
    # Test cases from the figure
    test_cases = [
        ((0, 1, 0), "Move forward to (0, 1)"),
        ((0, 1, np.pi/2), "Move to (0, 1) and turn 90°"),
        ((0, 0.5, 0), "Move forward to (0, 0.5)"),
        ((0.5, 0.5, -np.pi/2), "Move to (0.5, 0.5) and turn -90°")
    ]
    
    # Solve for each test case
    for x_final, description in test_cases:
        print(f"\nSolving for: {description}")
        print(f"Target state: p1={x_final[0]}, p2={x_final[1]}, theta={x_final[2]:.2f}")
        x_opt, u_opt = solve_car_control(x_final)
        if x_opt is not None and u_opt is not None:
            print("Optimization successful!")
            print(
                f"Final position: p1={x_opt[-1].value[0]:.3f}, "
                f"p2={x_opt[-1].value[1]:.3f}, "
                f"theta={x_opt[-1].value[2]:.3f}"
            )
            x_opt = np.array([xi.value for xi in x_opt])
            u_opt = np.array([ui.value for ui in u_opt])
            # Plot the trajectory
            fig, ax = plot_trajectory(x_opt, u_opt, L=0.1, h=0.1, title=description)
            plt.show()
        else:
            print("Optimization failed!")
        # Additional analysis: plot control inputs
        if x_opt is not None and u_opt is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

            time_steps = np.arange(len(u_opt)) * 0.1  # h = 0.1

            ax1.plot(time_steps, u_opt[:, 0], 'b-', linewidth=2)
            ax1.set_ylabel('Speed s(t)')
            ax1.set_xlabel('Time')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(time_steps, u_opt[:, 1], 'r-', linewidth=2)
            ax2.set_ylabel('Steering angle φ(t)')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
