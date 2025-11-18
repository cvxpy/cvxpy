import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp


def solve_car_control_vectorized(x_final, L=0.1, N=50, h=0.1, gamma=10):
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
    np.random.seed(858)
    x, u = cp.Variable((N+1, 3)), cp.Variable((N, 2))
    u.value = np.random.uniform(0, 1, size=(N,2))
    x_init = np.array([0, 0, 0])

    objective = cp.sum_squares(u)
    objective += gamma * cp.sum_squares(u[1:, :] - u[:-1, :])

    constraints = [x[0, :] == x_init, x[N, :] == x_final]
    # Extract state components for timesteps 0 to N-1
    x_curr, x_next = x[:-1, :], x[1:, :]
    v, delta, theta = u[:, 0], u[:, 1], x_curr[:, 2]

    constraints += [x_next[:, 0] == x_curr[:, 0] + h * cp.multiply(v, cp.cos(theta)),
                    x_next[:, 1] == x_curr[:, 1] + h * cp.multiply(v, cp.sin(theta)),
                    x_next[:, 2] == x_curr[:, 2] + h * cp.multiply(v / L, cp.tan(delta))]

    # speed limit bounds
    constraints += [u[:, 0] >= -1.0, u[:, 0] <= 2.0]
    # steering angle bounds
    constraints += [u[:, 1] >= -np.pi / 4, u[:, 1] <= np.pi / 8]
    # acceleration bounds
    constraints += [cp.abs(u[1:, 0] - u[:-1, 0]) <= 0.6 * h]
    # steering angle rate bounds
    constraints += [cp.abs(u[1:, 1] - u[:-1, 1]) <= np.pi / 4 * h]
    # Create and solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
    return x.value, u.value

def plot_trajectory(x_opt, u_opt, L, h, title="Car Trajectory"):
    """
    Plot the car trajectory with orientation indicators.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    car_length = L
    car_width = L * 0.6

    # Select time steps to show car outline (e.g., every 2nd step for more shadows)
    steps_to_show = np.arange(0, len(x_opt), max(1, len(x_opt)//20))
    n_shadows = len(steps_to_show)

    # Draw car as a fading rectangle (shadow) at each step
    for i, k in enumerate(steps_to_show):
        p1, p2, theta = x_opt[k]
        # Rectangle corners (centered at (p1, p2), rotated by theta)
        corners = np.array([
            [ car_length/2,  car_width/2],
            [ car_length/2, -car_width/2],
            [-car_length/2, -car_width/2],
            [-car_length/2,  car_width/2],
            [ car_length/2,  car_width/2],  # close the rectangle
        ])
        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated = (R @ corners.T).T + np.array([p1, p2])
        # Fade older shadows
        alpha = 0.15 + 0.7 * (i+1)/n_shadows
        ax.fill(rotated[:,0], rotated[:,1], color='dodgerblue', alpha=alpha, edgecolor='k', linewidth=0.7)

        # Draw steering angle indicator if not at final position
        if k < len(u_opt):
            phi = u_opt[k, 1]
            # Steering direction from front of car
            front_center = np.array([p1, p2]) + (car_length/2) * np.array([np.cos(theta), np.sin(theta)])
            steer_tip = front_center + (car_length/3) * np.array([np.cos(theta + phi), np.sin(theta + phi)])
            ax.plot([front_center[0], steer_tip[0]], [front_center[1], steer_tip[1]],
                    color='crimson', linewidth=1, alpha=alpha+0.1)

    # Mark start and end points
    ax.plot(x_opt[0, 0], x_opt[0, 1], 'go', markersize=10, label='Start')
    ax.plot(x_opt[-1, 0], x_opt[-1, 1], 'ro', markersize=10, label='Goal')

    #ax.set_xlabel('p1')
    #ax.set_ylabel('p2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return fig, ax


def plot_control_inputs(u_opt, h=0.1):
    """
    Plot control inputs and their derivatives.
    
    Parameters:
    - u_opt: optimal controls (N x 2) containing [speed, steering_angle]
    - h: time step size
    """
    # Plot control inputs in one plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    k_steps = np.arange(len(u_opt))  # time step indices
    
    # Normalize steering angle to [0, 1]
    # Assuming steering angle range is [-π/4, π/4] based on the constraints
    max_steering = np.pi / 4
    normalized_steering = (u_opt[:, 1] + max_steering) / (2 * max_steering)
    
    # Plot both controls on the same axes (smooth lines, no markers)
    line1 = ax.plot(k_steps, u_opt[:, 0], 'b-', linewidth=2, label='Speed')
    line2 = ax.plot(k_steps, normalized_steering, 'r-', linewidth=2, 
                   label='Steering Angle')
    
    ax.set_xlabel('k')
    ax.set_ylabel('$u_k$')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig("control_inputs.pdf", bbox_inches="tight", dpi=300)
    print("Saved control inputs plot as control_inputs.pdf")
    plt.show()
    
    # Plot acceleration and steering rate derivatives
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate derivatives (differences)
    if len(u_opt) > 1:
        # Acceleration (derivative of speed)
        acceleration = np.diff(u_opt[:, 0]) / h
        # Steering rate (derivative of steering angle)
        steering_rate = np.diff(u_opt[:, 1]) / h
        
        # Normalize steering rate to similar scale as acceleration
        max_steering_rate = np.pi / 2  # from constraints: ω_max
        normalized_steering_rate = steering_rate / max_steering_rate
        
        k_steps_diff = np.arange(len(acceleration))  # k for derivatives
        
        # Plot both derivatives (smooth lines, no markers)
        line1 = ax.plot(k_steps_diff, acceleration, 'g-', linewidth=2, 
                       label='Acceleration')
        line2 = ax.plot(k_steps_diff, normalized_steering_rate, 'm-', linewidth=2, 
                       label='Steering Rate')
        
        ax.set_xlabel('k')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        fig.savefig("acceleration_derivatives.pdf", bbox_inches="tight", dpi=300)
        print("Saved acceleration and derivatives plot as acceleration_derivatives.pdf")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Test cases - interesting maneuvers
    test_cases = [
        ((0, 0.1, np.pi), "U-turn"),
        ((0.3, 0.1, np.pi/2), "Parallel parking"), 
        ((0, 0.5, 0), "Forward motion"),
        ((0, 1, 0), "Long forward motion"),
    ]
    
    # Store results for all test cases
    results = []
    
    # Solve for each test case
    for i, (x_final, description) in enumerate(test_cases):
        print(f"\nSolving for: {description}")
        print(f"Target state: p1={x_final[0]}, p2={x_final[1]}, theta={x_final[2]:.2f}")
        
        try:
            x_opt, u_opt = solve_car_control_vectorized(x_final)
            
            if x_opt is not None and u_opt is not None:
                print("Optimization successful!")
                print(f"Final position: p1={x_opt[-1, 0]:.3f}, "
                      f"p2={x_opt[-1, 1]:.3f}, theta={x_opt[-1, 2]:.3f}")
                
                results.append({
                    'x_opt': x_opt,
                    'u_opt': u_opt, 
                    'description': description,
                    'x_final': x_final
                })
            else:
                print("Optimization failed!")
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Create combined plots
    if len(results) == 4:
        # Plot 1: Combined Trajectories (2x2 subplots)
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, result in enumerate(results):
            row, col = i // 2, i % 2
            ax = axes1[row, col]
            
            x_opt = result['x_opt']
            u_opt = result['u_opt']
            L = 0.1
            car_length = L
            car_width = L * 0.6
            
            # Simplified trajectory plot for subplot
            steps_to_show = np.arange(0, len(x_opt), max(1, len(x_opt)//20))
            n_shadows = len(steps_to_show)
            
            for j, k in enumerate(steps_to_show):
                p1, p2, theta = x_opt[k]
                corners = np.array([
                    [car_length/2, car_width/2], [car_length/2, -car_width/2],
                    [-car_length/2, -car_width/2], [-car_length/2, car_width/2],
                    [car_length/2, car_width/2]
                ])
                R = np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta), np.cos(theta)]])
                rotated = (R @ corners.T).T + np.array([p1, p2])
                alpha = 0.15 + 0.7 * (j+1)/n_shadows
                ax.fill(rotated[:,0], rotated[:,1], color='dodgerblue', 
                       alpha=alpha, edgecolor='k', linewidth=0.5)
                
                # Draw steering angle indicator if not at final position
                if k < len(u_opt):
                    phi = u_opt[k, 1]
                    # Steering direction from front of car
                    front_center = np.array([p1, p2]) + (car_length/2) * np.array([np.cos(theta), np.sin(theta)])
                    steer_tip = front_center + (car_length/3) * np.array([np.cos(theta + phi), np.sin(theta + phi)])
                    ax.plot([front_center[0], steer_tip[0]], [front_center[1], steer_tip[1]],
                            color='crimson', linewidth=1, alpha=alpha+0.1)
            
            ax.plot(x_opt[0, 0], x_opt[0, 1], 'go', markersize=8, label='Start')
            ax.plot(x_opt[-1, 0], x_opt[-1, 1], 'ro', markersize=8, label='Goal')
            
            # Add subtitle with x_final values in terms of pi
            x_final = result['x_final']
            
            # Convert theta to fraction of pi for display
            theta = x_final[2]
            if abs(theta - np.pi) < 0.01:
                theta_str = r'\pi'
            elif abs(theta - np.pi/2) < 0.01:
                theta_str = r'\pi/2'
            elif abs(theta) < 0.01:
                theta_str = '0'
            elif abs(theta - 3*np.pi/2) < 0.01:
                theta_str = r'3\pi/2'
            elif abs(theta - 2*np.pi) < 0.01:
                theta_str = r'2\pi'
            else:
                # For other values, show as decimal fraction of pi
                pi_fraction = theta / np.pi
                if abs(pi_fraction - round(pi_fraction)) < 0.01:
                    if round(pi_fraction) == 1:
                        theta_str = '\\pi'
                    else:
                        theta_str = f'{round(pi_fraction):.0f}\\pi'
                else:
                    theta_str = f'{pi_fraction:.2f}\\pi'
            
            ax.set_title(f'$x_{{final}} = ({x_final[0]:.1f}, {x_final[1]:.1f}, {theta_str})$', 
                        fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        fig1.savefig("combined_trajectories.pdf", bbox_inches="tight", dpi=300)
        print("Saved combined trajectories as combined_trajectories.pdf")
        plt.show()
        
        # Plot 2: Combined Control Inputs (2x2 subplots)
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
        
        for i, result in enumerate(results):
            row, col = i // 2, i % 2
            ax = axes2[row, col]
            
            u_opt = result['u_opt']
            k_steps = np.arange(len(u_opt))
            max_steering = np.pi / 4
            normalized_steering = (u_opt[:, 1] + max_steering) / (2 * max_steering)
            
            ax.plot(k_steps, u_opt[:, 0], 'b-', linewidth=2, label='Speed')
            ax.plot(k_steps, normalized_steering, 'r-', linewidth=2, label='Steering Angle')
            ax.set_xlabel('k')
            ax.set_ylabel('$u_k$')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        fig2.savefig("combined_control_inputs.pdf", bbox_inches="tight", dpi=300)
        print("Saved combined control inputs as combined_control_inputs.pdf")
        plt.show()
        
        # Plot 3: Combined Acceleration and Steering Rates (2x2 subplots)
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
        
        for i, result in enumerate(results):
            row, col = i // 2, i % 2
            ax = axes3[row, col]
            
            u_opt = result['u_opt']
            h = 0.1
            
            if len(u_opt) > 1:
                acceleration = np.diff(u_opt[:, 0]) / h
                steering_rate = np.diff(u_opt[:, 1]) / h
                max_steering_rate = np.pi / 2
                normalized_steering_rate = steering_rate / max_steering_rate
                k_steps_diff = np.arange(len(acceleration))
                
                ax.plot(k_steps_diff, acceleration, 'g-', linewidth=2, label='Acceleration')
                ax.plot(k_steps_diff, normalized_steering_rate, 'm-', 
                       linewidth=2, label='Steering Rate')
                ax.set_xlabel('k')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        fig3.savefig("combined_acceleration_derivatives.pdf", bbox_inches="tight", dpi=300)
        print("Saved combined acceleration and derivatives as " +
              "combined_acceleration_derivatives.pdf")
        plt.show()
    else:
        print(f"Only {len(results)} test cases succeeded. Need 4 for combined plots.")
