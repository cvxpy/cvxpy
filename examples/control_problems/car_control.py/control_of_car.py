import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp


def format_angle_with_pi(angle_rad):
    """Format angle in terms of pi for display using LaTeX."""
    if angle_rad == 0:
        return "0"
    elif angle_rad == np.pi:
        return r"$\pi$"
    elif angle_rad == -np.pi:
        return r"$-\pi$"
    elif angle_rad == np.pi/2:
        return r"$\pi/2$"
    elif angle_rad == -np.pi/2:
        return r"$-\pi/2$"
    elif angle_rad == np.pi/3:
        return r"$\pi/3$"
    elif angle_rad == -np.pi/3:
        return r"$-\pi/3$"
    elif angle_rad == np.pi/4:
        return r"$\pi/4$"
    elif angle_rad == -np.pi/4:
        return r"$-\pi/4$"
    elif angle_rad == np.pi/6:
        return r"$\pi/6$"
    elif angle_rad == -np.pi/6:
        return r"$-\pi/6$"
    else:
        # For other values, express as fraction of pi
        ratio = angle_rad / np.pi
        if abs(ratio - round(ratio)) < 1e-6:
            # Integer multiple of pi
            mult = int(round(ratio))
            if mult == 1:
                return r"$\pi$"
            elif mult == -1:
                return r"$-\pi$"
            else:
                return rf"${mult}\pi$"
        elif abs(ratio * 2 - round(ratio * 2)) < 1e-6:
            # Half-integer multiple of pi
            mult = int(round(ratio * 2))
            if mult == 1:
                return r"$\pi/2$"
            elif mult == -1:
                return r"$-\pi/2$"
            else:
                return rf"${mult}\pi/2$"
        else:
            # Fall back to decimal with pi
            return rf"${ratio:.2f}\pi$"


def solve_car_control_vectorized(x_final, L=0.1, N=50, h=0.1, gamma=10,
                                speed_bounds=(-0.6, 0.8), 
                                steering_bounds=(-np.pi/8, np.pi/6),
                                accel_bound=0.4, 
                                steering_rate_bound=np.pi/10):
    """
    Solve the nonlinear optimal control problem for car trajectory planning.
    
    Parameters:
    - x_final: tuple (p1, p2, theta) for final position and orientation
    - L: wheelbase length
    - N: number of time steps
    - h: time step size
    - gamma: weight for control smoothness term
    - speed_bounds: tuple (min_speed, max_speed) in m/s
    - steering_bounds: tuple (min_steering, max_steering) in radians
    - accel_bound: maximum acceleration magnitude in m/s²
    - steering_rate_bound: maximum steering rate magnitude in rad/s
    
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

    # speed limit bounds (parameterized)
    constraints += [u[:, 0] >= speed_bounds[0], u[:, 0] <= speed_bounds[1]]
    # steering angle bounds (parameterized)
    constraints += [u[:, 1] >= steering_bounds[0], u[:, 1] <= steering_bounds[1]]
    # acceleration bounds (parameterized)
    constraints += [cp.abs(u[1:, 0] - u[:-1, 0]) <= accel_bound * h]
    # steering angle rate bounds (parameterized)
    constraints += [cp.abs(u[1:, 1] - u[:-1, 1]) <= steering_rate_bound * h]
    
    # Create and solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
    return x.value, u.value

def plot_trajectory(x_opt, u_opt, L, h=0.1, title="Car Trajectory"):
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
            front_center = (np.array([p1, p2]) + 
                           (car_length/2) * np.array([np.cos(theta), np.sin(theta)]))
            steer_tip = (front_center + 
                        (car_length/3) * np.array([np.cos(theta + phi), np.sin(theta + phi)]))
            ax.plot([front_center[0], steer_tip[0]], [front_center[1], steer_tip[1]],
                    color='crimson', linewidth=1, alpha=alpha+0.1)

    # Mark start and end points
    ax.plot(x_opt[0, 0], x_opt[0, 1], 'go', markersize=10, label='Start')
    ax.plot(x_opt[-1, 0], x_opt[-1, 1], 'ro', markersize=10, label='Goal')
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
    results_file = "car_optimization_results.pkl"
    
    # Check if saved results exist
    if os.path.exists(results_file):
        print("Loading saved optimization results...")
        with open(results_file, 'rb') as f:
            results, test_cases = pickle.load(f)
        print(f"Loaded {len(results)} optimization results from {results_file}")
    else:
        print("No saved results found. Running optimization...")
        
        # Test cases with different constraint scenarios
    test_cases = [
        {
            "x_final": (0, 0.1, np.pi), 
            "description": "U-turn (tight steering)", 
            "speed_bounds": (-0.15, 0.6),
            "steering_bounds": (-np.pi/8, np.pi/8),  # ±15° - tight steering
            "accel_bound": 0.35,
            "steering_rate_bound": np.pi/10  # ±12°/s - slow steering changes
        },
        {
            "x_final": (0.5, 0.5, -np.pi/2), 
            "description": "Parallel parking (precise control)",
            "speed_bounds": (-0.15, 0.6),
            "steering_bounds": (-np.pi/8, np.pi/8),  # ±15° - tight steering
            "accel_bound": 0.35,
            "steering_rate_bound": np.pi/10  # ±12°/s - slow steering changes
        },
        {
            "x_final": (0, 0.5, 0), 
            "description": "Forward motion (high speed)",
            "speed_bounds": (-0.15, 0.8),  # higher speed limits
            "steering_bounds": (-np.pi/4, np.pi/4),  # [-18°, +22.5°] - moderate steering
            "accel_bound": 0.4,  # aggressive acceleration
            "steering_rate_bound": np.pi/4  # ±22.5°/s
        },
        {
            "x_final": (0, 1, np.pi/2), 
            "description": "Complex maneuver (aggressive)",
            "speed_bounds": (-0.15, 0.8),  # higher speed limits
            "steering_bounds": (-np.pi/4, np.pi/4),  # [-18°, +22.5°] - moderate steering
            "accel_bound": 0.4,  # aggressive acceleration
            "steering_rate_bound": np.pi/4  # ±22.5°/s
        }
    ]
    
    print("Car Control Optimization - Comparing Different Constraint Scenarios")
    print("=" * 70)
    
    # Store results for all test cases
    results = []
    
    # Solve for each test case with its specific constraints
    for i, case in enumerate(test_cases):
        x_final = case["x_final"]
        description = case["description"]
        
        print(f"\n[{i+1}/{len(test_cases)}] {description}")
        print(f"Target: p1={x_final[0]:.1f}, p2={x_final[1]:.1f}, theta={x_final[2]:.2f} rad")
        
        # Print the constraints being used
        steering_deg = (case["steering_bounds"][0]*180/np.pi, case["steering_bounds"][1]*180/np.pi)
        rate_deg = case["steering_rate_bound"]*180/np.pi
        print(f"Constraints:")
        print(f"  Speed: [{case['speed_bounds'][0]:.1f}, {case['speed_bounds'][1]:.1f}] m/s")
        print(f"  Steering: [{steering_deg[0]:.1f}°, {steering_deg[1]:.1f}°]")
        print(f"  Acceleration: ±{case['accel_bound']:.1f} m/s²")
        print(f"  Steering rate: ±{rate_deg:.1f}°/s")
        
        try:
            x_opt, u_opt = solve_car_control_vectorized(
                x_final,
                speed_bounds=case["speed_bounds"],
                steering_bounds=case["steering_bounds"],
                accel_bound=case["accel_bound"],
                steering_rate_bound=case["steering_rate_bound"]
            )
            
            if x_opt is not None and u_opt is not None:
                print("✓ Optimization successful!")
                print(f"Final: p1={x_opt[-1, 0]:.3f}, p2={x_opt[-1, 1]:.3f}, theta={x_opt[-1, 2]:.3f}")
                
                results.append({
                    'x_opt': x_opt,
                    'u_opt': u_opt,
                    'description': description,
                    'x_final': x_final,
                    'constraints': case  # Store constraint info for plotting
                })
            else:
                print("✗ Optimization failed!")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
        print(f"\nSuccessfully solved {len(results)}/{len(test_cases)} scenarios.")
        
        # Save results for future use
        with open(results_file, 'wb') as f:
            pickle.dump((results, test_cases), f)
        print(f"Saved optimization results to {results_file}")
    
    # Now create plots (this section runs regardless of whether we loaded or computed results)
    if len(results) == 4:
        print("\nCreating three separate plots...")
        
        # PLOT 1: Trajectories (2x2 layout)
        print("1. Creating trajectory plots...")
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, result in enumerate(results):
            row, col = i // 2, i % 2
            ax = axes1[row, col]
            
            x_opt = result['x_opt']
            u_opt = result['u_opt']
            constraints = result['constraints']
            L = 0.1
            car_length = L
            car_width = L * 0.6
            
            # Draw trajectory with car shadows
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
                
                # Draw steering angle indicator
                if k < len(u_opt):
                    phi = u_opt[k, 1]
                    front_center = (np.array([p1, p2]) + 
                                   (car_length/2) * np.array([np.cos(theta), np.sin(theta)]))
                    steer_tip = (front_center + 
                                (car_length/3) * np.array([np.cos(theta + phi), np.sin(theta + phi)]))
                    ax.plot([front_center[0], steer_tip[0]], [front_center[1], steer_tip[1]],
                            color='crimson', linewidth=1, alpha=alpha+0.1)
            
            ax.plot(x_opt[0, 0], x_opt[0, 1], 'go', markersize=8, label='Start')
            ax.plot(x_opt[-1, 0], x_opt[-1, 1], 'ro', markersize=8, label='Goal')
            
            # Simple title with x_final coordinates
            x_final = result['x_final']
            theta_str = format_angle_with_pi(x_final[2])
            ax.set_title(f'x_final: ({x_final[0]:.1f}, {x_final[1]:.1f}, {theta_str})', 
                        fontsize=14)
            
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        # Create common legend for trajectory plots
        handles, labels = axes1[0, 0].get_legend_handles_labels()
        fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
                   ncol=2, fontsize=15)
        
        plt.tight_layout()
        fig1.savefig("trajectories_comparison.pdf", bbox_inches="tight", dpi=300)
        print("✓ Saved trajectories_comparison.pdf")
        plt.show()
        
        # PLOT 2: First-order controls (Speed and Steering) (2x2 layout)
        print("2. Creating first-order control plots...")
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        
        for i, result in enumerate(results):
            row, col = i // 2, i % 2
            ax = axes2[row, col]
            
            u_opt = result['u_opt']
            constraints = result['constraints']
            k_steps = np.arange(len(u_opt))
            
            # Normalize both controls to [-0.5, 0.5] based on their bounds
            speed_bounds = constraints["speed_bounds"]
            steering_bounds = constraints["steering_bounds"]
            
            # Normalize speed: map from [min, max] to [-0.5, 0.5]
            speed_range = speed_bounds[1] - speed_bounds[0]
            speed_center = (speed_bounds[1] + speed_bounds[0]) / 2
            speed_normalized = (u_opt[:, 0] - speed_center) / speed_range
            
            # Normalize steering: map from [min, max] to [-0.5, 0.5] 
            steering_range = steering_bounds[1] - steering_bounds[0]
            steering_center = (steering_bounds[1] + steering_bounds[0]) / 2
            steering_normalized = (u_opt[:, 1] - steering_center) / steering_range
            
            # Plot both normalized controls on same axis
            ax.plot(k_steps, speed_normalized, 'b-', linewidth=2, label='Speed')
            ax.plot(k_steps, steering_normalized, 'r-', linewidth=2, label='Steering Angle')
            
            # Only add labels on outer edges
            if row == 1:  # Bottom row
                ax.set_xlabel('$k$', fontsize=16)
            if col == 0:  # Left column
                ax.set_ylabel('$u_k$', fontsize=16)
            
            ax.set_ylim(-0.6, 0.6)  # Slightly larger than [-0.5, 0.5] for visibility
            ax.grid(True, alpha=0.3)
        
        # Create common legend for first-order control plots
        handles, labels = axes2[0, 0].get_legend_handles_labels()
        fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
                   ncol=2, fontsize=15)
        
        plt.tight_layout()
        fig2.savefig("first_order_controls.pdf", bbox_inches="tight", dpi=300)
        print("✓ Saved first_order_controls.pdf")
        plt.show()
        
        # PLOT 3: Second-order controls (Acceleration and Steering Rate) (2x2 layout)
        print("3. Creating second-order control plots...")
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        
        for i, result in enumerate(results):
            row, col = i // 2, i % 2
            ax = axes3[row, col]
            
            u_opt = result['u_opt']
            constraints = result['constraints']
            
            if len(u_opt) > 1:
                # Calculate derivatives (differences)
                acceleration = np.diff(u_opt[:, 0]) / 0.1  # h = 0.1
                steering_rate = np.diff(u_opt[:, 1]) / 0.1  # in rad/s
                
                k_steps_diff = np.arange(len(acceleration))
                
                # Normalize derivatives to [-0.5, 0.5] based on their bounds
                accel_bound = constraints["accel_bound"]
                rate_bound = constraints["steering_rate_bound"]
                
                # Normalize acceleration: map from [-bound, bound] to [-0.5, 0.5]
                accel_normalized = acceleration / (2 * accel_bound)
                
                # Normalize steering rate: map from [-bound, bound] to [-0.5, 0.5]
                rate_normalized = steering_rate / (2 * rate_bound)
                
                # Plot both normalized derivatives on same axis
                ax.plot(k_steps_diff, accel_normalized, 'g-', linewidth=2, label='Acceleration')
                ax.plot(k_steps_diff, rate_normalized, 'm-', linewidth=2, 
                       label='Steering Angle Rate')
                
                # Only add labels on outer edges
                if row == 1:  # Bottom row
                    ax.set_xlabel('$k$', fontsize=16)
                if col == 0:  # Left column
                    ax.set_ylabel(r'$\frac{u_k - u_{k-1}}{h}$', fontsize=18)
                
                ax.set_ylim(-0.6, 0.6)  # Slightly larger than [-0.5, 0.5] for visibility
                ax.grid(True, alpha=0.3)
        
        # Create common legend for second-order control plots
        handles, labels = axes3[0, 0].get_legend_handles_labels()
        fig3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
                   ncol=2, fontsize=15)
        
        plt.tight_layout()
        fig3.savefig("second_order_controls.pdf", bbox_inches="tight", dpi=300)
        print("✓ Saved second_order_controls.pdf")
        plt.show()
        
        # Summary comparison table
        print("\n" + "="*80)
        print("CONSTRAINT COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Scenario':<25} {'Steering Range':<15} {'Max Speed':<12} {'Success':<8}")
        print("-"*80)
        for i, result in enumerate(results):
            constraints = result['constraints']
            steering_deg = (constraints["steering_bounds"][0]*180/np.pi, 
                           constraints["steering_bounds"][1]*180/np.pi)
            max_speed = constraints["speed_bounds"][1]
            
            print(f"{result['description']:<25} "
                  f"[{steering_deg[0]:>4.0f}°,{steering_deg[1]:>4.0f}°]    "
                  f"{max_speed:>6.1f} m/s    ✓")
    else:
        print(f"\nOnly {len(results)}/{len(test_cases)} scenarios succeeded.")
        print("Individual plots for successful cases:")
        
        for result in results:
            print(f"\nPlotting {result['description']}...")
            fig, ax = plot_trajectory(result['x_opt'], result['u_opt'], L=0.1, 
                                    title=result['description'])
            plt.show()
            
            # Show control inputs 
            plot_control_inputs(result['u_opt'])

    print("\nDone!")
