"""
Space Shuttle Reentry Trajectory Optimization using CVXPY

Adapted from JuMP example by Henrique Ferrolho
Original problem from Betts (2010), Chapter 6

This solves a nonlinear optimal control problem to maximize the cross-range
(final latitude) of a Space Shuttle during reentry, subject to dynamics
constraints and thermal heating limits.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import cvxpy as cp

# ============================================================================
# Physical Constants and Parameters
# ============================================================================

# Vehicle parameters
W = 203000.0      # weight (lb)
G0 = 32.174       # acceleration (ft/sec^2)
M = W / G0        # mass (slug)

# Atmospheric and aerodynamic constants
RHO0 = 0.002378   # sea level density
HR = 23800.0      # scale height (ft)
RE = 20902900.0   # Earth radius (ft)
MU = 0.14076539e17  # gravitational parameter
S = 2690.0        # reference area (ft^2)

# Aerodynamic coefficients for lift
A0 = -0.20704
A1 = 0.029244

# Aerodynamic coefficients for drag
B0 = 0.07854
B1 = -0.61592e-2
B2 = 0.621408e-3

# Heating coefficients
C0 = 1.0672181
C1 = -0.19213774e-1
C2 = 0.21286289e-3
C3 = -0.10117249e-5

# ============================================================================
# Initial and Final Conditions
# ============================================================================

# Initial conditions (scaled units where noted)
h_s = 2.6                    # altitude (ft) / 1e5
phi_s = np.deg2rad(0)        # longitude (rad)
theta_s = np.deg2rad(0)      # latitude (rad)
v_s = 2.56                   # velocity (ft/sec) / 1e4
gamma_s = np.deg2rad(-1)     # flight path angle (rad)
psi_s = np.deg2rad(90)       # azimuth (rad)
alpha_s = np.deg2rad(0)      # angle of attack (rad)
beta_s = np.deg2rad(0)       # bank angle (rad)

# Final conditions (TAEM interface)
h_t = 0.8                    # altitude (ft) / 1e5
v_t = 0.25                   # velocity (ft/sec) / 1e4
gamma_t = np.deg2rad(-5)     # flight path angle (rad)

# ============================================================================
# Discretization Parameters
# ============================================================================

N = 2  # number of mesh points
INTEGRATION_RULE = "rectangular"  # or "trapezoidal"

# ============================================================================
# Problem Setup
# ============================================================================

# Decision variables - state trajectory with bounds
scaled_h = cp.Variable(N, bounds=[0, None])  # altitude (ft) / 1e5
phi = cp.Variable(N)                          # longitude (rad), unbounded
theta = cp.Variable(N, bounds=[np.deg2rad(-89), np.deg2rad(89)])  # latitude (rad)
scaled_v = cp.Variable(N, bounds=[1e-4, None])  # velocity (ft/sec) / 1e4
gamma = cp.Variable(N, bounds=[np.deg2rad(-89), np.deg2rad(89)])  # flight path angle (rad)
psi = cp.Variable(N)                          # azimuth (rad), unbounded

# Control variables with bounds
alpha = cp.Variable(N, bounds=[np.deg2rad(-90), np.deg2rad(90)])  # angle of attack (rad)
beta = cp.Variable(N, bounds=[np.deg2rad(-89), np.deg2rad(1)])    # bank angle (rad)

# Time step - fixed at 4.0 seconds
dt = cp.Variable(N, bounds=[4.0, 4.0])  # time step (sec)

# ============================================================================
# Derived Quantities (as expressions)
# ============================================================================

# Restore true scale
h = scaled_h * 1e5
v = scaled_v * 1e4

# Aerodynamic coefficients
alpha_deg = alpha * 180 / np.pi
c_L = A0 + A1 * alpha_deg
c_D = B0 + B1 * alpha_deg + B2 * cp.square(alpha_deg)

# Atmospheric density
rho = RHO0 * cp.exp(-h / HR)

# Aerodynamic forces
D = 0.5 * cp.multiply(c_D, S * cp.multiply(rho, cp.square(v)))
L = 0.5 * cp.multiply(c_L, S * cp.multiply(rho, cp.square(v)))

# Gravitational quantities
r = RE + h
g = MU / cp.square(r)

# ============================================================================
# Dynamics (derivatives)
# ============================================================================

# State derivatives
delta_h = cp.multiply(v, cp.sin(gamma))
delta_phi = cp.multiply(
    cp.multiply(v / r, cp.cos(gamma)),
    cp.sin(psi) / cp.cos(theta)
)
delta_theta = cp.multiply(v / r, cp.multiply(cp.cos(gamma), cp.cos(psi)))
delta_v = -D / M - cp.multiply(g, cp.sin(gamma))
delta_gamma = (
    cp.multiply(L / (M * v), cp.cos(beta)) +
    cp.multiply(cp.cos(gamma), (v / r - g / v))
)
delta_psi = (
    cp.multiply(1 / (M * cp.multiply(v, cp.cos(gamma))), cp.multiply(L, cp.sin(beta))) +
    cp.multiply(
        cp.multiply(v / r, cp.cos(gamma)),
        cp.multiply(cp.sin(psi), cp.sin(theta)) / cp.cos(theta)
    )
)

# ============================================================================
# Constraints
# ============================================================================

constraints = []

# Initial conditions
constraints.append(scaled_h[0] == h_s)
constraints.append(phi[0] == phi_s)
constraints.append(theta[0] == theta_s)
constraints.append(scaled_v[0] == v_s)
constraints.append(gamma[0] == gamma_s)
constraints.append(psi[0] == psi_s)

# Final conditions
constraints.append(scaled_h[N-1] == h_t)
constraints.append(scaled_v[N-1] == v_t)
constraints.append(gamma[N-1] == gamma_t)

# Dynamics constraints (collocation) - vectorized
if INTEGRATION_RULE == "rectangular":
    # Rectangular (Euler) integration
    constraints.append(h[1:] == h[:-1] + cp.multiply(dt[:-1], delta_h[:-1]))
    constraints.append(phi[1:] == phi[:-1] + cp.multiply(dt[:-1], delta_phi[:-1]))
    constraints.append(theta[1:] == theta[:-1] + cp.multiply(dt[:-1], delta_theta[:-1]))
    constraints.append(v[1:] == v[:-1] + cp.multiply(dt[:-1], delta_v[:-1]))
    constraints.append(gamma[1:] == gamma[:-1] + cp.multiply(dt[:-1], delta_gamma[:-1]))
    constraints.append(psi[1:] == psi[:-1] + cp.multiply(dt[:-1], delta_psi[:-1]))

# ============================================================================
# Objective: Maximize Cross-Range (final latitude)
# ============================================================================

objective = cp.Maximize(theta[N-1])

# ============================================================================
# Problem Definition and Solve
# ============================================================================

problem = cp.Problem(objective, constraints)

# Initial guess (linear interpolation)
x_s = np.array([h_s, phi_s, theta_s, v_s, gamma_s, psi_s, alpha_s, beta_s])
x_t = np.array([h_t, phi_s, theta_s, v_t, gamma_t, psi_s, alpha_s, beta_s])
t_interp = np.linspace(0, 1, N)

for i, var in enumerate([scaled_h, phi, theta, scaled_v, gamma, psi, alpha, beta]):
    var.value = x_s[i] + t_interp * (x_t[i] - x_s[i])

# Solve with IPOPT
result = problem.solve(
    solver=cp.IPOPT,
    nlp=True,
    verbose=True,
    derivative_test='second-order',
)

print(f"\nOptimization Status: {problem.status}")
print(f"Final latitude θ = {np.rad2deg(theta.value[-1]):.2f}°")
print(f"Total time: {np.sum(dt.value):.1f} seconds")

# ============================================================================
# Helper Functions for Plotting
# ============================================================================

def compute_heating(h_val, v_val, alpha_val):
    """Compute aerodynamic heating on wing leading edge."""
    rho_val = RHO0 * np.exp(-h_val / HR)
    q_r = 17700 * np.sqrt(rho_val) * (0.0001 * v_val)**3.07
    alpha_deg_val = np.rad2deg(alpha_val)
    q_a = C0 + C1 * alpha_deg_val + C2 * alpha_deg_val**2 + C3 * alpha_deg_val**3
    return q_a * q_r

# ============================================================================
# Plotting Results
# ============================================================================

# Time vector
ts = np.cumsum(np.concatenate([[0], dt.value]))[:-1]

# Extract values
h_vals = scaled_h.value
phi_vals = phi.value
theta_vals = theta.value
v_vals = scaled_v.value
gamma_vals = gamma.value
psi_vals = psi.value
alpha_vals = alpha.value
beta_vals = beta.value

# Compute heating
h_full = h_vals * 1e5
v_full = v_vals * 1e4
heating = compute_heating(h_full, v_full, alpha_vals)

# Create comprehensive plot
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# State variables
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ts, h_vals, linewidth=2)
ax1.set_title('Altitude (100,000 ft)')
ax1.set_xlabel('Time (s)')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ts, np.rad2deg(phi_vals), linewidth=2)
ax2.set_title('Longitude (deg)')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(ts, np.rad2deg(theta_vals), linewidth=2)
ax3.set_title('Latitude (deg)')
ax3.set_xlabel('Time (s)')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(ts, v_vals, linewidth=2)
ax4.set_title('Velocity (10,000 ft/sec)')
ax4.set_xlabel('Time (s)')
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(ts, np.rad2deg(gamma_vals), linewidth=2)
ax5.set_title('Flight Path Angle (deg)')
ax5.set_xlabel('Time (s)')
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(ts, np.rad2deg(psi_vals), linewidth=2)
ax6.set_title('Azimuth (deg)')
ax6.set_xlabel('Time (s)')
ax6.grid(True, alpha=0.3)

# Control variables
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(ts, np.rad2deg(alpha_vals), linewidth=2)
ax7.set_title('Angle of Attack (deg)')
ax7.set_xlabel('Time (s)')
ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(ts, np.rad2deg(beta_vals), linewidth=2)
ax8.set_title('Bank Angle (deg)')
ax8.set_xlabel('Time (s)')
ax8.grid(True, alpha=0.3)

ax9 = fig.add_subplot(gs[2, 2])
ax9.plot(ts, heating, linewidth=2)
ax9.set_title('Heating (BTU/ft²/sec)')
ax9.set_xlabel('Time (s)')
ax9.grid(True, alpha=0.3)

plt.savefig('shuttle_reentry_states.png', dpi=150, bbox_inches='tight')
plt.show()

# 3D trajectory plot
fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection='3d')
ax.plot(np.rad2deg(phi_vals), np.rad2deg(theta_vals), h_vals, linewidth=2)
ax.set_xlabel('Longitude (deg)')
ax.set_ylabel('Latitude (deg)')
ax.set_zlabel('Altitude (100,000 ft)')
ax.set_title('Space Shuttle Reentry Trajectory')
plt.savefig('shuttle_reentry_3d.png', dpi=150, bbox_inches='tight')
plt.show()