import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.sparse import csr_matrix, diags


def branch_matrices():
    N = 9
    # Branch data: (from_bus, to_bus, resistance, reactance, susceptance)
    branch_data = np.array([
        [0, 3, 0.0, 0.0576, 0.0],
        [3, 4, 0.017, 0.092, 0.158],
        [5, 4, 0.039, 0.17, 0.358],
        [2, 5, 0.0, 0.0586, 0.0],
        [5, 6, 0.0119, 0.1008, 0.209],
        [7, 6, 0.0085, 0.072, 0.149],
        [1, 7, 0.0, 0.0625, 0.0],
        [7, 8, 0.032, 0.161, 0.306],
        [3, 8, 0.01, 0.085, 0.176],
    ], dtype=float)

    G_series = np.zeros((N,N))
    B_series = np.zeros((N,N))
    G_shunt = np.zeros((N,N))
    B_shunt = np.zeros((N,N))
    base_MVA = 100

    for f,t,r,x,bc in branch_data:
        z = (r + 1j*x) / base_MVA
        y = 1.0/z                     # pu admittance
        g, b = y.real, y.imag

        f, t = int(f), int(t)
        # series branch
        G_series[f,t] = G_series[t,f] = g
        B_series[f,t] = B_series[t,f] = b

        # line-charging shunt (split equally)
        b_shunt = (bc / 2.0) * base_MVA
        B_shunt[f,f] += b_shunt
        B_shunt[t,t] += b_shunt

    return G_series, B_series, G_shunt, B_shunt

def create_admittance_matrices():
    N = 9
    # Branch data: (from_bus, to_bus, resistance, reactance, susceptance)
    branch_data = np.array([
        [0, 3, 0.0, 0.0576, 0.0],
        [3, 4, 0.017, 0.092, 0.158],
        [5, 4, 0.039, 0.17, 0.358],
        [2, 5, 0.0, 0.0586, 0.0],
        [5, 6, 0.0119, 0.1008, 0.209],
        [7, 6, 0.0085, 0.072, 0.149],
        [1, 7, 0.0, 0.0625, 0.0],
        [7, 8, 0.032, 0.161, 0.306],
        [3, 8, 0.01, 0.085, 0.176],
    ])

    M = branch_data.shape[0]  # Number of branches
    base_MVA = 100

    # Build incidence matrix A
    from_bus = branch_data[:, 0].astype(int)
    to_bus = branch_data[:, 1].astype(int)
    A = csr_matrix((np.ones(M), (from_bus, np.arange(M))), shape=(N, M)) + \
        csr_matrix((-np.ones(M), (to_bus, np.arange(M))), shape=(N, M))

    # Network impedance
    z = (branch_data[:, 2] + 1j * branch_data[:, 3]) / base_MVA

    # Bus admittance matrix Y_0
    Y_0 = A @ diags(1.0 / z) @ A.T

    # Shunt admittance from line charging
    y_sh = 0.5 * (1j * branch_data[:, 4]) * base_MVA
    Y_sh_diag = np.array((A @ diags(y_sh) @ A.T).diagonal()).flatten()
    Y_sh = diags(Y_sh_diag)

    # Full bus admittance matrix
    #Y = Y_0 + Y_sh
    #Y_dense = Y.toarray()

    # Extract conductance and susceptance matrices
    G0 = np.real(Y_0.toarray())  # Conductance matrix
    B0 = np.imag(Y_0.toarray())  # Susceptance matrix
    G_sh = np.real(Y_sh.toarray())  # Shunt conductance
    B_sh = np.imag(Y_sh.toarray())  #

    return G0, B0, G_sh, B_sh

def plot_power_flows(P_mat, node_labels=None, pos=None, tol=1e-6):
    """
    Plot directed power flows given a matrix P_mat (NxN).

    Layout behavior:
    - If `pos` is None, place nodes 0,1,2 on an outer ring and nodes 3..N-1 on an inner ring.
    - Otherwise use the provided `pos` mapping (index -> (x,y)).

    Visual:
    - Draw directed arrow for every entry with abs(value) > tol.
    - Positive P_mat[i,j] is drawn as arrow from i -> j.
    - Self-flows (i==j) are drawn as loops under the node with both endpoints on the circumference.
    - Off-diagonal arrows are offset perpendicular to the edge so opposite-sign flows don't overlap.
    """
    P_mat = np.asarray(P_mat)
    N = P_mat.shape[0]

    # Default circular layout if no positions given
    if pos is None:
        # base radius scales with N (keeps previous behavior)
        base_radius = max(3, N / 1.5)

        # Outer ring: nodes 0,1,2 (if present)
        outer_indices = [0, 2, 1]
        outer_indices = [i for i in outer_indices if i < N]
        n_outer = len(outer_indices)

        # Inner ring: remaining nodes
        inner_indices = [i for i in range(N) if i not in outer_indices]
        n_inner = len(inner_indices)

        pos = {}
        # Place outer nodes evenly on circle of radius slightly larger than base
        if n_outer > 0:
            outer_radius = base_radius * 1.25
            outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
            for k, node in enumerate(outer_indices):
                a = outer_angles[k]
                pos[node] = (np.cos(a) * outer_radius, np.sin(a) * outer_radius)

        # Place inner nodes evenly on smaller circle
        if n_inner > 0:
            inner_radius = base_radius * 0.6
            # rotate inner ring slightly so nodes don't align with outer ones
            inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + \
                (np.pi / n_inner if n_inner>0 else 0.0)
            for k, node in enumerate(inner_indices):
                a = inner_angles[k]
                pos[node] = (np.cos(a) * inner_radius, np.sin(a) * inner_radius)

    fig, ax = plt.subplots(figsize=(8, 8))
    node_radius = 0.6

    # Draw nodes
    for i, (x, y) in pos.items():
        circ = Circle((x, y), node_radius, color='lightgray', zorder=2, ec='k', lw=0.6)
        ax.add_patch(circ)
        label = node_labels[i] if node_labels is not None else f"{i+1}"
        ax.text(x, y, label, ha='center', va='center', zorder=3, fontsize=9)

    # Determine scaling for linewidths
    max_flow = np.max(np.abs(P_mat))
    if max_flow <= 0:
        max_flow = 1.0

    # Base perpendicular offset (fraction of node_radius)
    base_offset = node_radius * 0.35

    # Draw directed edges for all nonzero entries (including self-loops)
    for i in range(N):
        for j in range(N):
            val = P_mat[i, j]
            if np.abs(val) <= tol:
                continue

            if i == j:
                # Self-loop: both start and end on the node circumference and drawn under the node.
                x, y = pos[i]
                # Start at lower-right quadrant, end at bottommost point (under the node)
                start_frac = 0.7
                loop_start = (x + node_radius * start_frac, y - node_radius * start_frac)
                loop_end = (x, y - node_radius * 0.98)
                loop = FancyArrowPatch(loop_start,
                                       loop_end,
                                       connectionstyle="arc3,rad=-0.9",
                                       arrowstyle='->',
                                       mutation_scale=18,
                                       shrinkA=0,
                                       shrinkB=0,
                                       linewidth=1 + 3 * np.abs(val) / max_flow,
                                       color='C0',
                                       zorder=0,   # underneath the node
                                       alpha=0.95)
                ax.add_patch(loop)

                # place label under the node near the loop
                label_x = x
                label_y = y - node_radius * 1.15
                ax.text(label_x, label_y, f"{val:.2f}", fontsize=8,
                        ha='center', va='center', zorder=1,
                        bbox=dict(facecolor='white', edgecolor='none', pad=0.5, alpha=0.9))
            else:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                vec = np.array([x2 - x1, y2 - y1])
                dist = np.linalg.norm(vec)
                if dist == 0:
                    continue
                dir_vec = vec / dist
                # Perpendicular (normalized)
                perp = np.array([-dir_vec[1], dir_vec[0]])

                # offset magnitude scales with node size and flow magnitude
                offset_mag = base_offset * (1.0 + 0.5 * np.abs(val) / max_flow)

                # extra side shift to separate opposite-sign arrows more clearly
                extra_side_frac = 0.25  # fraction of node_radius to add as extra separation
                extra_side = node_radius * extra_side_frac

                # shift direction depends on sign of the value: positives one side, negatives the
                #  other
                shift = perp * ((offset_mag + extra_side) * np.sign(val))

                # shorten so arrows don't overlap node circles, then apply perpendicular shift
                start = np.array([x1, y1]) + dir_vec * (node_radius * 0.95) + shift
                end = np.array([x2, y2]) - dir_vec * (node_radius * 0.95) + shift

                lw = 1 + 3 * np.abs(val) / max_flow
                # explicit colors: C1 = positive, C2 = negative (change if you prefer)
                color = 'C1' if val > 0 else 'C2'
        
                arrow = FancyArrowPatch(start, end,
                                        arrowstyle='-|>',
                                        mutation_scale=12,
                                        linewidth=lw,
                                        color=color,
                                        alpha=0.8)
                ax.add_patch(arrow)
                # label magnitude at shifted midpoint (label above arrows)
                mid = (start + end) / 2.0
                ax.text(mid[0], mid[1], f"{val:.2f}", fontsize=7,
                        ha='center', va='center', backgroundcolor='white', zorder=4)

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')
    plt.title("Real power flows (P_ij) — positive value shown as arrow i → j")
    plt.savefig("power_flows.pdf")

def plot_power_flows_OLD(P_mat, node_labels=None, pos=None, tol=1e-6):
    """
    Plot directed power flows given a matrix P_mat (NxN).
    - Draw directed arrow for every entry with abs(value) > tol.
    - Positive P_mat[i,j] is drawn as arrow from i -> j.
    - Self-flows (i==j) are drawn as loops next to the node with arrowhead pointing toward the node.
    """
    P_mat = np.asarray(P_mat)
    N = P_mat.shape[0]

    # Default circular layout if no positions given
    if pos is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radius = max(3, N / 1.5)  # scale radius a bit with N
        pos = {i: (np.cos(angles[i]) * radius, np.sin(angles[i]) * radius) for i in range(N)}

    fig, ax = plt.subplots(figsize=(8, 8))
    node_radius = 0.6

    # Draw nodes
    for i, (x, y) in pos.items():
        circ = Circle((x, y), node_radius, color='lightgray', zorder=2, ec='k', lw=0.6)
        ax.add_patch(circ)
        label = node_labels[i] if node_labels is not None else f"{i+1}"
        ax.text(x, y, label, ha='center', va='center', zorder=3, fontsize=9)

    # Determine scaling for linewidths
    max_flow = np.max(np.abs(P_mat))
    if max_flow <= 0:
        max_flow = 1.0

    # Draw directed edges for all nonzero entries (including self-loops)
    for i in range(N):
        for j in range(N):
            val = P_mat[i, j]
            if np.abs(val) <= tol:
                continue

            if i == j:
                # Self-loop: both start and end on the node circumference so arrowhead
                # appears to point into the node.
                x, y = pos[i]
                # Points placed on node circumference (angles: 0 rad and pi/2 rad)
                loop_start = (x + node_radius * 0.98, y)          # rightmost point on circumference
                loop_end = (x, y + node_radius * 0.98)            # topmost point on circumference
                # Draw an outward arc from loop_start to loop_end with arrowhead at loop_end
                loop = FancyArrowPatch(loop_start,
                                       loop_end,
                                       connectionstyle="arc3,rad=0.9",
                                       arrowstyle='->',
                                       mutation_scale=18,
                                       shrinkA=0,
                                       shrinkB=0,
                                       linewidth=1 + 3 * np.abs(val) / max_flow,
                                       color='C0',
                                       zorder=1,
                                       alpha=0.95)
                ax.add_patch(loop)
                # place label near the outer quadrant of the loop
                label_x = x + node_radius * 0.9
                label_y = y + node_radius * 0.6
                ax.text(label_x, label_y, f"{val:.2f}", fontsize=8,
                        ha='center', va='center', zorder=4,
                        bbox=dict(facecolor='white', edgecolor='none', pad=0.5, 
                                  alpha=0.8))
            else:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                vec = np.array([x2 - x1, y2 - y1])
                dist = np.linalg.norm(vec)
                if dist == 0:
                    continue
                dir_vec = vec / dist
                # shorten so arrows don't overlap node circles
                start = np.array([x1, y1]) + dir_vec * (node_radius * 0.95)
                end = np.array([x2, y2]) - dir_vec * (node_radius * 0.95)
                lw = 1 + 3 * np.abs(val) / max_flow
                color = 'C1' if val > 0 else 'C2'  # positive vs negative color
                arrow = FancyArrowPatch(start, end,
                                        arrowstyle='-|>',
                                        mutation_scale=12,
                                        linewidth=lw,
                                        color=color,
                                        alpha=0.8,
                                        zorder=1)
                ax.add_patch(arrow)
                # label magnitude at midpoint
                mid = (start + end) / 2.0
                ax.text(mid[0], mid[1], f"{val:.2f}", fontsize=7,
                        ha='center', va='center', backgroundcolor='white', zorder=4)

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')
    plt.title("Real power flows (P_ij) — positive value shown as arrow i → j")
    plt.savefig("power_flows.pdf")