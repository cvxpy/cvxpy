import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

def plot_circles(centers, radius, L, filename="circle_packing.pdf"):
    """Plot circle packing solution."""
    square_size = float(L) * 2
    pi = np.pi
    ratio = pi * np.sum(np.square(radius)) / (square_size**2)
    n = centers.shape[1]

    # create plot to visualize the packing
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal", adjustable="box")
    fig.set_dpi(150)

    # draw circles
    for i in range(n):
        x_val = centers[0, i]
        y_val = centers[1, i]
        if x_val is None or y_val is None:
            msg = f"Circle center value not assigned for index {i}."
            raise ValueError(msg)
        circle = Circle(
            (float(x_val), float(y_val)),  # (x, y) center
            radius[i],  # radius
            fill=False,  # outline only
            ec="b",
            lw=1.2,  # edge color/width
        )
        ax.add_patch(circle)

    # draw square border
    border = Rectangle(
    (-square_size / 2, -square_size / 2),  # bottom-left
    square_size,
    square_size,  # width, height
    fill=False,
    ec="g",
    lw=1.5,
    )
    ax.add_patch(border)

    # limits and cosmetics
    ax.set_xlim(float(-square_size / 2), float(square_size / 2))
    ax.set_ylim(float(-square_size / 2), float(square_size / 2))
    #ax.set_xlabel("x")
    #ax.set_ylabel("y")
    #ax.set_title(f"Circle packing (ratio={ratio:.3f})")
    print(f"Circle packing (ratio={ratio:.3f})")
    plt.savefig(filename)
