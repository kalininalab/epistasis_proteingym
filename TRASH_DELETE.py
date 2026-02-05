import numpy as np
import matplotlib.pyplot as plt

# Create grid
nx, ny = 300, 300
x = np.linspace(-3, 3, nx)
y = np.linspace(-3, 3, ny)
X, Y = np.meshgrid(x, y)

# Define three Gaussian peaks
peaks = [
    {"center": (-1.5, -1.0), "sigma": 0.6, "amp": 1.0},
    {"center": (0.5,  1.0),  "sigma": 0.7, "amp": 1.3},
    {"center": (1.8, -0.5),  "sigma": 0.5, "amp": 0.9},
]

Z = np.zeros_like(X)
for p in peaks:
    x0, y0 = p["center"]
    sigma = p["sigma"]
    amp = p["amp"]
    Z += amp * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

# Create contour plot (dotted black lines)
fig, ax = plt.subplots(figsize=(5, 4))
cs = ax.contour(X, Y, Z, levels=15, colors="black", linestyles="dotted")

# Mark the three peaks with black plus symbols
for p in peaks:
    x0, y0 = p["center"]
    ax.plot(x0, y0, marker="+", markersize=10, markeredgewidth=2, color="black")

ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(True)

plt.tight_layout()

png_path = "three_black_peaks_contour.png"
pdf_path = "three_black_peaks_contour.pdf"
fig.savefig("plot.svg", bbox_inches="tight")

fig.savefig(png_path, dpi=600)
fig.savefig(pdf_path)
plt.close(fig)

png_path, pdf_path