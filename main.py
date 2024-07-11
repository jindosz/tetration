import torch
import numpy as np
import matplotlib.pyplot as plt

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_ITER = 500
ESCAPE_RADIUS = 1e+10

def compute_tetration_divergence_torch(n, x0, y0, eps=5e-3, max_iter=MAX_ITER, escape_radius=ESCAPE_RADIUS, device=_device):
    nx, ny = n, n  # 화면 비율을 1:1로 맞춤
    x = np.linspace(x0 - eps, x0 + eps, nx)
    y = np.linspace(y0 - eps, y0 + eps, ny)
    c = x[:, None] + 1j * y[None, :]
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    z = c.clone()
    divergence_map = torch.zeros(c.shape, dtype=torch.bool, device=device)
    for _ in range(max_iter):
        z = c ** z
        divergence_map |= (torch.abs(z) > escape_radius)

    numpy_map = divergence_map.cpu().numpy().T.astype(np.uint8) * 255
    return numpy_map

def on_click(event):
    if event.inaxes is not None:
        x, y = event.xdata, event.ydata
        print(f"Clicked coordinates: x={x}, y={y}")

        current_xlim = event.inaxes.get_xlim()
        current_ylim = event.inaxes.get_ylim()
        current_eps = (current_xlim[1] - current_xlim[0]) / 2

        zoom_factor = 4
        new_eps = current_eps / zoom_factor
        zoom_image(x, y, new_eps)

def zoom_image(x, y, eps):
    map_zoomed = compute_tetration_divergence_torch(resolution, x, y, eps)
    
    fig, ax = plt.subplots()
    ax.imshow(map_zoomed, cmap='gray', extent=[x - eps, x + eps, y - eps, y + eps])
    ax.axis('off')
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__ == "__main__":
    resolution = 1080
    map_torch = compute_tetration_divergence_torch(resolution, 0, 0, 5)
    
    fig, ax = plt.subplots()
    ax.imshow(map_torch, cmap='gray', extent=[-5, 5, -5, 5])
    ax.axis('off')
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.show()
