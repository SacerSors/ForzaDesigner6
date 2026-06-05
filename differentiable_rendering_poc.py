import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DifferentiableRasterizer(nn.Module):
    def __init__(self, height, width, sharpness=40.0):
        super().__init__()
        self.height = height
        self.width = width
        self.sharpness = sharpness

        # Erstelle ein Grid für die Pixelkoordinaten [0, 1]
        y, x = torch.meshgrid(
            torch.linspace(0, 1, height),
            torch.linspace(0, 1, width),
            indexing='ij'
        )
        self.register_buffer('grid_x', x.unsqueeze(0)) # [1, H, W]
        self.register_buffer('grid_y', y.unsqueeze(0)) # [1, H, W]

    def forward(self, params, is_rect):
        """
        params: Tensor der Form [N, 9] -> (cx, cy, rx, ry, theta, r, g, b, a)
        is_rect: Tensor der Form [N] -> 1.0 für Rechtecke, 0.0 für Ellipsen
        """
        N = params.shape[0]
        device = params.device

        # 1. Parameter über Sigmoid-Aktivierungen in gültige Wertebereiche zwingen
        cx = torch.sigmoid(params[:, 0]).view(N, 1, 1)        # [0, 1]
        cy = torch.sigmoid(params[:, 1]).view(N, 1, 1)        # [0, 1]
        # Max-Radius ist 0.5 (Form füllt das ganze Canvas), + 1e-4 gegen Div by Zero
        rx = (torch.sigmoid(params[:, 2]) * 0.5 + 1e-4).view(N, 1, 1)
        ry = (torch.sigmoid(params[:, 3]) * 0.5 + 1e-4).view(N, 1, 1)

        theta = params[:, 4].view(N, 1, 1)                    # Unbeschränkt (Radiant)
        colors = torch.sigmoid(params[:, 5:8])                # [N, 3] -> [0, 1] RGB
        alphas = torch.sigmoid(params[:, 8]).view(N, 1, 1)    # [0, 1] Transparenz

        is_rect = is_rect.view(N, 1, 1).float()

        # 2. Pixelkoordinaten relativ zum Zentrum der Form berechnen
        dx = self.grid_x - cx
        dy = self.grid_y - cy

        # 3. Rotation anwenden (auf die lokalen Koordinaten)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x_loc = dx * cos_t + dy * sin_t
        y_loc = -dx * sin_t + dy * cos_t

        # --- SDF für Ellipsen ---
        # Analytische Distanz (approximiert für Ellipsen)
        ellipse_dist = (x_loc / rx)**2 + (y_loc / ry)**2
        # Wir rechnen das in eine distanzähnliche Metrik um. +1e-8 verhindert NaNs bei sqrt(0)
        d_ellipse = (torch.sqrt(ellipse_dist + 1e-8) - 1.0) * torch.min(rx, ry)
        alpha_ellipse = torch.sigmoid(-d_ellipse * self.sharpness)

        # --- SDF für Rechtecke (Box SDF) ---
        d_rect = torch.max(torch.abs(x_loc) - rx, torch.abs(y_loc) - ry)
        alpha_rect = torch.sigmoid(-d_rect * self.sharpness)

        # Mischen je nach gewählter Form
        shape_alpha = is_rect * alpha_rect + (1.0 - is_rect) * alpha_ellipse
        shape_alpha = shape_alpha * alphas # Mit genereller Transparenz multiplizieren

        # --- 4. Alpha-Compositing (Back-to-Front Rendering) ---
        canvas = torch.zeros((3, self.height, self.width), device=device) # Schwarzer Hintergrund

        for i in range(N):
            a_i = shape_alpha[i].unsqueeze(0) # [1, H, W]
            c_i = colors[i].view(3, 1, 1)     # [3, 1, 1]
            # Standard Überblendung: Neu * Alpha + Alt * (1 - Alpha)
            canvas = c_i * a_i + canvas * (1.0 - a_i)

        return canvas

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Laufe auf: {device}")

    H, W = 64, 64 # Kleine Auflösung für den PoC
    renderer = DifferentiableRasterizer(H, W, sharpness=40.0).to(device)

    # --- 1. Ein Dummy-Zielbild erstellen ---
    with torch.no_grad():
        def inv_sigmoid(x):
            return torch.log(x / (1.0 - x + 1e-5))

        gt_params = torch.zeros((2, 9), device=device)
        # Rotes Rechteck
        gt_params[0, 0:2] = inv_sigmoid(torch.tensor([0.3, 0.3])) # X, Y
        gt_params[0, 2:4] = inv_sigmoid(torch.tensor([0.15, 0.15]) * 2.0) # Scale X, Y
        gt_params[0, 4] = 0.2 # Rotation
        gt_params[0, 5:8] = inv_sigmoid(torch.tensor([1.0, 0.2, 0.2])) # RGB
        gt_params[0, 8] = inv_sigmoid(torch.tensor(1.0)) # Alpha

        # Blaue Ellipse
        gt_params[1, 0:2] = inv_sigmoid(torch.tensor([0.7, 0.6])) # X, Y
        gt_params[1, 2:4] = inv_sigmoid(torch.tensor([0.1, 0.2]) * 2.0) # Scale X, Y
        gt_params[1, 4] = -0.5 # Rotation
        gt_params[1, 5:8] = inv_sigmoid(torch.tensor([0.2, 0.2, 1.0])) # RGB
        gt_params[1, 8] = inv_sigmoid(torch.tensor(1.0)) # Alpha

        gt_is_rect = torch.tensor([1.0, 0.0], device=device)
        target_img = renderer(gt_params, gt_is_rect)

    # --- 2. Learnable Parameter initialisieren ---
    # Wir werfen 4 zufällige Formen (2 Rechtecke, 2 Ellipsen) ins Rennen
    num_shapes = 4
    params = torch.randn((num_shapes, 9), device=device) * 0.1
    params[:, 0:2] = 0.0 # Startposition in der Mitte
    params[:, 8] = 0.0   # Semi-transparent zu Beginn
    params = nn.Parameter(params) # PyTorch weiß nun: Hier müssen Gradienten berechnet werden!

    is_rect = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)

    # Initiales Bild für den Vorher/Nachher Plot speichern
    with torch.no_grad():
        initial_img = renderer(params, is_rect)

    # --- 3. Die Optimierung (Der ForzaDesigner Magie-Loop) ---
    optimizer = torch.optim.Adam([params], lr=0.05)
    criterion = nn.MSELoss()

    print("Starte Optimierung...")
    for epoch in range(500):
        optimizer.zero_grad()

        # 1. Bild zeichnen (über den Differentiable Renderer)
        rendered_img = renderer(params, is_rect)

        # 2. Fehler zum Zielbild berechnen
        loss = criterion(rendered_img, target_img)

        # 3. Den Gradienten berechnen (Backpropagation fließt durch Pixel in die Formen)
        loss.backward()

        # 4. Form-Parameter aktualisieren (Verschieben, drehen, färben)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:3d} - Loss: {loss.item():.6f}")

    # --- 4. Visualisierung ---
    target_np = target_img.detach().cpu().permute(1, 2, 0).numpy()
    initial_np = initial_img.detach().cpu().permute(1, 2, 0).numpy()
    final_np = rendered_img.detach().cpu().permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(target_np)
    axs[0].set_title("Zielbild (Target)")
    axs[1].imshow(initial_np)
    axs[1].set_title("Initiale Formen")
    axs[2].imshow(final_np)
    axs[2].set_title("Nach 500 Epochen")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('differentiable_rendering_result.png')
    print("Ergebnis als 'differentiable_rendering_result.png' gespeichert.")

if __name__ == "__main__":
    main()
