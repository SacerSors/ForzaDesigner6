import math
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from fd6.shapegen.shapes.ellipse import RotatedEllipse
from fd6.shapegen.scoring import score_shape

class DifferentiableRasterizer(nn.Module):
    def __init__(self, height, width, sharpness=40.0):
        super().__init__()
        self.height = height
        self.width = width
        self.sharpness = sharpness

        y, x = torch.meshgrid(
            torch.linspace(0, 1, height),
            torch.linspace(0, 1, width),
            indexing='ij'
        )
        self.register_buffer('grid_x', x.unsqueeze(0))
        self.register_buffer('grid_y', y.unsqueeze(0))

    def forward(self, params, base_canvas_tensor, edge_weight_tensor=None):
        N = params.shape[0]
        device = params.device

        cx = torch.sigmoid(params[:, 0]).view(N, 1, 1)
        cy = torch.sigmoid(params[:, 1]).view(N, 1, 1)
        rx = (torch.sigmoid(params[:, 2]) * 0.5 + 1e-4).view(N, 1, 1)
        ry = (torch.sigmoid(params[:, 3]) * 0.5 + 1e-4).view(N, 1, 1)

        theta = params[:, 4].view(N, 1, 1)
        # We don't optimize color in the renderer for now, just the shape itself.
        # But we need color for rendering to compute loss. Let's make it grayscale for shape matching if no target color,
        # or we just let it optimize color too.
        colors = torch.sigmoid(params[:, 5:8])
        alphas = torch.sigmoid(params[:, 8]).view(N, 1, 1)

        dx = self.grid_x - cx
        dy = self.grid_y - cy

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x_loc = dx * cos_t + dy * sin_t
        y_loc = -dx * sin_t + dy * cos_t

        ellipse_dist = (x_loc / rx)**2 + (y_loc / ry)**2
        d_ellipse = (torch.sqrt(ellipse_dist + 1e-8) - 1.0) * torch.min(rx, ry)
        shape_alpha = torch.sigmoid(-d_ellipse * self.sharpness) * alphas

        # If we just want to render each shape independently, we expand the canvas
        # and do batch operations to save memory and time
        # shape_alpha is [N, H, W]
        # colors is [N, 3] -> [N, 3, 1, 1]
        a_i = shape_alpha.unsqueeze(1) # [N, 1, H, W]
        c_i = colors.view(N, 3, 1, 1)

        # We assume base_canvas_tensor is [3, H, W]
        # Expand it to [N, 3, H, W]
        canvas = base_canvas_tensor.unsqueeze(0).expand(N, -1, -1, -1)

        canvas = c_i * a_i + canvas * (1.0 - a_i)

        return canvas

class PyTorchSearcher:
    """Experimental Differentiable Rendering backend for Forza Designer 6."""

    def __init__(self, target: np.ndarray, alpha_mask: Optional[np.ndarray], edge_weight: np.ndarray):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h, self.w = target.shape[:2]

        self.target_np = target
        self.alpha_mask_np = alpha_mask
        self.edge_weight_np = edge_weight

        # Normalize target to [0, 1] and shape to [C, H, W]
        target_norm = target.astype(np.float32) / 255.0
        self.target_tensor = torch.from_numpy(target_norm).permute(2, 0, 1).to(self.device)

        if edge_weight is not None:
            self.edge_weight_tensor = torch.from_numpy(edge_weight.astype(np.float32)).to(self.device)
        else:
            self.edge_weight_tensor = torch.ones((self.h, self.w), device=self.device)

        if alpha_mask is not None:
            alpha_norm = alpha_mask.astype(np.float32) / 255.0
            self.alpha_mask_tensor = torch.from_numpy(alpha_norm).unsqueeze(0).to(self.device)
        else:
            self.alpha_mask_tensor = torch.ones((1, self.h, self.w), device=self.device)

        self.renderer = DifferentiableRasterizer(self.h, self.w, sharpness=40.0).to(self.device)

    def search(self, canvas: np.ndarray, n_random: int, n_mutate: int,
               max_size_frac: Optional[float], rng: random.Random) -> tuple[float, Optional[RotatedEllipse]]:
        """
        Differentiable rendering search. We optimize a batch of random shapes simultaneously,
        but each shape is optimized independently against the current canvas to see which ONE shape
        improves the canvas the most.
        """
        # Convert current canvas
        canvas_norm = canvas.astype(np.float32) / 255.0
        canvas_tensor = torch.from_numpy(canvas_norm).permute(2, 0, 1).to(self.device)

        batch_size = max(1, n_random)

        def inv_sigmoid(x):
            return torch.log(x / (1.0 - x + 1e-5))

        # Initialize random params
        params = torch.randn((batch_size, 9), device=self.device) * 0.1
        params[:, 0:2] = inv_sigmoid(torch.rand((batch_size, 2), device=self.device)) # Random positions

        # Scale based on max_size_frac
        max_s = max_size_frac if max_size_frac else 0.5
        params[:, 2:4] = inv_sigmoid(torch.rand((batch_size, 2), device=self.device) * max_s + 0.01)

        params[:, 4] = torch.rand((batch_size,), device=self.device) * 3.14159 * 2.0 # Rotation
        params[:, 5:8] = inv_sigmoid(torch.rand((batch_size, 3), device=self.device)) # Random colors
        params[:, 8] = inv_sigmoid(torch.ones((batch_size,), device=self.device) * 0.5) # Alpha ~0.5

        params.requires_grad_(True)

        optimizer = torch.optim.Adam([params], lr=0.1)

        target_masked = self.target_tensor * self.alpha_mask_tensor

        # Limit batch size to avoid OOM
        max_batch = 32

        # We will split into chunks
        num_chunks = (batch_size + max_batch - 1) // max_batch

        # 1. First pass without gradients to find the top candidates (reduces work)
        with torch.no_grad():
            scores = []
            for i in range(num_chunks):
                start = i * max_batch
                end = min((i + 1) * max_batch, batch_size)
                p_chunk = params[start:end]

                rendered = self.renderer(p_chunk, canvas_tensor) # [B, 3, H, W]
                rendered_masked = rendered * self.alpha_mask_tensor.unsqueeze(0) # [B, 3, H, W]
                # target_masked is [3, H, W], broadcast to [B, 3, H, W]

                diff = (rendered_masked - target_masked.unsqueeze(0))**2 # [B, 3, H, W]
                loss = (diff.sum(dim=1) * self.edge_weight_tensor.unsqueeze(0)).mean(dim=(1, 2)) # [B]
                scores.append(loss)

            all_scores = torch.cat(scores)

        # 2. Select top K for gradient descent
        top_k = min(16, batch_size)
        _, top_indices = torch.topk(all_scores, top_k, largest=False)

        top_params = params[top_indices].clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([top_params], lr=0.1)

        # Optimize for 15 steps
        for step in range(15):
            optimizer.zero_grad()

            rendered = self.renderer(top_params, canvas_tensor) # [K, 3, H, W]
            rendered_masked = rendered * self.alpha_mask_tensor.unsqueeze(0)

            diff = (rendered_masked - target_masked.unsqueeze(0))**2
            loss_per_shape = (diff.sum(dim=1) * self.edge_weight_tensor.unsqueeze(0)).mean(dim=(1, 2))

            total_loss = loss_per_shape.sum()
            total_loss.backward()
            optimizer.step()

        # After optimization, evaluate only the top_k shapes on CPU to find the best one
        # using the exact same scoring logic as OpenCL/CPU paths.
        with torch.no_grad():
            final_params = top_params.detach()
            cx = torch.sigmoid(final_params[:, 0]).cpu().numpy() * self.w
            cy = torch.sigmoid(final_params[:, 1]).cpu().numpy() * self.h
            rx = (torch.sigmoid(final_params[:, 2]) * 0.5 + 1e-4).cpu().numpy() * self.w
            ry = (torch.sigmoid(final_params[:, 3]) * 0.5 + 1e-4).cpu().numpy() * self.h
            theta = final_params[:, 4].cpu().numpy() * (180.0 / math.pi)

        best_score = float('inf')
        best_shape = None

        for i in range(top_k):
            shape = RotatedEllipse(
                x=float(cx[i]), y=float(cy[i]),
                rx=float(rx[i]), ry=float(ry[i]),
                angle=float(theta[i]),
                color=(0, 0, 0, 128) # Color will be optimized by CPU score_shape
            )

            score, color = score_shape(
                shape, canvas, self.target_np, self.alpha_mask_np,
                edge_weight=self.edge_weight_np
            )
            if score < best_score:
                best_score = score
                if color is not None:
                    shape.color = color
                best_shape = shape

        if best_shape is None:
            return float('inf'), None

        return best_score, best_shape
