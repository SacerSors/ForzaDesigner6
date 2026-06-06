import math
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

try:
    import torch_directml
    HAS_DIRECTML = True
except (ImportError, OSError):
    HAS_DIRECTML = False

from fd6.shapegen.shapes.ellipse import RotatedEllipse
from fd6.shapegen.shapes.rectangle import Rectangle, RotatedRectangle
from fd6.shapegen.shapes.triangle import Triangle
from fd6.shapegen.shapes.base import Shape
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

    def forward(self, params, base_canvas_tensor, shape_type="rotated_ellipse", edge_weight_tensor=None):
        N = params.shape[0]
        device = params.device

        if shape_type in ("rotated_ellipse", "rotated_rectangle"):
            cx = torch.sigmoid(params[:, 0]).view(N, 1, 1)
            cy = torch.sigmoid(params[:, 1]).view(N, 1, 1)
            rx = (torch.sigmoid(params[:, 2]) * 0.5 + 1e-4).view(N, 1, 1)
            ry = (torch.sigmoid(params[:, 3]) * 0.5 + 1e-4).view(N, 1, 1)
            theta = params[:, 4].view(N, 1, 1)
            colors = torch.sigmoid(params[:, 5:8])
            alphas = torch.sigmoid(params[:, 8]).view(N, 1, 1)

            dx = self.grid_x - cx
            dy = self.grid_y - cy
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            x_loc = dx * cos_t + dy * sin_t
            y_loc = -dx * sin_t + dy * cos_t

            if shape_type == "rotated_ellipse":
                ellipse_dist = (x_loc / rx)**2 + (y_loc / ry)**2
                dist = (torch.sqrt(ellipse_dist + 1e-8) - 1.0) * torch.min(rx, ry)
            else:
                dist = torch.max(torch.abs(x_loc) - rx, torch.abs(y_loc) - ry)

            shape_alpha = torch.sigmoid(-dist * self.sharpness) * alphas

        elif shape_type == "rectangle":
            cx = torch.sigmoid(params[:, 0]).view(N, 1, 1)
            cy = torch.sigmoid(params[:, 1]).view(N, 1, 1)
            hw = (torch.sigmoid(params[:, 2]) * 0.5 + 1e-4).view(N, 1, 1)
            hh = (torch.sigmoid(params[:, 3]) * 0.5 + 1e-4).view(N, 1, 1)
            colors = torch.sigmoid(params[:, 4:7])
            alphas = torch.sigmoid(params[:, 7]).view(N, 1, 1)

            dx = self.grid_x - cx
            dy = self.grid_y - cy
            dist = torch.max(torch.abs(dx) - hw, torch.abs(dy) - hh)
            shape_alpha = torch.sigmoid(-dist * self.sharpness) * alphas

        elif shape_type == "triangle":
            x1 = torch.sigmoid(params[:, 0]).view(N, 1, 1)
            y1 = torch.sigmoid(params[:, 1]).view(N, 1, 1)
            x2 = torch.sigmoid(params[:, 2]).view(N, 1, 1)
            y2 = torch.sigmoid(params[:, 3]).view(N, 1, 1)
            x3 = torch.sigmoid(params[:, 4]).view(N, 1, 1)
            y3 = torch.sigmoid(params[:, 5]).view(N, 1, 1)
            colors = torch.sigmoid(params[:, 6:9])
            alphas = torch.sigmoid(params[:, 9]).view(N, 1, 1)

            # Signed distance to edge
            def edge_dist(px, py, ax, ay, bx, by):
                return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

            d1 = edge_dist(self.grid_x, self.grid_y, x1, y1, x2, y2)
            d2 = edge_dist(self.grid_x, self.grid_y, x2, y2, x3, y3)
            d3 = edge_dist(self.grid_x, self.grid_y, x3, y3, x1, y1)

            # Smooth step for the edges. In SDF, positive is outside, negative is inside.
            # However, orientation matters for triangles.
            # If d1, d2, d3 are all same sign, it's inside.
            mask1_pos = torch.sigmoid(d1 * self.sharpness)
            mask2_pos = torch.sigmoid(d2 * self.sharpness)
            mask3_pos = torch.sigmoid(d3 * self.sharpness)
            mask_all_pos = mask1_pos * mask2_pos * mask3_pos

            mask1_neg = torch.sigmoid(-d1 * self.sharpness)
            mask2_neg = torch.sigmoid(-d2 * self.sharpness)
            mask3_neg = torch.sigmoid(-d3 * self.sharpness)
            mask_all_neg = mask1_neg * mask2_neg * mask3_neg

            shape_alpha = (mask_all_pos + mask_all_neg) * alphas

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
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif HAS_DIRECTML and torch_directml.is_available():
            self.device = torch_directml.device()

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

    def search(self, canvas: np.ndarray, types: list[str], n_random: int, n_mutate: int,
               max_size_frac: Optional[float], rng: random.Random) -> tuple[float, Optional[Shape]]:
        """
        Differentiable rendering search. We optimize a batch of random shapes simultaneously,
        but each shape is optimized independently against the current canvas to see which ONE shape
        improves the canvas the most.
        """
        # Convert current canvas
        canvas_norm = canvas.astype(np.float32) / 255.0
        canvas_tensor = torch.from_numpy(canvas_norm).permute(2, 0, 1).to(self.device)

        batch_size = max(1, n_random)

        shape_type = types[0] if types else "rotated_ellipse"

        def inv_sigmoid(x):
            return torch.log(x / (1.0 - x + 1e-5))

        # Initialize random params.
        # For rotated_ellipse: 9 params (cx, cy, rx, ry, theta, r, g, b, a)
        # For rectangle: 8 params (cx, cy, hw, hh, r, g, b, a)
        # For rotated_rectangle: 9 params (cx, cy, hw, hh, theta, r, g, b, a)
        # For triangle: 10 params (x1, y1, x2, y2, x3, y3, r, g, b, a)

        num_params = 9
        if shape_type == "rectangle":
            num_params = 8
        elif shape_type == "triangle":
            num_params = 10

        # Initialize randomly on CPU first (DirectML often lacks native RNG implementations)
        cpu_dev = torch.device('cpu')
        params_cpu = torch.randn((batch_size, num_params), device=cpu_dev) * 0.1
        max_s = max_size_frac if max_size_frac else 0.5

        if shape_type in ("rotated_ellipse", "rotated_rectangle"):
            params_cpu[:, 0:2] = inv_sigmoid(torch.rand((batch_size, 2), device=cpu_dev))
            params_cpu[:, 2:4] = inv_sigmoid(torch.rand((batch_size, 2), device=cpu_dev) * max_s + 0.01)
            params_cpu[:, 4] = torch.rand((batch_size,), device=cpu_dev) * 3.14159 * 2.0
            params_cpu[:, 5:8] = inv_sigmoid(torch.rand((batch_size, 3), device=cpu_dev))
            params_cpu[:, 8] = inv_sigmoid(torch.ones((batch_size,), device=cpu_dev) * 0.5)
        elif shape_type == "rectangle":
            params_cpu[:, 0:2] = inv_sigmoid(torch.rand((batch_size, 2), device=cpu_dev))
            params_cpu[:, 2:4] = inv_sigmoid(torch.rand((batch_size, 2), device=cpu_dev) * max_s + 0.01)
            params_cpu[:, 4:7] = inv_sigmoid(torch.rand((batch_size, 3), device=cpu_dev))
            params_cpu[:, 7] = inv_sigmoid(torch.ones((batch_size,), device=cpu_dev) * 0.5)
        elif shape_type == "triangle":
            params_cpu[:, 0:6] = inv_sigmoid(torch.rand((batch_size, 6), device=cpu_dev))
            # keep points somewhat grouped by modifying initialization to be around a center
            cxcy = torch.rand((batch_size, 2), device=cpu_dev)
            spread = max_s * 0.5
            for v in range(3):
                pts = cxcy + (torch.rand((batch_size, 2), device=cpu_dev) - 0.5) * spread
                pts = torch.clamp(pts, 0.01, 0.99)
                params_cpu[:, v*2:v*2+2] = inv_sigmoid(pts)

            params_cpu[:, 6:9] = inv_sigmoid(torch.rand((batch_size, 3), device=cpu_dev))
            params_cpu[:, 9] = inv_sigmoid(torch.ones((batch_size,), device=cpu_dev) * 0.5)

        params = params_cpu.to(self.device).requires_grad_(True)

        optimizer = torch.optim.Adam([params], lr=0.1, foreach=False)

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

                rendered = self.renderer(p_chunk, canvas_tensor, shape_type) # [B, 3, H, W]
                rendered_masked = rendered * self.alpha_mask_tensor.unsqueeze(0) # [B, 3, H, W]
                # target_masked is [3, H, W], broadcast to [B, 3, H, W]

                diff = (rendered_masked - target_masked.unsqueeze(0))**2 # [B, 3, H, W]
                loss = (diff.sum(dim=1) * self.edge_weight_tensor.unsqueeze(0)).mean(dim=(1, 2)) # [B]
                scores.append(loss)

            all_scores = torch.cat(scores)

        # 2. Select top K for gradient descent
        top_k = min(16, batch_size)
        # DirectML does not implement topk for all data types. Using sort is safer.
        _, sorted_indices = torch.sort(all_scores, descending=False)
        top_indices = sorted_indices[:top_k]

        top_params = params[top_indices].clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([top_params], lr=0.1, foreach=False)

        # Optimize for 15 steps
        for step in range(15):
            optimizer.zero_grad()

            rendered = self.renderer(top_params, canvas_tensor, shape_type) # [K, 3, H, W]
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

            if shape_type in ("rotated_ellipse", "rotated_rectangle"):
                cx = torch.sigmoid(final_params[:, 0]).cpu().numpy() * self.w
                cy = torch.sigmoid(final_params[:, 1]).cpu().numpy() * self.h
                rx = (torch.sigmoid(final_params[:, 2]) * 0.5 + 1e-4).cpu().numpy() * self.w
                ry = (torch.sigmoid(final_params[:, 3]) * 0.5 + 1e-4).cpu().numpy() * self.h
                theta = final_params[:, 4].cpu().numpy() * (180.0 / math.pi)
            elif shape_type == "rectangle":
                cx = torch.sigmoid(final_params[:, 0]).cpu().numpy() * self.w
                cy = torch.sigmoid(final_params[:, 1]).cpu().numpy() * self.h
                hw = (torch.sigmoid(final_params[:, 2]) * 0.5 + 1e-4).cpu().numpy() * self.w
                hh = (torch.sigmoid(final_params[:, 3]) * 0.5 + 1e-4).cpu().numpy() * self.h
            elif shape_type == "triangle":
                x1 = torch.sigmoid(final_params[:, 0]).cpu().numpy() * self.w
                y1 = torch.sigmoid(final_params[:, 1]).cpu().numpy() * self.h
                x2 = torch.sigmoid(final_params[:, 2]).cpu().numpy() * self.w
                y2 = torch.sigmoid(final_params[:, 3]).cpu().numpy() * self.h
                x3 = torch.sigmoid(final_params[:, 4]).cpu().numpy() * self.w
                y3 = torch.sigmoid(final_params[:, 5]).cpu().numpy() * self.h

        best_score = float('inf')
        best_shape = None

        for i in range(top_k):
            if shape_type == "rotated_ellipse":
                shape = RotatedEllipse(
                    x=float(cx[i]), y=float(cy[i]),
                    rx=float(rx[i]), ry=float(ry[i]),
                    angle=float(theta[i]), color=(0, 0, 0, 128)
                )
            elif shape_type == "rectangle":
                shape = Rectangle(
                    x=float(cx[i]), y=float(cy[i]),
                    hw=float(hw[i]), hh=float(hh[i]), color=(0, 0, 0, 128)
                )
            elif shape_type == "rotated_rectangle":
                shape = RotatedRectangle(
                    x=float(cx[i]), y=float(cy[i]),
                    hw=float(rx[i]), hh=float(ry[i]),
                    angle=float(theta[i]), color=(0, 0, 0, 128)
                )
            elif shape_type == "triangle":
                shape = Triangle(
                    x1=float(x1[i]), y1=float(y1[i]),
                    x2=float(x2[i]), y2=float(y2[i]),
                    x3=float(x3[i]), y3=float(y3[i]), color=(0, 0, 0, 128)
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
