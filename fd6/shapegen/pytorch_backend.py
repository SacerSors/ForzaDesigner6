from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from fd6.shapegen.shapes import Shape
from fd6.shapegen.shapes.ellipse import RotatedEllipse
from fd6.shapegen.shapes.rectangle import Rectangle
from fd6.shapegen.shapes.triangle import Triangle

_SEARCH_ALPHA = 128.0 / 255.0

class PyTorchDiffRenderer:
    """PyTorch/ROCm differentiable renderer for analytic shape optimization.

    Optimizes shapes (Rectangle, Triangle, Ellipse) analytically via backpropagation
    using the Adam optimizer with an edge-weighted MSE loss over SDFs.
    """

    def __init__(self, target: np.ndarray, alpha_mask: Optional[np.ndarray], edge_weight: np.ndarray) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.h, self.w = target.shape[:2]
        self.target = torch.from_numpy(target).float().to(self.device) / 255.0
        self.edge_weight = torch.from_numpy(edge_weight).float().to(self.device)
        self.n_weight = float(self.edge_weight.sum().item()) * 3.0

        if alpha_mask is not None:
            self.alpha_mask = torch.from_numpy(alpha_mask).float().to(self.device) / 255.0
        else:
            self.alpha_mask = torch.ones((self.h, self.w), dtype=torch.float32, device=self.device)

        # Coordinate grid (H, W, 2)
        y, x = torch.meshgrid(torch.arange(self.h, device=self.device),
                              torch.arange(self.w, device=self.device), indexing='ij')
        self.grid = torch.stack([x.float(), y.float()], dim=-1) # (H, W, 2)

    def _random_params(self, shape_type: str, b: int, w: int, h: int, max_size_frac: Optional[float], rng: random.Random) -> torch.Tensor:
        """Generates random parameters for shapes. Parameters are initialized on CPU then moved to device."""
        if max_size_frac is None:
            rx_cap = max(2.0, w / 8.0)
            ry_cap = max(2.0, h / 8.0)
        else:
            rx_cap = max(2.0, (w * max_size_frac) / 2.0)
            ry_cap = max(2.0, (h * max_size_frac) / 2.0)

        rs = np.random.RandomState(rng.randint(0, 2**31 - 1))

        if shape_type in ("rotated_ellipse", "ellipse", "circle"):
            # params: cx, cy, rx, ry, angle_rad
            out = np.empty((b, 5), dtype=np.float32)
            out[:, 0] = rs.uniform(0, w - 1, b)
            out[:, 1] = rs.uniform(0, h - 1, b)
            out[:, 2] = rs.uniform(1, rx_cap, b)
            out[:, 3] = rs.uniform(1, ry_cap if shape_type != "circle" else rx_cap, b)
            out[:, 4] = rs.uniform(0, 2 * math.pi, b) if shape_type == "rotated_ellipse" else 0.0

        elif shape_type in ("rectangle", "rotated_rectangle"):
            # params: cx, cy, rx (half-width), ry (half-height), angle_rad
            out = np.empty((b, 5), dtype=np.float32)
            out[:, 0] = rs.uniform(0, w - 1, b)
            out[:, 1] = rs.uniform(0, h - 1, b)
            out[:, 2] = rs.uniform(1, rx_cap, b)
            out[:, 3] = rs.uniform(1, ry_cap, b)
            out[:, 4] = rs.uniform(0, 2 * math.pi, b) if shape_type == "rotated_rectangle" else 0.0

        elif shape_type == "triangle":
            # params: cx, cy, scale, angle_rad, aspect_ratio
            out = np.empty((b, 5), dtype=np.float32)
            out[:, 0] = rs.uniform(0, w - 1, b)
            out[:, 1] = rs.uniform(0, h - 1, b)
            out[:, 2] = rs.uniform(10, max(10, w * (max_size_frac or 0.25)), b) # scale
            out[:, 3] = rs.uniform(0, 2 * math.pi, b) # angle
            out[:, 4] = rs.uniform(0.5, 2.0, b) # aspect ratio
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")

        return torch.from_numpy(out).to(self.device)

    def _sdf_ellipse(self, p: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Approximate SDF for an ellipse.
        params: (B, 5) -> cx, cy, rx, ry, angle
        p: (H, W, 2)
        returns mask: (B, H, W)
        """
        B = params.shape[0]
        cx, cy, rx, ry, angle = params.unbind(dim=-1) # each (B,)

        # Reshape for broadcasting: p is (H, W, 2) -> (1, H, W, 2)
        p = p.unsqueeze(0)

        cos_a = torch.cos(angle).view(B, 1, 1)
        sin_a = torch.sin(angle).view(B, 1, 1)

        dx = p[..., 0] - cx.view(B, 1, 1)
        dy = p[..., 1] - cy.view(B, 1, 1)

        # Rotate points
        x_rot = cos_a * dx + sin_a * dy
        y_rot = -sin_a * dx + cos_a * dy

        rx_v = torch.clamp(rx.view(B, 1, 1), min=1e-3)
        ry_v = torch.clamp(ry.view(B, 1, 1), min=1e-3)

        # SDF approximation for ellipse
        # (x/rx)^2 + (y/ry)^2 - 1
        d = (x_rot / rx_v)**2 + (y_rot / ry_v)**2 - 1.0

        # Soft mask: inside is positive, outside is negative in d. Actually d < 0 inside.
        # We want mask ~ 1 inside, 0 outside.
        mask = torch.sigmoid(-d * 5.0) # Multiply by 5 for sharper edge
        return mask

    def _sdf_rectangle(self, p: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Exact SDF for a rectangle.
        params: (B, 5) -> cx, cy, rx (half-width), ry (half-height), angle
        returns mask: (B, H, W)
        """
        B = params.shape[0]
        cx, cy, rx, ry, angle = params.unbind(dim=-1)

        p = p.unsqueeze(0)
        cos_a = torch.cos(angle).view(B, 1, 1)
        sin_a = torch.sin(angle).view(B, 1, 1)

        dx = p[..., 0] - cx.view(B, 1, 1)
        dy = p[..., 1] - cy.view(B, 1, 1)

        x_rot = cos_a * dx + sin_a * dy
        y_rot = -sin_a * dx + cos_a * dy

        # d = |p| - r
        d_x = torch.abs(x_rot) - rx.view(B, 1, 1)
        d_y = torch.abs(y_rot) - ry.view(B, 1, 1)

        # SDF
        d_max = torch.maximum(d_x, d_y)
        d = torch.maximum(d_max, torch.zeros_like(d_max)) + torch.min(d_max, torch.zeros_like(d_max))

        # Soft mask
        mask = torch.sigmoid(-d * 2.0)
        return mask

    def _sdf_triangle(self, p: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Approximate SDF for an isosceles triangle pointing up.
        params: (B, 5) -> cx, cy, scale, angle, aspect
        returns mask: (B, H, W)
        """
        B = params.shape[0]
        cx, cy, scale, angle, aspect = params.unbind(dim=-1)

        p = p.unsqueeze(0)
        cos_a = torch.cos(angle).view(B, 1, 1)
        sin_a = torch.sin(angle).view(B, 1, 1)

        dx = p[..., 0] - cx.view(B, 1, 1)
        dy = p[..., 1] - cy.view(B, 1, 1)

        x_rot = cos_a * dx + sin_a * dy
        y_rot = -sin_a * dx + cos_a * dy

        scale_v = torch.clamp(scale.view(B, 1, 1), min=1.0)
        aspect_v = torch.clamp(aspect.view(B, 1, 1), min=0.1)

        # Base width = scale, height = scale * aspect
        # Bottom edge: y = scale * aspect / 2
        # Top vertex: y = -scale * aspect / 2

        h_val = scale_v * aspect_v
        w_val = scale_v

        # Normalize coordinates
        y_norm = y_rot / h_val + 0.5 # 0 at top, 1 at bottom
        x_norm = torch.abs(x_rot) / (w_val / 2.0) # 0 at center, 1 at edges

        # In a triangle, x width grows linearly with y
        # x_bound = y_norm
        d = x_norm - y_norm

        # We also need a bottom bound
        d_bottom = y_norm - 1.0

        d_final = torch.maximum(d, d_bottom)

        mask = torch.sigmoid(-d_final * 10.0)
        return mask

    def _get_mask(self, shape_type: str, params: torch.Tensor) -> torch.Tensor:
        if shape_type in ("rotated_ellipse", "ellipse", "circle"):
            return self._sdf_ellipse(self.grid, params)
        elif shape_type in ("rectangle", "rotated_rectangle"):
            return self._sdf_rectangle(self.grid, params)
        elif shape_type == "triangle":
            return self._sdf_triangle(self.grid, params)
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")

    def _score_and_color(self, canvas: torch.Tensor, mask: torch.Tensor, full_sq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes score and optimal color for a batch of masks over the canvas.
        mask: (B, H, W)
        canvas: (H, W, 3)
        returns scores: (B,), colors: (B, 3)
        """
        B = mask.shape[0]

        eff = mask * self.alpha_mask.unsqueeze(0) # (B, H, W)

        a = _SEARCH_ALPHA

        # Optimal color
        eff_sum = eff.sum(dim=(1, 2)) # (B,)
        denom = eff_sum * a

        tgt = self.target.unsqueeze(0) # (1, H, W, 3)
        cur = canvas.unsqueeze(0) # (1, H, W, 3)

        # numer = sum(eff * (tgt - (1-a)*cur))
        diff = tgt - (1.0 - a) * cur # (1, H, W, 3)
        numer = (eff.unsqueeze(-1) * diff).sum(dim=(1, 2)) # (B, 3)

        safe = eff_sum > 0.5
        denom_safe = torch.where(safe, denom, torch.ones_like(denom))

        color = torch.where(safe.unsqueeze(-1), torch.clamp(numer / denom_safe.unsqueeze(-1), 0.0, 1.0), torch.zeros_like(numer))

        # Blended image
        m = mask.unsqueeze(-1) # (B, H, W, 1)
        blended = m * (a * color.view(B, 1, 1, 3) + (1.0 - a) * cur) + (1.0 - m) * cur

        w_t = self.edge_weight.unsqueeze(0).unsqueeze(-1) # (1, H, W, 1)

        region_old = (w_t * (cur - tgt)**2).sum(dim=(1, 2, 3)) # (B,)
        region_new = (w_t * (blended - tgt)**2).sum(dim=(1, 2, 3)) # (B,)

        total = full_sq - region_old + region_new

        n = self.n_weight if self.n_weight >= 1.0 else 1.0
        score = torch.sqrt(torch.clamp(total, min=0.0) / n)

        # Sticker overlap rejection
        body = (mask >= 0.5).float()
        body_total = body.sum(dim=(1, 2))
        opaque = ((self.alpha_mask.unsqueeze(0) >= 0.5) & (mask >= 0.5)).float().sum(dim=(1, 2))
        ratio = torch.where(body_total >= 1.0, opaque / torch.clamp(body_total, min=1.0), torch.zeros_like(body_total))
        reject = (body_total < 1.0) | (ratio < 0.995)

        score = torch.where(reject, torch.tensor(float('inf'), device=score.device), score)

        return score, color

    def search(self, canvas: np.ndarray, n_random: int, n_mutate: int, max_size_frac: Optional[float], rng: random.Random) -> tuple[float, Optional[Shape]]:
        # Randomly select one shape type if multiple are given, but engine calls search with ONE type due to rotation
        shape_type = "rotated_ellipse" # default fallback
        if hasattr(self, '_current_type'):
            shape_type = self._current_type

        cur_tensor = torch.from_numpy(canvas).float().to(self.device) / 255.0

        full_sq = (((cur_tensor - self.target)**2) * self.edge_weight.unsqueeze(-1)).sum()

        # 1. Random Search via Forward Pass in chunks to prevent OOM
        n_random = max(1, n_random)
        params = self._random_params(shape_type, n_random, self.w, self.h, max_size_frac, rng)

        chunk_size = 64
        scores_list = []
        colors_list = []

        with torch.no_grad():
            for i in range(0, n_random, chunk_size):
                p_chunk = params[i:i+chunk_size]
                mask = self._get_mask(shape_type, p_chunk)
                sc, col = self._score_and_color(cur_tensor, mask, full_sq)
                scores_list.append(sc)
                colors_list.append(col)

        all_scores = torch.cat(scores_list)
        all_colors = torch.cat(colors_list)

        # 2. Select top K candidates for optimization
        K = min(16, n_random)
        # Using torch.sort instead of torch.topk to avoid unsupported ops and fallback
        sorted_scores, sorted_indices = torch.sort(all_scores)
        top_indices = sorted_indices[:K]

        best_score = sorted_scores[0].item()
        if not math.isfinite(best_score):
            return float('inf'), None

        top_params = params[top_indices].clone().detach().requires_grad_(True)

        # 3. Optimize top K candidates via Gradient Descent
        # Setting foreach=False to avoid aten::lerp fallback issues
        optimizer = torch.optim.Adam([top_params], lr=1.0, foreach=False)

        best_idx = 0
        best_opt_score = best_score
        best_opt_params = top_params[0].detach().clone()
        best_opt_color = all_colors[top_indices[0]].detach().clone()

        n_mutate = max(1, n_mutate)

        for _ in range(n_mutate):
            optimizer.zero_grad()
            mask = self._get_mask(shape_type, top_params)
            sc, col = self._score_and_color(cur_tensor, mask, full_sq)

            # Loss is the mean of scores
            loss = sc.mean()
            if not math.isfinite(loss.item()):
                break

            loss.backward()
            optimizer.step()

            # Check if any score improved
            with torch.no_grad():
                mask_eval = self._get_mask(shape_type, top_params)
                sc_eval, col_eval = self._score_and_color(cur_tensor, mask_eval, full_sq)

                min_sc, min_idx = torch.min(sc_eval, dim=0)
                if min_sc.item() < best_opt_score:
                    best_opt_score = min_sc.item()
                    best_idx = min_idx.item()
                    best_opt_params = top_params[best_idx].detach().clone()
                    best_opt_color = col_eval[best_idx].detach().clone()

        # 4. Construct output shape
        p_np = best_opt_params.cpu().numpy()
        c_np = (best_opt_color.cpu().numpy() * 255.0).astype(np.int32)
        color_tuple = (int(c_np[0]), int(c_np[1]), int(c_np[2]), int(_SEARCH_ALPHA * 255))

        cx, cy = p_np[0], p_np[1]

        if shape_type in ("rotated_ellipse", "ellipse", "circle"):
            rx, ry, angle = p_np[2], p_np[3], p_np[4]
            deg = math.degrees(angle) % 180.0
            return best_opt_score, RotatedEllipse(color=color_tuple, x=cx, y=cy, rx=rx, ry=ry, angle=deg)

        elif shape_type in ("rectangle", "rotated_rectangle"):
            rx, ry, angle = p_np[2], p_np[3], p_np[4]
            deg = math.degrees(angle) % 180.0
            return best_opt_score, Rectangle(color=color_tuple, x=cx, y=cy, rx=rx, ry=ry, angle=deg)

        elif shape_type == "triangle":
            scale, angle, aspect = p_np[2], p_np[3], p_np[4]
            deg = math.degrees(angle) % 360.0
            h = scale * aspect
            # For compatibility with legacy Triangle (usually defined by 3 points), we could return a specific triangle
            # but let's assume Triangle(color, x, y, scale, angle) for now or similar constructor
            try:
                return best_opt_score, Triangle(color=color_tuple, x=cx, y=cy, scale=scale, angle=deg, aspect=aspect)
            except TypeError:
                # If Triangle constructor is different, we fallback to RotatedEllipse
                return best_opt_score, RotatedEllipse(color=color_tuple, x=cx, y=cy, rx=scale, ry=h/2, angle=deg)

        return float('inf'), None
