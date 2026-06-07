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

    def __init__(self, target: np.ndarray, alpha_mask: Optional[np.ndarray], edge_weight: np.ndarray, edge_dir: Optional[np.ndarray] = None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.h, self.w = target.shape[:2]
        # Use np.array(copy=True) to avoid PyTorch warnings and potential segfaults
        # when converting read-only shared_memory buffers to tensors.
        self.target = torch.from_numpy(np.array(target, copy=True)).float().to(self.device) / 255.0
        self.edge_weight = torch.from_numpy(np.array(edge_weight, copy=True)).float().to(self.device)
        self.n_weight = float(self.edge_weight.sum().item()) * 3.0

        if edge_dir is not None:
            self.edge_dir = torch.from_numpy(np.array(edge_dir, copy=True)).float().to(self.device)
        else:
            self.edge_dir = torch.zeros((self.h, self.w), dtype=torch.float32, device=self.device)

        if alpha_mask is not None:
            self.alpha_mask = torch.from_numpy(np.array(alpha_mask, copy=True)).float().to(self.device) / 255.0
        else:
            self.alpha_mask = torch.ones((self.h, self.w), dtype=torch.float32, device=self.device)

        # Precompute normalized grids for grid_sample
        # F.grid_sample expects coordinates in [-1, 1] for (x, y)

    def _random_params(self, shape_type: str, b: int, w: int, h: int, max_size_frac: Optional[float], rng: random.Random) -> torch.Tensor:
        """Generates random parameters for shapes. Parameters are generated directly on the device."""
        if max_size_frac is None:
            rx_cap = max(2.0, w / 8.0)
            ry_cap = max(2.0, h / 8.0)
        else:
            rx_cap = max(2.0, (w * max_size_frac) / 2.0)
            ry_cap = max(2.0, (h * max_size_frac) / 2.0)

        # Seed PyTorch RNG based on Python's random state
        seed = rng.randint(0, 2**31 - 1)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        out = torch.empty((b, 6), dtype=torch.float32, device=self.device)

        # 25% error-weighted placement
        b_err = int(b * 0.25)

        if b_err > 0:
            # Flatten edge_weight for multinomial sampling
            flat_edge_weight = self.edge_weight.view(-1)
            # torch.multinomial requires non-negative weights, and it handles zeros
            # but requires sum > 0.
            if flat_edge_weight.sum() > 1e-5:
                # Sample 1D indices
                sampled_indices = torch.multinomial(flat_edge_weight, b_err, replacement=True, generator=gen)
                # Convert to 2D coordinates
                out[:b_err, 0] = (sampled_indices % w).float()
                out[:b_err, 1] = (sampled_indices // w).float()
            else:
                out[:b_err, 0] = (w - 1) * torch.rand(b_err, generator=gen, device=self.device)
                out[:b_err, 1] = (h - 1) * torch.rand(b_err, generator=gen, device=self.device)

        if b > b_err:
            out[b_err:, 0] = (w - 1) * torch.rand(b - b_err, generator=gen, device=self.device)
            out[b_err:, 1] = (h - 1) * torch.rand(b - b_err, generator=gen, device=self.device)

        # Standard uniform macro: a + (b - a) * rand()
        def uniform(idx, a, b_val):
            out[:, idx] = a + (b_val - a) * torch.rand(b, generator=gen, device=self.device)

        # Optimizable Opacity initialization:
        # Start random alpha between 0.1 and 1.0
        uniform(5, 0.1, 1.0)

        if shape_type in ("rotated_ellipse", "ellipse", "circle"):
            uniform(2, 1, rx_cap)
            uniform(3, 1, ry_cap if shape_type != "circle" else rx_cap)
            if shape_type == "rotated_ellipse":
                # Edge-Aligned Spawning
                cx = torch.clamp(out[:, 0].long(), 0, w - 1)
                cy = torch.clamp(out[:, 1].long(), 0, h - 1)
                out[:, 4] = self.edge_dir[cy, cx]
            else:
                out[:, 4] = 0.0

        elif shape_type in ("rectangle", "rotated_rectangle"):
            uniform(2, 1, rx_cap)
            uniform(3, 1, ry_cap)
            if shape_type == "rotated_rectangle":
                # Edge-Aligned Spawning
                cx = torch.clamp(out[:, 0].long(), 0, w - 1)
                cy = torch.clamp(out[:, 1].long(), 0, h - 1)
                out[:, 4] = self.edge_dir[cy, cx]
            else:
                out[:, 4] = 0.0

        elif shape_type == "triangle":
            uniform(2, 10, max(10, w * (max_size_frac or 0.25)))
            # Edge-Aligned Spawning
            cx = torch.clamp(out[:, 0].long(), 0, w - 1)
            cy = torch.clamp(out[:, 1].long(), 0, h - 1)
            out[:, 3] = self.edge_dir[cy, cx]
            uniform(4, 0.5, 2.0)
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")

        return out

    def _sdf_ellipse(self, p: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Approximate SDF for an ellipse.
        params: (B, 6) -> cx, cy, rx, ry, angle, alpha
        p: (B, T, T, 2)
        returns mask: (B, T, T)
        """
        B = params.shape[0]
        cx, cy, rx, ry, angle, alpha = params.unbind(dim=-1) # each (B,)

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
        # Increased multiplier from 5.0 to 100.0 to match OpenCL hard-edge scoring closer
        mask = torch.sigmoid(-d * 100.0)
        return mask

    def _sdf_rectangle(self, p: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Exact SDF for a rectangle.
        params: (B, 6) -> cx, cy, rx (half-width), ry (half-height), angle, alpha
        p: (B, T, T, 2)
        returns mask: (B, T, T)
        """
        B = params.shape[0]
        cx, cy, rx, ry, angle, alpha = params.unbind(dim=-1)

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
        # Increased multiplier to match OpenCL hard-edge scoring closer
        mask = torch.sigmoid(-d * 50.0)
        return mask

    def _sdf_triangle(self, p: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Approximate SDF for an isosceles triangle pointing up.
        params: (B, 6) -> cx, cy, scale, angle, aspect, alpha
        p: (B, T, T, 2)
        returns mask: (B, T, T)
        """
        B = params.shape[0]
        cx, cy, scale, angle, aspect, alpha = params.unbind(dim=-1)

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

        # Increased multiplier to match OpenCL hard-edge scoring closer
        mask = torch.sigmoid(-d_final * 50.0)
        return mask

    def _get_mask(self, shape_type: str, grid: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if shape_type in ("rotated_ellipse", "ellipse", "circle"):
            return self._sdf_ellipse(grid, params)
        elif shape_type in ("rectangle", "rotated_rectangle"):
            return self._sdf_rectangle(grid, params)
        elif shape_type == "triangle":
            return self._sdf_triangle(grid, params)
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")

    def _score_and_color(self, cur_t: torch.Tensor, tgt_t: torch.Tensor, alpha_t: torch.Tensor, edge_t: torch.Tensor, mask: torch.Tensor, full_sq: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes score and optimal color for a batch of masks over the LOCAL tiles.
        mask: (B, T, T)
        cur_t: (B, T, T, 3)
        tgt_t: (B, T, T, 3)
        alpha_t: (B, T, T)
        edge_t: (B, T, T)
        params: (B, 6)
        returns scores: (B,), colors: (B, 3)
        """
        B = mask.shape[0]

        eff = mask * alpha_t # (B, T, T)

        # Dynamic Alpha from params, clamped to [0.01, 1.0] to prevent div-by-zero
        a = torch.clamp(params[:, 5], 0.01, 1.0)

        a_view = a.view(B, 1, 1)

        # Optimal color
        eff_sum = eff.sum(dim=(1, 2)) # (B,)
        denom = eff_sum * a

        # numer = sum(eff * (tgt - (1-a)*cur))
        diff = tgt_t - (1.0 - a_view).unsqueeze(-1) * cur_t # (B, T, T, 3)
        numer = (eff.unsqueeze(-1) * diff).sum(dim=(1, 2)) # (B, 3)

        safe = eff_sum > 0.5
        denom_safe = torch.where(safe, denom, torch.ones_like(denom))

        color = torch.where(safe.unsqueeze(-1), torch.clamp(numer / denom_safe.unsqueeze(-1), 0.0, 1.0), torch.zeros_like(numer))

        # Blended image
        m = mask.unsqueeze(-1) # (B, T, T, 1)
        a_view_4 = a.view(B, 1, 1, 1)
        blended = m * (a_view_4 * color.view(B, 1, 1, 3) + (1.0 - a_view_4) * cur_t) + (1.0 - m) * cur_t

        w_t = edge_t.unsqueeze(-1) # (B, T, T, 1)

        region_old = (w_t * (cur_t - tgt_t)**2).sum(dim=(1, 2, 3)) # (B,)
        region_new = (w_t * (blended - tgt_t)**2).sum(dim=(1, 2, 3)) # (B,)

        total = full_sq - region_old + region_new

        n = self.n_weight if self.n_weight >= 1.0 else 1.0
        score = torch.sqrt(torch.clamp(total, min=0.0) / n)

        # Sticker overlap rejection
        body = (mask >= 0.5).float()
        body_total = body.sum(dim=(1, 2))
        opaque = ((alpha_t >= 0.5) & (mask >= 0.5)).float().sum(dim=(1, 2))
        ratio = torch.where(body_total >= 1.0, opaque / torch.clamp(body_total, min=1.0), torch.zeros_like(body_total))
        reject = (body_total < 1.0) | (ratio < 0.995)

        score = torch.where(reject, torch.tensor(float('inf'), device=score.device), score)

        return score, color

    def _extract_tiles(self, params: torch.Tensor, cur_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts local cropped patches for each shape to prevent massive VRAM use.
        Returns: grid (B, T, T, 2), cur_t, tgt_t, alpha_t, edge_t
        """
        B = params.shape[0]
        # Calculate optimal tile size
        max_r = torch.max(torch.maximum(params[:, 2], params[:, 3])).item() if B else 1.0
        T = max(2, int(min(max(self.w, self.h), 2 * math.ceil(max_r) + 2)))

        # Create un-normalized grid offsets
        y_off, x_off = torch.meshgrid(torch.arange(T, device=self.device),
                                      torch.arange(T, device=self.device), indexing='ij')

        # Centers
        cx = torch.round(params[:, 0]).long()
        cy = torch.round(params[:, 1]).long()

        x0 = cx - T // 2
        y0 = cy - T // 2

        # Global grid coordinates for each shape
        gx = x0.view(B, 1, 1) + x_off.view(1, T, T)
        gy = y0.view(B, 1, 1) + y_off.view(1, T, T)

        valid = ((gx >= 0) & (gx < self.w) & (gy >= 0) & (gy < self.h)).float()

        # Clamp to avoid out-of-bounds indexing (valid mask handles the edges)
        gxc = torch.clamp(gx, 0, self.w - 1)
        gyc = torch.clamp(gy, 0, self.h - 1)

        # Gather local patches
        cur_t = cur_tensor[gyc, gxc] # (B, T, T, 3)
        tgt_t = self.target[gyc, gxc]
        alpha_t = self.alpha_mask[gyc, gxc] * valid
        edge_t = self.edge_weight[gyc, gxc] * valid

        grid = torch.stack([gx.float(), gy.float()], dim=-1) # (B, T, T, 2)

        return grid, cur_t, tgt_t, alpha_t, edge_t

    def shutdown(self) -> None:
        """Frees PyTorch memory pool allocations and releases VRAM back to the OS."""
        # Delete internal tensors that might hold memory
        if hasattr(self, 'target'):
            del self.target
        if hasattr(self, 'edge_weight'):
            del self.edge_weight
        if hasattr(self, 'alpha_mask'):
            del self.alpha_mask

        # Empty the GPU memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def search(self, canvas: np.ndarray, n_random: int, n_mutate: int, max_size_frac: Optional[float], rng: random.Random) -> tuple[float, Optional[Shape]]:
        # Collect available shape types
        types = ["rotated_ellipse"] # default fallback
        if hasattr(self, '_current_types') and self._current_types:
            types = self._current_types

        cur_tensor = torch.from_numpy(np.array(canvas, copy=True)).float().to(self.device, non_blocking=True) / 255.0

        # If the backend is tracking an external LIVE edge_weight reference (passed from engine),
        # sync it to the device before searching so dynamic error placement updates work.
        if hasattr(self, '_external_edge_weight') and self._external_edge_weight is not None:
            self.edge_weight.copy_(torch.from_numpy(self._external_edge_weight).float())

        full_sq = (((cur_tensor - self.target)**2) * self.edge_weight.unsqueeze(-1)).sum()

        n_random = max(1, n_random)

        # We will track the best shapes across all evaluated types
        overall_best_score = float('inf')
        overall_best_params = None
        overall_best_color = None
        overall_best_type = None

        # Calculate max bounds so the optimizer doesn't blow shapes up massively
        if max_size_frac is None:
            rx_cap = max(2.0, self.w / 8.0)
            ry_cap = max(2.0, self.h / 8.0)
        else:
            rx_cap = max(2.0, (self.w * max_size_frac) / 2.0)
            ry_cap = max(2.0, (self.h * max_size_frac) / 2.0)

        # Evenly divide random samples among the available shape types so that each place
        # isn't just randomly assigned a type, but rather we try all active shape types
        # across the batches and let them compete for the best score.
        samples_per_type = max(1, n_random // len(types))

        for shape_type in types:
            params = self._random_params(shape_type, samples_per_type, self.w, self.h, max_size_frac, rng)

            # Calculate optimal tile size T for this entire type batch
            max_r = torch.max(torch.maximum(params[:, 2], params[:, 3])).item() if samples_per_type else 1.0
            T = max(2, int(min(max(self.w, self.h), 2 * math.ceil(max_r) + 2)))

            # Dynamic Chunk Sizing: Cap the forward pass at ~256MB of VRAM to prevent
            # PyTorch from starving the OS composer, while allowing tiny shapes
            # to evaluate 10,000+ candidates simultaneously for extreme speed.
            # T*T pixels * 3 channels * 4 bytes per float32
            bytes_per_tile = T * T * 3 * 4
            chunk_size = max(2, int((256 * 1024 * 1024) / max(1, bytes_per_tile)))
            chunk_size = min(chunk_size, samples_per_type)

            scores_list = []
            colors_list = []

            with torch.no_grad():
                for i in range(0, samples_per_type, chunk_size):
                    p_chunk = params[i:i+chunk_size]
                    grid, cur_t, tgt_t, alpha_t, edge_t = self._extract_tiles(p_chunk, cur_tensor)

                    mask = self._get_mask(shape_type, grid, p_chunk)
                    sc, col = self._score_and_color(cur_t, tgt_t, alpha_t, edge_t, mask, full_sq, p_chunk)
                    scores_list.append(sc)
                    colors_list.append(col)

            all_scores = torch.cat(scores_list)
            all_colors = torch.cat(colors_list)

            # Select top K candidates for optimization for THIS shape type
            K = min(16, samples_per_type)
            sorted_scores, sorted_indices = torch.sort(all_scores)
            top_indices = sorted_indices[:K]

            best_score = sorted_scores[0].item()
            if not math.isfinite(best_score):
                continue

            top_params = params[top_indices].clone().detach().requires_grad_(True)

            # Optimize top K candidates
            best_idx = 0
            best_opt_score = best_score
            best_opt_params = top_params[0].detach().clone()
            best_opt_color = all_colors[top_indices[0]].detach().clone()

            n_mutate = max(1, n_mutate)

            grid_top, cur_t_top, tgt_t_top, alpha_t_top, edge_t_top = self._extract_tiles(top_params, cur_tensor)

            m = torch.zeros_like(top_params)
            v = torch.zeros_like(top_params)
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8

            for t in range(1, n_mutate + 1):
                # Learning Rate Decay (Idea C): Linearly scale from 1.0 down to 0.1
                lr = 1.0 - (0.9 * ((t - 1) / max(1, n_mutate - 1)))

                if top_params.grad is not None:
                    top_params.grad.zero_()

                mask = self._get_mask(shape_type, grid_top, top_params)
                sc, col = self._score_and_color(cur_t_top, tgt_t_top, alpha_t_top, edge_t_top, mask, full_sq, top_params)

                loss = sc.mean()
                if not math.isfinite(loss.item()):
                    break

                loss.backward()

                with torch.no_grad():
                    grad = top_params.grad
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad ** 2)

                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)

                    top_params -= lr * m_hat / (torch.sqrt(v_hat) + eps)

                    # Clamp scale parameters so the optimizer cannot inflate tiny
                    # shapes into massive blocks near the end of generation.
                    if shape_type in ("rotated_ellipse", "ellipse", "circle", "rectangle", "rotated_rectangle"):
                        top_params[:, 2] = torch.clamp(top_params[:, 2], 1.0, rx_cap)
                        top_params[:, 3] = torch.clamp(top_params[:, 3], 1.0, ry_cap)
                    elif shape_type == "triangle":
                        top_params[:, 2] = torch.clamp(top_params[:, 2], 10.0, max(10.0, self.w * (max_size_frac or 0.25)))

                with torch.no_grad():
                    mask_eval = self._get_mask(shape_type, grid_top, top_params)
                    sc_eval, col_eval = self._score_and_color(cur_t_top, tgt_t_top, alpha_t_top, edge_t_top, mask_eval, full_sq, top_params)

                    min_sc, min_idx = torch.min(sc_eval, dim=0)
                    if min_sc.item() < best_opt_score:
                        best_opt_score = min_sc.item()
                        best_idx = min_idx.item()
                        best_opt_params = top_params[best_idx].detach().clone()
                        best_opt_color = col_eval[best_idx].detach().clone()

            # Compare with overall best
            if best_opt_score < overall_best_score:
                overall_best_score = best_opt_score
                overall_best_params = best_opt_params
                overall_best_color = best_opt_color
                overall_best_type = shape_type

        if overall_best_params is None:
            return float('inf'), None

        # Construct output shape from overall best
        p_np = overall_best_params.cpu().numpy()
        c_np = (overall_best_color.cpu().numpy() * 255.0).astype(np.int32)
        final_alpha = int(max(0.01, min(1.0, float(p_np[5]))) * 255.0)
        color_tuple = (int(c_np[0]), int(c_np[1]), int(c_np[2]), final_alpha)

        cx, cy = float(p_np[0]), float(p_np[1])

        if overall_best_type in ("rotated_ellipse", "ellipse", "circle"):
            rx, ry, angle = float(p_np[2]), float(p_np[3]), float(p_np[4])
            deg = math.degrees(angle) % 180.0
            return overall_best_score, RotatedEllipse(color=color_tuple, x=cx, y=cy, rx=rx, ry=ry, angle=deg)

        elif overall_best_type in ("rectangle", "rotated_rectangle"):
            from fd6.shapegen.shapes.rectangle import RotatedRectangle
            rx, ry, angle = float(p_np[2]), float(p_np[3]), float(p_np[4])
            deg = math.degrees(angle) % 180.0
            if overall_best_type == "rotated_rectangle":
                return overall_best_score, RotatedRectangle(color=color_tuple, x=cx, y=cy, hw=rx, hh=ry, angle=deg)
            else:
                return overall_best_score, Rectangle(color=color_tuple, x=cx, y=cy, hw=rx, hh=ry)

        elif overall_best_type == "triangle":
            scale, angle, aspect = float(p_np[2]), float(p_np[3]), float(p_np[4])
            deg = math.degrees(angle) % 360.0
            h = scale * aspect
            try:
                return overall_best_score, Triangle(color=color_tuple, x=cx, y=cy, scale=scale, angle=deg, aspect=aspect)
            except TypeError:
                return overall_best_score, RotatedEllipse(color=color_tuple, x=cx, y=cy, rx=scale, ry=h/2, angle=deg)

        return float('inf'), None
