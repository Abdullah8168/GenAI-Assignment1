"""Image-quality metrics used for losses and evaluation (images in [0, 1])."""
import torch
import torch.nn.functional as F


def _gauss_window(size=11, sigma=1.5, channels=3, device="cpu"):
    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(channels, 1, size, size).contiguous()


def ssim(x, y, data_range=1.0, reduce=True):
    """Structural similarity (Wang et al. 2004) with an 11x11 Gaussian window, sigma 1.5.

    x, y: [B, C, H, W]. Returns a scalar (reduce=True) or one value per image.
    Differentiable, so 1 - ssim() is used directly as a loss.
    """
    C = x.shape[1]
    w = _gauss_window(channels=C, device=x.device).to(x.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, w, groups=C)
    mu_y = F.conv2d(y, w, groups=C)
    sxx = F.conv2d(x * x, w, groups=C) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=C) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=C) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    per_img = s.flatten(1).mean(1)
    return per_img.mean() if reduce else per_img


def psnr(x, y, data_range=1.0, reduce=True):
    mse = ((x - y) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    v = 10 * torch.log10(data_range ** 2 / mse)
    return v.mean() if reduce else v


def l1(x, y, reduce=True):
    v = (x - y).abs().flatten(1).mean(1)
    return v.mean() if reduce else v


def restoration_loss(pred, target, alpha):
    """L = alpha * L1 + (1 - alpha) * (1 - SSIM)   (assignment Task 1 / Task 2)."""
    return alpha * F.l1_loss(pred, target) + (1 - alpha) * (1 - ssim(pred, target))


def restoration_score(psnr_val, ssim_val):
    """Single validation objective for Optuna: equal mix of SSIM and PSNR scaled to ~[0,1]."""
    return 0.5 * ssim_val + 0.5 * (psnr_val / 40.0)
