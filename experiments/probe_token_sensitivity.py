"""Stage 0 probe: how much perturbation of the scene tokens can the Gaussian head absorb?

Runs the ZipSplat pipeline up to the `[scene | color]` tokens that feed the Gaussian head,
caches them, then re-decodes + re-renders under a range of lossy transforms. The backbone
runs exactly once, so a full sweep costs a few head passes plus renders.

Distortion is measured against the *unperturbed* renders on a fixed orbit, i.e. it isolates
the damage caused by the transform rather than the model's own reconstruction error.

PCA and k-means are fit on the very scene they are evaluated on, so their numbers are an
optimistic bound on what a learned encoder could reach.

Usage:
    python experiments/probe_token_sensitivity.py assets/examples/drone.mp4
"""

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor

from zipsplat import Camera, Pose, ZipSplat, load_image, load_video, viz
from zipsplat.gaussians import Gaussians
from zipsplat.utils import kmeans

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TARGETS = ("scene", "color", "both")


def token_slice(model: ZipSplat, target: str) -> Tuple[int, int]:
    """Channel range of the head input `[scene (D) | color (D_c)]` to perturb."""
    d_scene = model.model.backbone.embed_dim
    d_color = model.model.conf["color_skip_dim"]
    return {
        "scene": (0, d_scene),
        "color": (d_scene, d_scene + d_color),
        "both": (0, d_scene + d_color),
    }[target]


# --------------------------------------------------------------------------- inputs


def load_inputs(path: str, num_frames: int) -> List[Tensor]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES:
        return load_video(p, num_frames=num_frames)
    files = (
        sorted(f for f in p.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES)
        if p.is_dir()
        else sorted(Path().glob(path))
    )
    if not files:
        raise ValueError(f"No images/video found at {path!r}.")
    return [load_image(f) for f in files]


@torch.no_grad()
def compute_tokens(model: ZipSplat, images: List[Tensor], compression: float) -> Tensor:
    """Run everything up to (but not including) the Gaussian head. Returns (B, K, 1664)."""
    imgs, _, _ = model._prepare_inputs(images, None, None)
    net = model.model
    features = net._backbone_features(imgs, None, None)
    layer_tokens = net._prepare(features)
    nearest_idx = net._cluster(layer_tokens[0], compression)
    scene = net._fuse(layer_tokens, nearest_idx)
    color = net._color(imgs, nearest_idx)
    return torch.cat([scene, color], dim=-1)


# --------------------------------------------------------------------------- metrics


def psnr(pred: Tensor, ref: Tensor) -> Tensor:
    mse = (pred.clamp(0, 1) - ref.clamp(0, 1)).pow(2).flatten(1).mean(1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def _gaussian_window(size: int, sigma: float, device) -> Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-coords.pow(2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g)


def ssim(pred: Tensor, ref: Tensor, size: int = 11, sigma: float = 1.5) -> Tensor:
    """Standard Gaussian-window SSIM over (N, 3, H, W) images in [0, 1]."""
    c = pred.shape[1]
    win = _gaussian_window(size, sigma, pred.device).expand(c, 1, size, size)
    conv = lambda x: torch.nn.functional.conv2d(x, win, groups=c)  # noqa: E731
    mu_p, mu_r = conv(pred), conv(ref)
    mu_p2, mu_r2, mu_pr = mu_p.pow(2), mu_r.pow(2), mu_p * mu_r
    sig_p = conv(pred * pred) - mu_p2
    sig_r = conv(ref * ref) - mu_r2
    sig_pr = conv(pred * ref) - mu_pr
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_pr + c1) * (2 * sig_pr + c2)
    den = (mu_p2 + mu_r2 + c1) * (sig_p + sig_r + c2)
    return (num / den).flatten(1).mean(1)


# --------------------------------------------------------------------------- transforms


def t_identity(x: Tensor) -> Tensor:
    return x


def t_noise(x: Tensor, snr_db: float) -> Tensor:
    signal_rms = x.pow(2).mean().sqrt()
    return x + torch.randn_like(x) * signal_rms * 10 ** (-snr_db / 20)


def t_pca(x: Tensor, d: int) -> Tensor:
    mu = x.mean(0, keepdim=True)
    xc = (x - mu).float()
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    basis = vh[:d].T
    return ((xc @ basis) @ basis.T + mu).to(x.dtype)


def t_randproj(x: Tensor, d: int, seed: int = 0) -> Tensor:
    gen = torch.Generator(device=x.device).manual_seed(seed)
    rand = torch.randn(x.shape[1], d, generator=gen, device=x.device, dtype=torch.float32)
    basis, _ = torch.linalg.qr(rand)
    xf = x.float()
    return ((xf @ basis) @ basis.T).to(x.dtype)


def t_vq(x: Tensor, k: int, n_iters: int = 15) -> Tensor: # 基于K-Means的向量量化 Vector Quantization
    centroids, _, assignments = kmeans(x[None].float(), k, n_iters=n_iters)
    return centroids[0][assignments[0]].to(x.dtype)


def _split_sizes(dim: int, groups: int) -> List[int]:
    base, rem = divmod(dim, groups)
    return [base + (1 if i < rem else 0) for i in range(groups)]


def t_product_vq(x: Tensor, groups: int, k: int, n_iters: int = 10) -> Tensor:
    """Split channels into `groups` blocks, quantise each with its own codebook."""
    out = torch.empty_like(x)
    start = 0
    for size in _split_sizes(x.shape[1], groups):
        seg = x[:, start : start + size].float()
        centroids, _, assignments = kmeans(seg[None], k, n_iters=n_iters)
        out[:, start : start + size] = centroids[0][assignments[0]].to(x.dtype)
        start += size
    return out


def t_residual_vq(x: Tensor, stages: int, k: int, n_iters: int = 10) -> Tensor:
    """Quantise, subtract, repeat — each stage refines the previous residual."""
    residual = x.float()
    recon = torch.zeros_like(residual)
    for _ in range(stages):
        centroids, _, assignments = kmeans(residual[None], k, n_iters=n_iters)
        quantised = centroids[0][assignments[0]]
        recon += quantised
        residual = residual - quantised
    return recon.to(x.dtype)


def t_scalar_quant(x: Tensor, bits: int) -> Tensor:
    lo = x.min(0, keepdim=True).values
    hi = x.max(0, keepdim=True).values
    levels = 2**bits - 1
    q = ((x - lo) / (hi - lo).clamp_min(1e-8) * levels).round()
    return q / levels * (hi - lo) + lo


def build_sweep(args) -> List[Tuple[str, Callable[[Tensor], Tensor], float]]:
    """Returns (name, transform, bits_per_token) triples. bits = nan when not a codec."""
    nan = float("nan")
    sweep: List[Tuple[str, Callable[[Tensor], Tensor], float]] = [("identity", t_identity, nan)]
    for snr in args.snr:
        sweep.append((f"noise@{snr}dB", lambda x, s=snr: t_noise(x, s), nan))
    for d in args.dims:
        sweep.append((f"pca@{d}", lambda x, d=d: t_pca(x, d), d * 32))
        sweep.append((f"randproj@{d}", lambda x, d=d: t_randproj(x, d), d * 32))
    for k in args.codes:
        sweep.append((f"vq@{k}", lambda x, k=k: t_vq(x, k), math.log2(k)))
    for g in args.pvq_groups:
        for k in args.pvq_codes:
            sweep.append(
                (f"pvq@{g}x{k}", lambda x, g=g, k=k: t_product_vq(x, g, k), g * math.log2(k))
            )
    for n in args.rvq_stages:
        for k in args.rvq_codes:
            sweep.append(
                (f"rvq@{n}x{k}", lambda x, n=n, k=k: t_residual_vq(x, n, k), n * math.log2(k))
            )
    for b in args.bits:
        sweep.append((f"uniform@{b}bit", lambda x, b=b: t_scalar_quant(x, b), b * args.token_dim))
    return sweep


# --------------------------------------------------------------------------- rendering


@torch.no_grad()
def render_orbit(
    gaussians: Gaussians, cameras: Camera, poses: Pose, chunk: int = 8
) -> Tensor:
    """Render a fixed trajectory, returning float (T, 3, H, W) in [0, 1]."""
    out = []
    bg = torch.ones(3, device=gaussians.device)
    for s in range(0, poses.shape[0], chunk):
        e = min(s + chunk, poses.shape[0])
        rgb, _ = gaussians.render(
            cameras[s:e], poses[s:e], mode="RGB", backgrounds=bg.expand(e - s, 3)
        )
        out.append(rgb.float().clamp(0, 1))
    return torch.cat(out, 0)


def gaussian_drift(pred: Gaussians, ref: Gaussians, radius: float) -> Dict[str, float]:
    """Parameter-space damage, normalised by scene scale where meaningful."""
    return {
        "mean_shift_pct": float((pred.means - ref.means).norm(dim=-1).mean() / radius * 100),
        "opacity_mae": float((pred.opacities - ref.opacities).abs().mean()),
        "scale_ratio": float((pred.scales.mean() / ref.scales.mean())),
        "rgb_mae": float((pred.rgb - ref.rgb).abs().mean()),
    }


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Video file, image directory, or glob.")
    ap.add_argument("--weights", default="zipsplat")
    ap.add_argument("--num-frames", type=int, default=24, help="Frames sampled from a video.")
    ap.add_argument("--compression", type=float, default=1.0, help="k-means token ratio.")
    ap.add_argument(
        "--target",
        default="both",
        choices=TARGETS,
        help="Which part of the head input to perturb.",
    )
    ap.add_argument("--eval-views", type=int, default=16)
    ap.add_argument("--render-size", type=int, default=384)
    ap.add_argument("--sweep-deg", type=float, default=360.0)
    ap.add_argument("--snr", type=float, nargs="*", default=[40, 30, 20, 15, 10, 5])
    ap.add_argument("--dims", type=int, nargs="*", default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--codes", type=int, nargs="*", default=[64, 256, 1024, 4096])
    ap.add_argument("--pvq-groups", type=int, nargs="*", default=[8, 16, 32])
    ap.add_argument("--pvq-codes", type=int, nargs="*", default=[256, 1024])
    ap.add_argument("--rvq-stages", type=int, nargs="*", default=[4, 8, 16])
    ap.add_argument("--rvq-codes", type=int, nargs="*", default=[1024])
    ap.add_argument("--bits", type=int, nargs="*", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--out", default="outputs/probe_token_sensitivity.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = ZipSplat(weights=args.weights).cuda().eval()
    images = load_inputs(args.input, args.num_frames)
    print(f"Loaded {len(images)} view(s) from {args.input}")

    tokens = compute_tokens(model, images, args.compression)
    b, k, dim = tokens.shape
    print(f"Head input tokens: {(b, k, dim)}  (perturbing '{args.target}' slice)")

    head = model.model.gaussian_head
    with torch.no_grad():
        ref_gaussians = head(tokens)[0]

    # Trajectory is derived from the baseline scene once, so every variant is
    # scored from identical viewpoints.
    center, radius = viz.scene_center_radius(ref_gaussians)
    poses = viz.orbit_poses(center, radius, args.eval_views, sweep_deg=args.sweep_deg)
    poses = poses.to(ref_gaussians.device)
    cam = Camera.from_fov(
        torch.tensor(math.radians(55.0)), w=args.render_size, h=args.render_size
    ).to(ref_gaussians.device)
    cameras = Camera(cam.data_.unsqueeze(0).expand(args.eval_views, -1).clone())
    ref_render = render_orbit(ref_gaussians, cameras, poses)
    print(f"Baseline: {ref_gaussians.num_gaussians} gaussians, radius {radius:.3f}")

    lo, hi = token_slice(model, args.target)
    args.token_dim = hi - lo
    flat = tokens.reshape(-1, dim)
    results = []

    for name, transform, bits in build_sweep(args):
        with torch.no_grad():
            perturbed = flat.clone()
            perturbed[:, lo:hi] = transform(flat[:, lo:hi])
            rel_err = float(
                (perturbed[:, lo:hi] - flat[:, lo:hi]).norm() / flat[:, lo:hi].norm()
            )
            gaussians = head(perturbed.reshape(b, k, dim))[0]
            render = render_orbit(gaussians, cameras, poses)

        row = {
            "name": name,
            "bits_per_token": bits,
            "token_rel_err": rel_err,
            "psnr": float(psnr(render, ref_render).mean()),
            "ssim": float(ssim(render, ref_render).mean()),
            **gaussian_drift(gaussians, ref_gaussians, radius),
        }
        results.append(row)
        print(
            f"{name:>16}  relerr {rel_err:6.3f}  PSNR {row['psnr']:7.2f}  "
            f"SSIM {row['ssim']:.4f}  drift {row['mean_shift_pct']:6.2f}%"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "input": args.input,
                "views": len(images),
                "target": args.target,
                "token_shape": [b, k, dim],
                "num_gaussians": ref_gaussians.num_gaussians,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
