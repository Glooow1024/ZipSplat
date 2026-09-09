"""Real released-checkpoint P2 shape/gradient/profile smoke; no optimization/training.

Examples:
  python tools/scene_token/validate_scene_model.py --backend inference --views 1 2 8 16 --output /path/inference.json
  python tools/scene_token/validate_scene_model.py --backend training --views 1 2 8 16 --backward --output /path/warmup.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["inference", "training"], required=True)
    parser.add_argument("--views", type=int, nargs="+", default=[1, 2, 8, 16])
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--joint", action="store_true", help="Include existing DA3 parameter gradients"
    )
    parser.add_argument(
        "--render-loss",
        action="store_true",
        help="Training backend: 64px rasterized RGB loss instead of parameter loss",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.render_loss and (args.backend != "training" or not args.backward):
        parser.error("--render-loss requires --backend training --backward")
    torch.manual_seed(19)
    torch.set_num_threads(4)
    if args.backend == "inference":
        from zipsplat.predictor import ZipSplat

        model = ZipSplat(
            scene_tokens_enabled=True,
            num_scene_tokens=256,
            scene_token_init_from_base=True,
            use_checkpoint=True,
        ).model
        if not args.joint:
            model.backbone.requires_grad_(False)
            model.backbone.backbone.scene_tokens.requires_grad_(True)
    else:
        from splatfactory.models.networks.zipsplat import ZipSplat

        model = ZipSplat(
            dict(
                scene_tokens_enabled=True,
                num_scene_tokens=256,
                scene_token_init_from_base=True,
                weights="zipsplat",
                backbone=dict(weights=None, use_checkpoint=True),
                freeze_backbone_except_scene=not args.joint,
                query_sample_ratio=[1.0, 1.0],
                query_ratio_schedule=0.0,
                query_scale_with_views=0.0,
                train_prior_probability=0.0,
                eval_use_priors=False,
                use_checkpoint=True,
                return_attention=False,
                skip_head_render=True,
                compile=False,
            )
        )
    model.cuda().train(args.backward)
    records = []
    report = dict(
        backend=args.backend,
        num_scene_tokens=256,
        backward=args.backward,
        joint=args.joint,
        dtype="autocast_bfloat16",
        render_loss=args.render_loss,
        note="Random RGB; untrained new slots; not quality or optimizer throughput. Render-loss mode includes a 64px rasterization; otherwise Gaussian parameter loss.",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        records=records,
    )
    for views in args.views:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        x = torch.rand(1, views, 3, 252, 252, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.set_grad_enabled(args.backward), torch.autocast("cuda", dtype=torch.bfloat16):
            features = (
                model._backbone_features(x, None, None)
                if args.backend == "inference"
                else model._backbone_features({"context": {"image": x}})
            )
            shapes = [list(f.shape) for f in features]
            assert shapes == [[1, views, 256, 3072]] * 3, shapes
        del features  # The actual end-to-end pass below is measured separately.
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.set_grad_enabled(args.backward), torch.autocast("cuda", dtype=torch.bfloat16):
            g = (
                model(x)
                if args.backend == "inference"
                else model({"context": {"image": x}})["gaussians"]
            )
            assert g.num_gaussians == views * 256 * 32
            loss = g.means.float().square().mean() + g.sh_coeffs.float().square().mean()
        if args.render_loss:
            from splatfactory.geometry import Camera, Pose

            points = g.means.detach()[0]
            center = points.median(dim=0).values
            extent = (
                (points.quantile(0.95, dim=0) - points.quantile(0.05, dim=0)).max().clamp(min=1.0)
            )
            eye = center.clone()
            eye[2] = points[:, 2].min() - extent
            pose = Pose.from_Rt(torch.eye(3, device="cuda")[None, None], eye[None, None])
            fov = torch.ones(1, 1, device="cuda")
            camera = Camera.from_fov(fov, fov, w=64, h=64)
            rendering = g.render_view(camera, pose)["rendering"]
            target = torch.nn.functional.interpolate(
                x[:, 0], size=(64, 64), mode="bilinear", align_corners=False
            )
            loss = (rendering[:, 0] - target).square().mean()
        torch.cuda.synchronize()
        forward_s = time.perf_counter() - start
        grads = {}
        if args.backward:
            start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_s = time.perf_counter() - start
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all(), name
            for name in [
                "backbone.backbone.scene_tokens",
                "scene_color_query.weight",
                "downscale.0.weight",
                "gaussian_head.gaussian_head.1.weight",
            ]:
                parameter = dict(model.named_parameters())[name]
                assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
                grads[name] = float(parameter.grad.norm())
            if args.joint:
                parameter = model.backbone.backbone.blocks[-1].attn.qkv.weight
                assert parameter.grad is not None and parameter.grad.abs().sum() > 0
                grads["last_backbone_qkv"] = float(parameter.grad.norm())
        else:
            backward_s = None
        record = dict(
            views=views,
            features=shapes,
            K=views * 256,
            gaussians=g.num_gaussians,
            forward_seconds=forward_s,
            backward_seconds=backward_s,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            gradients=grads,
        )
        records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(record), flush=True)
        del g, loss, x


if __name__ == "__main__":
    main()
