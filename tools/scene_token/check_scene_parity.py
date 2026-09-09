"""Check full Giant training/inference parity after identical scene initialization."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from omegaconf import OmegaConf
from splatfactory.models.networks.zipsplat import ZipSplat as TrainingModel
from zipsplat.zipsplat import ZipSplat as InferenceModel
from zipsplat.predictor import _model_conf_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(25)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    train = (
        TrainingModel(
            dict(
                scene_tokens_enabled=True,
                num_scene_tokens=256,
                scene_token_init_from_base=True,
                weights="zipsplat",
                backbone=dict(weights=None),
                query_sample_ratio=[1.0, 1.0],
                query_scale_with_views=0.0,
                query_ratio_schedule=0.0,
                train_prior_probability=0.0,
                return_attention=False,
                skip_head_render=True,
                compile=False,
            )
        )
        .cuda()
        .eval()
    )
    model_conf = _model_conf_from_checkpoint(
        {"conf": {"model": OmegaConf.to_container(train.conf, resolve=True)}}
    )
    infer = InferenceModel(**model_conf).cuda().eval()
    infer.flexible_load(train.state_dict())
    images = torch.rand(1, 2, 3, 252, 252, device="cuda")
    results = {}
    for mixed in (False, True):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
            a = train({"context": {"image": images}})["gaussians"]
            b = infer(images)
        errors = {}
        for key in ("means", "scales", "quats", "opacities", "sh_coeffs"):
            x, y = getattr(a, key), getattr(b, key)
            torch.testing.assert_close(x, y, atol=5e-5, rtol=1e-4)
            errors[key] = float((x - y).abs().max())
        results["bfloat16" if mixed else "float32"] = errors
    args.output.write_text(
        json.dumps(dict(model_conf=model_conf, max_abs_errors=results), indent=2)
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
