"""P2 contracts, using small real attention blocks instead of the Giant for unit tests.

Run in the ZipSplat environment: python -m unittest discover -s tests -v
"""
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from zipsplat.backbone import DAV3DinoVisionTransformer as InferenceBackbone
from zipsplat.scene_tokens import load_scene_state, scene_count
from zipsplat.zipsplat import ZipSplat as InferenceModel
from splatfactory.models.encoders.dav3_dino import DAV3DinoVisionTransformer as TrainBackbone
from splatfactory.models.networks.zipsplat import ZipSplat as TrainModel

BASE = dict(
    img_size=28,
    patch_size=14,
    embed_dim=64,
    depth=6,
    num_heads=1,
    alt_start=3,
    qknorm_start=3,
    rope_start=3,
    out_layers=[3, 5],
)


def infer_backbone(**kwargs):
    return InferenceBackbone(**(BASE | kwargs))


def train_backbone(**kwargs):
    return TrainBackbone(BASE | kwargs)


def training_conf(count=256, enabled=True):
    return dict(
        scene_tokens_enabled=enabled,
        num_scene_tokens=count,
        backbone=dict(
            vit_name="vits",
            embed_dim=64,
            out_layers=[3, 5],
            alt_start=3,
            qknorm_start=3,
            rope_start=3,
            weights=None,
        ),
        num_heads=1,
        img_size=28,
        weights=None,
        compile=False,
        return_attention=False,
        query_sample_ratio=[1.0, 1.0],
        query_ratio_schedule=0.0,
        query_scale_with_views=0.0,
        train_prior_probability=0.0,
        skip_head_render=True,
    )


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def test_disabled_backbone_matches_git_baseline(self):
        root = Path(__file__).resolve().parents[1]
        for path, new_cls in [
            ("zipsplat/backbone.py", InferenceBackbone),
            ("splatfactory/models/encoders/dav3_dino.py", TrainBackbone),
        ]:
            source = subprocess.check_output(["git", "show", "c674892:" + path], cwd=root)
            module = types.ModuleType("scene_test_baseline")
            exec(compile(source, path, "exec"), module.__dict__)
            old_cls = module.DAV3DinoVisionTransformer
            old = old_cls(**BASE) if new_cls is InferenceBackbone else old_cls(BASE)
            new = new_cls(**BASE) if new_cls is InferenceBackbone else new_cls(BASE)
            new.load_state_dict(old.state_dict(), strict=True)
            image = torch.rand(1, 2, 3, 28, 42)
            a = (
                old(image)
                if new_cls is InferenceBackbone
                else old.get_intermediate_layers(image, [3, 5])[0]
            )
            b = (
                new(image)
                if new_cls is InferenceBackbone
                else new.get_intermediate_layers(image, [3, 5])[0]
            )
            for x, y in zip(a, b):
                torch.testing.assert_close(x[0], y[0], rtol=0, atol=0)

    def test_layout_rope_layers_and_cross_view_information(self):
        for count in (3, 256):
            model = infer_backbone(num_scene_tokens=count, num_register_tokens=2)
            for views in (1, 2, 8, 16):
                image = torch.rand(1, views, 3, 28, 42)
                tokens = model.prepare_tokens(image)
                self.assertEqual(tokens.shape, (1, views, 3 + count + 6, 64))
                torch.testing.assert_close(tokens[0, 0, 3 : 3 + count], model.scene_tokens)
                pos, glob = model._prepare_rope(1, views, 28, 42, image.device)
                self.assertEqual(pos.shape[-2], tokens.shape[-2])
                self.assertEqual(pos[:, :, : 3 + count].count_nonzero(), 0)
                self.assertTrue((glob[:, :, 3 + count :] == 1).all())
                with torch.no_grad():
                    out = model(image)
                self.assertEqual([x[0].shape for x in out], [(1, views, count, 128)] * 2)
            image = torch.rand(1, 2, 3, 28, 28)
            changed = image.clone()
            changed[:, 1] += 0.4
            with torch.no_grad():
                a, b = model(image)[-1][0], model(changed)[-1][0]
            self.assertGreater((a[:, 0] - b[:, 0]).abs().max().item(), 1e-6)

    def test_train_inference_backbone_and_checkpoint_gradients(self):
        infer = infer_backbone(num_scene_tokens=7, num_register_tokens=2)
        train = train_backbone(num_scene_tokens=7, num_register_tokens=2, use_checkpoint=True)
        train.load_state_dict(infer.state_dict(), strict=True)
        image = torch.rand(2, 2, 3, 28, 42)
        cam = torch.rand(2, 2, 64)
        a = infer(image, cam_token=cam)
        b, aux = train.get_intermediate_layers(image, [3, 5], cam_token=cam, export_feat_layers=[5])
        self.assertEqual(aux[0].shape, (2, 2, 7, 64))
        for x, y in zip(a, b):
            torch.testing.assert_close(x[0], y[0], rtol=1e-5, atol=2e-6)
        sum(x[0].square().mean() for x in a).backward()
        sum(x[0].square().mean() for x in b).backward()
        torch.testing.assert_close(
            infer.scene_tokens.grad, train.scene_tokens.grad, rtol=1e-4, atol=2e-6
        )
        for model in (infer, train):
            self.assertTrue(torch.isfinite(model.scene_tokens.grad).all())
            self.assertGreater(model.scene_tokens.grad.abs().sum(), 0)

    def test_models_color_contract_and_freeze_gradient(self):
        with patch.dict(
            "zipsplat.backbone._VARIANT_FACTORIES", {"vits": infer_backbone}
        ), patch.dict(
            "splatfactory.models.encoders.dav3_encoder.encoder_map",
            {"vits": lambda **kw: train_backbone(**(kw | {"img_size": 28}))},
        ):
            infer = InferenceModel(
                vit_name="vits", out_layers=[3, 5], scene_tokens_enabled=True, num_scene_tokens=256
            )
            train = TrainModel(training_conf())
            train.flexible_load(infer.state_dict())
            image = torch.rand(1, 2, 3, 28, 28)
            colors = []
            hook = infer.color_cross_attention.register_forward_pre_hook(
                lambda _m, args: colors.append((args[0].shape, args[1].shape))
            )
            infer.eval()
            train.eval()
            with torch.no_grad():
                a = infer(image)
                b = train({"context": {"image": image}})["gaussians"]
            hook.remove()
            self.assertEqual(colors, [(torch.Size([1, 512, 128]), torch.Size([1, 8, 128]))])
            self.assertEqual(a.num_gaussians, 2 * 256 * 32)
            for key in ("means", "scales", "quats", "opacities", "sh_coeffs"):
                torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=1e-4, atol=2e-5)
            # Decoder parameters frozen; gradients must still cross decoder to slots.
            infer.requires_grad_(False)
            infer.backbone.backbone.scene_tokens.requires_grad_(True)
            g = infer(image)
            (g.means.square().mean() + g.sh_coeffs.square().mean()).backward()
            self.assertGreater(infer.backbone.backbone.scene_tokens.grad.abs().sum(), 0)
            train.train()
            train.zero_grad(set_to_none=True)
            g = train({"context": {"image": image}})["gaussians"]
            (g.means.square().mean() + g.sh_coeffs.square().mean()).backward()
            for name in (
                "backbone.backbone.scene_tokens",
                "scene_color_query.weight",
                "color_cross_attention.attn.q.weight",
            ):
                # Projection and slots are mandatory; CA names vary across module implementations.
                if name in dict(train.named_parameters()):
                    grad = dict(train.named_parameters())[name].grad
                    self.assertIsNotNone(grad, name)
                    self.assertTrue(torch.isfinite(grad).all(), name)
                    self.assertGreater(grad.abs().sum(), 0, name)
            with self.assertRaises(ValueError):
                infer(image, compression=0.75)
            bad = training_conf()
            bad["query_sample_ratio"] = [0.5, 1.0]
            with self.assertRaises(ValueError):
                TrainModel(bad)

    def test_checkpoint_guards_and_roundtrip(self):
        from zipsplat.predictor import ZipSplat as Predictor

        with patch.dict("zipsplat.backbone._VARIANT_FACTORIES", {"vits": infer_backbone}):
            base = InferenceModel(vit_name="vits", out_layers=[3, 5])
            scene = InferenceModel(vit_name="vits", out_layers=[3, 5], scene_tokens_enabled=True)
            with self.assertRaises(RuntimeError):
                load_scene_state(scene, base.state_dict())
            missing, extra = load_scene_state(scene, base.state_dict(), allow_base_init=True)
            self.assertEqual(len(missing), 3)
            self.assertFalse(extra)
            load_scene_state(
                scene, {"module._orig_mod." + k: v for k, v in scene.state_dict().items()}
            )
            bad = dict(base.state_dict())
            bad.pop("downscale.0.weight")
            with self.assertRaises(RuntimeError):
                load_scene_state(scene, bad, allow_base_init=True)
            with self.assertRaises(RuntimeError):
                base.flexible_load(scene.state_dict())
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.pt"
                torch.save({"model": scene.state_dict(), "model_conf": scene.conf}, path)
                loaded = Predictor(str(path))
                self.assertEqual(loaded.model.conf["num_scene_tokens"], 256)
                loaded.save_checkpoint(path)
                again = Predictor(str(path))
                for key, value in scene.state_dict().items():
                    torch.testing.assert_close(value, again.model.state_dict()[key], rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    Predictor(str(path), num_scene_tokens=128)
        for count in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                scene_count(True, count)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_mixed_precision_backward(self):
        for dtype in (torch.float16, torch.bfloat16):
            for factory in (infer_backbone, train_backbone):
                model = factory(num_scene_tokens=256, use_checkpoint=True).cuda().train()
                with torch.autocast("cuda", dtype=dtype):
                    x = torch.rand(1, 2, 3, 28, 28, device="cuda")
                    out = (
                        model(x)
                        if factory is infer_backbone
                        else model.get_intermediate_layers(x, [3, 5])[0]
                    )
                    loss = sum(y[0].float().square().mean() for y in out)
                loss.backward()
                self.assertTrue(torch.isfinite(model.scene_tokens.grad).all())
                self.assertGreater(model.scene_tokens.grad.abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
