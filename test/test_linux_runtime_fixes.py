import importlib.util
import math
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import setup
from generator import _stage_external_images

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


class OptionalNattenPipCheckTests(unittest.TestCase):
    def test_accepts_only_missing_natten_when_wheelhouse_omits_it(self):
        lines = (
            "naf 0.1.0+modly requires natten, which is not installed.\n"
            "pixal3d-core 0.1.0+modly requires natten, which is not installed.\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            wheelhouse = Path(temp)
            self.assertTrue(setup._is_optional_natten_absent_check({"stdout_tail": lines}, wheelhouse))
            self.assertFalse(setup._is_optional_natten_absent_check(
                {"stdout_tail": lines + "foo 1.0 requires bar, which is not installed.\n"}, wheelhouse
            ))
            self.assertFalse(setup._is_optional_natten_absent_check({"stdout_tail": ""}, wheelhouse))
            (wheelhouse / "natten-0.21.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"")
            self.assertFalse(setup._is_optional_natten_absent_check({"stdout_tail": lines}, wheelhouse))


class StageExternalImagesTests(unittest.TestCase):
    def test_copies_outside_images_into_workspace_and_keeps_workspace_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            outside = Path(temp) / "Desktop" / "view.webp"
            outside.parent.mkdir()
            outside.write_bytes(b"image bytes")
            inside = workspace / "already.png"
            inside.write_bytes(b"png")

            staged = _stage_external_images([str(outside), str(inside), None], workspace)

            copied = Path(staged[0])
            self.assertEqual(copied.parent, workspace.resolve() / "Inputs" / "pixal3d-mv")
            self.assertEqual(copied.suffix, ".webp")
            self.assertEqual(copied.read_bytes(), b"image bytes")
            self.assertEqual(staged[1:], [str(inside), None])
            self.assertEqual(_stage_external_images([str(outside)], workspace), staged[:1])


class ReturnedLatentLifetimeTests(unittest.TestCase):
    def test_spills_the_tracebacks_four_gib_cuda_output_without_changing_dtype(self):
        from pixal3d_extension import runtime

        shape = (1, 1024, 1024, 4, 256)
        self.assertEqual(runtime._natten_output_size_bytes(shape, 4), 4 * 1024**3)
        self.assertTrue(runtime._should_spill_natten_output("cuda", shape, 4))
        self.assertFalse(runtime._should_spill_natten_output("cpu", shape, 4))

    def test_texture_fallback_tiles_columns_instead_of_allocating_a_full_row(self):
        from pixal3d_extension import runtime

        # Representative 1024 texture-stage dimensions. The old row-only
        # fallback could not go below a full 1024-column allocation.
        bytes_per_position = 4 * 81 * (2 * 1024 + 2 * 256 + 2) * 4
        tile_height, tile_width = runtime._natten_tile_shape(
            1024, 1024, bytes_per_position
        )

        self.assertEqual(tile_height, 1)
        self.assertGreater(tile_width, 1)
        self.assertLess(tile_width, 1024)
        self.assertLessEqual(
            tile_height * tile_width * bytes_per_position,
            runtime._NATTEN_CHUNK_TARGET_BYTES,
        )

    def test_strips_unused_latents_but_preserves_meshes_and_resolution(self):
        from pixal3d_extension import runtime

        meshes = [object()]
        result = runtime._strip_returned_latents(
            (meshes, (object(), object(), 1536))
        )

        self.assertIs(result[0], meshes)
        self.assertEqual(result[1], (None, None, 1536))


class PipelineMemoryPatchTests(unittest.TestCase):
    def test_run_uses_inference_mode_offloads_models_and_drops_latents(self):
        from pixal3d_extension import runtime

        calls = []

        class Model:
            def cpu(self):
                calls.append("model.cpu")
                return self

        class Pipeline:
            image_cond_model_ss = Model()
            image_cond_model_shape_512 = Model()
            image_cond_model_shape_1024 = Model()
            image_cond_model_tex_1024 = Model()
            rembg_model = Model()

            def cpu(self):
                calls.append("pipeline.cpu")
                return self

            def run(self, *args, **kwargs):
                del args, kwargs
                calls.append("run")
                return ["mesh"], ("shape", "texture", 1536)

        class MultiViewPipeline(Pipeline):
            def run_mv(self, *args, **kwargs):
                del args, kwargs
                calls.append("run_mv")
                return ["mv-mesh"], ("mv-shape", "mv-texture", 1536)

        for name in (
            "preprocess_image", "get_proj_cond_ss", "get_proj_cond_shape",
            "sample_sparse_structure", "sample_shape_slat", "sample_tex_slat",
            "decode_shape_slat", "decode_tex_slat",
        ):
            setattr(Pipeline, name, lambda self: None)

        pipeline_module = types.ModuleType("pixal3d.pipelines.pixal3d_image_to_3d")
        pipeline_module.Pixal3DImageTo3DPipeline = Pipeline
        pipelines_package = types.ModuleType("pixal3d.pipelines")
        pipelines_package.Pixal3DMVImageTo3DPipeline = MultiViewPipeline
        projection_module = types.ModuleType(
            "pixal3d.trainers.flow_matching.mixins.image_conditioned_proj"
        )

        class ProjGrid:
            def forward(self):
                return None

        projection_module.ProjGrid = ProjGrid

        class Extractor:
            use_naf_upsample = True

            def __init__(self):
                self.model = Model()

            def extract_features(self, image):
                calls.append("extract_features")
                return image

        projection_module.DinoV3ProjFeatureExtractor = Extractor
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = type("Tensor", (), {})
        fake_torch.inference_mode = nullcontext
        with patch.dict(sys.modules, {
            "torch": fake_torch,
            "pixal3d.pipelines": pipelines_package,
            "pixal3d.pipelines.pixal3d_image_to_3d": pipeline_module,
            "pixal3d.trainers.flow_matching.mixins.image_conditioned_proj": projection_module,
        }), patch.object(runtime, "_release_cuda_memory", side_effect=lambda: calls.append("release")):
            runtime._install_pipeline_memory_patches()
            pipeline = Pipeline()
            pipeline.preprocess_image()
            result = pipeline.run(return_latent=True)
            mv_result = MultiViewPipeline().run_mv(return_latent=True)
            extracted = Extractor().extract_features("features")

        self.assertEqual(result, (["mesh"], (None, None, 1536)))
        self.assertEqual(mv_result, (["mv-mesh"], (None, None, 1536)))
        self.assertIn("run", calls)
        self.assertIn("run_mv", calls)
        self.assertEqual(extracted, "features")
        self.assertIn("extract_features", calls)
        self.assertIn("pipeline.cpu", calls)
        self.assertGreaterEqual(calls.count("model.cpu"), 10)
        self.assertGreaterEqual(calls.count("release"), 6)


@unittest.skipUnless(HAS_TORCH, "torch is required for the natten fallback")
class NattenFallbackChunkingTests(unittest.TestCase):
    def _fallback(self):
        import torch  # noqa: F401
        from pixal3d_extension import runtime

        saved = sys.modules.pop("natten", None)

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name == "natten" or name.startswith("natten."):
                    raise ModuleNotFoundError("blocked", name="natten")

        blocker = Block()
        sys.meta_path.insert(0, blocker)
        try:
            runtime._install_natten_fallback()
            return sys.modules.pop("natten").na2d
        finally:
            sys.meta_path.remove(blocker)
            if saved is not None:
                sys.modules["natten"] = saved

    @staticmethod
    def _reference(q, k, v, kernel_size, dilation):
        import torch

        b, h, w, n, d = q.shape
        kh, kw = kernel_size
        dh, dw = dilation
        out = torch.zeros(b, h, w, n, v.shape[-1], dtype=q.dtype)
        for y in range(h):
            for x in range(w):
                ys = [y + (i - kh // 2) * dh for i in range(kh)]
                xs = [x + (j - kw // 2) * dw for j in range(kw)]
                keys = [(yy, xx) for yy in ys for xx in xs if 0 <= yy < h and 0 <= xx < w]
                kk = torch.stack([k[:, yy, xx] for yy, xx in keys], dim=2)
                vv = torch.stack([v[:, yy, xx] for yy, xx in keys], dim=2)
                scores = (q[:, y, x].unsqueeze(2) * kk).sum(-1) / math.sqrt(d)
                out[:, y, x] = (torch.softmax(scores, dim=-1).unsqueeze(-1) * vv).sum(2)
        return out

    def test_chunked_fallback_matches_reference_including_borders(self):
        import torch

        na2d = self._fallback()
        torch.manual_seed(0)
        for h, w, kernel, dilation, d_v in ((7, 9, (3, 3), (1, 1), 4), (11, 8, (3, 5), (2, 1), 6)):
            with self.subTest(h=h, w=w, kernel=kernel, dilation=dilation):
                q = torch.randn(2, h, w, 3, 4)
                k = torch.randn(2, h, w, 3, 4)
                v = torch.randn(2, h, w, 3, d_v)
                expected = self._reference(q, k, v, kernel, dilation)
                actual = na2d(q, k, v, kernel_size=kernel, dilation=dilation)
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertLess((actual - expected).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
