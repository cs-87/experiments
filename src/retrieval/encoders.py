"""
Pretrained frame encoders behind one interface, so the experiment swaps models
by name and nothing else changes.

Every encoder returns L2-normalised float32 rows, which makes inner product
equal cosine similarity and lets both the torch and FAISS backends use the same
IndexFlatIP semantics.

Model choice is driven by the video-copy-detection literature rather than by
general-purpose retrieval leaderboards: the VSC22 baseline and all three top
VSC22 finishers extract a descriptor PER FRAME with an image model and do
temporal reasoning separately, so per-frame image encoders are what belong here
and clip-level video encoders (VideoMAE, InternVideo2, V-JEPA2) do not -- they
pool 8-16 frames into one vector, which is the wrong temporal granularity for
frame-exact retrieval by construction.
"""
from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_CACHE = os.path.expanduser("~/.cache/leak_retrieval")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_size(*candidates, default: int = 224) -> int:
    """
    Pull a square input size out of whatever an image processor exposes.

    transformers 5.x returns a SizeDict object rather than a plain dict, and the
    useful field varies by model (crop_size.height for DINOv2, size.height for
    SigLIP2, shortest_edge for others), so this probes them in preference order.
    """
    for c in candidates:
        if c is None:
            continue
        if isinstance(c, (int, float)):
            return int(c)
        for key in ("height", "shortest_edge", "width", "longest_edge"):
            v = c.get(key) if isinstance(c, dict) else getattr(c, key, None)
            if v:
                return int(v)
    return default


def _to_tensor(frames_bgr: list[np.ndarray], size: int,
               mean: tuple, std: tuple, device: str) -> torch.Tensor:
    """BGR uint8 HxWx3 list -> normalised NCHW float tensor at size x size."""
    batch = np.stack([
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (size, size),
                   interpolation=cv2.INTER_AREA)
        for f in frames_bgr
    ])
    x = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


class Encoder:
    """Interface: encode a list of BGR uint8 frames into L2-normalised rows."""

    name: str = "base"
    dim: int = 0
    input_size: int = 224

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        return _to_tensor(frames, self.input_size, IMAGENET_MEAN,
                          IMAGENET_STD, self.device)

    @torch.no_grad()
    def encode(self, frames: Iterable[np.ndarray], batch_size: int = 32,
               tile: int = 1) -> np.ndarray:
        """
        tile > 1 splits each frame into a tile x tile grid, embeds every cell,
        and concatenates. Region embeddings are the standard lever for
        fine-grained retrieval: a global vector summarises the whole scene, and
        on near-static content the between-frame difference lives in a small
        region that global pooling averages away.
        """
        frames = list(frames)
        out = []
        for i in range(0, len(frames), batch_size):
            chunk = frames[i:i + batch_size]
            if tile == 1:
                v = self._embed_batch(chunk)
            else:
                cells = []
                for f in chunk:
                    h, w = f.shape[:2]
                    for r in range(tile):
                        for c in range(tile):
                            cells.append(f[r * h // tile:(r + 1) * h // tile,
                                           c * w // tile:(c + 1) * w // tile])
                e = self._embed_batch(cells).reshape(len(chunk), tile * tile, -1)
                # Renormalise after concatenation so every frame is unit length
                # and no single cell can dominate the inner product.
                v = e.reshape(len(chunk), -1)
                v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
            out.append(v)
        return np.concatenate(out, axis=0).astype(np.float32)

    def _embed_batch(self, frames: list[np.ndarray]) -> np.ndarray:
        x = self._preprocess(frames)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=self.device.startswith("cuda")):
            f = self._forward(x)
        f = F.normalize(f.float(), dim=-1)
        return f.cpu().numpy()


class SSCDEncoder(Encoder):
    """
    Self-Supervised Copy Detection descriptor (Pizzi et al., CVPR 2022).

    The most on-task pretrained model that exists: it is trained specifically so
    that an edited/re-encoded copy of an image lands near its source, and it is
    the per-frame descriptor Meta's own VSC22 video-copy-detection baseline uses.
    Its training augmentations do NOT include perspective warp or moire, so it is
    tested against but not trained for camcording -- which is exactly what this
    experiment measures.

    Shipped as TorchScript, so there is no architecture code to install.
    """

    URLS = {
        "sscd_disc_mixup": ("https://dl.fbaipublicfiles.com/sscd-copy-detection/"
                            "sscd_disc_mixup.torchscript.pt", 512),
        "sscd_disc_large": ("https://dl.fbaipublicfiles.com/sscd-copy-detection/"
                            "sscd_disc_large.torchscript.pt", 1024),
        "sscd_imagenet_mixup": ("https://dl.fbaipublicfiles.com/sscd-copy-detection/"
                                "sscd_imagenet_mixup.torchscript.pt", 512),
    }

    def __init__(self, variant: str = "sscd_disc_mixup", device: str = "cuda",
                 input_size: int = 288):
        import urllib.request

        url, dim = self.URLS[variant]
        os.makedirs(_CACHE, exist_ok=True)
        path = os.path.join(_CACHE, os.path.basename(url))
        if not os.path.exists(path):
            print(f"downloading {variant} ...")
            urllib.request.urlretrieve(url, path)

        self.device = device
        self.model = torch.jit.load(path, map_location=device).eval()
        self.name = variant
        self.dim = dim
        # 288 is what the SSCD repo recommends for DISC-style evaluation; the
        # descriptor is resolution-sensitive and 224 measurably underperforms.
        self.input_size = input_size

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _embed_batch(self, frames):
        # TorchScript module traced in fp32; autocast around it is unreliable.
        x = self._preprocess(frames)
        f = self.model(x)
        return F.normalize(f.float(), dim=-1).cpu().numpy()


class HFVisionEncoder(Encoder):
    """
    Any HuggingFace vision backbone exposing pooled/CLS features.

    Covers DINOv2 (Apache-2.0, the clean general-purpose choice) and DINOv3
    (better dense/correspondence scores but gated weights under a custom,
    non-Apache licence -- check the terms before shipping anything with it).
    """

    def __init__(self, model_id: str = "facebook/dinov2-base", device: str = "cuda",
                 input_size: int | None = None, pool: str = "cls_avg"):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(model_id)

        self.name = model_id.split("/")[-1]
        self.pool = pool

        self.input_size = input_size or _resolve_size(
            getattr(proc, "crop_size", None), getattr(proc, "size", None))

        self._mean = tuple(getattr(proc, "image_mean", IMAGENET_MEAN))
        self._std = tuple(getattr(proc, "image_std", IMAGENET_STD))

        hidden = self.model.config.hidden_size
        # cls_avg concatenates the CLS token with the mean of the patch tokens.
        # The patch mean is what carries spatial detail, and on near-static
        # content that is where the between-frame signal lives.
        self.dim = hidden * 2 if pool == "cls_avg" else hidden

    def _preprocess(self, frames):
        return _to_tensor(frames, self.input_size, self._mean, self._std, self.device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x).last_hidden_state
        cls, patches = out[:, 0], out[:, 1:]
        if self.pool == "cls":
            return cls
        if self.pool == "avg":
            return patches.mean(1)
        return torch.cat([F.normalize(cls, dim=-1),
                          F.normalize(patches.mean(1), dim=-1)], dim=-1)


class CLIPLikeEncoder(Encoder):
    """
    CLIP / SigLIP2 image tower.

    Included as a control, not a contender. Multiple benchmarks (MMVP's
    CLIP-blind pairs, ILIAS, the erroneous-agreements study showing >0.99 cosine
    between an image and its mirror) show CLIP-family embeddings collapse
    exactly the fine visual distinctions that separate adjacent frames. Measuring
    that here turns a literature claim into a number on your own data.
    """

    def __init__(self, model_id: str = "google/siglip2-base-patch16-224",
                 device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.model = getattr(model, "vision_model", model)
        self._head = getattr(model, "visual_projection", None)

        proc = AutoImageProcessor.from_pretrained(model_id)
        self.input_size = _resolve_size(
            getattr(proc, "size", None), getattr(proc, "crop_size", None))
        self._mean = tuple(getattr(proc, "image_mean", (0.5, 0.5, 0.5)))
        self._std = tuple(getattr(proc, "image_std", (0.5, 0.5, 0.5)))

        self.name = model_id.split("/")[-1]
        with torch.no_grad():
            probe = torch.zeros(1, 3, self.input_size, self.input_size, device=device)
            self.dim = int(self._forward(probe).shape[-1])

    def _preprocess(self, frames):
        return _to_tensor(frames, self.input_size, self._mean, self._std, self.device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        f = getattr(out, "pooler_output", None)
        if f is None:
            f = out.last_hidden_state.mean(1)
        return self._head(f) if self._head is not None else f


class PixelBaseline(Encoder):
    """
    High-pass luma, downsampled, mean-removed and L2-normalised: cosine
    similarity here is exactly ZNCC on high-pass luma.

    The floor every learned encoder must clear. On near-static content this
    baseline is not merely weak, it is actively misleading -- measured on
    4_sec_source, a MILDLY degraded frame scores 0.237 against its own original
    while an undistorted frame 20 positions away scores 0.270, so the argmax is
    wrong by construction. Reporting it makes that failure legible.
    """

    def __init__(self, device: str = "cuda", size: int = 64, sigma: float = 2.0):
        self.device, self.name = device, f"pixel_hp{size}"
        self.input_size, self.sigma = size, sigma
        self.dim = size * size

    def encode(self, frames, batch_size: int = 32, tile: int = 1) -> np.ndarray:
        out = []
        for f in frames:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
            g = g - cv2.GaussianBlur(g, (0, 0), self.sigma)
            v = g.ravel() - g.mean()
            out.append(v / max(np.linalg.norm(v), 1e-8))
        return np.stack(out).astype(np.float32)


REGISTRY = {
    "sscd": lambda d: SSCDEncoder("sscd_disc_mixup", d),
    "sscd_large": lambda d: SSCDEncoder("sscd_disc_large", d),
    "dinov2_base": lambda d: HFVisionEncoder("facebook/dinov2-base", d),
    "dinov2_large": lambda d: HFVisionEncoder("facebook/dinov2-large", d),
    "dinov3_base": lambda d: HFVisionEncoder(
        "facebook/dinov3-vitb16-pretrain-lvd1689m", d),
    "siglip2": lambda d: CLIPLikeEncoder("google/siglip2-base-patch16-224", d),
    "clip": lambda d: CLIPLikeEncoder("openai/clip-vit-base-patch16", d),
    "pixel": lambda d: PixelBaseline(d),
}


def build_encoder(name: str, device: str = "cuda") -> Encoder:
    if name not in REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](device)
