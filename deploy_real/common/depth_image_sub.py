"""D435i depth image DDS subscriber with online encoding.

Self-contained: includes DDS message types, VAENet encoder model, and
DepthImageObserver. No dependency on the d435i_test directory.

Typical usage
-------------
    # ChannelFactoryInitialize must be called before creating the observer.
    observer = DepthImageObserver(
        topic="rt/depth_image",
        min_depth=0.25,
        max_depth=10.0,
        target_resolution=(64, 40),   # (width, height)
        encoder_path="pre_train/depth_encoder/vae_pretrain_new.pth",
        feature_dim=64,
    )

    # In the control loop:
    depth_feature = observer.get_latest()   # np.ndarray [feature_dim]
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import regnet_x_400mf
from torchvision.ops import Conv2dNormActivation, FeaturePyramidNetwork

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_


# ---------------------------------------------------------------------------
# DDS message types  (mirrors depth_image_dds.py in d435i_test)
# ---------------------------------------------------------------------------

@annotate.final
@annotate.autoid("sequential")
class DepthIntrinsics_(idl.IdlStruct, typename="DepthIntrinsics_"):
    """Camera intrinsic parameters."""
    fx: types.float64
    fy: types.float64
    cx: types.float64
    cy: types.float64


@annotate.final
@annotate.autoid("sequential")
class DepthImage_(idl.IdlStruct, typename="DepthImage_"):
    """Raw uint16 z16 depth frame published by the D435i bridge node."""
    header: Header_
    width: types.uint32
    height: types.uint32
    depth_scale: types.float32    # metres per uint16 step
    intrinsics: DepthIntrinsics_
    data: types.sequence[types.uint8]  # raw uint16 bytes, len = width*height*2


def _decode_depth_message(msg: DepthImage_) -> np.ndarray:
    """Decode raw bytes to uint16 depth array [H, W]."""
    return np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)


# ---------------------------------------------------------------------------
# Depth encoder model  (inference-only subset of VAENet from isaaclab_depth_noise.py)
# The VAEDecoder and DepthNoise modules are not needed at deployment time.
# ---------------------------------------------------------------------------

class _VAESampler(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.conv = Conv2dNormActivation(
            input_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.mean_layers = nn.Sequential(
            Conv2dNormActivation(
                latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.mean_layers(x)


class _DepthEncoder(nn.Module):
    def __init__(self, out_channel: int):
        super().__init__()
        backbone = regnet_x_400mf(weights=None)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Replace first conv to accept single-channel depth input
        backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc   = backbone[0]
        self.enc_1 = backbone[1][:2]
        self.enc_2 = backbone[1][2]
        self.enc_3 = backbone[1][3]
        self.fpn   = FeaturePyramidNetwork([64, 160, 400], out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = OrderedDict()
        x           = self.enc(x)
        out["feat1"] = self.enc_1(x)
        out["feat2"] = self.enc_2(out["feat1"])
        out["feat3"] = self.enc_3(out["feat2"])
        return self.fpn(out)["feat1"]


class _VAENet(nn.Module):
    """VAE encoder stack (without decoder); matches checkpoint keys."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.depth_encoder = _DepthEncoder(latent_dim)
        self.vae_sampler   = _VAESampler(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns spatial feature map [B, latent_dim, H', W']."""
        return self.vae_sampler(self.depth_encoder(x))


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class DepthImageObserver:
    """Subscribes to D435i depth images over DDS, encodes each frame with
    VAENet, and caches the resulting flat feature vector.

    ChannelFactoryInitialize() must have been called before constructing this.

    Parameters
    ----------
    topic:             DDS topic string, e.g. "rt/depth_image"
    min_depth:         Minimum valid depth in metres (pixels below → 0)
    max_depth:         Maximum valid depth in metres (pixels above → 0)
    target_resolution: (width, height) to crop the incoming frame to
    encoder_path:      Path to vae_pretrain_new.pth checkpoint
    feature_dim:       Latent dimension (must match checkpoint, typically 64)
    device:            Torch device string, e.g. "cpu" or "cuda:0"
    """

    def __init__(
        self,
        topic: str,
        min_depth: float,
        max_depth: float,
        target_resolution: tuple[int, int],
        encoder_path: str,
        feature_dim: int,
        device: str = "cpu",
    ):
        self.min_depth  = min_depth
        self.max_depth  = max_depth
        self.target_w, self.target_h = target_resolution  # (W, H)
        self.device     = torch.device(device)
        self.feature_dim = feature_dim

        # Initialise cache with zeros so callers always get a valid array
        self._latest_feature: np.ndarray = np.zeros(feature_dim, dtype=np.float32)

        # Load encoder weights.  Only keep keys for depth_encoder and
        # vae_sampler; the checkpoint may also contain depth_decoder keys
        # that are not needed for inference.
        self._encoder = _VAENet(feature_dim).to(self.device)
        self._encoder.eval()
        checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=True)
        encoder_keys = {
            k: v for k, v in checkpoint.items()
            if k.startswith("depth_encoder.") or k.startswith("vae_sampler.")
        }
        self._encoder.load_state_dict(encoder_keys, strict=True)
        print(f"[DepthImageObserver] Loaded encoder from: {encoder_path}")

        # DDS subscriber
        self._sub = ChannelSubscriber(topic, DepthImage_)
        self._sub.Init(self._on_message, 10)
        print(f"[DepthImageObserver] Subscribed to topic: {topic}")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, depth_image: np.ndarray, depth_scale: float) -> torch.Tensor:
        """uint16 depth → float32 metres tensor [1, 1, H, W] with validity mask."""
        depth_m = depth_image.astype(np.float32) * depth_scale
        t = torch.from_numpy(depth_m)
        t = torch.nan_to_num(t, nan=50.0, posinf=50.0, neginf=0.0)

        h, w = t.shape
        t = t.view(1, 1, h, w)

        # Crop to target resolution if larger
        if h > self.target_h or w > self.target_w:
            t = t[:, :, : self.target_h, : self.target_w]

        # Zero out pixels outside valid range
        t = t.clone()
        t[t > self.max_depth] = 0.0
        t[t < self.min_depth] = 0.0
        return t

    # ------------------------------------------------------------------
    # DDS callback
    # ------------------------------------------------------------------

    def _on_message(self, msg: DepthImage_):
        try:
            depth_image = _decode_depth_message(msg)
            depth_tensor = self._preprocess(depth_image, msg.depth_scale).to(self.device)
            with torch.no_grad():
                feat = self._encoder(depth_tensor)       # [1, C, H', W']
                feat = feat.mean(dim=[2, 3]).squeeze(0)  # [C]  – global avg pool
            self._latest_feature = feat.cpu().numpy().astype(np.float32)
        except Exception as exc:
            print(f"[DepthImageObserver] Error processing frame: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest(self) -> np.ndarray:
        """Return the most recently encoded depth feature (copy, shape [feature_dim])."""
        return self._latest_feature.copy()
