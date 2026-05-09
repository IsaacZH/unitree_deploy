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
    depth_feature = observer.get_latest()   # np.ndarray [feature_dim * H' * W']
"""

from collections import OrderedDict
from dataclasses import dataclass
import threading
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import regnet_x_400mf
from torchvision.ops import Conv2dNormActivation, FeaturePyramidNetwork

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_

from .depth_noise import DepthNoise


# ---------------------------------------------------------------------------
# DDS message types  (mirrors depth_image_dds.py in d435i_test)
# ---------------------------------------------------------------------------

@dataclass
@annotate.final
@annotate.autoid("sequential")
class DepthIntrinsics_(idl.IdlStruct, typename="DepthIntrinsics_"):
    """Camera intrinsic parameters."""
    fx: types.float64
    fy: types.float64
    cx: types.float64
    cy: types.float64


@dataclass
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
        # Keep this branch to stay checkpoint-compatible with training modules.
        self.logvar_layers = nn.Sequential(
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
    """Subscribes to D435i depth images over DDS.

    The DDS callback only caches the latest raw depth frame. Encoding is done
    lazily when get_latest() is called.

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
        enable_noise: bool = False,
        focal_length: float | None = None,
        baseline: float | None = None,
        use_jit_precompiled: bool = False,
        use_amp: bool = False,
        compile_encoder: bool = False,
        visualize_depth: bool = False,
        visualize_topic: str = "rt/depth_image_noisy",
    ):
        """Initialize depth image observer with optional noise injection.
        
        Parameters
        ----------
        topic:             DDS topic string, e.g. "rt/depth_image"
        min_depth:         Minimum valid depth in metres (pixels below → 0)
        max_depth:         Maximum valid depth in metres (pixels above → 0)
        target_resolution: (width, height) to crop the incoming frame to
        encoder_path:      Path to vae_pretrain_new.pth checkpoint
        feature_dim:       Latent dimension (must match checkpoint, typically 64)
        device:            Torch device string, e.g. "cpu" or "cuda:0"
        enable_noise:      If True, apply stereo depth noise before encoding (default False)
        focal_length:      Camera focal length in pixels (required if enable_noise=True)
        baseline:          Stereo baseline in metres (required if enable_noise=True)
        use_jit_precompiled: If True, use JIT-compiled encoder for speed (default False)
        use_amp:            If True, use FP16 autocast during encoder inference (Jetson Orin Tensor Core)
        compile_encoder:    If True, apply torch.compile to encoder after loading (PyTorch >= 2.0)
        visualize_depth:   If True, publish noisy depth frames to DDS for external visualization
        visualize_topic:   DDS topic used to publish noisy depth frames when visualize_depth=True
        """
        self.min_depth  = min_depth
        self.max_depth  = max_depth
        self.target_w, self.target_h = target_resolution  # (W, H)
        self.device     = torch.device(device)
        self.feature_dim = feature_dim
        self.enable_noise = enable_noise
        self.use_jit_precompiled = use_jit_precompiled
        self.use_amp = use_amp
        self.visualize_depth = visualize_depth
        self.visualize_topic = visualize_topic

        # Initialize noise simulator if enabled
        if enable_noise:
            if focal_length is None or baseline is None:
                raise ValueError(
                    f"enable_noise=True requires focal_length and baseline. "
                    f"Got focal_length={focal_length}, baseline={baseline}"
                )
            self._noise_simulator = DepthNoise(
                focal_length=focal_length,
                baseline=baseline,
                min_depth=min_depth,
                max_depth=max_depth,
            ).to(self.device)
            self._noise_simulator.eval()
            print(f"[DepthImageObserver] Depth noise enabled: focal_length={focal_length}, baseline={baseline}")
        else:
            self._noise_simulator = None

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

        if compile_encoder:
            try:
                self._encoder = torch.compile(self._encoder, mode="reduce-overhead")
                print("[DepthImageObserver] Encoder compiled with torch.compile (mode=reduce-overhead).")
            except Exception as exc:
                print(f"[DepthImageObserver] torch.compile skipped: {exc}")

        # Infer encoded map shape from target resolution so output dim matches
        # training-style flattened feature map.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.target_h, self.target_w, device=self.device, dtype=torch.float32)
            dummy_map = self._encoder(dummy)
        self._encoded_map_shape = tuple(int(x) for x in dummy_map.shape)  # (1, C, H', W')
        self._encoded_flat_dim = int(dummy_map.numel())

        # Initialise cache with zeros so callers always get a valid array
        self._latest_feature: np.ndarray = np.zeros(self._encoded_flat_dim, dtype=np.float32)
        self._latest_depth_image: np.ndarray | None = None
        self._latest_depth_scale: float = 0.0
        self._latest_viz_depth: torch.Tensor | None = None  # For visualization publish (raw or noisy)
        self._latest_intrinsics: DepthIntrinsics_ | None = None
        self._frame_version: int = 0
        self._encoded_version: int = -1
        self._lock = threading.Lock()

        self._viz_publisher = None
        if self.visualize_depth:
            self._viz_publisher = ChannelPublisher(self.visualize_topic, DepthImage_)
            self._viz_publisher.Init()
            print(f"[DepthImageObserver] Depth visualization publisher enabled: topic={self.visualize_topic}")

        # DDS subscriber
        self._sub = ChannelSubscriber(topic, DepthImage_)
        self._sub.Init(self._on_message, 10)
        print(f"[DepthImageObserver] Subscribed to topic: {topic}")
        if self.visualize_depth:
            print("[DepthImageObserver] Depth visualization mode: DDS publish (non-blocking)")

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

    def _apply_noise(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Apply stereo depth noise if enabled.
        
        Parameters
        ----------
        depth_tensor:  Preprocessed depth tensor [1, 1, H, W]
        
        Returns
        -------
        Noisy depth tensor [1, 1, H, W], or original if noise disabled.
        """
        if not self.enable_noise or self._noise_simulator is None:
            # When noise is disabled, visualize preprocessed raw depth.
            self._latest_viz_depth = depth_tensor.detach()
            return depth_tensor
        
        with torch.no_grad():
            noisy_depth = self._noise_simulator(depth_tensor, add_noise=True)
        
        # When noise is enabled, visualize noisy depth.
        self._latest_viz_depth = noisy_depth.detach()
        return noisy_depth

    def _publish_noisy_depth(self):
        """Publish latest visualization depth as DDS message for external visualization."""
        if not (self.visualize_depth and self._viz_publisher is not None and self._latest_viz_depth is not None):
            return

        try:
            viz_np = self._latest_viz_depth[0, 0].detach().cpu().numpy()
            # Enforce deploy-side depth validity before DDS publish:
            # values outside [min_depth, max_depth] are set to zero.
            valid = (viz_np >= float(self.min_depth)) & (viz_np <= float(self.max_depth))
            viz_np = np.where(valid, viz_np, 0.0)
            depth_uint16 = np.clip(np.rint(viz_np / 0.001), 0, np.iinfo(np.uint16).max).astype(np.uint16)
            height, width = depth_uint16.shape

            if self._latest_intrinsics is not None:
                intr = self._latest_intrinsics
            else:
                intr = DepthIntrinsics_(
                    fx=0.0,
                    fy=0.0,
                    cx=float(width) / 2.0,
                    cy=float(height) / 2.0,
                )

            now = time.time()
            msg = DepthImage_(
                header=Header_(
                    stamp=Time_(sec=int(now), nanosec=int((now % 1) * 1e9)),
                    frame_id="depth_noisy_viz",
                ),
                width=width,
                height=height,
                depth_scale=0.001,
                intrinsics=intr,
                data=depth_uint16.tobytes(),
            )
            self._viz_publisher.Write(msg)
        except Exception as exc:
            print(f"[DepthImageObserver] Noisy depth publish error: {exc}")

    # ------------------------------------------------------------------
    # DDS callback
    # ------------------------------------------------------------------

    def _on_message(self, msg: DepthImage_):
        try:
            depth_image = _decode_depth_message(msg).copy()
            with self._lock:
                self._latest_depth_image = depth_image
                self._latest_depth_scale = float(msg.depth_scale)
                self._latest_intrinsics = msg.intrinsics
                self._frame_version += 1
        except Exception as exc:
            print(f"[DepthImageObserver] Error processing frame: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest(self) -> np.ndarray:
        """Return latest flattened encoded depth feature (copy)."""
        with self._lock:
            if self._encoded_version == self._frame_version:
                self._publish_noisy_depth()
                return self._latest_feature.copy()
            if self._latest_depth_image is None:
                return self._latest_feature.copy()
            depth_image = self._latest_depth_image.copy()
            depth_scale = self._latest_depth_scale
            frame_version = self._frame_version

        # Preprocess: uint16 → float32 metres in [B, 1, H, W]
        depth_tensor = self._preprocess(depth_image, depth_scale).to(self.device)
        
        # Optional noise injection before encoding
        depth_tensor = self._apply_noise(depth_tensor)
        
        # Publish noisy depth for external visualizer if enabled (non-blocking)
        self._publish_noisy_depth()
        
        # Encode to feature
        with torch.no_grad():
            if self.use_amp and self.device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    feat_map = self._encoder(depth_tensor)   # [1, C, H', W']
            else:
                feat_map = self._encoder(depth_tensor)       # [1, C, H', W']
            feat = feat_map.contiguous().view(-1).float()    # [C * H' * W'] fp32
        feature = feat.cpu().numpy().astype(np.float32)

        with self._lock:
            if frame_version >= self._encoded_version:
                self._latest_feature = feature
                self._encoded_version = frame_version
            return self._latest_feature.copy()
