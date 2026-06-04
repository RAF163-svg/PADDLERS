#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlers.models.paddleseg.models.unet import Decoder, Encoder


class ClusterRoutedUNet(nn.Layer):
    """
    Shared UNet encoder/decoder trunk with cluster-routed segmentation heads.

    Routing semantics:
    - `cluster_id` selects exactly one head for each sample.
    - All heads predict the same 16-class GID15 label space.
    - `cluster_id` is never used as pixel supervision.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        num_heads: int,
        in_channels: int = 3,
        use_deconv: bool = False,
        align_corners: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_heads = int(num_heads)
        self.in_channels = int(in_channels)
        self.use_deconv = bool(use_deconv)
        self.align_corners = bool(align_corners)

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")

        self.encode = Encoder(self.in_channels)
        self.decode = Decoder(self.align_corners, use_deconv=self.use_deconv)
        self.heads = nn.LayerList(
            [
                nn.Conv2D(
                    in_channels=64,
                    out_channels=self.num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                for _ in range(self.num_heads)
            ]
        )

    def forward_shared(self, x):
        encoded, short_cuts = self.encode(x)
        decoded = self.decode(encoded, short_cuts)
        return decoded

    def route_logits(self, all_logits, cluster_ids, return_mask: bool = False):
        if all_logits.ndim != 5:
            raise ValueError(f"Expected all_logits shape [B, H, C, H, W], got {all_logits.shape}")

        batch_size = int(all_logits.shape[0])
        if self.num_heads == 1:
            routing_mask = paddle.ones([batch_size, 1], dtype=all_logits.dtype)
            routed_logits = all_logits[:, 0]
            if return_mask:
                return routed_logits, routing_mask
            return routed_logits

        if cluster_ids is None:
            raise ValueError("cluster_ids must be provided when num_heads > 1")
        if cluster_ids.ndim != 1:
            cluster_ids = paddle.reshape(cluster_ids, [-1])
        routing_mask = F.one_hot(cluster_ids.astype("int64"), num_classes=self.num_heads).astype(all_logits.dtype)
        routed_logits = paddle.sum(
            all_logits * routing_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            axis=1,
        )
        if return_mask:
            return routed_logits, routing_mask
        return routed_logits

    def forward(
        self,
        x,
        cluster_ids=None,
        return_all_heads: bool = False,
        retain_all_head_grads: bool = False,
    ):
        decoded = self.forward_shared(x)
        all_head_logits = paddle.stack([head(decoded) for head in self.heads], axis=1)
        if retain_all_head_grads:
            all_head_logits.retain_grads()
        routed_logits, routing_mask = self.route_logits(all_head_logits, cluster_ids, return_mask=True)

        payload = {
            "logits": routed_logits,
            "aux_logits": None,
            "cluster_ids": cluster_ids,
        }
        if return_all_heads:
            payload["routing_mask"] = routing_mask
            payload["all_head_logits"] = list(paddle.unstack(all_head_logits, axis=1))
            payload["all_head_logits_tensor"] = all_head_logits
        return payload


def load_pretrained_weights(model: ClusterRoutedUNet, pretrained_path: str | Path) -> dict[str, object]:
    pretrained_path = str(Path(pretrained_path).resolve())
    state_dict = paddle.load(pretrained_path)
    model_state = model.state_dict()

    num_loaded = 0
    loaded_keys: list[str] = []
    head_broadcast_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, tensor in model_state.items():
        source_key = None
        if key in state_dict and list(state_dict[key].shape) == list(tensor.shape):
            source_key = key
        elif key.startswith("heads."):
            suffix = ".".join(key.split(".")[2:])
            for candidate in (f"cls.{suffix}", f"conv.{suffix}"):
                if candidate in state_dict and list(state_dict[candidate].shape) == list(tensor.shape):
                    source_key = candidate
                    head_broadcast_keys.append(key)
                    break

        if source_key is None:
            skipped_keys.append(key)
            continue

        model_state[key] = state_dict[source_key]
        loaded_keys.append(key)
        num_loaded += 1

    model.set_state_dict(model_state)
    return {
        "pretrained_path": pretrained_path,
        "num_model_tensors": len(model_state),
        "num_loaded_tensors": num_loaded,
        "loaded_keys_preview": loaded_keys[:50],
        "head_broadcast_keys_preview": head_broadcast_keys[:50],
        "skipped_keys_preview": skipped_keys[:50],
    }


def build_model_structure(config: dict, num_classes: int) -> dict[str, object]:
    routing_cfg = config["routing"]
    model_cfg = config["model"]
    return {
        "model_name": "ClusterRoutedUNet",
        "num_heads": int(routing_cfg["num_heads"]),
        "head_names": list(routing_cfg["cluster_names"]),
        "selected_k": int(routing_cfg["selected_k"]),
        "cluster_id_handling": routing_cfg.get("cluster_id_handling", "route"),
        "sample_path_source": routing_cfg.get("sample_path_source", "input"),
        "shared_modules": ["UNetEncoder", "UNetDecoder"],
        "segmentation_head_type": "Conv2D(kernel=3)",
        "segmentation_head_count": int(routing_cfg["num_heads"]),
        "segmentation_num_classes": int(num_classes),
        "in_channels": int(model_cfg.get("in_channels", 3)),
        "use_deconv": bool(model_cfg.get("use_deconv", False)),
        "align_corners": bool(model_cfg.get("align_corners", False)),
        "routing_rule": "Each sample uses exactly one segmentation head selected by cluster_id.",
        "supervision_rule": "The selected head is supervised by the sample's original segmentation mask.",
        "evaluation_rule": "OA / mAcc / mIoU / Kappa / per-class metrics are computed in the original 16-class GID15 label space.",
    }
