#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlers.models.paddleseg.models.backbones.hrnet import HRNet_W18, HRNet_W48
from paddlers.models.paddleseg.models.fcn import FCNHead


BACKBONE_FACTORY = {
    18: HRNet_W18,
    48: HRNet_W48,
}


class ClusterRoutedHRNet(nn.Layer):
    """
    Shared HRNet trunk with cluster-routed FCN segmentation heads.

    Routing semantics:
    - `cluster_id` selects exactly one segmentation head for each sample.
    - All heads predict the same 16-class GID15 label space.
    - `cluster_id` is never used as pixel supervision.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        num_heads: int,
        width: int = 48,
        in_channels: int = 3,
        head_channels: int | None = None,
        align_corners: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_heads = int(num_heads)
        self.width = int(width)
        self.in_channels = int(in_channels)
        self.align_corners = bool(align_corners)

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.width not in BACKBONE_FACTORY:
            raise ValueError(f"width must be one of {sorted(BACKBONE_FACTORY)}, got {self.width}")

        self.backbone = BACKBONE_FACTORY[self.width](
            in_channels=self.in_channels,
            pretrained=None,
            align_corners=self.align_corners,
        )
        backbone_channels = tuple(self.backbone.feat_channels)
        self.head_channels = int(head_channels) if head_channels is not None else int(backbone_channels[0])
        self.heads = nn.LayerList(
            [
                FCNHead(
                    num_classes=self.num_classes,
                    backbone_indices=(-1,),
                    backbone_channels=backbone_channels,
                    channels=self.head_channels,
                )
                for _ in range(self.num_heads)
            ]
        )

    def forward_shared(self, x):
        return self.backbone(x)

    def _upsample_logits(self, logits, input_size):
        return F.interpolate(
            logits,
            input_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )

    def route_logits(self, all_logits, cluster_ids):
        if all_logits.ndim != 5:
            raise ValueError(f"Expected all_logits with shape [B, H, C, H, W], got {all_logits.shape}")
        if self.num_heads == 1:
            return all_logits[:, 0]
        if cluster_ids is None:
            raise ValueError("cluster_ids must be provided when num_heads > 1")
        if cluster_ids.ndim != 1:
            cluster_ids = paddle.reshape(cluster_ids, [-1])
        routing_mask = F.one_hot(cluster_ids.astype("int64"), num_classes=self.num_heads)
        routing_mask = routing_mask.astype(all_logits.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return paddle.sum(all_logits * routing_mask, axis=1)

    def forward(self, x, cluster_ids=None, return_all_heads=False):
        input_size = paddle.shape(x)[2:]
        feat_list = self.forward_shared(x)
        main_logits = [self._upsample_logits(head(feat_list)[0], input_size) for head in self.heads]
        all_main_logits = paddle.stack(main_logits, axis=1)
        routed_logits = self.route_logits(all_main_logits, cluster_ids)

        payload = {
            "logits": routed_logits,
            "aux_logits": None,
            "cluster_ids": cluster_ids,
        }
        if return_all_heads:
            payload["all_head_logits"] = list(paddle.unstack(all_main_logits, axis=1))
        return payload


def load_pretrained_weights(model: ClusterRoutedHRNet, pretrained_path: str | Path) -> dict[str, object]:
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
            candidate = f"head.{suffix}"
            if candidate in state_dict and list(state_dict[candidate].shape) == list(tensor.shape):
                source_key = candidate
                head_broadcast_keys.append(key)

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
    width = int(model_cfg.get("width", 48))
    return {
        "model_name": "ClusterRoutedHRNet",
        "hrnet_width": width,
        "backbone_name": f"HRNet_W{width}",
        "num_heads": int(routing_cfg["num_heads"]),
        "head_names": list(routing_cfg["cluster_names"]),
        "selected_k": int(routing_cfg["selected_k"]),
        "cluster_id_handling": routing_cfg.get("cluster_id_handling", "route"),
        "shared_modules": ["HRNetBackbone"],
        "segmentation_head_type": "FCNHead",
        "segmentation_head_count": int(routing_cfg["num_heads"]),
        "segmentation_num_classes": int(num_classes),
        "in_channels": int(model_cfg.get("in_channels", 3)),
        "head_channels": int(model_cfg.get("head_channels", 0) or 0),
        "align_corners": bool(model_cfg.get("align_corners", False)),
        "routing_rule": "Each sample uses exactly one segmentation head selected by cluster_id.",
        "supervision_rule": "The selected head is supervised by the sample's original segmentation mask.",
        "evaluation_rule": "OA / Mean Acc / per-class Accuracy / per-class Recall / per-class IoU / mIoU / Kappa are all computed in the original 16-class GID15 label space.",
    }
