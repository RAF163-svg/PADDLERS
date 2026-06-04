#!/usr/bin/env python3

from __future__ import annotations

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlers.models.paddleseg.models import layers
from paddlers.models.paddleseg.models.fast_scnn import (
    Classifier,
    FeatureFusionModule,
    GlobalFeatureExtractor,
    LearningToDownsample,
)


class ClusterRoutedFastSCNN(nn.Layer):
    """
    Shared FastSCNN trunk with cluster-routed segmentation heads.

    Routing semantics:
    - `cluster_id` selects exactly one segmentation head for each sample.
    - All heads predict the same 16-class GID15 label space.
    - `cluster_id` is never used as pixel supervision.
    """

    def __init__(
        self,
        num_classes: int,
        num_heads: int,
        in_channels: int = 3,
        enable_auxiliary_loss: bool = True,
        align_corners: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_heads = int(num_heads)
        self.in_channels = int(in_channels)
        self.enable_auxiliary_loss = bool(enable_auxiliary_loss)
        self.align_corners = bool(align_corners)

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")

        self.learning_to_downsample = LearningToDownsample(self.in_channels, 32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=64,
            block_channels=[64, 96, 128],
            out_channels=128,
            expansion=6,
            num_blocks=[3, 3, 3],
            align_corners=True,
        )
        self.feature_fusion = FeatureFusionModule(64, 128, 128, self.align_corners)
        self.main_heads = nn.LayerList([Classifier(128, self.num_classes) for _ in range(self.num_heads)])

        if self.enable_auxiliary_loss:
            self.aux_heads = nn.LayerList(
                [layers.AuxLayer(64, 32, self.num_classes) for _ in range(self.num_heads)]
            )
        else:
            self.aux_heads = None

    def forward_shared(self, x):
        higher_res_features = self.learning_to_downsample(x)
        low_res_features = self.global_feature_extractor(higher_res_features)
        fused_features = self.feature_fusion(higher_res_features, low_res_features)
        return higher_res_features, fused_features

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
        routing_mask = routing_mask.astype(all_logits.dtype)
        routing_mask = routing_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return paddle.sum(all_logits * routing_mask, axis=1)

    def forward(self, x, cluster_ids=None, return_all_heads=False):
        input_size = paddle.shape(x)[2:]
        higher_res_features, fused_features = self.forward_shared(x)

        main_logits = [self._upsample_logits(head(fused_features), input_size) for head in self.main_heads]
        all_main_logits = paddle.stack(main_logits, axis=1)
        routed_logits = self.route_logits(all_main_logits, cluster_ids)

        payload = {
            "logits": routed_logits,
            "aux_logits": None,
            "cluster_ids": cluster_ids,
        }

        if self.enable_auxiliary_loss and self.aux_heads is not None:
            aux_logits = [self._upsample_logits(head(higher_res_features), input_size) for head in self.aux_heads]
            all_aux_logits = paddle.stack(aux_logits, axis=1)
            payload["aux_logits"] = self.route_logits(all_aux_logits, cluster_ids)
            if return_all_heads:
                payload["all_aux_head_logits"] = list(paddle.unstack(all_aux_logits, axis=1))

        if return_all_heads:
            payload["all_head_logits"] = list(paddle.unstack(all_main_logits, axis=1))
        return payload


def build_model_structure(config: dict, num_classes: int) -> dict[str, object]:
    routing_cfg = config["routing"]
    model_cfg = config["model"]
    return {
        "model_name": "ClusterRoutedFastSCNN",
        "num_heads": int(routing_cfg["num_heads"]),
        "head_names": list(routing_cfg["cluster_names"]),
        "selected_k": int(routing_cfg["selected_k"]),
        "cluster_id_handling": routing_cfg.get("cluster_id_handling", "route"),
        "shared_modules": [
            "LearningToDownsample",
            "GlobalFeatureExtractor",
            "FeatureFusion"
        ],
        "main_head_count": int(routing_cfg["num_heads"]),
        "aux_head_count": int(routing_cfg["num_heads"]) if bool(model_cfg.get("enable_auxiliary_loss", True)) else 0,
        "segmentation_num_classes": int(num_classes),
        "in_channels": int(model_cfg.get("in_channels", 3)),
        "align_corners": bool(model_cfg.get("align_corners", False)),
        "enable_auxiliary_loss": bool(model_cfg.get("enable_auxiliary_loss", True)),
        "routing_rule": "Each sample uses exactly one segmentation head selected by cluster_id.",
        "supervision_rule": "The selected head is supervised by the sample's original segmentation mask.",
        "evaluation_rule": "OA / Mean Acc / per-class Accuracy / per-class IoU / mIoU are all computed in the original 16-class GID15 label space."
    }

