#!/usr/bin/env python3

from __future__ import annotations

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlers.rs_models.seg.farseg import (
    AsymmetricDecoder,
    DefaultConvBlock,
    FPN,
    FSRelation,
    ResNetEncoder,
)


class ClusterRoutedFarSeg(nn.Layer):
    """
    Shared FarSeg trunk with cluster-routed segmentation heads.

    Routing semantics:
    - `cluster_id` selects exactly one head for each sample.
    - All heads predict the same 16-class GID15 label space.
    - `cluster_id` is never used as pixel supervision.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_heads: int,
        backbone: str = "resnet50",
        backbone_pretrained: bool = True,
        fpn_out_channels: int = 256,
        fsr_out_channels: int = 256,
        scale_aware_proj: bool = True,
        decoder_out_channels: int = 128,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_heads = int(num_heads)
        self.backbone_name = str(backbone).lower()

        self.encoder = ResNetEncoder(
            backbone=self.backbone_name,
            in_channels=self.in_channels,
            pretrained=backbone_pretrained,
        )

        fpn_max_in_channels = 2048
        if self.backbone_name in {"resnet18", "resnet34"}:
            fpn_max_in_channels = 512

        self.fpn = FPN(
            in_channels_list=[fpn_max_in_channels // (2 ** (3 - idx)) for idx in range(4)],
            out_channels=fpn_out_channels,
        )
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.fsr = FSRelation(
            in_channels=fpn_max_in_channels,
            channels_list=[fpn_out_channels] * 4,
            out_channels=fsr_out_channels,
            scale_aware_proj=scale_aware_proj,
        )
        self.decoder = AsymmetricDecoder(
            in_channels=fsr_out_channels,
            out_channels=decoder_out_channels,
        )
        self.heads = nn.LayerList(
            [
                nn.Sequential(
                    DefaultConvBlock(decoder_out_channels, num_classes, 1),
                    nn.UpsamplingBilinear2D(scale_factor=4),
                )
                for _ in range(self.num_heads)
            ]
        )

    def forward_shared(self, x):
        feature_list = self.encoder(x)
        fpn_feature_list = self.fpn(feature_list)
        scene_feature = self.gap(feature_list[-1])
        refined_feature_list = self.fsr(scene_feature, fpn_feature_list)
        return self.decoder(refined_feature_list)

    def route_logits(self, all_logits, cluster_ids):
        if cluster_ids is None:
            raise ValueError("cluster_ids must be provided for routed forward")
        if cluster_ids.ndim != 1:
            cluster_ids = paddle.reshape(cluster_ids, [-1])
        routing_mask = F.one_hot(cluster_ids.astype("int64"), num_classes=self.num_heads)
        routing_mask = routing_mask.astype(all_logits.dtype)
        routing_mask = routing_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return paddle.sum(all_logits * routing_mask, axis=1)

    def forward(self, x, cluster_ids=None, return_all_heads=False):
        decoder_feature = self.forward_shared(x)
        head_logits = [head(decoder_feature) for head in self.heads]
        all_logits = paddle.stack(head_logits, axis=1)

        routed_logits = None
        if cluster_ids is not None:
            routed_logits = self.route_logits(all_logits, cluster_ids)

        if return_all_heads:
            return {
                "routed_logits": routed_logits,
                "all_head_logits": list(paddle.unstack(all_logits, axis=1)),
                "cluster_ids": cluster_ids,
            }
        return routed_logits


def build_model_structure(config: dict, num_classes: int) -> dict:
    routing_cfg = config["routing"]
    model_cfg = config["model"]
    return {
        "model_name": "ClusterRoutedFarSeg",
        "backbone": model_cfg["backbone"],
        "backbone_pretrained": model_cfg["backbone_pretrained"],
        "shared_encoder": True,
        "shared_fpn": True,
        "shared_fsr": True,
        "shared_decoder": True,
        "num_heads": routing_cfg["num_heads"],
        "head_names": routing_cfg["cluster_names"],
        "segmentation_num_classes": num_classes,
        "routing_rule": "Each sample uses exactly one segmentation head selected by cluster_id.",
        "supervision_rule": "The selected head is supervised by the sample's original segmentation mask.",
        "evaluation_rule": "Metrics are computed on original GID15 classes, with foreground 15-class metrics as primary.",
        "fpn_out_channels": model_cfg["fpn_out_channels"],
        "fsr_out_channels": model_cfg["fsr_out_channels"],
        "decoder_out_channels": model_cfg["decoder_out_channels"],
        "scale_aware_proj": model_cfg["scale_aware_proj"],
    }
