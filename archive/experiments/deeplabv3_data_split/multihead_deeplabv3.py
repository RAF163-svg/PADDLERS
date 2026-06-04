#!/usr/bin/env python3

from __future__ import annotations

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlers.models.paddleseg.models import backbones, layers


DEFAULT_BACKBONE_PRETRAINED = {
    "ResNet50_vd": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams",
    "ResNet101_vd": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_vd_pretrained.pdparams",
}


def normalize_backbone_pretrained(name: str, pretrained):
    if isinstance(pretrained, bool):
        if pretrained:
            return DEFAULT_BACKBONE_PRETRAINED.get(name)
        return None
    if pretrained in {"", "none", "null", "False", "false"}:
        return None
    return pretrained


def build_backbone(name: str, pretrained=True):
    pretrained = normalize_backbone_pretrained(name, pretrained)
    if name == "ResNet50_vd":
        return backbones.ResNet50_vd(pretrained=pretrained)
    if name == "ResNet101_vd":
        return backbones.ResNet101_vd(pretrained=pretrained)
    raise ValueError(f"Unsupported backbone for ClusterRoutedDeepLabV3: {name}")


class ClusterRoutedDeepLabV3(nn.Layer):
    """
    Shared DeepLabV3 trunk with cluster-routed segmentation heads.

    Routing semantics:
    - `cluster_id` selects which segmentation head a sample should use.
    - All heads predict the same GID15 segmentation classes.
    - `cluster_id` is never treated as pixel supervision.
    """

    def __init__(
        self,
        num_classes: int,
        num_heads: int,
        backbone_name: str = "ResNet50_vd",
        backbone_pretrained: bool = True,
        backbone_indices=(3,),
        aspp_ratios=(1, 6, 12, 18),
        aspp_out_channels: int = 256,
        trunk_channels: int = 256,
        dropout_prob: float = 0.1,
        align_corners: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.align_corners = align_corners
        self.backbone_indices = tuple(backbone_indices)

        self.backbone = build_backbone(backbone_name, pretrained=backbone_pretrained)
        backbone_channels = [self.backbone.feat_channels[i] for i in self.backbone_indices]

        self.aspp = layers.ASPPModule(
            aspp_ratios,
            backbone_channels[0],
            aspp_out_channels,
            align_corners,
            use_sep_conv=False,
            image_pooling=True,
        )
        self.shared_trunk = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=aspp_out_channels,
                out_channels=trunk_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout2D(p=dropout_prob),
        )
        self.heads = nn.LayerList(
            [nn.Conv2D(trunk_channels, num_classes, kernel_size=1) for _ in range(num_heads)]
        )

    def forward_shared(self, x):
        feat_list = self.backbone(x)
        feat = feat_list[self.backbone_indices[0]]
        feat = self.aspp(feat)
        feat = self.shared_trunk(feat)
        return feat

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
        trunk = self.forward_shared(x)
        head_logits = [head(trunk) for head in self.heads]
        head_logits = paddle.stack(head_logits, axis=1)

        routed = None
        if cluster_ids is not None:
            routed = self.route_logits(head_logits, cluster_ids)
            routed = F.interpolate(
                routed,
                paddle.shape(x)[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        if return_all_heads:
            resized = [
                F.interpolate(logit, paddle.shape(x)[2:], mode="bilinear", align_corners=self.align_corners)
                for logit in paddle.unstack(head_logits, axis=1)
            ]
            return {
                "routed_logits": routed,
                "all_head_logits": resized,
                "cluster_ids": cluster_ids,
            }

        return routed
