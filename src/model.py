from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GlobalConfig, ExperimentConfig


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def normalization_layer(channels: int) -> nn.Module:
    """
    GroupNorm works well for segmentation models
    trained with relatively small batch sizes.
    """

    groups = min(8, channels)

    while channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(
        groups,
        channels,
    )


# ---------------------------------------------------------
# Standard convolution block
# ---------------------------------------------------------

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            normalization_layer(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            normalization_layer(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


# ---------------------------------------------------------
# Residual block
# ---------------------------------------------------------

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.main = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            normalization_layer(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            normalization_layer(
                out_channels
            ),
        )

        self.shortcut = (
            nn.Identity()

            if in_channels == out_channels

            else nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),

                normalization_layer(
                    out_channels
                ),
            )
        )

        self.activation = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.activation(
            self.main(x)
            + self.shortcut(x)
        )


def build_feature_block(
    in_channels: int,
    out_channels: int,
    use_residual: bool,
) -> nn.Module:

    block_class = (
        ResidualBlock
        if use_residual
        else ConvBlock
    )

    return block_class(
        in_channels,
        out_channels,
    )


# ---------------------------------------------------------
# CBAM - Channel Attention
# ---------------------------------------------------------

class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ):
        super().__init__()

        hidden_channels = max(
            channels // reduction,
            1,
        )

        self.shared_mlp = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        average_attention = self.shared_mlp(
            F.adaptive_avg_pool2d(
                x,
                output_size=1,
            )
        )

        maximum_attention = self.shared_mlp(
            F.adaptive_max_pool2d(
                x,
                output_size=1,
            )
        )

        attention = torch.sigmoid(
            average_attention
            + maximum_attention
        )

        return x * attention


# ---------------------------------------------------------
# CBAM - Spatial Attention
# ---------------------------------------------------------

class SpatialAttention(nn.Module):

    def __init__(
        self,
        kernel_size: int = 7,
    ):
        super().__init__()

        padding = (
            kernel_size // 2
        )

        self.convolution = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        average_map = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        maximum_map = torch.amax(
            x,
            dim=1,
            keepdim=True,
        )

        attention = torch.sigmoid(
            self.convolution(
                torch.cat(
                    [
                        average_map,
                        maximum_map,
                    ],
                    dim=1,
                )
            )
        )

        return x * attention


# ---------------------------------------------------------
# Complete CBAM
# ---------------------------------------------------------

class CBAM(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()

        self.channel_attention = (
            ChannelAttention(
                channels=channels,
                reduction=reduction,
            )
        )

        self.spatial_attention = (
            SpatialAttention(
                kernel_size=spatial_kernel_size,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


def build_attention(
    channels: int,
    use_cbam: bool,
) -> nn.Module:

    if use_cbam:
        return CBAM(channels)

    return nn.Identity()


# ---------------------------------------------------------
# ASPP
# ---------------------------------------------------------

class ASPPConv(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
    ):
        super().__init__(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),

            normalization_layer(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )


class ASPP(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_rates: Tuple[int, ...] = (
            1,
            2,
            4,
            6,
        ),
    ):
        super().__init__()

        branch_channels = max(
            out_channels
            // len(dilation_rates),
            1,
        )

        self.branches = nn.ModuleList([

            ASPPConv(
                in_channels=in_channels,
                out_channels=branch_channels,
                dilation=rate,
            )

            for rate
            in dilation_rates
        ])

        merged_channels = (
            branch_channels
            * len(dilation_rates)
        )

        self.project = nn.Sequential(

            nn.Conv2d(
                merged_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),

            normalization_layer(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout2d(
                p=0.1
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        features = [
            branch(x)
            for branch
            in self.branches
        ]

        return self.project(
            torch.cat(
                features,
                dim=1,
            )
        )


# ---------------------------------------------------------
# Encoder
# ---------------------------------------------------------

class EncoderBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_residual: bool,
        use_cbam: bool,
    ):
        super().__init__()

        self.features = (
            build_feature_block(
                in_channels,
                out_channels,
                use_residual,
            )
        )

        self.attention = (
            build_attention(
                out_channels,
                use_cbam,
            )
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        features = self.attention(
            self.features(x)
        )

        pooled = self.pool(
            features
        )

        return (
            features,
            pooled,
        )


# ---------------------------------------------------------
# Decoder
# ---------------------------------------------------------

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_residual: bool,
        use_cbam: bool,
    ):
        super().__init__()

        self.upsample = (
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            )
        )

        self.features = (
            build_feature_block(
                out_channels
                + skip_channels,
                out_channels,
                use_residual,
            )
        )

        self.attention = (
            build_attention(
                out_channels,
                use_cbam,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.upsample(x)

        if (
            x.shape[-2:]
            != skip.shape[-2:]
        ):
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [skip, x],
            dim=1,
        )

        x = self.features(x)
        x = self.attention(x)

        return x


# ---------------------------------------------------------
# Configurable Residual CBAM U-Net
# ---------------------------------------------------------

class ConfigurableCBAMUNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        use_residual: bool = True,
        use_cbam: bool = True,
        use_aspp: bool = True,
        use_deep_supervision: bool = True,
    ):
        super().__init__()

        self.use_deep_supervision = (
            use_deep_supervision
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.encoder1 = EncoderBlock(
            in_channels,
            c1,
            use_residual,
            use_cbam,
        )

        self.encoder2 = EncoderBlock(
            c1,
            c2,
            use_residual,
            use_cbam,
        )

        self.encoder3 = EncoderBlock(
            c2,
            c3,
            use_residual,
            use_cbam,
        )

        self.encoder4 = EncoderBlock(
            c3,
            c4,
            use_residual,
            use_cbam,
        )

        # Bottleneck
        if use_aspp:

            self.bottleneck = ASPP(
                in_channels=c4,
                out_channels=c5,
            )

        else:

            self.bottleneck = (
                build_feature_block(
                    c4,
                    c5,
                    use_residual,
                )
            )

        self.bottleneck_attention = (
            build_attention(
                c5,
                use_cbam,
            )
        )

        # Decoder
        self.decoder4 = DecoderBlock(
            c5,
            c4,
            c4,
            use_residual,
            use_cbam,
        )

        self.decoder3 = DecoderBlock(
            c4,
            c3,
            c3,
            use_residual,
            use_cbam,
        )

        self.decoder2 = DecoderBlock(
            c3,
            c2,
            c2,
            use_residual,
            use_cbam,
        )

        self.decoder1 = DecoderBlock(
            c2,
            c1,
            c1,
            use_residual,
            use_cbam,
        )

        # Final segmentation head
        self.output_head = nn.Conv2d(
            c1,
            num_classes,
            kernel_size=1,
        )

        # Deep supervision
        if self.use_deep_supervision:

            self.deep_head_3 = nn.Conv2d(
                c3,
                num_classes,
                kernel_size=1,
            )

            self.deep_head_2 = nn.Conv2d(
                c2,
                num_classes,
                kernel_size=1,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[
        str,
        torch.Tensor,
    ]:

        input_size = (
            x.shape[-2:]
        )

        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        x = self.bottleneck(x)

        x = (
            self.bottleneck_attention(x)
        )

        x = self.decoder4(
            x,
            skip4,
        )

        x = self.decoder3(
            x,
            skip3,
        )

        deep_feature_3 = x

        x = self.decoder2(
            x,
            skip2,
        )

        deep_feature_2 = x

        x = self.decoder1(
            x,
            skip1,
        )

        main_output = (
            self.output_head(x)
        )

        outputs = {
            "main": main_output
        }

        if self.use_deep_supervision:

            outputs["deep_2"] = (
                F.interpolate(
                    self.deep_head_2(
                        deep_feature_2
                    ),
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

            outputs["deep_3"] = (
                F.interpolate(
                    self.deep_head_3(
                        deep_feature_3
                    ),
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        return outputs


# ---------------------------------------------------------
# Model factory
# ---------------------------------------------------------

def build_model(
    experiment: ExperimentConfig,
    config: GlobalConfig,
) -> nn.Module:
    """
    Create a model using the selected experiment settings.
    """

    return ConfigurableCBAMUNet(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        base_channels=config.base_channels,
        use_residual=experiment.use_residual,
        use_cbam=experiment.use_cbam,
        use_aspp=experiment.use_aspp,
        use_deep_supervision=experiment.use_deep_supervision,
    )