import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXt3DBlock(nn.Module):
    """Minimal 3D ConvNeXt block with depthwise 3D convolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        # layer norm expects channels last
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + self.gamma.view(1, -1, 1, 1, 1) * x


class ConvNeXt3DTiny(nn.Module):
    """Lightweight ConvNeXt style network for 5-frame video classification."""

    def __init__(self, in_channels: int = 3, num_classes: int = 7, dims=(32, 64, 128)):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.BatchNorm3d(dims[0]),
            nn.GELU(),
        )
        self.downsample_layers.append(stem)
        for i in range(2):
            layer = nn.Sequential(
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.BatchNorm3d(dims[i + 1]),
                nn.GELU(),
            )
            self.downsample_layers.append(layer)
        self.stages = nn.ModuleList([
            nn.Sequential(ConvNeXt3DBlock(dims[0])),
            nn.Sequential(ConvNeXt3DBlock(dims[1])),
            nn.Sequential(ConvNeXt3DBlock(dims[2])),
        ])
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage, down in zip(self.stages, self.downsample_layers):
            x = down(x)
            x = stage(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class InvertedResidual3D(nn.Module):
    """3D version of MobileNetV2 inverted residual block."""

    def __init__(self, inp: int, oup: int, stride, expand_ratio: float):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == (1, 1, 1) and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv3d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm3d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_3D(nn.Module):
    """Simplified MobileNetV2 adapted for 5-frame video."""

    def __init__(self, in_channels: int = 3, num_classes: int = 7, width_mult: float = 0.5):
        super().__init__()
        cfgs = [
            # t, c, n, s
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (1, 2, 2)],
            [6, 32, 3, (2, 2, 2)],
            [6, 64, 4, (2, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]
        input_channel = int(32 * width_mult)
        layers = [
            nn.Conv3d(in_channels, input_channel, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(input_channel),
            nn.ReLU6(inplace=True),
        ]
        for t, c, n, s in cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                layers.append(InvertedResidual3D(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(input_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ViViTTiny(nn.Module):
    """Tiny ViViT-style transformer for video classification."""

    def __init__(self, in_channels: int = 3, num_classes: int = 7, embed_dim: int = 128, depth: int = 4, num_heads: int = 4, patch_size: int = 12):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))
        num_patches = 5 * (120 // patch_size) * (180 // patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # B, C, T, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)
