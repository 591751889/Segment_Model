import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ViTEncoder(nn.Module):
    """Vision Transformer编码器"""

    def __init__(self, in_channels=3, patch_size=16, emb_dim=768, num_layers=6, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.patch_emb = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, (256 // patch_size) ** 2, emb_dim))

    def forward(self, x):
        # 生成图像块
        x = self.patch_emb(x)  # [B, C, H/p, W/p]
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        # 添加位置编码
        x += self.pos_embedding[:, :(h * w)]

        # 经过Transformer
        x = self.transformer(x)

        # 恢复空间维度
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class DecoderBlock(nn.Module):
    """解码器块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + out_channels, out_channels, 3, padding=1),  # 拼接后通道数加倍
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.up(x)  # 上采样
        # 调整skip的尺寸以匹配x
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # 拼接
        return self.conv(x)


class UNetTransformer(nn.Module):
    """结合Transformer的UNet"""

    def __init__(self, in_channels=3, num_classes=21, emb_dim=768):
        super().__init__()
        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.encoder2 = ViTEncoder(in_channels=64, patch_size=16, emb_dim=emb_dim)

        # 解码器
        self.decoder1 = DecoderBlock(emb_dim, 64)
        self.final_conv = nn.Conv2d(64, num_classes, 1)

        # 中间层
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # 编码阶段
        s1 = self.encoder1(x)  # [B, 64, H, W]
        s2 = self.pool(s1)
        s2 = self.encoder2(s2)  # [B, emb_dim, H/16, W/16]

        # 解码阶段
        x = self.decoder1(s2, s1)  # 上采样并拼接
        x = self.final_conv(x)
        x = F.interpolate(x, size=(x.shape[2] * 16, x.shape[3] * 16), mode='bilinear', align_corners=False)  # 恢复到原图尺寸

        return x


# 测试输入输出
if __name__ == "__main__":
    # 参数设置
    batch_size = 16
    in_channels = 15
    num_classes = 1
    H, W = 32, 32

    # 创建模型
    model = UNetTransformer(in_channels=in_channels, num_classes=num_classes)

    # 生成测试输入
    dummy_input = torch.randn(batch_size, in_channels, H, W)

    # 前向传播
    output = model(dummy_input)

    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")  # 应该为 [batch_size, num_classes, H, W]