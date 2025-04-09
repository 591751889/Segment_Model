import torch
import torch.nn as nn
import torch.nn.functional as F
import math


############################################################
# VGGBlock
############################################################
class VGGBlock(nn.Module):
    """VGG Block: (Conv => ReLU) * num_convs"""

    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class Up(nn.Module):
    """
    上采样模块：先对输入进行上采样（使用双线性或转置卷积），再用 VGGBlock 进行卷积，并与跳跃连接特征拼接
    """

    def __init__(self, in_channels, out_channels, num_convs, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = VGGBlock(in_channels, out_channels, num_convs=num_convs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 尺寸对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


#############################################################################
#  定义 swin-unet
#############################################################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, expect {H * W} got {L}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 添加填充处理
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 移除填充
        x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # 添加填充处理
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)])

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x)
        return x


#############################################
# 更深的 SwinUnet_Deep 订正网络（4个 Encoder 阶段 + 对称 Decoder）
#############################################
class SwinUnet_Deep(nn.Module):
    def __init__(self, in_chans=19, num_classes=1,
                 embed_dim=128,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=8,
                 decoder_conv_num=6,
                 bilinear=True):
        """
        参数说明：
          in_chans: 输入通道数（这里与上采样输出通道一致，默认为 19，即原始 11+8）
          embed_dim: patch embedding 输出通道数
          depths: 每个阶段中 Swin Transformer Block 的数量（这里4个阶段）
          num_heads: 每个阶段中多头注意力的头数
          window_size: 局部注意力窗口大小
          decoder_conv_num: 每个上采样模块中内部卷积层数（类似 UNet_VGG 的 VGGBlock 层数）
          bilinear: 是否使用双线性上采样
        """
        super().__init__()
        self.embed_dim = embed_dim
        # Encoder
        # self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)  # (B, embed_dim, H/4, W/4)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        # Stage 1
        self.stage1 = BasicLayer(dim=embed_dim, depth=depths[0], num_heads=num_heads[0], window_size=window_size)
        self.patch_merge1 = PatchMerging(dim=embed_dim)  # 输出通道: 2*embed_dim
        # Stage 2
        self.stage2 = BasicLayer(dim=embed_dim * 2, depth=depths[1], num_heads=num_heads[1], window_size=window_size)
        self.patch_merge2 = PatchMerging(dim=embed_dim * 2)  # 输出通道: 4*embed_dim
        # Stage 3
        self.stage3 = BasicLayer(dim=embed_dim * 4, depth=depths[2], num_heads=num_heads[2], window_size=window_size)
        self.patch_merge3 = PatchMerging(dim=embed_dim * 4)  # 输出通道: 8*embed_dim
        # Stage 4 (Bottleneck)
        self.stage4 = BasicLayer(dim=embed_dim * 8, depth=depths[3], num_heads=num_heads[3], window_size=window_size)

        # Decoder —— 对称上采样，每个 Up 模块使用跳跃连接
        self.up1 = Up(in_channels=embed_dim * 8 + embed_dim * 4, out_channels=embed_dim * 4, num_convs=decoder_conv_num,
                      bilinear=bilinear)
        self.up2 = Up(in_channels=embed_dim * 4 + embed_dim * 2, out_channels=embed_dim * 2, num_convs=decoder_conv_num,
                      bilinear=bilinear)
        self.up3 = Up(in_channels=embed_dim * 2 + embed_dim, out_channels=embed_dim, num_convs=decoder_conv_num,
                      bilinear=bilinear)
        self.up4 = Up(in_channels=embed_dim + embed_dim, out_channels=embed_dim, num_convs=decoder_conv_num,
                      bilinear=bilinear)
        self.outc = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.size(0)
        # Encoder
        x0 = self.patch_embed(x)  # (B, embed_dim, H/4, W/4)
        B, C, H, W = x0.shape
        x0_tokens = x0.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        x1_tokens = self.stage1(x0_tokens, H, W)  # (B, H*W, embed_dim)
        x1 = x1_tokens.transpose(1, 2).view(B, self.embed_dim, H, W)  # (B, embed_dim, H, W)
        skip1 = x1
        x1_merge = self.patch_merge1(x1_tokens, H, W)
        H1, W1 = math.ceil(H / 2), math.ceil(W / 2)

        x2_tokens = self.stage2(x1_merge, H1, W1)  # (B, H1*W1, 2*embed_dim)
        x2 = x2_tokens.transpose(1, 2).view(B, self.embed_dim * 2, H1, W1)
        skip2 = x2
        x2_merge = self.patch_merge2(x2_tokens, H1, W1)  # (B, H2*W2, 4*embed_dim)
        H2, W2 = math.ceil(H1 / 2), math.ceil(W1 / 2)

        x3_tokens = self.stage3(x2_merge, H2, W2)  # (B, H2*W2, 4*embed_dim)
        x3 = x3_tokens.transpose(1, 2).view(B, self.embed_dim * 4, H2, W2)
        skip3 = x3
        x3_merge = self.patch_merge3(x3_tokens, H2, W2)  # (B, H3*W3, 8*embed_dim)
        H3, W3 = math.ceil(H2 / 2), math.ceil(W2 / 2)

        x4_tokens = self.stage4(x3_merge, H3, W3)  # (B, H3*W3, 8*embed_dim)
        x4 = x4_tokens.transpose(1, 2).view(B, self.embed_dim * 8, H3, W3)

        # Decoder
        x_up1 = self.up1(x4, skip3)  # (B, 4*embed_dim, H2, W2)
        x_up2 = self.up2(x_up1, skip2)  # (B, 2*embed_dim, H1, W1)
        x_up3 = self.up3(x_up2, skip1)  # (B, embed_dim, H, W)
        x_up4 = self.up4(x_up3, x0)  # (B, embed_dim, H, W)
        output = self.outc(x_up4)
        return output


#############################################
# 测试整个网络
#############################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUnet_Deep(in_chans=3, window_size=8).to(device)
    input_tensor = torch.randn(4, 3, 256, 256).to(device)
    output = model(input_tensor)
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")



