import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ScSEBlock(nn.Module):
    def __init__(self, in_channels):
        super(ScSEBlock, self).__init__()
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_se = self.spatial_se(x)
        channel_se = self.channel_se(x)
        return x * spatial_se + x * channel_se

class ScSEUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(ScSEUNet, self).__init__()
        # 编码器
        self.conv1 = ConvBlock(in_channels, 64)
        self.scse1 = ScSEBlock(64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128)
        self.scse2 = ScSEBlock(128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256)
        self.scse3 = ScSEBlock(256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(256, 512)
        self.scse4 = ScSEBlock(512)
        self.pool4 = nn.MaxPool2d(2)
        # 瓶颈层
        self.conv5 = ConvBlock(512, 1024)
        self.scse5 = ScSEBlock(1024)
        # 解码器
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(1024, 512)
        self.scse6 = ScSEBlock(512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(512, 256)
        self.scse7 = ScSEBlock(256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = ConvBlock(256, 128)
        self.scse8 = ScSEBlock(128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = ConvBlock(128, 64)
        self.scse9 = ScSEBlock(64)
        self.conv10 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器
        c1 = self.conv1(x)
        c1 = self.scse1(c1)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        c2 = self.scse2(c2)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        c3 = self.scse3(c3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        c4 = self.scse4(c4)
        p4 = self.pool4(c4)
        # 瓶颈层
        c5 = self.conv5(p4)
        c5 = self.scse5(c5)
        # 解码器
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        c6 = self.scse6(c6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        c7 = self.scse7(c7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        c8 = self.scse8(c8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c9 = self.scse9(c9)
        c10 = self.conv10(c9)
        return c10

# 测试输入输出尺寸
if __name__ == "__main__":
    model = ScSEUNet(in_channels=3, num_classes=2)
    input_data = torch.randn(4, 3, 256, 256)
    output = model(input_data)
    print("输入尺寸:", input_data.shape)
    print("输出尺寸:", output.shape)