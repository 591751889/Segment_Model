import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

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

class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_channels, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, out_channels * M, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(x))
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V

class SKU_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(SKU_Net, self).__init__()
        # 编码器
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        # 瓶颈层
        self.conv5 = ConvBlock(512, 1024)
        # 解码器
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = ConvBlock(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = ConvBlock(128, 64)
        self.conv10 = nn.Conv2d(64, num_classes, kernel_size=1)
        # SKConv
        self.skconv1 = SKConv(64, 64)
        self.skconv2 = SKConv(128, 128)
        self.skconv3 = SKConv(256, 256)
        self.skconv4 = SKConv(512, 512)
        self.skconv5 = SKConv(1024, 1024)
        self.skconv6 = SKConv(512, 512)
        self.skconv7 = SKConv(256, 256)
        self.skconv8 = SKConv(128, 128)
        self.skconv9 = SKConv(64, 64)

    def forward(self, x):
        # 编码器
        c1 = self.conv1(x)
        c1 = self.skconv1(c1)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        c2 = self.skconv2(c2)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        c3 = self.skconv3(c3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        c4 = self.skconv4(c4)
        p4 = self.pool4(c4)
        # 瓶颈层
        c5 = self.conv5(p4)
        c5 = self.skconv5(c5)
        # 解码器
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        c6 = self.skconv6(c6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        c7 = self.skconv7(c7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        c8 = self.skconv8(c8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c9 = self.skconv9(c9)
        c10 = self.conv10(c9)
        return c10

# 测试输入输出尺寸
if __name__ == "__main__":
    model = SKU_Net(in_channels=3, num_classes=2)
    input_data = torch.randn(4, 3, 256, 256)
    output = model(input_data)
    print("输入尺寸:", input_data.shape)
    print("输出尺寸:", output.shape)