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

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gate_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(AttentionUNet, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = ConvBlock(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att6 = AttentionGate(512, 512, 512)
        self.conv6 = ConvBlock(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att7 = AttentionGate(256, 256, 256)
        self.conv7 = ConvBlock(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att8 = AttentionGate(128, 128, 128)
        self.conv8 = ConvBlock(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att9 = AttentionGate(64, 64, 64)
        self.conv9 = ConvBlock(128, 64)
        self.conv10 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        att_6 = self.att6(up_6, c4)
        merge6 = torch.cat([up_6, att_6], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        att_7 = self.att7(up_7, c3)
        merge7 = torch.cat([up_7, att_7], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        att_8 = self.att8(up_8, c2)
        merge8 = torch.cat([up_8, att_8], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        att_9 = self.att9(up_9, c1)
        merge9 = torch.cat([up_9, att_9], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10

# 测试输入输出尺寸
if __name__ == "__main__":
    model = AttentionUNet(in_channels=3, num_classes=2)
    input_data = torch.randn(4, 3, 256, 256)
    output = model(input_data)
    print("输入尺寸:", input_data.shape)
    print("输出尺寸:", output.shape)