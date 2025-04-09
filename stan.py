import torch
import torch.nn as nn
import torch.nn.functional as F

class STAN(nn.Module):
    def __init__(self, num_classes):
        super(STAN, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 测试网络
if __name__ == "__main__":
    # 创建网络实例
    num_classes = 2  # 假设有2个类别
    model = STAN(num_classes)

    # 创建输入数据
    input_data = torch.randn(4, 3, 256, 256)  # 4个样本，3通道，256x256大小

    # 前向传播
    output = model(input_data)

    # 打印输入和输出的尺寸
    print("输入尺寸:", input_data.shape)
    print("输出尺寸:", output.shape)