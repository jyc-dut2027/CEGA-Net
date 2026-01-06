import torch
import torch.nn as nn
import torchvision.models as models
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 你原来的权重缓存路径保留
os.environ['TORCH_HOME'] = 'E:\CEGANet\resnet\ResNet34_model_path'

class RainNet(nn.Module):
    def __init__(self, in_channels: int = 1):
        super(RainNet, self).__init__()
        # 加载预训练 ResNet34
        try:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        except Exception:
            # 兼容老版本 torchvision
            resnet = models.resnet34(pretrained=True)

        # 首层根据通道数自适应
        if in_channels == 3:
            self.conv1 = resnet.conv1  # 直接使用预训练三通道
        else:
            # 单通道：用预训练权重的均值来初始化 1->64
            conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                conv.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))
            self.conv1 = conv

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
