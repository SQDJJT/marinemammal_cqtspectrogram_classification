import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from NAM_Module import NAM
from FPN_Module import FPN


class ResNet34(nn.Module):
    def __init__(self, num_classes=32):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.nam1 = NAM(channel=self.resnet.layer4[1].conv2.out_channels)
        self.nam2 = NAM(channel=self.resnet.layer3[1].conv2.out_channels)
        self.nam3 = NAM(channel=self.resnet.layer2[1].conv2.out_channels)
        self.nam4 = NAM(channel=self.resnet.layer1[1].conv2.out_channels)
        self.fpn = FPN([self.resnet.layer1[1].conv2.out_channels,
                        self.resnet.layer2[1].conv2.out_channels,
                        self.resnet.layer3[1].conv2.out_channels,
                        self.resnet.layer4[1].conv2.out_channels], 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x1_nam, _ = self.nam4(x1, x1)
        x2_nam, _ = self.nam3(x2, x2)
        x3_nam, _ = self.nam2(x3, x3)
        x4_nam, _ = self.nam1(x4, x4)

        fpn_features = self.fpn([x1_nam, x2_nam, x3_nam, x4_nam])

        # 将高层特征图上采样到与底层特征图相同的大小
        p1_upsampled = F.interpolate(fpn_features[0], size=fpn_features[3].shape[2:], mode='nearest')
        p2_upsampled = F.interpolate(fpn_features[1], size=fpn_features[3].shape[2:], mode='nearest')
        p3_upsampled = F.interpolate(fpn_features[2], size=fpn_features[3].shape[2:], mode='nearest')

        # 将特征图相加
        fused_feature = fpn_features[3] + p1_upsampled + p2_upsampled + p3_upsampled
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 全局平均池化
        pooled_feature = torch.mean(fused_feature, dim=(2, 3)).to(device)  # 对特征图的宽和高进行平均池化

        # 全连接层
        out = self.fc(pooled_feature)
        return out
