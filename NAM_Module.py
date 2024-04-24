import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

class NAM(nn.Module):
    def __init__(self, channel):
        super(NAM, self).__init__()
        self.channel = channel
        self.norm2 = nn.InstanceNorm2d(self.channel, affine=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # First path: Process the output of the previous layer (x1)
        residual1 = x1
        x1 = self.norm2(x1)
        out1 = self.sigmoid(x1) * residual1

        # Second path: Concatenate the output of the previous layer (x1) with the input of the current layer (x2)
        x2 = torch.cat((x1, x2), dim=1)
        x2 = self.norm2(x2)
        out2 = self.sigmoid(x2) * x2

        return out1, out2

if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    x1 = input.clone()
    x2 = input.clone()
    nam = NAM(512)
    output1, output2 = nam(x1, x2)
    print(output1.shape)
    print(output2.shape)
