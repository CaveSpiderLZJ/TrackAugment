import torch
import torch.nn as nn

import config as cf


class MBConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(MBConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 如果输入的特征图大小和输出的特征图大小不一致，需要使用1x1卷积来进行调整
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return out


class MBConv6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, expand_ratio=6):
        super(MBConv6, self).__init__()
        self.expand_ratio = expand_ratio
        mid_channels = round(expand_ratio * in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride,
                      padding=padding, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.conv(x)
        if x.shape == out.shape:
            out += x
        return out


class ConvBlock(nn.Module):
    """定义基本的卷积单元"""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class CompoundBlock(nn.Module):
    """定义复合的卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, num_blocks):
        super(CompoundBlock, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(
                ConvBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SeparableConv, self).__init__()

        # depthwise convolution
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                                        padding=0,
                                        groups=in_channels, bias=False)
        # pointwise convolution
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class GestureNet(nn.Module):
    def __init__(self, num_classes=5):
        super(GestureNet, self).__init__()
        self.conv1 = MBConv1(6, 12, (1, 9), stride=1,
                             padding=(0, 4))  
        self.conv2 = MBConv6(12, 24, (1, 10), stride=2,
                             padding=(0, 4)) 
        # self.conv3 = MBConv6(24, 48, (1, 10), stride=2, padding=(0, 4))
        # self.conv4 = MBConv6(48, 96, (1, 10), stride=2, padding=(0, 4))
        # self.conv5 = MBConv6(96, 192, (1, 10), stride=2, padding=(0, 4))
        # self.conv6 = MBConv6(192, 384, (1, 10), stride=2, padding=(0, 4))
        # self.conv7 = MBConv6(384, 768, (1, 10), stride=2, padding=(0, 4))
        # self.conv8 = MBConv6(768, 1024, (1, 10), stride=2, padding=(0, 4))
        # self.fc = nn.Linear(1024, num_classes)
        self.conv3 = SeparableConv(
            24, 24, (1, 10), stride=1) 
        self.maxpool1 = nn.MaxPool1d(
            kernel_size=9, stride=8) 
        self.flattern = nn.Flatten()
        self.linear1 = nn.Linear(120, 80)
        self.linear2 = nn.Linear(80, 40)
        self.linear3 = nn.Linear(40, 20)
        self.linear4 = nn.Linear(20, 10)
        self.linear5 = nn.Linear(10, num_classes)

    def forward(self, x):
        out = self.conv1(x) # shape: (12, 1, 100)
        out = self.conv2(out) # shape: (24, 1, 50)
        out = self.conv3(out) # shape: (24, 1, 41)
        out = out.view(out.size(0), out.size(1), -1) # shape: (24, 41)
        out = self.maxpool1(out) # shape: (24, 5)
        out = self.flattern(out)
        bottleneck = self.flattern(out).clone()
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        out = self.linear5(out)
        return out, bottleneck
    
class GestureNet3(nn.Module):
    def __init__(self, num_classes=5):
        super(GestureNet3, self).__init__()
        self.conv1 = MBConv1(3, 12, (1, 9), stride=1,
                             padding=(0, 4))  
        self.conv2 = MBConv6(12, 24, (1, 10), stride=2,
                             padding=(0, 4)) 
        # self.conv3 = MBConv6(24, 48, (1, 10), stride=2, padding=(0, 4))
        # self.conv4 = MBConv6(48, 96, (1, 10), stride=2, padding=(0, 4))
        # self.conv5 = MBConv6(96, 192, (1, 10), stride=2, padding=(0, 4))
        # self.conv6 = MBConv6(192, 384, (1, 10), stride=2, padding=(0, 4))
        # self.conv7 = MBConv6(384, 768, (1, 10), stride=2, padding=(0, 4))
        # self.conv8 = MBConv6(768, 1024, (1, 10), stride=2, padding=(0, 4))
        # self.fc = nn.Linear(1024, num_classes)
        self.conv3 = SeparableConv(
            24, 24, (1, 10), stride=1) 
        self.maxpool1 = nn.MaxPool1d(
            kernel_size=9, stride=8) 
        self.flattern = nn.Flatten()
        self.linear1 = nn.Linear(120, 80)
        self.linear2 = nn.Linear(80, 40)
        self.linear3 = nn.Linear(40, 20)
        self.linear4 = nn.Linear(20, 10)
        self.linear5 = nn.Linear(10, num_classes)

    def forward(self, x):
        out = self.conv1(x) # shape: (12, 1, 100)
        out = self.conv2(out) # shape: (24, 1, 50)
        out = self.conv3(out) # shape: (24, 1, 41)
        out = out.view(out.size(0), out.size(1), -1) # shape: (24, 41)
        out = self.maxpool1(out) # shape: (24, 5)
        out = self.flattern(out)
        bottleneck = self.flattern(out).clone()
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        out = self.linear5(out)
        return out, bottleneck


# !!!
class Model6(nn.Module):
    def __init__(self):
        super(Model6, self).__init__()
        self.conv1 = MBConv1(6, 12, (1, 9), stride=1,
                             padding=(0, 4))  
        self.conv2 = MBConv6(12, 24, (1, 10), stride=2,
                             padding=(0, 4)) 
        self.conv3 = MBConv6(24, 24, (1, 10), stride=2,
                             padding=(0, 4)) 
        self.conv4 = SeparableConv(
            24, 24, (1, 10), stride=1) 
        self.maxpool1 = nn.MaxPool1d(
            kernel_size=9, stride=8)
        self.flattern = nn.Flatten()
        self.linear1 = nn.Linear(120, 80)
        self.linear2 = nn.Linear(80, 40)
        self.linear3 = nn.Linear(40, cf.N_CLASSES)

    def forward(self, x):
        out = self.conv1(x) # shape: (12, 1, 200)
        out = self.conv2(out) # shape: (24, 1, 100)
        out = self.conv3(out) # shape: (24, 1, 50)
        out = self.conv4(out) # shape: (24, 1, 41)
        out = out.view(out.size(0), out.size(1), -1) # shape: (24, 41)
        out = self.maxpool1(out) # shape: (24, 5)
        out = self.flattern(out)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out


class Encoder(nn.Module):
    def __init__(self, feature_dim=256, encoder_size=[8192], z_dim=16, dropout=0.5, dropout_input=0.0, leak=0.2):
        super(Encoder, self).__init__()
        self.first_linear = nn.Linear(feature_dim*2, encoder_size[0])

        linear = []
        for i in range(len(encoder_size) - 1):
            linear.append(nn.Linear(encoder_size[i], encoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)
        self.final_linear = nn.Linear(encoder_size[-1], z_dim)
        self.lrelu = nn.LeakyReLU(leak)
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, reference_features):
        features = self.dropout_input(features)
        x = torch.cat([features, reference_features], 1)

        # print("features shape is:", features.shape, reference_features.shape)
        # print(x.shape)

        x = self.first_linear(x)
        x = self.linear(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.final_linear(x)

        return x

class Decoder(nn.Module):
    def __init__(self, feature_dim=256, decoder_size=[8192], z_dim=16, dropout=0.5, leak=0.2):
        super(Decoder, self).__init__()
        self.first_linear = nn.Linear(z_dim+feature_dim, decoder_size[0])

        linear = []
        for i in range(len(decoder_size) - 1):
            linear.append(nn.Linear(decoder_size[i], decoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)

        self.final_linear = nn.Linear(decoder_size[-1], feature_dim)
        self.lrelu = nn.LeakyReLU(leak)
        self.dropout = nn.Dropout(dropout)

    def forward(self, reference_features, code):
        x = torch.cat([reference_features, code], 1)

        x = self.first_linear(x)
        x = self.linear(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.final_linear(x)

        return x

class DeltaEncoder(nn.Module):
    def __init__(self, feature_dim=256, encoder_size=[8192], decoder_size=[8192], z_dim=16, dropout=0.5, dropout_input=0.0, leak=0.2):
        super(DeltaEncoder, self).__init__()
        self.encoder = Encoder(feature_dim, encoder_size, z_dim, dropout, dropout_input, leak)
        self.decoder = Decoder(feature_dim, decoder_size, z_dim, dropout, leak)

    def forward(self, features, reference_features):
        # assert features.shape == reference_features.shape
        code = self.encoder(features, reference_features)
        reconstructed_features = self.decoder(reference_features, code)
        return reconstructed_features, code
