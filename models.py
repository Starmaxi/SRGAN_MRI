import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
#from torchvision.models import efficientnet_b0
import math

class CoordRegressionNet(nn.Module):
    def __init__(self):
        super(CoordRegressionNet, self).__init__()
        # input images 1 * 200 * 200
        self.L1 = nn.Sequential(nn.Conv2d(1, 32, 5), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))  # 32 * 98 * 98
        self.L2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))  # 64 * 48 * 48
        self.L3 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))  # 128 * 23 * 23
        self.L4 = nn.Sequential(nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.ReLU())  # 128 * 21 * 21
        self.L5 = nn.Sequential(nn.Conv2d(128, 4, 1), nn.BatchNorm2d(4), nn.ReLU())  # 4 * 21 * 21
        self.FC = nn.Sequential(nn.Linear(4 * 28* 28, 256), nn.ReLU(), nn.Linear(256, 16), nn.ReLU())
        self.Last = nn.Linear(16, 4)

    def forward(self, x):
        x = self.L5(self.L4(self.L3(self.L2(self.L1(x)))))
        B, C, H, W = x.shape
        x = x.view(-1, C * H * W)
        x = self.FC(x)
        x = self.Last(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.PReLU(),
            nn.Conv2d(512, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualDenseBlock, self).__init__()
        self.res_block = ResidualBlock(in_features)
        #self.conv_1x1 = nn.Conv2d(in_features, out_features, 1)

    def forward(self, x):
        #return torch.cat([x ,self.conv_1x1(self.res_block(x))],1)
        y = self.res_block(x)
        return torch.cat([x + y,y],1)

class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseBlock, self).__init__()
        self.dense_block = nn.Sequential(
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, x):
        return torch.cat([x ,self.dense_block(x)],1)

class GeneratorVGG(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_dense_blocks=1):
        super(GeneratorVGG, self).__init__()

        # first layer

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())


        # DenseBlock

        self.feature_extractor = nn.Sequential(*list(vgg19(pretrained=True).features.children())[:18])

        for index, layer in enumerate(self.feature_extractor):
            if isinstance(layer, nn.MaxPool2d):
                self.feature_extractor[index] = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)



        # Upsampling layers

        #self.pixel_shuffle = nn.Sequential(nn.PixelShuffle(upscale_factor=4), #nn.PixelShuffle(upscale_factor=4))

        #self.pixel_shuffle = nn.Sequential(nn.PixelShuffle(upscale_factor=4))

        self.conv2 =  nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.PReLU())

        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 64*4, 3, 1, 1),
                nn.BatchNorm2d(64*4),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        #print("x: ", x.size())
        out1 = self.conv1(x)
        out = self.feature_extractor(x.expand(-1, 3, -1, -1))
        #print("feature extractor: ", out.size())
        #out = self.pixel_shuffle(out)
        #print("pixle shuffler: ", out.size())
        out = self.conv2(out)
        #print("out: ", out.size())
        #print("out1: ", out1.size())
        out = torch.add(out1, out)
        #print("conv1:", out.size())
        out = self.upsampling(out)
        #print("upsampling :", out.size())
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = ()
        self.output_shape = ()
        self.in_channels = 0
        self.calc_output_shape(input_shape)


        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = self.in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def calc_output_shape(self, input_shape):
        self.input_shape = input_shape
        self.in_channels, in_height, in_width = self.input_shape
        #print("in_channels: ", self.input_shape)
        #print(math.ceil(in_height / 2 ** 4), math.ceil(in_width / 2 ** 4))
        #print("")
        patch_h, patch_w = math.ceil(in_height / 2 ** 4), math.ceil(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)
        return self.output_shape

    def forward(self, img):
        return self.model(img)
