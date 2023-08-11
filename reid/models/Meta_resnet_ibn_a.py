import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


from .MetaModules import *
from .resnet_ibn_a import ResNet_IBN, Bottleneck_IBN
from .resnet_ibn_a import *
import  sys
sys.path.append('../')

__all__ =['meta_resnet50_ibn_b']
model_urls = {
    'ibn_resnet50a': '/home/ckx/QAconv/pretrained/resnet50_ibn_a.pth.tar',
    'ibn_resnet50b': '/home/ckx/QAconv/pretrained/resnet50_ibn_b-9ca61e85.pth',
}


class MetaIBN(MetaModule):
    def __init__(self, planes):
        super(MetaIBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = MetaInstanceNorm2d(half1, affine=True)
        self.BN = MetaBatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class MetaBottleneck_IBN(MetaModule):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(MetaBottleneck_IBN, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = MetaIBN(planes)
        else:
            self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * self.expansion)
        self.IN = MetaInstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class MetaResNet_IBN(MetaModule):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000):
        self.inplanes = 64
        super(MetaResNet_IBN, self).__init__()
        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = MetaInstanceNorm2d(64, affine=True)
        else:
            self.bn1 = MetaBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = MetaLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d) or isinstance(m, MetaInstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks-1) else ibn))

        return nn.Sequential(*layers)

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


from torchvision.models import resnet50, resnet34
def meta_resnet50_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MetaResNet_IBN(block=MetaBottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    model_nometa = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        state_dict = torch.load(model_urls['ibn_resnet50b'], map_location=torch.device('cpu'))
        state_dict = remove_module_key(state_dict)
        print("??????")
        print(state_dict.keys())
        # model_nometa.load_state_dict(state_dict)
        model.copyWeight(resnet50(pretrained=pretrained).state_dict())
        print(len(list(model.params())), len(list(model_nometa.parameters())))
        print(len(model.state_dict()))
    return model


def remove_module_key(state_dict):
    for key in list(state_dict.keys()):
        if 'module' in key:
            state_dict[key.replace('module.','')] = state_dict.pop(key)
    return state_dict
