
import torch
from torch import nn
from torch.nn import Module


class QAConv(Module):
    def __init__(self, num_features, height, width):
        """
        Inputs:
            num_features: the number of feature channels in the final feature map.
            height: height of the final feature map
            width: width of the final feature map
            num_features = neck = 128
            height = 384 // 16 = 24
            weight = 128 // 16 = 8
        """

        super(QAConv, self).__init__()
        self.num_features = num_features
        self.height = height
        self.width = width
        self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(self.height * self.width, 1)
        self.logit_bn = nn.BatchNorm1d(1)
        self.kernel = None
        self.reset_parameters()
        self.patch_size = 3
        self.padding_size = self.patch_size // 2
        self.pad_layer = torch.nn.ReflectionPad2d(self.padding_size)
        self.unfold = nn.Unfold(kernel_size=self.patch_size, padding=0, stride=1)


    def reset_running_stats(self):
        self.bn.reset_running_stats()
        self.logit_bn.reset_running_stats()

    def reset_parameters(self):
        self.bn.reset_parameters()
        self.logit_bn.reset_parameters()
        with torch.no_grad():
            self.fc.weight.fill_(1. / (self.height * self.width))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def make_kernel(self, features): # probe features
        self.kernel = features

    def forward(self, features):  # gallery features
        # if self.patch_size==1:
        #     score=self.pixel_forward(features)
        # else:
        #score=self.pixel_forward(features)
        score=self.patch_forward(features)
        #print(score2==score)
        return score

    def pixel_forward(self, features):
        self._check_input_dim(features)

        hw = self.height * self.width
        batch_size = features.size(0)

        score = torch.einsum('g c h w, p c y x -> g p y x h w', features, self.kernel)

        score = score.view(batch_size, -1, hw, hw)  # b,b,hw, hw
        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1)
        score = score.view(-1, 1, hw)  # bb*2 hw
        score = self.bn(score).view(-1, hw)  # bb*2 hw

        score = self.fc(score)  # bb*2 1
        score = score.view(-1, 2).sum(dim=-1, keepdim=True)  # bb * 1
        score = self.logit_bn(score)
        score = score.view(batch_size, -1).t()  # [p, g] #bb
        return score

    def patch_forward(self, features):
        # print("d")
        hw = self.height * self.width
        b,c,h,w = features.shape
        unfold_features = self.unfold(self.pad_layer(features)) # B, C*ps*ps, hw
        unfold_kernels = self.unfold(self.pad_layer(self.kernel))  # B, C*ps*ps, hw
        score = torch.einsum('g c n, p c m-> g p m n', unfold_features, unfold_kernels)
        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1)
        score = score.view(-1, 1, hw)  # bb*2 hw
        score = self.bn(score).view(-1, hw)  # bb*2 hw
        score = self.fc(score)  # bb*2 1
        score = score.view(-1, 2).sum(dim=-1, keepdim=True)  # bb * 1
        score = self.logit_bn(score)
        score = score.view(b, -1).t()  # [p, g] #bb
        return score





