import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Classes and functions required for Set encoding in adaptation networks. Many of the ideas and classes here are 
    closely related to DeepSets (https://arxiv.org/abs/1703.06114).
"""


def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)


class SetEncoder(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """
    def __init__(self, dynamic_mod_enable, output_dim_model):
        super(SetEncoder, self).__init__()
        self.pre_pooling_fn = SimplePrePoolNet()
        self.pooling_fn = mean_pooling
        self.dynamic_mod_enable = dynamic_mod_enable
        if self.dynamic_mod_enable == True:
            print('Added lineal to dynamic module')
            print("output dim model: ",output_dim_model)
            self.linear_dyna = nn.Sequential(
                nn.Linear(64, output_dim_model),
                nn.ReLU()
                )
        self.post_pooling_fn = Identity()

    def forward(self, x):
        """
        Forward pass through DeepSet SetEncoder. Implements the following computation:

                g(X) = rho ( mean ( phi(x) ) )
                Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
                and the mean is a pooling operation over elements in the set.

        :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x TC x H x W ).
        :return: (torch.tensor) Representation of the set, single vector in Rk.
        """
        x = x.view((x.size(0), -1, 3) + x.size()[-2:]).permute(0, 2, 1, 3, 4)
        x = self.pre_pooling_fn(x)
        prepooling_emb = x
        x = self.pooling_fn(x)
        x = self.post_pooling_fn(x)

        if self.dynamic_mod_enable == True:
            prepooling_emb = self.linear_dyna(prepooling_emb)
        return x, prepooling_emb

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AttModule(nn.Module):
    def __init__(self, ratio, temp_dim, emb):
        super(AttModule, self).__init__()
        out_temp_dim = int(temp_dim/ratio)

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(temp_dim, out_temp_dim),
            nn.ReLU(),
            nn.Linear(out_temp_dim, temp_dim)
        )

        self.maxPool = nn.MaxPool1d(emb, stride=emb)
        self.avgPool = nn.AvgPool1d(emb, stride=emb)
    
    def forward(self, x):

        x_max = self.maxPool(x)
        x_max = self.mlp(x_max)

        x_avg = self.avgPool(x)
        x_avg = self.mlp(x_avg)

        x_sum = x_max + x_avg

        scale = F.sigmoid(x_sum)

        return scale.unsqueeze(2)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimplePrePoolNet(nn.Module):
    """
    Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
    multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
    """
    def __init__(self):
        super(SimplePrePoolNet, self).__init__()
        # self.layer1 = self._make_conv2d_layer(3, 64)
        # self.layer2 = self._make_conv2d_layer(64, 64)
        # self.layer3 = self._make_conv2d_layer(64, 64)
        # self.layer4 = self._make_conv2d_layer(64, 64)
        # self.layer5 = self._make_conv2d_layer(64, 64)
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.layer1 = nn.Sequential(
                        nn.Conv3d(3,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                         nn.MaxPool3d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))


    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.layer5(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        return x

    @property
    def output_size(self):
        return 64