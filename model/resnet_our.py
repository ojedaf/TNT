"""
This code was based on the file resnet.py (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
from the pytorch/vision library (https://github.com/pytorch/vision).

The original license is included below:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        
class BottleneckFilm(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckFilm, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, gamma1, beta1, gamma2, beta2, gamma3, beta3):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._film(out, gamma1, beta1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self._film(out, gamma2, beta2)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self._film(out, gamma3, beta3)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        
    def _film(self, x, gamma, beta):
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        return gamma * x + beta




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlockFilm(nn.Module):
    """
    Extension to standard ResNet block (https://arxiv.org/abs/1512.03385) with FiLM layer adaptation. After every batch
    normalization layer, we add a FiLM layer (which applies an affine transformation to each channel in the hidden
    representation). As we are adapting the feature extractor with an external adaptation network, we expect parameters
    to be passed as an argument of the forward pass.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockFilm, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        """
        Implements a forward pass through the FiLM adapted ResNet block. FiLM parameters for adaptation are passed
        through to the method, one gamma / beta set for each convolutional layer in the block (2 for the blocks we are
        working with).
        :param x: (torch.tensor) Batch of images to apply computation to.
        :param gamma1: (torch.tensor) Multiplicative FiLM parameter for first conv layer (one for each channel).
        :param beta1: (torch.tensor) Additive FiLM parameter for first conv layer (one for each channel).
        :param gamma2: (torch.tensor) Multiplicative FiLM parameter for second conv layer (one for each channel).
        :param beta2: (torch.tensor) Additive FiLM parameter for second conv layer (one for each channel).
        :return: (torch.tensor) Resulting representation after passing through layer.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._film(out, gamma1, beta1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self._film(out, gamma2, beta2)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _film(self, x, gamma, beta):
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        return gamma * x + beta

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, param_dict):
        # See note [TorchScript super()]
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

    def forward(self, x, param_dict=None):
        return self._forward_impl(x, param_dict)
    
    def get_layer_output(self, x, param_dict, layer_to_return):
        if layer_to_return == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return x
        else:
            resnet_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            layer = layer_to_return - 1
            for block in range(self.layers[layer]):
                x = resnet_layers[layer][block](x, param_dict[layer][block]['gamma1'], param_dict[layer][block]['beta1'],
                                       param_dict[layer][block]['gamma2'], param_dict[layer][block]['beta2'])
            return x

    
    @property
    def output_size(self):
        return 512

class FilmResNet(ResNet):
    """
    Wrapper object around BasicBlockFilm that constructs a complete ResNet with FiLM layer adaptation. Inherits from
    ResNet object, and works with identical logic.
    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        ResNet.__init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None)

        self.layers = layers
        self.type_block = block

    def forward(self, x, param_dict):
        """
        Forward pass through ResNet. Same logic as standard ResNet, but expects a dictionary of FiLM parameters to be
        provided (by adaptation network objects).
        :param x: (torch.tensor) Batch of images to pass through ResNet.
        :param param_dict: (list::dict::torch.tensor) One dictionary for each block in each layer of the ResNet,
                           containing the FiLM adaptation parameters for each conv layer in the model.
        :return: (torch.tensor) Feature representation after passing through adapted network.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for block in range(self.layers[0]):
            if self.type_block == BottleneckFilm:
                x = self.layer1[block](x, param_dict[0][block]['gamma1'], param_dict[0][block]['beta1'], param_dict[0][block]['gamma2'], param_dict[0][block]['beta2'], param_dict[0][block]['gamma3'], param_dict[0][block]['beta3'])
            else:   
                x = self.layer1[block](x, param_dict[0][block]['gamma1'], param_dict[0][block]['beta1'], param_dict[0][block]['gamma2'], param_dict[0][block]['beta2'])

        for block in range(self.layers[1]):
            if self.type_block == BottleneckFilm:
                x = self.layer2[block](x, param_dict[1][block]['gamma1'], param_dict[1][block]['beta1'], param_dict[1][block]['gamma2'], param_dict[1][block]['beta2'], param_dict[1][block]['gamma3'], param_dict[1][block]['beta3'])
            else:   
                x = self.layer2[block](x, param_dict[1][block]['gamma1'], param_dict[1][block]['beta1'], param_dict[1][block]['gamma2'], param_dict[1][block]['beta2'])

        for block in range(self.layers[2]):
            if self.type_block == BottleneckFilm:
                x = self.layer3[block](x, param_dict[2][block]['gamma1'], param_dict[2][block]['beta1'], param_dict[2][block]['gamma2'], param_dict[2][block]['beta2'], param_dict[2][block]['gamma3'], param_dict[2][block]['beta3'])
            else:   
                x = self.layer3[block](x, param_dict[2][block]['gamma1'], param_dict[2][block]['beta1'], param_dict[2][block]['gamma2'], param_dict[2][block]['beta2'])

        for block in range(self.layers[3]):
            if self.type_block == BottleneckFilm:
                x = self.layer4[block](x, param_dict[3][block]['gamma1'], param_dict[3][block]['beta1'], param_dict[3][block]['gamma2'], param_dict[3][block]['beta2'], param_dict[3][block]['gamma3'], param_dict[3][block]['beta3'])
            else:   
                x = self.layer4[block](x, param_dict[3][block]['gamma1'], param_dict[3][block]['beta1'], param_dict[3][block]['gamma2'], param_dict[3][block]['beta2'])
                                 

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    @property
    def output_size(self):
        if self.type_block == BottleneckFilm:
            return 2048
        else:
            return 512

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def _resnet_film(arch, block, layers, pretrained, progress, **kwargs):
    model = FilmResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    """ResNet-18 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    """ResNet-34 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def film_resnet18(pretrained=False, progress=True, **kwargs):
    """
        Constructs a FiLM adapted ResNet-18 model.
    """

    return _resnet_film('resnet18', BasicBlockFilm, [2, 2, 2, 2], pretrained, progress, **kwargs)

def film_resnet34(pretrained=False, progress=True, **kwargs):
    """
        Constructs a FiLM adapted ResNet-18 model.
    """

    return _resnet_film('resnet34', BasicBlockFilm, [3, 4, 6, 3], pretrained, progress, **kwargs)

def film_resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_film('resnet50', BottleneckFilm, [3, 4, 6, 3], pretrained, progress, **kwargs)

# import torch.nn as nn
# import torch


# __all__ = ['ResNet', 'resnet18']


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class BasicBlockFilm(nn.Module):
#     """
#     Extension to standard ResNet block (https://arxiv.org/abs/1512.03385) with FiLM layer adaptation. After every batch
#     normalization layer, we add a FiLM layer (which applies an affine transformation to each channel in the hidden
#     representation). As we are adapting the feature extractor with an external adaptation network, we expect parameters
#     to be passed as an argument of the forward pass.
#     """
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlockFilm, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x, gamma1, beta1, gamma2, beta2):
#         """
#         Implements a forward pass through the FiLM adapted ResNet block. FiLM parameters for adaptation are passed
#         through to the method, one gamma / beta set for each convolutional layer in the block (2 for the blocks we are
#         working with).
#         :param x: (torch.tensor) Batch of images to apply computation to.
#         :param gamma1: (torch.tensor) Multiplicative FiLM parameter for first conv layer (one for each channel).
#         :param beta1: (torch.tensor) Additive FiLM parameter for first conv layer (one for each channel).
#         :param gamma2: (torch.tensor) Multiplicative FiLM parameter for second conv layer (one for each channel).
#         :param beta2: (torch.tensor) Additive FiLM parameter for second conv layer (one for each channel).
#         :return: (torch.tensor) Resulting representation after passing through layer.
#         """
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self._film(out, gamma1, beta1)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self._film(out, gamma2, beta2)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

#     def _film(self, x, gamma, beta):
#         gamma = gamma[None, :, None, None]
#         beta = beta[None, :, None, None]
#         return gamma * x + beta


# class ResNet(nn.Module):

#     def __init__(self, block, layers):
#         super(ResNet, self).__init__()
#         self.initial_pool = False
#         inplanes = self.inplanes = 64
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, inplanes, layers[0])
#         self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x, param_dict=None):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         if self.initial_pool:
#             x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)

#         return x

#     def get_layer_output(self, x, param_dict, layer_to_return):
#         if layer_to_return == 0:
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x = self.relu(x)
#             if self.initial_pool:
#                 x = self.maxpool(x)
#             return x
#         else:
#             resnet_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
#             layer = layer_to_return - 1
#             for block in range(self.layers[layer]):
#                 x = resnet_layers[layer][block](x, param_dict[layer][block]['gamma1'], param_dict[layer][block]['beta1'],
#                                        param_dict[layer][block]['gamma2'], param_dict[layer][block]['beta2'])
#             return x

#     @property
#     def output_size(self):
#         return 512


# class FilmResNet(ResNet):
#     """
#     Wrapper object around BasicBlockFilm that constructs a complete ResNet with FiLM layer adaptation. Inherits from
#     ResNet object, and works with identical logic.
#     """

#     def __init__(self, block, layers):
#         ResNet.__init__(self, block, layers)
#         self.layers = layers

#     def forward(self, x, param_dict):
#         """
#         Forward pass through ResNet. Same logic as standard ResNet, but expects a dictionary of FiLM parameters to be
#         provided (by adaptation network objects).
#         :param x: (torch.tensor) Batch of images to pass through ResNet.
#         :param param_dict: (list::dict::torch.tensor) One dictionary for each block in each layer of the ResNet,
#                            containing the FiLM adaptation parameters for each conv layer in the model.
#         :return: (torch.tensor) Feature representation after passing through adapted network.
#         """
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         if self.initial_pool:
#             x = self.maxpool(x)

#         for block in range(self.layers[0]):
#             x = self.layer1[block](x, param_dict[0][block]['gamma1'], param_dict[0][block]['beta1'],
#                                    param_dict[0][block]['gamma2'], param_dict[0][block]['beta2'])
#         for block in range(self.layers[1]):
#             x = self.layer2[block](x, param_dict[1][block]['gamma1'], param_dict[1][block]['beta1'],
#                                    param_dict[1][block]['gamma2'], param_dict[1][block]['beta2'])
#         for block in range(self.layers[2]):
#             x = self.layer3[block](x, param_dict[2][block]['gamma1'], param_dict[2][block]['beta1'],
#                                    param_dict[2][block]['gamma2'], param_dict[2][block]['beta2'])
#         for block in range(self.layers[3]):
#             x = self.layer4[block](x, param_dict[3][block]['gamma1'], param_dict[3][block]['beta1'],
#                                    param_dict[3][block]['gamma2'], param_dict[3][block]['beta2'])

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)

#         return x


# def resnet18(pretrained=False, pretrained_model_path=None, **kwargs):
#     """
#         Constructs a ResNet-18 model.
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         ckpt_dict = torch.load(pretrained_model_path)
#         model.load_state_dict(ckpt_dict['state_dict'])
#     return model

# def resnet34(pretrained=False, pretrained_model_path=None, **kwargs):
#     """
#         Constructs a ResNet-34 model.
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         ckpt_dict = torch.load(pretrained_model_path)
#         model.load_state_dict(ckpt_dict['state_dict'])
#     return model

# def film_resnet18(pretrained=False, pretrained_model_path=None, **kwargs):
#     """
#         Constructs a FiLM adapted ResNet-18 model.
#     """

#     model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         ckpt_dict = torch.load(pretrained_model_path)
#         model.load_state_dict(ckpt_dict['state_dict'])
#     return model

# def film_resnet34(pretrained=False, pretrained_model_path=None, **kwargs):
#     """
#         Constructs a FiLM adapted ResNet-34 model.
#     """

#     model = FilmResNet(BasicBlockFilm, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         ckpt_dict = torch.load(pretrained_model_path)
#         model.load_state_dict(ckpt_dict['state_dict'])
#     return model
