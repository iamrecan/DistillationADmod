import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from models.RememberingNormality.memoryModule import memoryModule
from models.utilsResnet import deconv2x2, deBasicBlock, deBottleneck, init_weights,RESNET_LAYERS

class deResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[deBasicBlock, deBottleneck]],
        layers: List[int],
        embedDim: int,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(deResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512 * block.expansion
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        
        if (block==deBasicBlock):
            self.memory0 = memoryModule(L=embedDim,channel=512) 
            self.memory1 = memoryModule(L=embedDim,channel=256)
            self.memory2=  memoryModule(L=embedDim,channel=128)
        else :
            self.memory0 = memoryModule(L=embedDim,channel=2048) 
            self.memory1 = memoryModule(L=embedDim,channel=1024)
            self.memory2=  memoryModule(L=embedDim,channel=512)

        init_weights(self)
    

    def _make_layer(self, block: Type[Union[deBasicBlock, deBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        inplanes=self.inplanes*2
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        
        x_mem=self.memory0(x)
        x_cat=torch.cat((x,x_mem),1)
        
        feature_a = self.layer1(x_cat)  # 512*8*8->256*16*16
        feature_mem_a = self.memory1(feature_a)
        feature_a_cat=torch.cat((feature_a,feature_mem_a),1)
        
        feature_b = self.layer2(feature_a_cat)  # 256*16*16->128*32*32
        feature_mem_b = self.memory2(feature_b)
        feature_b_cat=torch.cat((feature_b,feature_mem_b),1)
        
        feature_c = self.layer3(feature_b_cat)  # 128*32*32->64*64*64

        return [feature_c, feature_b, feature_a]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _deresnet(block: Type[Union[deBasicBlock, deBottleneck]],layers: List[int],embedDim: int,**kwargs: Any) -> deResNet:
    model = deResNet(block, layers,embedDim, **kwargs)
    return model


def de_resnetMemory(backbone_name,embedDim=50,**kwargs):
    if backbone_name=="resnet18" or backbone_name=="resnet34":
        return _deresnet(deBasicBlock, RESNET_LAYERS[backbone_name],embedDim)
    elif backbone_name=="resnet50" or backbone_name=="resnet101" or backbone_name=="resnet152":
        return _deresnet(deBottleneck, RESNET_LAYERS[backbone_name],embedDim)
    elif backbone_name=="wide_resnet50_2" or backbone_name=="wide_resnet101_2":
        kwargs['width_per_group'] = 64 * 2
        return _deresnet(deBottleneck, RESNET_LAYERS[backbone_name],embedDim,**kwargs)
    else:
        raise Exception("Invalid model name :  Choices are ['resnet18', 'resnet34','resnet50', \
                        'resnet101', 'resnet152', 'wide_resnet50_2', 'wide_resnet101_2']")